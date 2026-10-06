"""n107-hc: Triton kernels for SGLang's hyper-connection path (layers/hyperconnection.py, GatedResidual) on Intel XPU.

Model: Qwen3.8-Flash-Next (qwen4_exp), hc_count 4 x hidden 2560 = 10240-wide bf16 residual stream, hc_lowrank 320,
hc_per_branch_norm=True (GroupedGemmaRMSNorm(10240, group_size=2560), one weight of 10240 = per-branch weights).

Reference semantics reproduced here (what runs on XPU today, SGLang 0.5.20 image 24c872759256):

  norm     GroupedGemmaRMSNorm.forward eager path (the JIT kernel is gated on x.is_cuda):
             xf = x.float(); var = xf.view(.., G).pow(2).mean(-1)      # torch mean = sum * (1/G), fp32
             y  = (xf * rsqrt(var + eps)) * (1.0 + w.float())          # fp32, then .to(x.dtype)
  mix      torch.compile(_mix_compute) (inductor; fused_hc_mix / hc_mix JIT are is_cuda/sm100 gated):
             t = F.linear(n, Wd)                   extern mm, bf16 out
             a = silu(t.f32 / hc) -> bf16          one inductor kernel, single rounding (feeds an extern mm)
             u = F.linear(a, Wu)                   extern mm, bf16 out
             o = (sum_j sigmoid(u_j.f32) * n_j.f32) / hc -> bf16     one inductor reduction kernel, no
                                                                       intermediate rounding
  combine  torch.compile(_combine_compute):
             l = F.linear(n, Wi)                   extern mm, bf16 out [M, hc]
             g = 2 * sigmoid(l.f32 / hc) -> bf16   inductor REALIZES g as a bf16 buffer (in place on the mm output;
                                                   checked in the generated code, torch 2.10 CPU inductor)
             o = (R.f32 + bo.f32 * g.f32) -> bf16  one pointwise kernel

  The rounding points above were verified offline: a pure-torch emulation with exactly these roundings is bit-identical
  to torch.compile(_mix_compute) / torch.compile(_combine_compute) on CPU inductor (100.000 % of elements); the eager
  (uncompiled) functions differ from the compiled ones in ~49 % (mix) / ~15 % (combine) of elements by 1 bf16 ulp.

Kernels (all fp32 math, one rounding to the I/O dtype at the store):
  gemma_rmsnorm_grouped  one read of x, one write of y; program = (row, group); single pass when the group fits a
                         register block (G=2560 -> BLOCK 4096 masked), else a two-pass loop (2nd pass hits cache).
  hc_mix(mode="epilogue") down/up GEMMs stay in torch (oneDNN, identical calls to the reference -> identical bf16
                         GEMM outputs); Triton does silu(t/hc) and the sigmoid*x mean-over-hc epilogue.
  hc_mix(mode="fused")   down GEMM in torch; Triton does silu, the up-projection with tl.dot (K = lowrank = 320) per
                         branch and the epilogue in registers, so the [M, 10240] up-projection output never touches
                         memory (-336 MB per mix at M=8192). The fp32 dot result is rounded to bf16 before sigmoid to keep
                         the reference rounding point; the GEMM summation order differs from oneDNN.
  hc_combine(inject="torch") the 10240->hc GEMM in torch (identical to the reference), Triton does the rest in one pass.
  hc_combine(inject="fused") the 10240->hc GEMM inside the kernel (broadcast-multiply + tl.sum per K chunk).

Block sizes target Xe2 (B70 / BMG-G31): small register tiles (<= 32 fp32 accumulator values per lane for the dot kernel
at num_warps 32 / 16-wide subgroups), no autotune (autotune would benchmark inside graph capture / first serve call);
every launch config can be overridden by env (see _env_cfg) for sweeps. Any M >= 1 works (decode M=1..4 included).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:  # import-safe without Triton (lets patch_hc fall back cleanly and lets the reference code run anywhere)
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False

_IO_DTYPES = (torch.bfloat16, torch.float16)


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _env_cfg(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """EXL3_HC_<NAME>=a,b,c overrides a launch config tuple (same arity)."""
    v = os.environ.get(f"EXL3_HC_{name}")
    if not v:
        return default
    t = tuple(int(s) for s in v.split(","))
    if len(t) != len(default):
        raise ValueError(f"EXL3_HC_{name}={v!r}: expected {len(default)} ints")
    return t


def _launch_kw(num_warps: int) -> dict:
    kw = {"num_warps": int(num_warps)}
    tpw = os.environ.get("EXL3_HC_TPW")  # 16 or 32 (Intel backend 'threads_per_warp'); unset = backend default
    if tpw and os.environ.get("TRITON_INTERPRET", "0") != "1":
        kw["threads_per_warp"] = int(tpw)
    return kw


def _need_triton():
    if not HAVE_TRITON:
        raise RuntimeError("hc_xpu: triton is not importable")


# ---------------------------------------------------------------------------------------------------------------------
# Kernels

if HAVE_TRITON:

    @triton.jit
    def _gemma_rmsnorm_kernel(X, W, Y, stride_x, stride_y, G, inv_g, eps,
                              BLOCK: tl.constexpr, SINGLE: tl.constexpr):
        # program (row, group): y[row, g*G:(g+1)*G] = x * rsqrt(mean(x^2) + eps) * (1 + w)
        row = tl.program_id(0).to(tl.int64)
        grp = tl.program_id(1)
        col0 = grp * G
        xb = X + row * stride_x + col0
        yb = Y + row * stride_y + col0
        wb = W + col0
        if SINGLE:
            offs = tl.arange(0, BLOCK)
            m = offs < G
            x = tl.load(xb + offs, mask=m, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=0) * inv_g
            r = tl.math.rsqrt(var + eps)
            w = tl.load(wb + offs, mask=m, other=0.0).to(tl.float32)
            y = (x * r) * (1.0 + w)
            tl.store(yb + offs, y.to(Y.dtype.element_ty), mask=m)
        else:
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for c in range(0, G, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                x = tl.load(xb + offs, mask=offs < G, other=0.0).to(tl.float32)
                acc += x * x
            var = tl.sum(acc, axis=0) * inv_g
            r = tl.math.rsqrt(var + eps)
            for c in range(0, G, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                m = offs < G
                x = tl.load(xb + offs, mask=m, other=0.0).to(tl.float32)
                w = tl.load(wb + offs, mask=m, other=0.0).to(tl.float32)
                y = (x * r) * (1.0 + w)
                tl.store(yb + offs, y.to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _silu_div_kernel(T, A, n, HC: tl.constexpr, BLOCK: tl.constexpr):
        # a = silu(t / hc), fp32 math, one rounding (inductor: x * sigmoid(x))
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        t = tl.load(T + offs, mask=m, other=0.0).to(tl.float32)
        t = t / HC
        a = t * tl.sigmoid(t)
        tl.store(A + offs, a.to(A.dtype.element_ty), mask=m)

    @triton.jit
    def _mix_epilogue_kernel(U, X, O, M, stride_u, stride_x, stride_o, HS,
                             HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        # o[m, n] = (sum_j sigmoid(u[m, j*HS+n]) * x[m, j*HS+n]) / HC
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        m = (rows < M)[:, None] & (cols < HS)[None, :]
        r64 = rows.to(tl.int64)[:, None]
        c = cols[None, :]
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for j in tl.static_range(HC):
            u = tl.load(U + r64 * stride_u + j * HS + c, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + r64 * stride_x + j * HS + c, mask=m, other=0.0).to(tl.float32)
            acc += tl.sigmoid(u) * x
        o = acc / HC
        tl.store(O + r64 * stride_o + c, o.to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _mix_up_fused_kernel(A, WU, X, O, M, LR, stride_a, stride_x, stride_o, HS,
                             HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                             APPLY_SILU: tl.constexpr, ROUND_U: tl.constexpr, USE_BP: tl.constexpr):
        # For each branch j: u_j = a[BM, LR] @ WU[j*HS + n, :]^T  (tl.dot, fp32 acc), round u_j to the I/O dtype
        # (reference: extern mm with bf16 output), then o += sigmoid(u_j) * x_j.  o /= HC.
        # A is t (APPLY_SILU: silu(t/HC) applied on load) or the already activated a.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < M
        cmask = cols < HS
        r64 = rows.to(tl.int64)
        out = tl.zeros([BM, BN], dtype=tl.float32)
        for j in tl.static_range(HC):
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            if USE_BP:
                a_bp = tl.make_block_ptr(A, shape=(M, LR), strides=(stride_a, 1), offsets=(pid_m * BM, 0),
                                         block_shape=(BM, BK), order=(1, 0))
                # W_up is [HC*HS, LR] row-major; B = W_up^T viewed as [LR, HC*HS] (column-major)
                b_bp = tl.make_block_ptr(WU, shape=(LR, HC * HS), strides=(1, LR), offsets=(0, j * HS + pid_n * BN),
                                         block_shape=(BK, BN), order=(0, 1))
                for k in range(0, LR, BK):
                    a = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
                    if APPLY_SILU:
                        af = a.to(tl.float32) / HC
                        a = (af * tl.sigmoid(af)).to(WU.dtype.element_ty)
                    b = tl.load(b_bp, boundary_check=(0, 1), padding_option="zero")
                    acc = tl.dot(a, b, acc)
                    a_bp = tl.advance(a_bp, (0, BK))
                    b_bp = tl.advance(b_bp, (BK, 0))
            else:
                wrow = (j * HS + cols).to(tl.int64)
                for k in range(0, LR, BK):
                    ks = k + tl.arange(0, BK)
                    kmask = ks < LR
                    a = tl.load(A + r64[:, None] * stride_a + ks[None, :],
                                mask=rmask[:, None] & kmask[None, :], other=0.0)
                    if APPLY_SILU:
                        af = a.to(tl.float32) / HC
                        a = (af * tl.sigmoid(af)).to(WU.dtype.element_ty)
                    b = tl.load(WU + wrow[None, :] * LR + ks[:, None],
                                mask=kmask[:, None] & cmask[None, :], other=0.0)
                    acc = tl.dot(a, b, acc)
            if ROUND_U:
                acc = acc.to(O.dtype.element_ty).to(tl.float32)
            x = tl.load(X + r64[:, None] * stride_x + j * HS + cols[None, :],
                        mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            out += tl.sigmoid(acc) * x
        o = out / HC
        tl.store(O + r64[:, None] * stride_o + cols[None, :], o.to(O.dtype.element_ty),
                 mask=rmask[:, None] & cmask[None, :])

    @triton.jit
    def _combine_kernel(R, BO, L, O, M, stride_r, stride_b, stride_l, stride_o, HS,
                        HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, ROUND_G: tl.constexpr):
        # o[m, j*HS+n] = R[m, j*HS+n] + bo[m, n] * g[m, j],  g = 2*sigmoid(l[m, j] / HC)  (g rounded like inductor)
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        rmask = rows < M
        m = rmask[:, None] & (cols < HS)[None, :]
        r64 = rows.to(tl.int64)
        c = cols[None, :]
        bo = tl.load(BO + r64[:, None] * stride_b + c, mask=m, other=0.0).to(tl.float32)
        for j in tl.static_range(HC):
            lj = tl.load(L + r64 * stride_l + j, mask=rmask, other=0.0).to(tl.float32)
            g = tl.sigmoid(lj / HC) * 2.0
            if ROUND_G:
                g = g.to(O.dtype.element_ty).to(tl.float32)
            r = tl.load(R + r64[:, None] * stride_r + j * HS + c, mask=m, other=0.0).to(tl.float32)
            o = r + bo * g[:, None]
            tl.store(O + r64[:, None] * stride_o + j * HS + c, o.to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _combine_fused_kernel(R, BO, N, WI, O, M, K, stride_r, stride_b, stride_n, stride_o, HS,
                              HC: tl.constexpr, HCP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                              BK: tl.constexpr, NSPLIT: tl.constexpr, ROUND_L: tl.constexpr, ROUND_G: tl.constexpr):
        # Same as _combine_kernel, but l = normed @ WI^T is computed here (fp32, K chunks of BK).
        # Program = BM rows x a 1/NSPLIT share of the HS columns (each program recomputes l for its rows).
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rmask = rows < M
        r64 = rows.to(tl.int64)
        jj = tl.arange(0, HCP)
        jmask = jj < HC
        lacc = tl.zeros([BM, HCP], dtype=tl.float32)
        for k in range(0, K, BK):
            ks = k + tl.arange(0, BK)
            kmask = ks < K
            nv = tl.load(N + r64[:, None] * stride_n + ks[None, :], mask=rmask[:, None] & kmask[None, :],
                         other=0.0).to(tl.float32)
            wv = tl.load(WI + jj[:, None].to(tl.int64) * K + ks[None, :], mask=jmask[:, None] & kmask[None, :],
                         other=0.0).to(tl.float32)
            lacc += tl.sum(nv[:, None, :] * wv[None, :, :], axis=2)
        if ROUND_L:
            lacc = lacc.to(O.dtype.element_ty).to(tl.float32)
        g = tl.sigmoid(lacc / HC) * 2.0
        if ROUND_G:
            g = g.to(O.dtype.element_ty).to(tl.float32)
        for n0 in range(tl.program_id(1) * BN, HS, NSPLIT * BN):
            cols = n0 + tl.arange(0, BN)
            m = rmask[:, None] & (cols < HS)[None, :]
            c = cols[None, :]
            bo = tl.load(BO + r64[:, None] * stride_b + c, mask=m, other=0.0).to(tl.float32)
            for j in tl.static_range(HC):
                gj = tl.sum(tl.where(jj[None, :] == j, g, 0.0), axis=1)
                r = tl.load(R + r64[:, None] * stride_r + j * HS + c, mask=m, other=0.0).to(tl.float32)
                o = r + bo * gj[:, None]
                tl.store(O + r64[:, None] * stride_o + j * HS + c, o.to(O.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------------------------------------------------
# Host wrappers

def _rows2d(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.reshape(-1, n)


def gemma_rmsnorm_grouped(x: torch.Tensor, weight: torch.Tensor, group_size: Optional[int], eps: float,
                          block: Optional[int] = None, num_warps: Optional[int] = None) -> torch.Tensor:
    """GroupedGemmaRMSNorm.forward (eager semantics) for contiguous x [..., N], weight [N], group_size | N or None."""
    _need_triton()
    N = x.shape[-1]
    G = N if group_size is None else int(group_size)
    assert weight.numel() == N and N % G == 0 and x.is_contiguous()
    x2 = _rows2d(x, N)
    rows = x2.shape[0]
    y = torch.empty_like(x)
    if rows == 0:
        return y
    y2 = _rows2d(y, N)
    d_block, d_warps = _env_cfg("NORM_CFG", (0, 0))
    block = block or d_block or (_next_pow2(G) if _next_pow2(G) <= 8192 else 2048)
    single = block >= G
    if num_warps is None:
        num_warps = d_warps or max(1, min(16, block // 512))
    inv_g = float(torch.tensor(1.0 / G, dtype=torch.float32).item())  # torch mean factor, rounded to fp32
    w = weight if weight.is_contiguous() else weight.contiguous()
    _gemma_rmsnorm_kernel[(rows, N // G)](x2, w, y2, x2.stride(0), y2.stride(0), G, inv_g, float(eps),
                                          BLOCK=block, SINGLE=single, **_launch_kw(num_warps))
    return y


def silu_div(t: torch.Tensor, hc: int) -> torch.Tensor:
    """silu(t / hc) with fp32 math and a single rounding (= inductor's fused kernel)."""
    _need_triton()
    t = t.contiguous()
    a = torch.empty_like(t)
    n = t.numel()
    if n:
        BLOCK = 1024
        _silu_div_kernel[(triton.cdiv(n, BLOCK),)](t, a, n, HC=int(hc), BLOCK=BLOCK, **_launch_kw(4))
    return a


def mix_epilogue(u: torch.Tensor, x: torch.Tensor, hc: int, hs: int) -> torch.Tensor:
    """(sigmoid(u) * x).unflatten(-1, (hc, hs)).mean(-2) with fp32 math, one rounding. u, x: [M, hc*hs]."""
    _need_triton()
    M = u.shape[0]
    o = torch.empty((M, hs), dtype=x.dtype, device=x.device)
    if M == 0:
        return o
    BM, BN, W = _env_cfg("EPI_CFG", (1, 256, 4) if M < 64 else (4, 512, 4))
    grid = (triton.cdiv(M, BM), triton.cdiv(hs, BN))
    _mix_epilogue_kernel[grid](u, x, o, M, u.stride(0), x.stride(0), o.stride(0), hs,
                               HC=int(hc), BM=BM, BN=BN, **_launch_kw(W))
    return o


def _up_cfg(M: int) -> Tuple[int, int, int, int]:
    if M <= 16:
        d = (16, 64, 32, 4)
    elif M <= 64:
        d = (32, 64, 32, 4)
    elif M <= 512:
        d = (64, 128, 32, 16)
    else:
        d = (128, 128, 32, 32)
    return _env_cfg("UP_CFG", d)


def mix_up_fused(a: torch.Tensor, w_up: torch.Tensor, x: torch.Tensor, hc: int, hs: int,
                 apply_silu: bool = False, round_u: bool = True, use_bp: Optional[bool] = None,
                 cfg: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
    """o = mean_j sigmoid(bf16(a @ w_up[j*hs:(j+1)*hs].T)) * x_j ;  a [M, LR], w_up [hc*hs, LR], x [M, hc*hs]."""
    _need_triton()
    M, LR = a.shape
    assert w_up.shape == (hc * hs, LR) and w_up.is_contiguous() and a.stride(1) == 1 and x.stride(1) == 1
    o = torch.empty((M, hs), dtype=x.dtype, device=x.device)
    if M == 0:
        return o
    BM, BN, BK, W = cfg or _up_cfg(M)
    if use_bp is None:
        use_bp = os.environ.get("EXL3_HC_UP_BLOCKPTR", "1") == "1"
    grid = (triton.cdiv(M, BM), triton.cdiv(hs, BN))
    _mix_up_fused_kernel[grid](a, w_up, x, o, M, LR, a.stride(0), x.stride(0), o.stride(0), hs,
                               HC=int(hc), BM=BM, BN=BN, BK=BK, APPLY_SILU=bool(apply_silu),
                               ROUND_U=bool(round_u), USE_BP=bool(use_bp), **_launch_kw(W))
    return o


def hc_mix(normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, hc: int, hs: int,
           mode: str = "epilogue") -> torch.Tensor:
    """Drop-in for GatedResidual._mix_compute (same args, returns [..., hs] in normed.dtype)."""
    lead = normed.shape[:-1]
    n2 = _rows2d(normed, hc * hs)
    t = F.linear(n2, w_down)                       # identical call to the reference -> identical bf16 output
    if mode == "fused":
        a = silu_div(t, hc)
        o = mix_up_fused(a, w_up.contiguous(), n2, hc, hs)
    elif mode == "fused_silu":                     # silu applied inside the dot kernel (no [M, LR] round trip)
        o = mix_up_fused(t, w_up.contiguous(), n2, hc, hs, apply_silu=True)
    elif mode == "epilogue":
        a = silu_div(t, hc)
        u = F.linear(a, w_up)                      # identical call to the reference
        o = mix_epilogue(u, n2, hc, hs)
    else:
        raise ValueError(f"hc_mix mode {mode!r}")
    return o.reshape(*lead, hs)


def hc_combine(block_output: torch.Tensor, residual: torch.Tensor, normed: torch.Tensor, w_inject: torch.Tensor,
               hc: int, hs: int, inject: str = "torch", round_g: bool = True) -> torch.Tensor:
    """Drop-in for GatedResidual._combine_compute (same args, returns [..., hc*hs])."""
    _need_triton()
    lead = residual.shape[:-1]
    R = _rows2d(residual, hc * hs)
    n2 = _rows2d(normed, hc * hs)
    bo = _rows2d(block_output, hs)
    M = R.shape[0]
    o = torch.empty((M, hc * hs), dtype=residual.dtype, device=residual.device)
    if M == 0:
        return o.reshape(*lead, hc * hs)
    if inject == "torch":
        lg = F.linear(n2, w_inject)                # identical call to the reference, [M, hc]
        BM, BN, W = _env_cfg("COMB_CFG", (1, 256, 4) if M < 64 else (4, 512, 4))
        grid = (triton.cdiv(M, BM), triton.cdiv(hs, BN))
        _combine_kernel[grid](R, bo, lg, o, M, R.stride(0), bo.stride(0), lg.stride(0), o.stride(0), hs,
                              HC=int(hc), BM=BM, BN=BN, ROUND_G=bool(round_g), **_launch_kw(W))
    elif inject == "fused":
        wi = w_inject.contiguous()
        K = hc * hs
        if M < 64:
            d = (1, 256, 512, triton.cdiv(hs, 256), 4)      # many column programs, l recomputed per program
        else:
            d = (4, 512, 256, 1, 4)
        BM, BN, BK, NSPLIT, W = _env_cfg("COMBF_CFG", d)
        grid = (triton.cdiv(M, BM), NSPLIT)
        _combine_fused_kernel[grid](R, bo, n2, wi, o, M, K, R.stride(0), bo.stride(0), n2.stride(0), o.stride(0), hs,
                                    HC=int(hc), HCP=_next_pow2(hc), BM=BM, BN=BN, BK=BK, NSPLIT=NSPLIT,
                                    ROUND_L=True, ROUND_G=bool(round_g), **_launch_kw(W))
    else:
        raise ValueError(f"hc_combine inject {inject!r}")
    return o.reshape(*lead, hc * hs)


# ---------------------------------------------------------------------------------------------------------------------
# Support checks used by patch_hc (anything else -> original path)

def norm_supported(x: torch.Tensor, weight: torch.Tensor, group_size: Optional[int]) -> bool:
    if not HAVE_TRITON or x.device.type != "xpu" or x.dtype not in _IO_DTYPES or x.numel() == 0:
        return False
    N = x.shape[-1]
    G = N if group_size is None else group_size
    return (x.is_contiguous() and weight.device == x.device and weight.numel() == N and N % G == 0
            and weight.dtype in (torch.float32, torch.bfloat16, torch.float16))


def mix_supported(normed, w_down, w_up, hc, hs) -> bool:
    if not HAVE_TRITON or normed.device.type != "xpu" or normed.dtype not in _IO_DTYPES:
        return False
    if normed.shape[-1] != hc * hs or not normed.is_contiguous() or normed.numel() == 0:
        return False
    lr = w_down.shape[0]
    return (w_down.dtype == normed.dtype and w_up.dtype == normed.dtype and w_down.shape == (lr, hc * hs)
            and w_up.shape == (hc * hs, lr) and w_up.is_contiguous() and 0 < lr <= 4096
            and w_down.device == normed.device and w_up.device == normed.device)


def combine_supported(block_output, residual, normed, w_inject, hc, hs) -> bool:
    if not HAVE_TRITON or residual.device.type != "xpu" or residual.dtype not in _IO_DTYPES:
        return False
    if not (block_output.dtype == residual.dtype == normed.dtype == w_inject.dtype):
        return False
    if residual.shape[-1] != hc * hs or normed.shape != residual.shape or block_output.shape[-1] != hs:
        return False
    if block_output.shape[:-1] != residual.shape[:-1] or w_inject.shape != (hc, hc * hs):
        return False
    return (residual.is_contiguous() and normed.is_contiguous() and block_output.is_contiguous()
            and residual.numel() > 0 and block_output.device == residual.device == normed.device == w_inject.device)
