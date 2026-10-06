#!/usr/bin/env python3
"""n107-hc test: hc_xpu Triton kernels vs the exact PyTorch reference of SGLang layers/hyperconnection.py.

  XPU (B70):  python3 test_hc_xpu.py                      M in {1,2,4,8,256,8192}, correctness + timing + graph + patch
  CPU:        TRITON_INTERPRET=1 python3 test_hc_xpu.py --device cpu     (M in {1,2,4,8}; correctness only)
  options:    --ms 1,8,8192   --iters 20   --sweep (launch-config sweep at the largest M)   --json PATH
              --no-graph   --no-patch   --hc 4 --hs 2560 --lr 320

References (copied verbatim from hyperconnection.py, SGLang 0.5.20 in image 24c872759256; drift is checked against the
installed sglang when importable):
  norm      GroupedGemmaRMSNorm.forward eager branch                                  (what XPU runs today)
  mix       torch.compile(_mix_compute)  [+ eager _mix_compute for the noise floor]   (what XPU runs today)
  combine   torch.compile(_combine_compute) [+ eager]                                 (what XPU runs today)
Pass rules: norm  max |ulp| <= 1 vs eager (fp32 reduction order / rsqrt may move a bf16 rounding by one step);
            mix/combine  bit-equal, or max|ours - compiled| <= 2 * max|eager - compiled| (the reference's own
            eager-vs-compiled spread is the noise floor), and max ulp reported.
Exit status 0 = all PASS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=None, help="xpu (default when available) or cpu (TRITON_INTERPRET=1)")
ap.add_argument("--ms", default=None)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--no-patch", action="store_true")
ap.add_argument("--no-compile", action="store_true", help="skip torch.compile references (use the emulation)")
ap.add_argument("--hc", type=int, default=4)
ap.add_argument("--hs", type=int, default=2560)
ap.add_argument("--lr", type=int, default=320)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default=None)
args = ap.parse_args()

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
DEV = args.device or ("xpu" if HAS_XPU else "cpu")
if DEV == "cpu":
    os.environ.setdefault("TRITON_INTERPRET", "1")   # must be set before triton is imported
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
MS = [int(s) for s in args.ms.split(",")] if args.ms else ([1, 2, 4, 8, 256, 8192] if DEV == "xpu" else [1, 2, 4, 8])

import hc_xpu  # noqa: E402  (after TRITON_INTERPRET)


def _interpreter_bf16_shims():
    """Test-only fixes for two Triton-interpreter (3.7) bf16 bugs; compiled kernels are unaffected:
      * fp32 -> bf16 casts truncate (cast_impl passes rounding None) instead of round-to-nearest-even;
      * tl.dot on bf16 operands runs np.matmul on the raw uint16 bit patterns.
    Without them every bf16 output differs by up to 1 ulp and bf16 tl.dot results are garbage under the interpreter."""
    import numpy as np
    from triton.runtime import interpreter as I
    B = I.InterpreterBuilder
    if getattr(B, "_n107_shim", False):
        return True

    def to_f32(h):
        if h.dtype.scalar == tl_.bfloat16:
            return (np.asarray(h.data).astype(np.uint32) << 16).view(np.float32)
        return h.data

    orig_cast = B.cast_impl
    orig_dot = B.create_dot

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl_.float32 and dst_type.scalar == tl_.bfloat16:
            a = np.ascontiguousarray(np.asarray(src.data, dtype=np.float32))
            u = a.view(np.uint32).astype(np.uint64)
            r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            r = np.where(np.isnan(a), np.uint16(0x7FC0), r).astype(np.uint16)
            return I.TensorHandle(r, dst_type.scalar)
        return orig_cast(self, src, dst_type)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype.scalar == tl_.bfloat16 or b.dtype.scalar == tl_.bfloat16:
            return I.TensorHandle(np.matmul(to_f32(a), to_f32(b), dtype=np.float32) + d.data, d.dtype.scalar)
        return orig_dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    B.cast_impl = cast_impl
    B.create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    B.create_dot = create_dot
    B._n107_shim = True
    return True


if INTERP:
    import triton.language as tl_
    try:
        _interpreter_bf16_shims()
    except Exception as _e:  # pragma: no cover
        print(f"WARNING: interpreter bf16 shims not applied ({_e}); expect 1-ulp rounding diffs and bad bf16 dots",
              flush=True)

HC, HS, LR = args.hc, args.hs, args.lr
N = HC * HS
EPS = 1e-6
BF = torch.bfloat16
for _k in ("recompile_limit", "cache_size_limit"):   # one compile per M is expected; keep them all cached
    try:
        setattr(torch._dynamo.config, _k, max(64, getattr(torch._dynamo.config, _k)))
    except Exception:
        pass
RESULTS = {"device": DEV, "interpret": INTERP, "torch": torch.__version__, "hc": HC, "hs": HS, "lr": LR,
           "triton": getattr(hc_xpu.triton, "__version__", None), "rows": [], "timing": [], "notes": []}
FAILS = []


def note(s):
    print(s, flush=True)
    RESULTS["notes"].append(s)


# ---------------------------------------------------------------------------------------------------------------------
# Verbatim references (hyperconnection.py)

def ref_norm_eager(x, weight, group_size, variance_epsilon):
    # GroupedGemmaRMSNorm.forward, non-CUDA branch, verbatim
    input_dtype = x.dtype
    x_float = x.float()
    if group_size is None:
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + variance_epsilon)
    else:
        x_grouped = x_float.reshape(
            *x_float.shape[:-1],
            x_float.shape[-1] // group_size,
            group_size,
        )
        variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
        x_norm = (
            x_grouped * torch.rsqrt(variance + variance_epsilon)
        ).flatten(-2)
    return (x_norm * (1.0 + weight.float())).to(input_dtype)


def _mix_compute(
    hyper_input_normed: torch.Tensor,
    input_mix_weight_down: torch.Tensor,
    input_mix_weight_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    input_mix_weight = F.silu(
        F.linear(hyper_input_normed, input_mix_weight_down) / hc
    )
    input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
    input_mix_weight = torch.sigmoid(input_mix_weight)
    input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
    output = (
        input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
    ).mean(dim=-2)
    return output


def _combine_compute(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    block_inject_weight: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    R = residual.unflatten(-1, (hc, hs))
    block_inject_weight_out = 2 * torch.sigmoid(
        F.linear(normed_residual, block_inject_weight) / hc
    )
    injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
        -1
    )
    return (R + injection).flatten(-2)


# What inductor computes (rounding points verified bit-exact against torch.compile on CPU inductor, torch 2.10):
def emul_mix(n, wd, wu, hc, hs):
    t = F.linear(n, wd).float() / hc
    a = (t * torch.sigmoid(t)).to(n.dtype)
    s = (torch.sigmoid(F.linear(a, wu).float()) * n.float()).unflatten(-1, (hc, hs))
    acc = s[..., 0, :]
    for j in range(1, hc):
        acc = acc + s[..., j, :]
    return (acc / hc).to(n.dtype)


def emul_combine(bo, R, n, wi, hc, hs):
    g = (torch.sigmoid(F.linear(n, wi).float() / hc) * 2.0).to(bo.dtype).float()
    out = R.float().unflatten(-1, (hc, hs)) + bo.float().unsqueeze(-2) * g.unsqueeze(-1)
    return out.flatten(-2).to(bo.dtype)


def check_ref_drift():
    try:
        import inspect
        import sglang.srt.layers.hyperconnection as h
        src = inspect.getsource(h)
    except Exception as e:
        note(f"[drift] sglang hyperconnection not importable here ({type(e).__name__}); drift check skipped")
        return None
    keys = [
        "variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)",
        "return (x_norm * (1.0 + self.weight.float())).to(input_dtype)",
        "F.linear(hyper_input_normed, input_mix_weight_down) / hc",
        "input_mix_weight = torch.sigmoid(input_mix_weight)",
        "input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))",
        "block_inject_weight_out = 2 * torch.sigmoid(",
        "return (R + injection).flatten(-2)",
        "self._mix_compute = torch.compile(_mix_compute)",
        "self._combine_compute = torch.compile(_combine_compute)",
        "and x.is_cuda",
    ]
    missing = [k for k in keys if k not in src]
    if missing:
        note(f"[drift] WARNING installed hyperconnection.py differs from the reference copy: missing {missing}")
        FAILS.append("ref_drift")
    else:
        note(f"[drift] installed {h.__file__} matches the reference copy ({len(keys)} anchor lines)")
    return not missing


# ---------------------------------------------------------------------------------------------------------------------
# Metrics

def _ord(t):
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        u = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        mag = u & 0x7FFF
        return torch.where((u & 0x8000) != 0, -mag, mag)
    raise TypeError(t.dtype)


def compare(name, M, out, ref, floor=None, rule="floor"):
    out = out.detach()
    ref = ref.detach()
    d = (out.float() - ref.float()).abs()
    max_abs = d.max().item() if d.numel() else 0.0
    denom = ref.float().abs().clamp_min(1e-6)
    max_rel = (d / denom).max().item() if d.numel() else 0.0
    bit = (_ord(out) == _ord(ref)).float().mean().item() if out.numel() else 1.0
    ulp = (_ord(out) - _ord(ref)).abs()
    max_ulp = ulp.max().item() if ulp.numel() else 0
    finite = bool(torch.isfinite(out.float()).all().item())
    if rule == "ulp1":
        ok = finite and max_ulp <= 1
    elif rule == "exact":
        ok = finite and bit == 1.0
    else:
        ok = finite and (bit == 1.0 or (floor is not None and max_abs <= 2.0 * floor + 1e-6))
    row = {"check": name, "M": M, "max_abs": max_abs, "max_rel": max_rel, "bit_equal": bit, "max_ulp": int(max_ulp),
           "floor": floor, "rule": rule, "pass": ok}
    RESULTS["rows"].append(row)
    fl = "" if floor is None else f" floor={floor:.3e}"
    print(f"  {'PASS' if ok else 'FAIL'} {name:<34} M={M:<5} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"bit_eq={bit * 100:8.4f}% max_ulp={int(max_ulp)}{fl}", flush=True)
    if not ok:
        FAILS.append(f"{name}@M={M}")
    return max_abs


def info(name, M, out, ref):
    d = (out.float() - ref.float()).abs()
    bit = (_ord(out) == _ord(ref)).float().mean().item()
    print(f"  info {name:<34} M={M:<5} max_abs={d.max().item():.3e} bit_eq={bit * 100:8.4f}%", flush=True)
    RESULTS["rows"].append({"check": name, "M": M, "max_abs": d.max().item(), "bit_equal": bit, "pass": None})
    return d.max().item()


# ---------------------------------------------------------------------------------------------------------------------
# Timing

def sync():
    if DEV == "xpu":
        torch.xpu.synchronize()


def bench(fn, iters=None, warmup=None):
    iters = iters or args.iters
    warmup = args.warmup if warmup is None else warmup
    for _ in range(warmup):
        fn()
    sync()
    if DEV == "xpu":
        s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


# ---------------------------------------------------------------------------------------------------------------------

def make_inputs(M, g):
    x = torch.randn(M, N, generator=g) * 0.7
    if N >= 64:
        cols = torch.randperm(N, generator=g)[:16]
        x[:, cols] *= 40.0                                   # residual-stream outlier channels
    w_norm = (torch.randn(N, generator=g) * 0.2)
    wd = torch.randn(LR, N, generator=g) / math.sqrt(N)
    wu = torch.randn(N, LR, generator=g) * (2.0 / math.sqrt(LR))
    wi = torch.randn(HC, N, generator=g) * (4.0 / math.sqrt(N))
    bo = torch.randn(M, HS, generator=g)
    t = lambda v: v.to(BF).to(DEV).contiguous()  # noqa: E731
    return t(x), t(w_norm), t(wd), t(wu), t(wi), t(bo)


def main():
    print(f"n107-hc test: device={DEV} interpret={INTERP} torch={torch.__version__} "
          f"triton={RESULTS['triton']} hc={HC} hs={HS} lr={LR} Ms={MS}", flush=True)
    if DEV == "xpu":
        p = torch.xpu.get_device_properties(0)
        RESULTS["gpu"] = str(p)
        print(f"gpu: {p}", flush=True)
    if not hc_xpu.HAVE_TRITON:
        print("FAIL: triton not importable", flush=True)
        return 2
    check_ref_drift()

    cmix = ccomb = None
    if not args.no_compile:
        try:
            cmix = torch.compile(_mix_compute)
            ccomb = torch.compile(_combine_compute)
        except Exception as e:  # pragma: no cover
            note(f"torch.compile unavailable ({e}); emulation used as the compiled reference")
    g = torch.Generator().manual_seed(args.seed)
    timing = DEV == "xpu"

    for M in MS:
        print(f"--- M={M}", flush=True)
        x, w_norm, wd, wu, wi, bo = make_inputs(M, g)

        # norm (hc_per_branch_norm=True: GroupedGemmaRMSNorm(10240, group_size=2560))
        n_ref = ref_norm_eager(x, w_norm, HS, EPS)
        n_tri = hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS)
        compare("norm grouped vs eager", M, n_tri, n_ref, rule="ulp1")
        n_ref32 = ref_norm_eager(x, w_norm.float(), HS, EPS)          # fp32 weight variant
        compare("norm grouped fp32-w vs eager", M, hc_xpu.gemma_rmsnorm_grouped(x, w_norm.float(), HS, EPS),
                n_ref32, rule="ulp1")
        if M <= 8:  # per-branch=False layout: [M, hc, hs] with a shared [hs] weight, group None
            x3 = x.view(M, HC, HS)
            compare("norm shared-w (per_branch=False)", M, hc_xpu.gemma_rmsnorm_grouped(x3, w_norm[:HS], None, EPS),
                    ref_norm_eager(x3, w_norm[:HS], None, EPS), rule="ulp1")
            if HS >= 2048:
                compare("norm loop variant (BLOCK 512)", M,
                        hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS, block=512, num_warps=4), n_ref, rule="ulp1")

        # mix: reference is the compiled function on the eager-normed input
        n = n_ref
        m_eager = _mix_compute(n, wd, wu, HC, HS)
        m_emul = emul_mix(n, wd, wu, HC, HS)
        if cmix is not None:
            try:
                m_ref = cmix(n, wd, wu, HC, HS)
            except Exception as e:
                note(f"compiled _mix_compute failed ({type(e).__name__}: {e}); emulation used")
                m_ref = m_emul
        else:
            m_ref = m_emul
        floor = info("mix eager vs compiled (floor)", M, m_eager, m_ref)
        info("mix emulation vs compiled", M, m_emul, m_ref)
        for mode in ("epilogue", "fused", "fused_silu"):
            try:
                compare(f"mix[{mode}] vs compiled", M, hc_xpu.hc_mix(n, wd, wu, HC, HS, mode=mode), m_ref, floor)
            except Exception as e:
                print(f"  FAIL mix[{mode}] raised {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                FAILS.append(f"mix[{mode}]@M={M} raised")
        if M <= 8:
            try:
                compare("mix[fused] plain-ptr vs compiled", M,
                        hc_xpu.mix_up_fused(hc_xpu.silu_div(F.linear(n, wd), HC), wu, n, HC, HS, use_bp=False),
                        m_ref, floor)
            except Exception as e:
                print(f"  FAIL mix[fused] plain-ptr raised {type(e).__name__}: {e}", flush=True)
                FAILS.append(f"mix plain-ptr@M={M} raised")

        # combine: residual = raw stream x, normed = n
        c_eager = _combine_compute(bo, x, n, wi, HC, HS)
        c_emul = emul_combine(bo, x, n, wi, HC, HS)
        if ccomb is not None:
            try:
                c_ref = ccomb(bo, x, n, wi, HC, HS)
            except Exception as e:
                note(f"compiled _combine_compute failed ({type(e).__name__}: {e}); emulation used")
                c_ref = c_emul
        else:
            c_ref = c_emul
        cfloor = info("combine eager vs compiled (floor)", M, c_eager, c_ref)
        info("combine emulation vs compiled", M, c_emul, c_ref)
        for inject in ("torch", "fused"):
            for rg in (True, False):
                try:
                    out = hc_xpu.hc_combine(bo, x, n, wi, HC, HS, inject=inject, round_g=rg)
                    if rg:
                        compare(f"combine[{inject}] vs compiled", M, out, c_ref, cfloor)
                    else:
                        info(f"combine[{inject},round_g=0] vs compiled", M, out, c_ref)
                except Exception as e:
                    print(f"  FAIL combine[{inject},round_g={int(rg)}] raised {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    FAILS.append(f"combine[{inject}]@M={M} raised")

        if timing:
            t = {"M": M}
            t["norm_eager"] = bench(lambda: ref_norm_eager(x, w_norm, HS, EPS))
            t["norm_tri"] = bench(lambda: hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS))
            if cmix is not None:
                t["mix_compiled"] = bench(lambda: cmix(n, wd, wu, HC, HS))
            t["mix_eager"] = bench(lambda: _mix_compute(n, wd, wu, HC, HS))
            for mode in ("epilogue", "fused", "fused_silu"):
                try:
                    t[f"mix_{mode}"] = bench(lambda: hc_xpu.hc_mix(n, wd, wu, HC, HS, mode=mode))
                except Exception:
                    t[f"mix_{mode}"] = None
            t["gemm_down"] = bench(lambda: F.linear(n, wd))
            a = hc_xpu.silu_div(F.linear(n, wd), HC)
            t["gemm_up"] = bench(lambda: F.linear(a, wu))
            if ccomb is not None:
                t["combine_compiled"] = bench(lambda: ccomb(bo, x, n, wi, HC, HS))
            t["combine_eager"] = bench(lambda: _combine_compute(bo, x, n, wi, HC, HS))
            for inject in ("torch", "fused"):
                try:
                    t[f"combine_{inject}"] = bench(lambda: hc_xpu.hc_combine(bo, x, n, wi, HC, HS, inject=inject))
                except Exception:
                    t[f"combine_{inject}"] = None
            t["gemm_inject"] = bench(lambda: F.linear(n, wi))
            RESULTS["timing"].append(t)
            nb = M * N * x.element_size()
            print("  time ms: " + " ".join(f"{k}={v:.4f}" for k, v in t.items() if k != "M" and v is not None),
                  flush=True)
            print(f"  norm eff. bandwidth: eager {2 * nb / t['norm_eager'] / 1e6:.0f} GB/s (2 passes counted), "
                  f"triton {2 * nb / t['norm_tri'] / 1e6:.0f} GB/s", flush=True)

    if timing and RESULTS["timing"]:
        summarize()
    if DEV == "xpu" and not args.no_graph:
        graph_test()
    if not args.no_patch:
        patch_test()
    if args.sweep and DEV == "xpu":
        sweep(max(MS))
    return 0


def summarize():
    t = RESULTS["timing"][-1]
    M = t["M"]
    old_mix = t["norm_eager"] + (t.get("mix_compiled") or t["mix_eager"])
    old_comb = t.get("combine_compiled") or t["combine_eager"]
    best_mix_mode = min((k for k in ("mix_epilogue", "mix_fused", "mix_fused_silu") if t.get(k)), key=lambda k: t[k])
    best_comb = min((k for k in ("combine_torch", "combine_fused") if t.get(k)), key=lambda k: t[k])
    new_mix = t["norm_tri"] + t[best_mix_mode]
    new_comb = t[best_comb]
    old = 97 * old_mix + 96 * old_comb
    new = 97 * new_mix + 96 * new_comb
    s = (f"[projection M={M}] per forward (97 mix incl. norm + 96 combine): reference {old:.1f} ms -> n107-hc "
         f"{new:.1f} ms (x{old / new:.2f}); best mix mode {best_mix_mode[4:]} ({t[best_mix_mode]:.3f} ms), "
         f"best combine {best_comb[8:]} ({new_comb:.3f} ms)")
    note(s)
    for tt in RESULTS["timing"]:
        if tt["M"] <= 8:
            o = tt["norm_eager"] + (tt.get("mix_compiled") or tt["mix_eager"]) + (tt.get("combine_compiled") or
                                                                                   tt["combine_eager"])
            nn_ = tt["norm_tri"] + min(v for k, v in tt.items() if k.startswith("mix_") and k[4:] in
                                       ("epilogue", "fused", "fused_silu") and v) + \
                min(v for k, v in tt.items() if k in ("combine_torch", "combine_fused") and v)
            note(f"[decode M={tt['M']}] one layer-half (norm+mix+combine): reference {o * 1e3:.1f} us -> "
                 f"{nn_ * 1e3:.1f} us (eager-launch timing; graph replay removes most launch cost)")


def graph_test():
    print("--- XPU graph capture (decode shapes)", flush=True)
    if not (hasattr(torch.xpu, "XPUGraph") and hasattr(torch.xpu, "graph")):
        note("[graph] torch.xpu.XPUGraph not available: SKIP")
        return
    g = torch.Generator().manual_seed(123)
    for M in (1, 4):
        try:
            x, w_norm, wd, wu, wi, bo = make_inputs(M, g)

            def step():
                nn_ = hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS)
                mm = hc_xpu.hc_mix(nn_, wd, wu, HC, HS, mode=os.environ.get("EXL3_HC_MIX", "epilogue"))
                cc = hc_xpu.hc_combine(bo + mm, x, nn_, wi, HC, HS,
                                       inject=os.environ.get("EXL3_HC_COMBINE", "torch"))
                return cc

            for _ in range(2):          # warm-up / JIT compile outside capture, like FullXPUGraphBackend
                step()
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(xpu_graph=graph):
                out = step()
            torch.xpu.synchronize()
            # new values in the static input buffers; the replay must see them
            x2, _, _, _, _, bo2 = make_inputs(M, g)
            x.copy_(x2)
            bo.copy_(bo2)
            graph.replay()
            torch.xpu.synchronize()
            got = out.clone()
            want = step()
            compare("graph replay vs eager (norm+mix+combine)", M, got, want, rule="exact")
        except Exception as e:
            print(f"  FAIL graph capture M={M}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            FAILS.append(f"graph@M={M}")


def patch_test():
    print("--- patch_hc end-to-end (GatedResidual from the installed sglang)", flush=True)
    os.environ["EXL3_HC_XPU"] = "1"
    try:
        import patch_hc
        patch_hc.install(import_now=True, force=True)
        from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
    except Exception as e:
        note(f"[patch] sglang not importable here ({type(e).__name__}: {e}): SKIP")
        return
    restore = None
    if not torch.cuda.is_available():
        restore = getattr(torch.cuda, "current_device")
        if DEV == "xpu":
            torch.cuda.current_device = lambda: torch.device("xpu", torch.xpu.current_device())
        else:
            torch.cuda.current_device = lambda: torch.device("cpu")
    try:
        cfg = HyperConnectionConfig(hc_count=HC, hidden_size=HS, params_dtype=BF, hc_lowrank=LR,
                                    rms_norm_eps=EPS, hc_per_branch_norm=True)
        mod = GatedResidual(cfg, use_mix=True, use_combine=True).to(DEV).to(BF)
    except Exception as e:
        note(f"[patch] could not construct GatedResidual ({type(e).__name__}: {e}): SKIP")
        return
    finally:
        if restore is not None:
            torch.cuda.current_device = restore
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p, scale in ((mod.hc_norm.weight, 0.2), (mod.input_mix_weight_down.weight, 1 / math.sqrt(N)),
                         (mod.input_mix_weight_up.weight, 2 / math.sqrt(LR)), (mod.block_inject_weight.weight,
                                                                               4 / math.sqrt(N))):
            p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.dtype))
    wrapped = getattr(mod._mix_compute, "_n107", False) and getattr(mod._combine_compute, "_n107", False)
    print(f"  instance wrapped: {wrapped}", flush=True)
    if not wrapped:
        FAILS.append("patch: instance not wrapped")
    for M in [m for m in MS if m <= 256] + ([8192] if DEV == "xpu" and 8192 in MS else []):
        x = (torch.randn(M, N, generator=g) * 0.7).to(BF).to(DEV)
        bo = torch.randn(M, HS, generator=g).to(BF).to(DEV)
        with torch.no_grad():
            before = dict(patch_hc.STATS)
            mi, res = mod.mix(x)
            out = mod.combine(bo, res)
            after = dict(patch_hc.STATS)
            saved = dict(patch_hc._state)
            for k in saved:
                patch_hc._state[k] = False
            mi0, res0 = mod.mix(x)                      # original path: eager norm + torch.compile'd mix/combine
            out0 = mod.combine(bo, res0)
            patch_hc._state.update(saved)
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        expect = ("norm_xpu", "mix_xpu", "combine_xpu") if DEV == "xpu" else ("norm_orig", "mix_orig", "combine_orig")
        ok = all(delta.get(k, 0) >= 1 for k in expect)
        print(f"  dispatch M={M}: {delta} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            FAILS.append(f"patch dispatch@M={M}")
        compare("patched norm vs original path", M, res[1], res0[1], rule="ulp1")
        # mix/combine see inputs that may differ by one bf16 ulp (norm); tolerance: 2 bf16 ulps of the largest output
        compare("patched mix vs original path", M, mi, mi0, floor=mi0.float().abs().max().item() * 2.0 ** -8)
        compare("patched combine vs original path", M, out, out0, floor=out0.float().abs().max().item() * 2.0 ** -8)


def sweep(M):
    print(f"--- launch-config sweep at M={M}", flush=True)
    g = torch.Generator().manual_seed(99)
    x, w_norm, wd, wu, wi, bo = make_inputs(M, g)
    n = hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS)
    res = []

    def run(name, fn):
        try:
            ms = bench(fn)
            res.append((name, ms))
            print(f"  {name:<40} {ms:.4f} ms", flush=True)
        except Exception as e:
            print(f"  {name:<40} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)

    for blk in (4096, 2048, 1024, 512):
        for w in (4, 8, 16, 32):
            if blk // w < 32:
                continue
            run(f"norm BLOCK={blk} warps={w}", lambda: hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS, blk, w))
    for tpw in (None, "16", "32"):
        if tpw:
            os.environ["EXL3_HC_TPW"] = tpw
        else:
            os.environ.pop("EXL3_HC_TPW", None)
        run(f"norm default tpw={tpw}", lambda: hc_xpu.gemma_rmsnorm_grouped(x, w_norm, HS, EPS))
    os.environ.pop("EXL3_HC_TPW", None)
    t = F.linear(n, wd)
    a = hc_xpu.silu_div(t, HC)
    u = F.linear(a, wu)
    for cfg in ("1,512,4", "2,512,4", "4,512,4", "4,512,8", "8,256,8", "8,512,8", "16,256,8"):
        os.environ["EXL3_HC_EPI_CFG"] = cfg
        run(f"mix epilogue BM,BN,W={cfg}", lambda: hc_xpu.mix_epilogue(u, n, HC, HS))
    os.environ.pop("EXL3_HC_EPI_CFG", None)
    for cfg in ((64, 64, 32, 8), (64, 128, 32, 8), (64, 128, 32, 16), (128, 64, 32, 16), (128, 128, 32, 16),
                (128, 128, 32, 32), (128, 128, 64, 32), (256, 64, 32, 32), (256, 128, 32, 32)):
        for bp in (True, False):
            run(f"mix up-fused cfg={cfg} blockptr={int(bp)}",
                lambda: hc_xpu.mix_up_fused(a, wu, n, HC, HS, use_bp=bp, cfg=cfg))
    for cfg in ("1,512,4", "4,512,4", "4,512,8", "8,512,8", "8,256,8", "16,256,8"):
        os.environ["EXL3_HC_COMB_CFG"] = cfg
        run(f"combine[torch] BM,BN,W={cfg}", lambda: hc_xpu.hc_combine(bo, x, n, wi, HC, HS, inject="torch"))
    os.environ.pop("EXL3_HC_COMB_CFG", None)
    for cfg in ("4,512,256,1,4", "8,512,256,1,8", "2,512,512,1,4", "4,256,256,2,4", "16,256,128,1,8"):
        os.environ["EXL3_HC_COMBF_CFG"] = cfg
        run(f"combine[fused] BM,BN,BK,NSPLIT,W={cfg}",
            lambda: hc_xpu.hc_combine(bo, x, n, wi, HC, HS, inject="fused"))
    os.environ.pop("EXL3_HC_COMBF_CFG", None)
    RESULTS["sweep"] = res


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        FAILS.append("exception")
    RESULTS["fails"] = FAILS
    RESULTS["pass"] = not FAILS and rc == 0
    print(f"=== {'ALL PASS' if RESULTS['pass'] else 'FAILURES: ' + ', '.join(FAILS)}", flush=True)
    out = args.json or os.path.join(HERE, f"results_{DEV}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    try:
        with open(out, "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
        print(f"results: {out}", flush=True)
    except OSError as e:
        print(f"results not written ({e})", flush=True)
    sys.exit(0 if RESULTS["pass"] else 1)
