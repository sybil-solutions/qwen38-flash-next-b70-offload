#!/usr/bin/env python3
"""N107 QSA prefill test + benchmark: new Triton kernels vs the reference path, on XPU (or CPU for the tiny cases).

  # B70 (inside image 24c872759256, see README.md): accuracy + timing at 8k and 32k context, 8192 query rows
  python3 test_qsa_xpu.py --ctx 8192 32768 --rows 8192 --variants ref,union,row,row_cast,compact,tile --out res.jsonl
  python3 test_qsa_xpu.py --sweep --ctx 8192 --rows 8192                # launch-config sweep (row + tile)
  python3 test_qsa_xpu.py --nosync --expand                             # sync-removal + expander exactness
  python3 test_qsa_xpu.py --dump /dumps/qsa_topk_L3_*.pt --variants ref,union,row,tile   # real selections
  # CPU, no GPU: Triton interpreter at tiny sizes (Linux, triton installed) or the torch emulators (no triton)
  TRITON_INTERPRET=1 python3 test_qsa_xpu.py --device cpu --tiny
  python3 test_qsa_xpu.py --device cpu --tiny --emulate --nosync

Reference = SGLang qsa_sparse_attention_reference semantics in fp32 (ref_fp32, cross-checked against the verbatim
per-row loop on a row sample). "union" = exl3xpu qsa_sparse_attention_union, the path the server runs today.
Exit code 1 if any checked variant exceeds the tolerances.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import qsa_ref as R  # noqa: E402

try:
    import triton  # noqa: F401
    import qsa_sparse_triton as K
    HAVE_TRITON = True
except Exception as _e:  # pragma: no cover
    K = None
    HAVE_TRITON = False
    _TRITON_ERR = repr(_e)

try:
    import qsa_sycl as SY          # N108 SYCL per-query kernel (Strata port); needs build/qsa_row_sycl.so on XPU
except Exception as _e:  # pragma: no cover
    SY = None

TOL_MAX_ABS = 3e-2      # bf16 output, P rounded to bf16 (the union path does the same)
TOL_MEAN_ABS = 2e-3
FAILS = []


def dev_sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, iters, warmup):
    t0 = time.perf_counter()
    out = fn()
    dev_sync(device)
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(max(0, warmup - 1)):
        fn()
    dev_sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        dev_sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    return out, first, (statistics.median(ts) if ts else first), (min(ts) if ts else first)


def err(out, ref):
    d = (out.float() - ref.float()).abs()
    zero_rows = (ref.float().abs().amax(dim=(1, 2)) == 0)
    zero_ok = bool((out[zero_rows].float().abs().max() == 0) if zero_rows.any() else True)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), ref_absmax=float(ref.float().abs().max()),
                zero_rows=int(zero_rows.sum()), zero_rows_exact=zero_ok, finite=bool(torch.isfinite(out.float()).all()))


def emit(rec, fh):
    print(json.dumps(rec), flush=True)
    if fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def flops(case):
    """Useful FLOPs of sparse attention (QK + PV over valid slots, all q heads)."""
    nv = int((case["slots"] >= 0).sum())
    return 4.0 * nv * R.HQ * case["q"].shape[-1]


def run_variants(case, args, device, fh, tag):
    q, kc, vc, slots, logical = case["q"], case["k_cache"], case["v_cache"], case["slots"], case["logical"]
    scale = case["scale"]
    S, Rr = case["S"], case["R"]
    tbl = case["table"]
    seqs = case["seqs"]
    t2b = torch.zeros((Rr,), dtype=torch.int32, device=device)
    for i, (_, r0, n, _, _) in enumerate(seqs):
        t2b[r0:r0 + n] = i
    seq_lens_dev = torch.tensor([s[4] for s in seqs], dtype=torch.int32, device=device)
    st = R.selection_stats(logical, S)
    emit(dict(tag=tag, kind="case", S=S, kv=str(kc.dtype), sel=args.sel, **st,
              useful_gflop=flops(case) / 1e9), fh)

    variants = [v for v in args.variants.split(",") if v]
    ref = None
    results = {}
    if True:   # the fp32 reference is always computed (every variant is checked against it)
        ref, first, med, mn = bench(lambda: R.ref_fp32(q, kc, vc, slots, scale), device, 1 if Rr > 1024 else 2, 1)
        emit(dict(tag=tag, kind="time", variant="ref_fp32", first_ms=first, ms=med), fh)
        # cross-check the vectorised reference against SGLang's verbatim per-row loop on a row sample
        nchk = min(args.ref_rows, Rr)
        rows = sorted(set([0, 1, 2, Rr - 1] + torch.linspace(0, Rr - 1, nchk).long().tolist()))
        loop = R.ref_sglang_rows(q, kc, vc, slots, scale, rows=rows)
        e = err(loop, ref[rows])
        emit(dict(tag=tag, kind="check", variant="ref_fp32_vs_sglang_loop", rows=len(rows), **e), fh)
        if e["max_abs"] > 8e-3:     # both fp32; only output bf16 rounding of near-ties differs
            FAILS.append(f"{tag} ref_fp32 vs sglang loop max_abs {e['max_abs']:.3e}")

    def add(name, fn, check=True):
        try:
            out, first, med, mn = bench(fn, device, args.iters, args.warmup)
        except Exception as ex:
            emit(dict(tag=tag, kind="error", variant=name, error=f"{type(ex).__name__}: {ex}"[:2000]), fh)
            FAILS.append(f"{tag} {name}: {type(ex).__name__}")
            return
        e = err(out, ref)
        rec = dict(tag=tag, kind="result", variant=name, first_ms=round(first, 3), ms=round(med, 3),
                   min_ms=round(mn, 3), tflops=round(flops(case) / (med * 1e-3) / 1e12, 3), **e)
        emit(rec, fh)
        results[name] = rec
        if check and (e["max_abs"] > TOL_MAX_ABS or e["mean_abs"] > TOL_MEAN_ABS or not e["finite"]
                      or not e["zero_rows_exact"]):
            FAILS.append(f"{tag} {name}: max_abs {e['max_abs']:.3e} mean_abs {e['mean_abs']:.3e} "
                         f"finite {e['finite']} zero_rows_exact {e['zero_rows_exact']}")

    use_emu = args.emulate or not HAVE_TRITON
    for v in variants:
        if v == "ref":
            continue
        if v == "union":
            add("union_exl3xpu", lambda: R.union_exl3xpu(q, kc, vc, slots, scale), check=False)
        elif v == "row":
            if use_emu:
                add("emu_row", lambda: R.emulate_row_kernel(q, kc, vc, slots, scale, block_n=args.block_n))
            else:
                add("row", lambda: K.qsa_sparse_attention_triton(q, kc, vc, slots, scale, fp8_mode="bits"))
        elif v in ("sycl_row", "sycl_strata", "sycl_bcast"):
            if SY is None or device.type != "xpu" or not SY.load():
                emit(dict(tag=tag, kind="warn", msg=f"{v} skipped: {None if SY is None else SY.error()}"), fh)
            elif not SY.supports(q, kc, vc, slots):
                emit(dict(tag=tag, kind="warn", msg=f"{v} skipped: unsupported shape/dtype"), fh)
            elif v == "sycl_row":           # default config (EXL3_QSA_SYCL_CFG), what EXL3_QSA_IMPL=sycl_row runs
                add("sycl_row", lambda: SY.qsa_sparse_attention_sycl(q, kc, vc, slots, scale))
            elif v == "sycl_strata":        # Strata's launch shape: 64-cell split-K chunks + merge, sub-group 32
                add("sycl_strata", lambda: SY.qsa_sparse_attention_sycl(q, kc, vc, slots, scale, kernel=1, cpw=1,
                                                                          sg=32, grf=128, fp8_mode=0))
            else:
                add("sycl_bcast", lambda: SY.qsa_sparse_attention_sycl(q, kc, vc, slots, scale, kernel=2, cpw=0,
                                                                         sg=16, grf=128, fp8_mode=1))
        elif v == "row_cast":
            if not use_emu and kc.dtype in K._FP8:
                add("row_cast", lambda: K.qsa_sparse_attention_triton(q, kc, vc, slots, scale, fp8_mode="cast"))
        elif v == "compact":
            if not use_emu:
                add("compact", lambda: K.qsa_compact_row_attention(q, kc, vc, logical, tbl, t2b, seq_lens_dev,
                                                                    [s[4] for s in seqs], scale))
        elif v == "tile":
            if use_emu:
                if Rr * S <= 4096 * 4096:
                    add("emu_tile", lambda: R.emulate_tile_kernel(q, kc, vc, logical, tbl, seqs, scale,
                                                                  BM=args.tile_bm, BN=args.tile_bn))
            else:
                add("tile", lambda: K.qsa_tile_attention(q, kc, vc, logical, tbl, seqs, scale))
        else:
            emit(dict(tag=tag, kind="warn", msg=f"unknown variant {v}"), fh)
    if "union_exl3xpu" in results and not use_emu:
        base = results["union_exl3xpu"]["ms"]
        for name, r in results.items():
            if name != "union_exl3xpu":
                emit(dict(tag=tag, kind="speedup", variant=name, vs="union_exl3xpu", per_layer_ms_saved=base - r["ms"],
                          per_8k_chunk_s_saved_12_layers=round(12 * (base - r["ms"]) * 1e-3 * (8192 / Rr), 3),
                          speedup=round(base / r["ms"], 2)), fh)
    return results


def run_sweep(case, args, device, fh, tag):
    """Launch-config sweep. Every config is checked against the fp32 reference; compile failures are recorded."""
    q, kc, vc, slots, logical = case["q"], case["k_cache"], case["v_cache"], case["slots"], case["logical"]
    scale = case["scale"]
    ref = R.ref_fp32(q, kc, vc, slots, scale)
    best = {}

    def one(kernel, cfg, fn):
        try:
            out, first, med, _ = bench(fn, device, args.iters, 1)
            e = err(out, ref)
            emit(dict(tag=tag, kind="sweep", kernel=kernel, **cfg, ms=med, first_ms=first, max_abs=e["max_abs"]), fh)
            if e["max_abs"] <= TOL_MAX_ABS and e["finite"] and (kernel not in best or med < best[kernel]["ms"]):
                best[kernel] = dict(ms=med, **cfg)
        except Exception as ex:
            emit(dict(tag=tag, kind="sweep", kernel=kernel, **cfg, error=f"{type(ex).__name__}: {str(ex)[:300]}"), fh)

    row_cfgs = [(bn, nw, 0, tpw) for bn in (32, 64, 128) for nw in (4, 8, 16) for tpw in (0, 16, 32)]
    row_cfgs += [(64, 8, 2, 0), (64, 8, 3, 0), (32, 4, 2, 0)]
    for bn, nw, ns, tpw in row_cfgs:
        for fp8 in (("bits", "cast") if kc.dtype in K._FP8 else ("bits",)):
            one("row", dict(block_n=bn, warps=nw, stages=ns, tpw=tpw, fp8=fp8),
                lambda: K.qsa_sparse_attention_triton(q, kc, vc, slots, scale, block_n=bn, num_warps=nw,
                                                      num_stages=ns, threads_per_warp=tpw or None, fp8_mode=fp8))
    for bm in (32, 64, 128):
        for bn in (32, 64):
            for nw in (4, 8, 16):
                for tpw in (0, 16):
                    one("tile", dict(BM=bm, BN=bn, warps=nw, tpw=tpw),
                        lambda: K.qsa_tile_attention(q, kc, vc, logical, case["table"], case["seqs"], scale, BM=bm,
                                                     BN=bn, num_warps=nw, threads_per_warp=tpw or None))
    emit(dict(tag=tag, kind="sweep_best", best=best), fh)


def run_sycl_sweep(case, args, device, fh, tag):
    """N108 launch-shape sweep of the SYCL kernel (kernel, chunks per work-group, sub-group, GRF, fp8 decode).
    Every config is checked against the fp32 reference with the full pass criteria."""
    q, kc, vc, slots = case["q"], case["k_cache"], case["v_cache"], case["slots"]
    scale = case["scale"]
    ref = R.ref_fp32(q, kc, vc, slots, scale)
    if SY is None or not SY.load():
        emit(dict(tag=tag, kind="error", variant="sycl_sweep", error=str(None if SY is None else SY.error())), fh)
        FAILS.append(f"{tag} sycl library")
        return
    fp8 = kc.dtype == torch.float8_e4m3fn
    cfgs = []
    for kern, cpw, sg, grf, f8 in [
            (1, 1, 32, 128, 0),          # Strata's shape (split-K over 64-cell chunks + merge)
            (1, 1, 32, 128, 1), (1, 4, 32, 128, 1), (1, 0, 32, 128, 1), (1, 0, 16, 128, 1), (1, 0, 32, 256, 1),
            (2, 0, 16, 128, 1), (2, 0, 16, 128, 0), (2, 0, 16, 256, 1), (2, 0, 32, 128, 1), (2, 0, 32, 256, 1),
            (2, 1, 16, 128, 1), (2, 2, 16, 128, 1), (2, 4, 16, 128, 1)]:
        if not fp8 and f8 == 1 and (kern, cpw, sg, grf) in [(c[0], c[1], c[2], c[3]) for c in cfgs]:
            continue
        cfgs.append((kern, cpw, sg, grf, f8))
    best = None
    for kern, cpw, sg, grf, f8 in cfgs:
        cfg = dict(kernel=kern, cpw=cpw, sg=sg, grf=grf, fp8_mode=f8)
        try:
            out, first, med, mn = bench(lambda: SY.qsa_sparse_attention_sycl(q, kc, vc, slots, scale, **cfg), device,
                                        args.iters, 1)
            e = err(out, ref)
            ok = (e["max_abs"] <= TOL_MAX_ABS and e["mean_abs"] <= TOL_MEAN_ABS and e["finite"] and e["zero_rows_exact"])
            emit(dict(tag=tag, kind="sycl_sweep", **cfg, ms=round(med, 3), min_ms=round(mn, 3), first_ms=round(first, 3),
                      tflops=round(flops(case) / (med * 1e-3) / 1e12, 3), ok=ok, **e), fh)
            if not ok:
                FAILS.append(f"{tag} sycl {cfg}: max_abs {e['max_abs']:.3e} mean_abs {e['mean_abs']:.3e}")
            elif best is None or med < best["ms"]:
                best = dict(ms=round(med, 3), **cfg)
            del out
        except Exception as ex:
            emit(dict(tag=tag, kind="sycl_sweep", **cfg, error=f"{type(ex).__name__}: {str(ex)[:500]}"), fh)
            FAILS.append(f"{tag} sycl {cfg}: {type(ex).__name__}")
    emit(dict(tag=tag, kind="sycl_sweep_best", best=best), fh)


def test_fp8_decode_sycl(device, fh):
    """Device fp8 decode (both routes of the SYCL kernels) == torch e4m3fn for all 254 non-NaN codes."""
    if SY is None or device.type != "xpu" or not SY.load():
        emit(dict(kind="fp8_decode_sycl", skipped=str(None if SY is None else SY.error())), fh)
        return
    codes = torch.arange(256, dtype=torch.uint8)
    nan = (codes & 0x7F) == 0x7F
    want = codes.view(torch.float8_e4m3fn).float()
    got = SY.fp8_decode_test(codes.to(device)).cpu()
    res = {}
    for i, name in enumerate(("exact", "via_fp16")):
        a = got[i]
        res[name] = bool(torch.equal(a[~nan], want[~nan]) and torch.equal(torch.signbit(a[~nan]), torch.signbit(want[~nan])))
    emit(dict(kind="fp8_decode_sycl", codes=256, **res), fh)
    if not all(res.values()):
        FAILS.append(f"fp8 sycl decode {res}")


def edge_case(case):
    """Rows with no valid slot (-> exact zeros), holes in the middle of rows (-1 between valid entries) and
    duplicate-free shuffles of the valid entries (order must not matter beyond fp32 summation order)."""
    c = dict(case)
    logical = case["logical"].clone()
    Rr = logical.shape[0]
    for r in (0, 3, Rr // 2):
        logical[r] = -1
    holes = torch.zeros_like(logical, dtype=torch.bool)
    holes[1::7, 5::3] = True
    logical = torch.where(holes, torch.full_like(logical, -1), logical)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(logical.shape[1], generator=g).to(logical.device)
    logical[Rr // 3:] = logical[Rr // 3:, perm]
    table = case["table"][0]
    c["logical"] = logical
    c["slots"] = torch.where(logical >= 0, table[logical.clamp_min(0).long()], torch.full_like(logical, -1)).to(torch.int32)
    return c


def case_from_dump(path, args, device):
    d = torch.load(path, map_location="cpu")
    logical = d["logical"].to(device)
    seqs = [tuple(int(x) for x in s) for s in d["seqs"]]
    Rr = logical.shape[0]
    L = max(s[4] for s in seqs)
    g = torch.Generator().manual_seed(1)
    page = 64
    n_pages = (L + page - 1) // page
    tables = []
    used = 1
    for s in seqs:
        np_ = (s[4] + page - 1) // page
        pages = torch.arange(used, used + np_)[torch.randperm(np_, generator=g)]
        used += np_
        pos = torch.arange(L)
        tables.append((pages[(pos // page).clamp_max(np_ - 1)] * page + pos % page).to(torch.int32))
    table = torch.stack(tables).to(device)
    N = (used + 1) * page
    kv_dtype = torch.float8_e4m3fn if args.kv == "fp8" else torch.bfloat16
    k = torch.randn((N, R.HK, R.HEAD_DIM), generator=g).to(kv_dtype).to(device)
    v = torch.randn((N, R.HK, R.HEAD_DIM), generator=g).to(kv_dtype).to(device)
    q = torch.randn((Rr, R.HQ, R.HEAD_DIM), generator=g).to(torch.bfloat16).to(device)
    t2b = torch.cat([torch.full((s[2],), i, dtype=torch.long) for i, s in enumerate(seqs)]).to(device)
    valid = logical >= 0
    phys = torch.where(valid, table[t2b[:, None], logical.clamp_min(0).long()], torch.full_like(logical, -1))
    return dict(q=q, k_cache=k, v_cache=v, logical=logical.to(torch.int32), slots=phys.to(torch.int32),
                table=table, S=L, R=Rr, scale=R.HEAD_DIM ** -0.5, seqs=seqs)


# ------------------------------------------------------------------------------------------------------------------
# sync-removal tests (CPU is fine; uses SGLang's real functions when importable, verbatim copies otherwise)

def _orig_apply_rope_copy(get_is_capture_mode, apply_rotary_emb):
    def apply_rope(self, positions, tensor):
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = positions.shape[-1] if positions.ndim == 2 else positions.numel()
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if not get_is_capture_mode() and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length"):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[:, :half_rotary_dim]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[:, :half_rotary_dim]
        rotated = apply_rotary_emb(tensor[..., :rotary_dim], cos, sin, self.rotary_emb.is_neox_style)
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)
    return apply_rope


def _apply_rotary_emb_copy(x, cos, sin, is_neox_style):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class StubRope:
    """SGLang RotaryEmbedding (base.py) cache semantics: cos|sin cache rows, grow-only guard, 1-D position lookup."""

    def __init__(self, rotary_dim=64, max_pos=4096, base=10000.0, device="cpu"):
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = True
        inv = self._compute_inv_freq(base)
        t = torch.arange(max_pos, dtype=torch.float)
        f = torch.einsum("i,j -> ij", t, inv)
        self.cos_sin_cache = torch.cat((f.cos(), f.sin()), dim=-1).to(device)

    def _compute_inv_freq(self, base):
        return 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

    def _ensure_cos_sin_cache_length(self, needed_max_pos: int):
        cur_len = int(self.cos_sin_cache.shape[0])
        if needed_max_pos < cur_len:
            return
        align = 128
        new_len = ((needed_max_pos + align) // align) * align
        device = self.cos_sin_cache.device
        dtype = self.cos_sin_cache.dtype
        inv_freq = self._compute_inv_freq(self.base).to(device=device)
        start = cur_len
        t_new = torch.arange(start, new_len, dtype=inv_freq.dtype, device=device)
        if t_new.numel() == 0:
            return
        freqs_new = torch.einsum("i,j->ij", t_new, inv_freq)
        new_rows = torch.cat((freqs_new.cos(), freqs_new.sin()), dim=-1).to(dtype=dtype)
        self.cos_sin_cache = torch.cat((self.cos_sin_cache, new_rows), dim=0).to(device=device, dtype=dtype)

    def get_cos_sin_with_position(self, positions):
        cos_sin = self.cos_sin_cache.index_select(0, positions.flatten())
        last_dim = cos_sin.size()[-1]
        cos, sin = cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)
        self.position_cos, self.position_sin = (cos.view(-1, 1, 1, last_dim).contiguous(),
                                                sin.view(-1, 1, 1, last_dim).contiguous())


class SyncCounter:
    """Counts Tensor.item / Tensor.tolist calls (the host reads the patch removes)."""

    def __enter__(self):
        self.n = 0
        self._item, self._tolist = torch.Tensor.item, torch.Tensor.tolist
        c = self

        def item(t, *a, **k):
            c.n += 1
            return c._item(t, *a, **k)

        def tolist(t, *a, **k):
            c.n += 1
            return c._tolist(t, *a, **k)
        torch.Tensor.item, torch.Tensor.tolist = item, tolist
        return self

    def __exit__(self, *exc):
        torch.Tensor.item, torch.Tensor.tolist = self._item, self._tolist


def _build_qsa_row_ranges_copy(sequence_lengths, query_positions, query_sequence_ids, compress_ratio):
    sequence_lengths = sequence_lengths.to(dtype=torch.int32)
    compressed_lengths = torch.div(sequence_lengths, compress_ratio, rounding_mode="floor")
    compressed_cu_seqlens = torch.nn.functional.pad(compressed_lengths.cumsum(0), (1, 0)).to(torch.int32)
    query_sequence_ids = query_sequence_ids.to(device=sequence_lengths.device, dtype=torch.long)
    row_starts = compressed_cu_seqlens.index_select(0, query_sequence_ids)
    visible_blocks = torch.div(query_positions.to(device=sequence_lengths.device, dtype=torch.int32) + 1,
                               compress_ratio, rounding_mode="floor")
    max_blocks = compressed_lengths.index_select(0, query_sequence_ids)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    return row_starts, row_ends, compressed_cu_seqlens


def _orig_get_prefill_mqa_inputs_copy(build_qsa_row_ranges):
    def get_prefill_mqa_inputs(self, layer_id, positions):
        pool = self.token_to_kv_pool
        ratio = self.compress_ratio
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        parts = []
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        sequence_lengths_list = sequence_lengths.tolist()
        for sequence_id in range(len(sequence_lengths_list)):
            complete_blocks = int(sequence_lengths_list[sequence_id]) // ratio
            if complete_blocks == 0:
                continue
            compressed_locs = (self.token_slot_table[sequence_id, : complete_blocks * ratio: ratio].long() // ratio)
            parts.append(compressed_buffer.index_select(0, compressed_locs))
        compressed_keys = (torch.cat(parts, dim=0) if parts else
                           compressed_buffer.new_empty((0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim)))
        num_valid_tokens = self.token_to_batch_idx.numel()
        positions = positions[:num_valid_tokens]
        row_starts, row_ends, _ = build_qsa_row_ranges(sequence_lengths, positions.to(sequence_lengths.device),
                                                       self.token_to_batch_idx.to(sequence_lengths.device),
                                                       self.compress_ratio)
        return compressed_keys, row_starts, row_ends, sequence_lengths
    return get_prefill_mqa_inputs


def test_nosync(device, fh):
    import qsa_nosync as NS
    src = "copy"
    capture = lambda: False  # noqa: E731
    rot = _apply_rotary_emb_copy
    orig_apply_rope = _orig_apply_rope_copy(capture, rot)
    brr = _build_qsa_row_ranges_copy
    orig_gpmi = _orig_get_prefill_mqa_inputs_copy(brr)
    try:   # prefer SGLang's real functions (inside the image)
        from sglang.srt.layers.attention.qsa import qsa_indexer as qi, metadata as md
        f = qi.QSAIndexer.apply_rope
        g = md.QSAIndexerMetadata.get_prefill_mqa_inputs
        if not getattr(f, "_n107", False) and not getattr(g, "_n107", False):
            orig_apply_rope, orig_gpmi = f, g
            rot, brr = qi.apply_rotary_emb, md.build_qsa_row_ranges
            capture = qi.get_is_capture_mode
            src = "sglang"
    except Exception:
        pass
    new_apply_rope = NS.make_apply_rope(capture, rot)
    new_gpmi = NS.make_get_prefill_mqa_inputs(brr)
    ok = True
    # --- apply_rope: q path (S tokens, prefix P) and compressed-key path; cache covered and cache growth cases
    for cache_len, S, P in ((4096, 1000, 200), (300, 1000, 500), (998, 1000, 500), (1024, 1024, 0)):
        for kind in ("q", "ck"):
            torch.manual_seed(0)
            pos = torch.arange(P, S, device=device) if kind == "q" else (torch.arange(P // 4, S // 4, device=device) * 4)
            x = torch.randn((pos.numel(), 4 if kind == "q" else 1, 128), device=device, dtype=torch.bfloat16)
            a = types.SimpleNamespace(rotary_emb=StubRope(64, cache_len, device=device))
            b = types.SimpleNamespace(rotary_emb=StubRope(64, cache_len, device=device))
            with SyncCounter() as c0:
                ya = orig_apply_rope(a, pos, x)
            NS._tls.ctx = NS.HostCtx([S], S - 1)
            try:
                with SyncCounter() as c1:
                    yb = new_apply_rope(b, pos, x)
            finally:
                NS._tls.ctx = None
            same = torch.equal(ya, yb)
            n_common = min(a.rotary_emb.cos_sin_cache.shape[0], b.rotary_emb.cos_sin_cache.shape[0])
            cache_same = torch.equal(a.rotary_emb.cos_sin_cache[:n_common], b.rotary_emb.cos_sin_cache[:n_common])
            rec = dict(kind="nosync", test=f"apply_rope[{kind}] cache={cache_len} S={S} P={P}", impl=src,
                       bitwise_equal=same, cache_rows_equal=cache_same, syncs_orig=c0.n, syncs_new=c1.n)
            emit(rec, fh)
            ok &= same and cache_same and c1.n == 0 and c0.n >= 1
    # --- get_prefill_mqa_inputs: 1 and 3 sequences
    for lens in ([8192], [5000, 1203, 64]):
        bs = len(lens)
        Lmax = max(lens)
        torch.manual_seed(1)
        table = torch.stack([(torch.randperm(Lmax // 4 + 8)[: (Lmax + 3) // 4].repeat_interleave(4) * 4
                              + torch.arange(4).repeat((Lmax + 3) // 4))[:Lmax] for _ in range(bs)]).int().to(device)
        ext = [min(L, 512) for L in lens]
        t2b = torch.cat([torch.full((e,), i, dtype=torch.int32) for i, e in enumerate(ext)]).to(device)
        positions = torch.cat([torch.arange(L - e, L) for L, e in zip(lens, ext)]).to(device)
        cbuf = torch.randn((int(table.max()) // 4 + 8, 1, 128), device=device, dtype=torch.bfloat16)
        pool = types.SimpleNamespace(get_qsa_compressed_k_buffer=lambda lid: cbuf, qsa_index_kv_heads=1,
                                     qsa_index_head_dim=128)
        meta = types.SimpleNamespace(token_to_kv_pool=pool, compress_ratio=4,
                                     sequence_lengths=torch.tensor(lens, dtype=torch.int32, device=device),
                                     token_slot_table=table, token_to_batch_idx=t2b)
        with SyncCounter() as c0:
            ra = orig_gpmi(meta, 3, positions)
        NS._tls.ctx = NS.HostCtx(list(lens), max(lens) - 1)
        try:
            with SyncCounter() as c1:
                rb = new_gpmi(meta, 3, positions)
        finally:
            NS._tls.ctx = None
        same = all(torch.equal(x, y) for x, y in zip(ra, rb))
        emit(dict(kind="nosync", test=f"get_prefill_mqa_inputs lens={lens}", impl=src, bitwise_equal=same,
                  syncs_orig=c0.n, syncs_new=c1.n), fh)
        ok &= same and c1.n == 0 and c0.n >= 1
    # --- host context from a ForwardBatch-like object
    Mode = types.SimpleNamespace
    fb = types.SimpleNamespace(forward_mode=Mode(is_decode=lambda: False, is_extend=lambda: True,
                                                 is_target_verify=lambda: False, is_draft_extend=lambda: False),
                               seq_lens=torch.tensor([10, 20]), seq_lens_cpu=torch.tensor([10, 20]),
                               contains_mm_inputs=lambda: False)
    ctx = NS.host_ctx_from_forward_batch(fb)
    good = ctx is not None and ctx.seq_lens == [10, 20] and ctx.max_pos == 19
    fb.contains_mm_inputs = lambda: True
    ctx2 = NS.host_ctx_from_forward_batch(fb)
    good &= ctx2 is not None and ctx2.max_pos is None
    fb.forward_mode = Mode(is_decode=lambda: True, is_extend=lambda: False)
    good &= NS.host_ctx_from_forward_batch(fb) is None
    emit(dict(kind="nosync", test="host_ctx_from_forward_batch", ok=bool(good)), fh)
    ok &= bool(good)
    if not ok:
        FAILS.append("nosync tests")


def _fake_kmod():
    """Kernel-module stand-in built from the torch emulators (machines without Triton)."""
    def compact(q, kc, vc, logical, table, t2b, seq_lens, seq_lens_host, scale=None, **kw):
        ks = [kc.index_select(0, table[i, :L].long()) for i, L in enumerate(seq_lens_host)]
        vs = [vc.index_select(0, table[i, :L].long()) for i, L in enumerate(seq_lens_host)]
        k2, v2 = torch.cat(ks).to(q.dtype), torch.cat(vs).to(q.dtype)
        lens = seq_lens.long()
        base = (torch.cumsum(lens, 0) - lens).index_select(0, t2b.long())
        cidx = torch.where(logical >= 0, logical.long() + base[:, None], torch.full_like(logical, -1, dtype=torch.long))
        return R.emulate_row_kernel(q, k2, v2, cidx.int(), scale)
    return types.SimpleNamespace(
        supports=lambda *a, **k: True,
        qsa_sparse_attention_triton=lambda q, k, v, s, scale=None, **kw: R.emulate_row_kernel(q, k, v, s, scale),
        qsa_tile_attention=lambda q, k, v, li, tbl, seqs, scale=None, **kw: R.emulate_tile_kernel(q, k, v, li, tbl,
                                                                                                    seqs, scale),
        qsa_compact_row_attention=compact, FP8_MODE="bits", ROW_CFG=(64, 8, 0), TILE_CFG=(64, 64, 8, 0))


def test_patch_mock(device, fh, use_emu):
    """Drive patch_qsa.install() against fake SGLang modules: dispatcher + forward_extend wrappers (row / tile /
    compact) must reproduce the reference on a 3-sequence chunked-prefill batch (prefix > 0, DP-padded q rows)."""
    import importlib
    import qsa_nosync as NS
    os.environ.update(EXL3_QSA_TRITON="1", EXL3_QSA_DEVICES=device.type, EXL3_QSA_TRITON_MIN_ROWS="1",
                      EXL3_QSA_TRITON_EXPAND="1", EXL3_QSA_NOSYNC="1")
    kmod = _fake_kmod() if use_emu else K
    # batch: 3 sequences, (prefix, extend) = (300, 100), (0, 64), (700, 60); KV pool with shuffled pages
    seq_spec = [(300, 100), (0, 64), (700, 60)]
    g = torch.Generator().manual_seed(3)
    page = 16
    Lmax = max(p + e for p, e in seq_spec)
    tables, used = [], 1
    for p, e in seq_spec:
        L = p + e
        npg = (L + page - 1) // page
        pages = torch.arange(used, used + npg)[torch.randperm(npg, generator=g)]
        used += npg
        pos = torch.arange(Lmax)
        tables.append((pages[(pos // page).clamp_max(npg - 1)] * page + pos % page).int())
    table = torch.stack(tables).to(device)
    N = (used + 2) * page
    kc = torch.randn((N, R.HK, R.HEAD_DIM), generator=g).to(torch.float8_e4m3fn).to(device)
    vc = torch.randn((N, R.HK, R.HEAD_DIM), generator=g).to(torch.float8_e4m3fn).to(device)
    budget = 128
    logical, rows_q = [], 0
    for i, (p, e) in enumerate(seq_spec):
        L = p + e
        qpos = torch.arange(p, L)
        nb = L // R.RATIO
        logits = torch.rand((e, max(nb, 1)), generator=g)
        ends = torch.minimum((qpos + 1) // R.RATIO, torch.tensor(nb))
        bi = R.qsa_fast_topk_xpu(logits, torch.zeros_like(ends), ends, budget // R.RATIO)
        logical.append(R.torch_expand_qsa_block_indices(bi, qpos, torch.full((e,), L), R.RATIO, budget))
    logical = torch.cat(logical).to(device)
    n = logical.shape[0]
    pad_rows = 5                                           # DP padding: q has more rows than the indexer
    q = torch.randn((n + pad_rows, R.HQ * R.HEAD_DIM), generator=g).to(torch.bfloat16).to(device)
    t2b = torch.cat([torch.full((e,), i, dtype=torch.int32) for i, (p, e) in enumerate(seq_spec)]).to(device)
    seq_lens = torch.tensor([p + e for p, e in seq_spec], dtype=torch.int32, device=device)

    md = types.SimpleNamespace(token_to_batch_idx=t2b, sequence_lengths=seq_lens, token_slot_table=table,
                               is_cuda_graph=False)

    def l2p(li, metadata):     # SGLang QwenSparseAttnBackend._logical_to_physical, verbatim
        sequence_ids = metadata.token_to_batch_idx.long()
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(0, sequence_ids)
        valid = (li >= 0) & (li < row_lengths.unsqueeze(1))
        safe = li.clamp(min=0, max=metadata.token_slot_table.shape[1] - 1).long()
        slots = metadata.token_slot_table[sequence_ids[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)

    def pad(output, num_rows):
        output = output.reshape(output.shape[0], -1)
        if output.shape[0] == num_rows:
            return output
        padded = output.new_zeros((num_rows, output.shape[1]))
        padded[: output.shape[0]].copy_(output)
        return padded

    prev_calls = []

    def prev_attention(q3, k, v, slots, scale=None):     # what exl3xpu installed before us
        prev_calls.append(q3.shape[0])
        return R.ref_fp32(q3, k, v, slots, scale)

    class FakeBackend:
        def __init__(self):
            self.token_to_kv_pool = types.SimpleNamespace(
                set_kv_buffer=lambda layer, loc, k, v: None, get_key_buffer=lambda lid: kc,
                get_value_buffer=lambda lid: vc)

        @staticmethod
        def _is_speculative_paged_mode(mode):
            return False

        def _resolve_metadata(self, fb):
            return md

        _logical_to_physical = staticmethod(l2p)
        _pad_extend_output = staticmethod(pad)

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, topk_indices=None, **kw):
            q3 = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
            rows = q3.shape[0]
            q3 = q3[: topk_indices.shape[0]]
            slots = self._logical_to_physical(topk_indices, md)
            return self._pad_extend_output(fake_qb.qsa_sparse_attention(q3, kc, vc, slots, layer.scaling), rows)

    fake_qb = types.ModuleType("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    fake_qb.qsa_sparse_attention = prev_attention
    fake_qb.QwenSparseAttnBackend = FakeBackend
    fake_qi = types.ModuleType("sglang.srt.layers.attention.qsa.qsa_indexer")

    class FakeIndexer:
        def forward_cuda(self, hidden_states, positions, forward_batch, indexer_metadata):
            return NS.current()

        def apply_rope(self, positions, tensor):
            return tensor
    fake_qi.QSAIndexer = FakeIndexer
    fake_qi.get_is_capture_mode = lambda: False
    fake_qi.apply_rotary_emb = _apply_rotary_emb_copy
    fake_qi.expand_qsa_block_indices = R.torch_expand_qsa_block_indices
    fake_md = types.ModuleType("sglang.srt.layers.attention.qsa.metadata")

    class FakeMeta:
        def get_prefill_mqa_inputs(self, layer_id, positions):
            return None
    fake_md.QSAIndexerMetadata = FakeMeta
    fake_md.build_qsa_row_ranges = _build_qsa_row_ranges_copy
    fake_km = types.ModuleType("sglang.srt.layers.attention.qsa.kernel")
    fake_km.triton_expand_qsa_block_indices = R.torch_expand_qsa_block_indices
    saved = {}
    names = {"sglang.srt.layers.attention.qwen_sparse_attn_backend": fake_qb,
             "sglang.srt.layers.attention.qsa.qsa_indexer": fake_qi,
             "sglang.srt.layers.attention.qsa.metadata": fake_md,
             "sglang.srt.layers.attention.qsa.kernel": fake_km}
    for k_, m_ in names.items():
        saved[k_] = sys.modules.get(k_)
        sys.modules[k_] = m_
    layer = types.SimpleNamespace(tp_q_head_num=R.HQ, head_dim=R.HEAD_DIM, scaling=R.HEAD_DIM ** -0.5, layer_id=3)
    fb = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True,
                                           is_target_verify=lambda: False, is_draft_extend=lambda: False),
        extend_seq_lens_cpu=[e for p, e in seq_spec], seq_lens_cpu=torch.tensor([p + e for p, e in seq_spec]),
        seq_lens=seq_lens, out_cache_loc=None, contains_mm_inputs=lambda: False)
    slots = l2p(logical, md)
    q3 = q.view(-1, R.HQ, R.HEAD_DIM)
    expect = pad(R.ref_fp32(q3[:n], kc, vc, slots, layer.scaling), n + pad_rows)
    ok = True
    try:
        for variant in ("row", "tile", "compact"):
            os.environ["EXL3_QSA_TRITON_VARIANT"] = variant
            fake_qb.qsa_sparse_attention = prev_attention
            FakeBackend.forward_extend = FakeBackend.__dict__["forward_extend"].__wrapped__ \
                if hasattr(FakeBackend.__dict__["forward_extend"], "__wrapped__") else FakeBackend.forward_extend
            import patch_qsa
            patch_qsa = importlib.reload(patch_qsa)
            patch_qsa.install(kmod=kmod)
            prev_calls.clear()
            be = FakeBackend()
            out = be.forward_extend(q, None, None, layer, fb, topk_indices=logical)
            e = err(out.view(-1, R.HQ, R.HEAD_DIM), expect.view(-1, R.HQ, R.HEAD_DIM))
            ctx = FakeIndexer().forward_cuda(None, None, fb, None)
            wrapped = getattr(FakeBackend.forward_extend, "_n107", False)
            good = (e["max_abs"] <= TOL_MAX_ABS and e["zero_rows_exact"] and not prev_calls
                    and isinstance(fake_qb.qsa_sparse_attention, patch_qsa._Dispatch)
                    and (wrapped == (variant != "row")) and ctx is not None and ctx.max_pos == 759
                    and getattr(fake_qi.expand_qsa_block_indices, "_n107", False))
            emit(dict(kind="patch_mock", variant=variant, forward_extend_wrapped=wrapped,
                      fallback_calls=len(prev_calls), host_ctx_max_pos=getattr(ctx, "max_pos", None), ok=good, **e), fh)
            ok &= good
            # unsupported input -> previous implementation
            prev_calls.clear()
            kmod_sup = kmod.supports
            kmod.supports = lambda *a, **k: False
            fake_qb.qsa_sparse_attention(q3[:n], kc, vc, slots, layer.scaling)
            kmod.supports = kmod_sup
            ok &= prev_calls == [n]
            # undo wrappers for the next variant
            FakeIndexer.forward_cuda = FakeIndexer.forward_cuda.__wrapped__
            fake_qi.expand_qsa_block_indices = R.torch_expand_qsa_block_indices
            FakeMeta.get_prefill_mqa_inputs = lambda self, layer_id, positions: None
            if hasattr(FakeBackend.forward_extend, "__wrapped__"):
                FakeBackend.forward_extend = FakeBackend.forward_extend.__wrapped__
    finally:
        for k_, m_ in saved.items():
            if m_ is None:
                sys.modules.pop(k_, None)
            else:
                sys.modules[k_] = m_
        for k_ in ("EXL3_QSA_TRITON", "EXL3_QSA_DEVICES", "EXL3_QSA_TRITON_VARIANT", "EXL3_QSA_TRITON_MIN_ROWS"):
            os.environ.pop(k_, None)
    if not ok:
        FAILS.append("patch mock")


def test_patch_mock_sycl(device, fh):
    """EXL3_QSA_IMPL=sycl_row install path against a fake SGLang backend module: the attention entry point becomes the
    sycl_row dispatcher (real qsa_sycl on XPU, a torch stand-in elsewhere), unsupported inputs and a missing library
    keep the previous function, and the indexer patches stay off unless set explicitly."""
    import importlib
    g = torch.Generator().manual_seed(5)
    case = R.make_case(512, 96, device=device, kv_dtype=torch.float8_e4m3fn, budget=256)
    q, kc, vc, slots, scale = case["q"], case["k_cache"], case["v_cache"], case["slots"], case["scale"]
    real = device.type == "xpu" and SY is not None and SY.load()
    fake = types.ModuleType("qsa_sycl")
    fake.CFG = dict(kernel=2, cpw=0, sg=16, grf=128, fp8_mode=1)
    fake.lib_path = lambda: "<fake>"
    fake.error = lambda: None
    fake.load = lambda: True
    fake.supports = lambda q_, k_, v_, s_=None: q_.shape[-1] == 256
    fake.qsa_sparse_attention_sycl = lambda q_, k_, v_, s_, sc_=None: R.emulate_row_kernel(q_, k_, v_, s_, sc_)
    prev_calls = []

    def prev_attention(q3, k, v, sl, sc=None):
        prev_calls.append(q3.shape[0])
        return R.ref_fp32(q3, k, v, sl, sc)
    fake_qb = types.ModuleType("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    saved = {k_: sys.modules.get(k_) for k_ in ("sglang.srt.layers.attention.qwen_sparse_attn_backend", "qsa_sycl")}
    sys.modules["sglang.srt.layers.attention.qwen_sparse_attn_backend"] = fake_qb
    if not real:
        sys.modules["qsa_sycl"] = fake
    env_keys = ("EXL3_QSA_IMPL", "EXL3_QSA_DEVICES", "EXL3_QSA_SYCL_MIN_ROWS", "EXL3_QSA_TRITON", "EXL3_QSA_NOSYNC",
                "EXL3_QSA_TRITON_EXPAND")
    saved_env = {k_: os.environ.get(k_) for k_ in env_keys}
    ok = True
    try:
        for k_ in env_keys:
            os.environ.pop(k_, None)
        os.environ.update(EXL3_QSA_IMPL="sycl_row", EXL3_QSA_DEVICES=device.type, EXL3_QSA_SYCL_MIN_ROWS="1")
        import patch_qsa
        patch_qsa = importlib.reload(patch_qsa)
        fake_qb.qsa_sparse_attention = prev_attention
        patch_qsa.install()
        d = fake_qb.qsa_sparse_attention
        installed = isinstance(d, patch_qsa._Dispatch) and d.name == "sycl_row"
        out = d(q, kc, vc, slots, scale) if installed else None
        e = err(out, R.ref_fp32(q, kc, vc, slots, scale)) if out is not None else {}
        good = installed and not prev_calls and e.get("max_abs", 1) <= TOL_MAX_ABS and e.get("zero_rows_exact", False)
        # unsupported -> previous
        mod = d.K if installed else None
        if installed:
            sup = mod.supports
            mod.supports = lambda *a, **k: False
            d(q, kc, vc, slots, scale)
            mod.supports = sup
            good &= prev_calls == [q.shape[0]]
        emit(dict(kind="patch_mock_sycl", real_lib=real, installed=installed, fallback_ok=prev_calls == [q.shape[0]],
                  ok=bool(good), **e), fh)
        ok &= bool(good)
        # missing library -> nothing replaced
        broken = types.ModuleType("qsa_sycl")
        broken.load = lambda: False
        broken.error = lambda: "missing"
        broken.lib_path = lambda: "<none>"
        sys.modules["qsa_sycl"] = broken
        patch_qsa = importlib.reload(patch_qsa)
        fake_qb.qsa_sparse_attention = prev_attention
        patch_qsa.install()
        kept = fake_qb.qsa_sparse_attention is prev_attention
        emit(dict(kind="patch_mock_sycl", test="missing library keeps previous", ok=kept), fh)
        ok &= kept
    finally:
        for k_, m_ in saved.items():
            if m_ is None:
                sys.modules.pop(k_, None)
            else:
                sys.modules[k_] = m_
        for k_, v_ in saved_env.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    if not ok:
        FAILS.append("patch mock sycl")


def test_expand(device, fh):
    """SGLang's Triton expander == the torch expander for exl3xpu top-k output (needs SGLang + Triton device)."""
    try:
        from sglang.srt.layers.attention.qsa.kernel import triton_expand_qsa_block_indices as texp
    except Exception as e:
        emit(dict(kind="expand", skipped=f"sglang not importable ({e!r})"), fh)
        return
    for S, Rr in ((8192, 8192), (32768, 2048), (700, 700)):
        case = R.make_case(S, Rr, device=device, kv_dtype=torch.bfloat16)
        a = R.torch_expand_qsa_block_indices(case["block_idx"], case["qpos"], case["seq_lens"], R.RATIO, R.BUDGET)
        b = texp(case["block_idx"].contiguous(), case["qpos"].contiguous(), case["seq_lens"].contiguous(),
                 R.RATIO, R.BUDGET)
        same = torch.equal(a, b)
        emit(dict(kind="expand", S=S, rows=Rr, equal=same), fh)
        if not same:
            FAILS.append(f"expand S={S}")


def test_fp8_decode(fh):
    codes = torch.arange(256, dtype=torch.uint8)
    nan = (codes & 0x7F) == 0x7F
    a = R.decode_e4m3fn_bits(codes)
    b = codes.view(torch.float8_e4m3fn).float()
    same = torch.equal(a[~nan], b[~nan]) and torch.equal(torch.signbit(a[~nan]), torch.signbit(b[~nan]))
    emit(dict(kind="fp8_decode", codes=256, equal_excluding_nan=bool(same)), fh)
    if not same:
        FAILS.append("fp8 bit decode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--tiny", action="store_true", help="small shapes (CPU / TRITON_INTERPRET)")
    ap.add_argument("--variants", default="ref,union,row,row_cast,compact,tile")
    ap.add_argument("--kv", default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--sel", default="structured", choices=["structured", "uniform"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ref-rows", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--tile-bm", type=int, default=64)
    ap.add_argument("--tile-bn", type=int, default=64)
    ap.add_argument("--emulate", action="store_true", help="torch emulators of the kernels instead of Triton")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sycl-sweep", action="store_true", help="N108: launch-shape sweep of the SYCL kernel")
    ap.add_argument("--nosync", action="store_true")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--patch-mock", action="store_true", help="drive patch_qsa.install() against fake SGLang modules")
    ap.add_argument("--dump", nargs="*", default=None)
    ap.add_argument("--edge", action="store_true", help="also run the edge-case rows (always on with --tiny)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    device = torch.device(args.device)
    fh = open(args.out, "a") if args.out else None
    emit(dict(kind="env", torch=torch.__version__, device=str(device), triton=HAVE_TRITON,
              triton_interpret=os.environ.get("TRITON_INTERPRET", "0"),
              triton_err=None if HAVE_TRITON else _TRITON_ERR,
              sycl=(None if SY is None else dict(lib=SY.lib_path(), cfg=SY.CFG)),
              xpu_name=(torch.xpu.get_device_name(0) if device.type == "xpu" else None),
              cfg=dict(row=getattr(K, "ROW_CFG", None), tile=getattr(K, "TILE_CFG", None),
                       fp8=getattr(K, "FP8_MODE", None))), fh)
    if device.type == "cpu" and HAVE_TRITON and os.environ.get("TRITON_INTERPRET", "0") != "1" and not args.emulate:
        emit(dict(kind="warn", msg="CPU without TRITON_INTERPRET=1: using the torch emulators"), fh)
        args.emulate = True
    kv_dtype = torch.float8_e4m3fn if args.kv == "fp8" else torch.bfloat16

    test_fp8_decode(fh)
    if device.type == "xpu" and SY is not None:
        test_fp8_decode_sycl(device, fh)
    if args.nosync:
        test_nosync(device, fh)
    if args.expand:
        test_expand(device, fh)
    if args.patch_mock:
        test_patch_mock(device, fh, args.emulate or not HAVE_TRITON)
        test_patch_mock_sycl(device, fh)

    if args.dump is not None:
        files = [f for p in args.dump for f in sorted(glob.glob(p))]
        for f in files:
            case = case_from_dump(f, args, device)
            run_variants(case, args, device, fh, tag=f"dump:{os.path.basename(f)}")
    else:
        if args.tiny:
            shapes = [(256, 128, 128), (640, 200, 256)]     # (S, R, budget): budget 128 -> 32 blocks -> 131 slots
        else:
            shapes = [(S, min(args.rows, S), R.BUDGET) for S in args.ctx]
        for S, Rr, budget in shapes:
            case = R.make_case(S, Rr, device=device, kv_dtype=kv_dtype, sel=args.sel, budget=budget)
            tag = f"S={S} R={Rr} kv={args.kv} sel={args.sel}"
            if args.sycl_sweep:
                run_sycl_sweep(case, args, device, fh, tag)
            elif args.sweep and HAVE_TRITON and not args.emulate:
                run_sweep(case, args, device, fh, tag)
            else:
                run_variants(case, args, device, fh, tag)
                if args.tiny or args.edge:
                    run_variants(edge_case(case), args, device, fh, tag + " edge")
            del case
            if device.type == "xpu":
                torch.xpu.empty_cache()
    emit(dict(kind="summary", fails=FAILS, ok=not FAILS), fh)
    if fh:
        fh.close()
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
