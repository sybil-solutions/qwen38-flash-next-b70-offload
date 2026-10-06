#!/usr/bin/env python3
"""N114-hc GPU test (B70 84:00.0 via run_xpu.sh, which mounts n107-hc at /hc): SYCL hyper-connection decode kernels
vs the image's GatedResidual (sglang, torch.compile'd _mix_compute / _combine_compute = what XPU runs today) and vs
n107-hc (Triton norm + oneDNN GEMMs + Triton epilogues), at decode shapes M = 1, 2, 4, 8.

    bench/xpu_run.sh 0000:84:00.0 n114-hc kernels/xpu_bmg/n114-decode/run_xpu.sh test_hc_dec_xpu.py [--ms 1,2,4,8]
        [--iters 100] [--no-graph] [--no-n107] [--sweep]

Checks (exit 0 only if all pass):
  norm     n114 n vs the reference normed residual: max ulp <= 1, bit_eq >= 99.9 %
  mix      n114 vs compiled: bit-equal or max_abs <= 2 * max|eager - compiled| (the reference's own spread)
  combine  n114 (its own l) vs compiled combine (given the reference normed): same floor rule
  fold     combine_mix == combine + mix bit for bit (r_new and mixed)
  graph    96 chained halves captured in one XPU graph, replay == eager n114 bit for bit (new inputs before replay)
  patch    patch_hc_dec routes M <= 8 XPU calls to n114 (STATS) with outputs == the direct ops; M = 16 -> original
Timing: per layer-half (mix + combine) in us, eager launches (torch.xpu.Event) and inside an XPU graph of 96 chained
halves over 8 distinct weight sets (106 MB of HC weights, larger than L2, so the weights stream like in the model).
Writes results/hc_<time>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="1,2,4,8")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--graph-halves", type=int, default=96)
ap.add_argument("--nsets", type=int, default=8)
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--no-n107", action="store_true")
ap.add_argument("--no-patch", action="store_true")
ap.add_argument("--sweep", action="store_true", help="launch-config sweep (ks, uc, mode) of the n114 half")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

HC, HS, LR, EPS = 4, 2560, 320, 1e-6
D = HC * HS
DEV = "xpu"
FAILS: list = []
RES: dict = {"time": time.strftime("%Y%m%d-%H%M%S"), "checks": [], "timing": [], "graph": [], "sweep": []}


def note(s):
    print(s, flush=True)


import hc_dec as hd  # noqa: E402

if not hd.load():
    note(f"FAIL: n114 hc library not loadable: {hd.error()}")
    sys.exit(2)

from sglang.srt.layers import hyperconnection as H  # noqa: E402

ORIG_MIX, ORIG_COMB = H.GatedResidual.mix, H.GatedResidual.combine

hx = None
if not args.no_n107:
    try:
        sys.path.insert(0, "/hc")
        import hc_xpu as hx  # noqa: E402
    except Exception as e:  # pragma: no cover
        note(f"[n107] hc_xpu not importable ({type(e).__name__}: {e}); n107 columns skipped")
        hx = None


# ---- verbatim eager reference (noise floor) ----
def mix_eager(n, wd, wu, hc, hs):
    a = F.silu(F.linear(n, wd) / hc)
    a = F.linear(a, wu)
    a = torch.sigmoid(a).unflatten(-1, (hc, hs))
    return (a * n.unflatten(-1, (hc, hs))).mean(dim=-2)


def comb_eager(bo, r, n, wi, hc, hs):
    R = r.unflatten(-1, (hc, hs))
    g = 2 * torch.sigmoid(F.linear(n, wi) / hc)
    return (R + bo.unsqueeze(-2) * g.unsqueeze(-1)).flatten(-2)


def make_gr(gen):
    cfg = H.HyperConnectionConfig(hc_count=HC, hidden_size=HS, params_dtype=torch.bfloat16, hc_lowrank=LR,
                                  rms_norm_eps=EPS, hc_per_branch_norm=True)
    gr = H.GatedResidual(cfg, use_mix=True, use_combine=True)
    with torch.no_grad():
        gr.hc_norm.weight.data = (0.2 * torch.randn(D, generator=gen)).to(torch.bfloat16).to(DEV)
        gr.input_mix_weight_down.weight.copy_((0.02 * torch.randn(LR, D, generator=gen)).to(torch.bfloat16))
        gr.input_mix_weight_up.weight.copy_((0.06 * torch.randn(D, LR, generator=gen)).to(torch.bfloat16))
        gr.block_inject_weight.weight.copy_((0.02 * torch.randn(HC, D, generator=gen)).to(torch.bfloat16))
    return gr.to(DEV)


def bits(t):
    return t.contiguous().view(torch.int16)


def ulp_max(a, b):
    ka = bits(a).to(torch.int32)
    kb = bits(b).to(torch.int32)
    ka = torch.where(ka < 0, -(ka & 0x7FFF), ka)
    kb = torch.where(kb < 0, -(kb & 0x7FFF), kb)
    return int((ka - kb).abs().max().item())


def check(name, M, got, want, floor=None, rule="floor"):
    got, want = got.to(torch.bfloat16), want.to(torch.bfloat16)
    d = (got.float() - want.float()).abs()
    beq = float((bits(got) == bits(want)).float().mean().item())
    mx = float(d.max().item())
    finite = bool(torch.isfinite(got.float()).all().item())
    if rule == "exact":
        ok = beq == 1.0
    elif rule == "ulp1":
        ok = ulp_max(got, want) <= 1 and beq >= 0.999
    else:
        ok = beq == 1.0 or (floor is not None and mx <= 2 * max(floor, 1e-30))
    ok = ok and finite
    rec = dict(name=name, M=M, bit_eq=beq, max_abs=mx, floor=floor, ok=ok)
    RES["checks"].append(rec)
    note(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} M={M}: bit_eq {100 * beq:8.4f}% max_abs {mx:.3e}"
         + (f" (floor {floor:.3e})" if floor is not None else ""))
    if not ok:
        FAILS.append(f"{name}@M={M}")
    return ok


def ev_time(fn, iters, warm=10):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) * 1e3 / iters   # us


# ---- one layer-half per path: mix -> block (bo = mixed, identity) -> combine ----
def half_ref(gr, x):
    m, res = ORIG_MIX(gr, x)
    return ORIG_COMB(gr, m, res)


def half_n107(gr, x):
    n = hx.gemma_rmsnorm_grouped(x, gr.hc_norm.weight, HS, EPS)
    m = hx.hc_mix(n, gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight, HC, HS, mode="epilogue")
    return hx.hc_combine(m, x, n, gr.block_inject_weight.weight, HC, HS, inject="torch", round_g=True)


def half_n114(gr, x):
    m, n, l = hd.mix(gr, x)
    return hd.combine(x, m, l, HC)


def correctness(M, gen):
    note(f"--- correctness M={M}")
    gr = make_gr(gen)
    x = (1.5 * torch.randn(M, D, generator=gen)).to(torch.bfloat16).to(DEV)
    bo = (0.8 * torch.randn(M, HS, generator=gen)).to(torch.bfloat16).to(DEV)
    mixed_ref, (xr, n_ref) = ORIG_MIX(gr, x)
    wd, wu, wi = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight, gr.block_inject_weight.weight
    floor_mix = float((mix_eager(n_ref, wd, wu, HC, HS).to(torch.bfloat16).float() - mixed_ref.float()).abs().max())
    comb_ref = ORIG_COMB(gr, bo, (x, n_ref))
    floor_comb = float((comb_eager(bo, x, n_ref, wi, HC, HS).to(torch.bfloat16).float() - comb_ref.float()).abs().max())
    m4, n4, l4 = hd.mix(gr, x)
    torch.xpu.synchronize()
    check("n114 norm vs reference normed", M, n4, n_ref, rule="ulp1")
    check("n114 mix vs compiled", M, m4, mixed_ref, floor_mix)
    check("n114 combine vs compiled", M, hd.combine(x, bo, l4, HC), comb_ref, floor_comb)
    if hx is not None:
        n7 = hx.gemma_rmsnorm_grouped(x, gr.hc_norm.weight, HS, EPS)
        m7 = hx.hc_mix(n7, wd, wu, HC, HS, mode="epilogue")
        check("n107 mix vs compiled (info)", M, m7, mixed_ref, floor_mix)
        check("n114 mix vs n107 (info)", M, m4, m7, floor_mix)
        FAILS[:] = [f for f in FAILS if "(info)" not in f]
    # fold: combine_mix(r, bo, l) == mix(combine(r, bo, l))
    gr2 = make_gr(gen)
    r1 = hd.combine(x, bo, l4, HC)
    mu, nu, lu = hd.mix(gr2, r1)
    mf, rf, nf, lf = hd.combine_mix(gr2, x, bo, l4)
    check("fold r_new == combine", M, rf, r1, rule="exact")
    check("fold mixed == combine + mix", M, mf, mu, rule="exact")
    check("fold l == unfolded l", M, lf.to(torch.bfloat16), lu.to(torch.bfloat16), rule="exact")
    # norm weight dtypes (the checkpoint stores hc_norm.weight as F16; runtime may be fp32 / bf16 / fp16)
    for dt in (torch.float16, torch.float32):
        w0 = gr.hc_norm.weight.data
        gr.hc_norm.weight.data = w0.to(dt)
        _, (_, n_r) = ORIG_MIX(gr, x)
        _, n_d, _ = hd.mix(gr, x)
        check(f"n114 norm ({str(dt)[6:]} weight) vs reference", M, n_d, n_r, rule="ulp1")
        gr.hc_norm.weight.data = w0
    # every launch config agrees (fp32 sums differ only in the split order of the down partials)
    for ks, uc, mode in ((1, 16, 0), (4, 64, 1), (8, 32, 1), (2, 8, 0)):
        mm, _, _ = hd.mix(gr, x, ks=ks, uc=uc, mode=mode)
        check(f"n114 mix cfg ks{ks} uc{uc} mode{mode} vs compiled", M, mm, mixed_ref, floor_mix)


def timing(M, gen):
    gr = make_gr(gen)
    x = (1.5 * torch.randn(M, D, generator=gen)).to(torch.bfloat16).to(DEV)
    rec = {"M": M}
    rec["ref_compiled_us"] = ev_time(lambda: half_ref(gr, x), args.iters)
    if hx is not None:
        rec["n107_us"] = ev_time(lambda: half_n107(gr, x), args.iters)
    rec["n114_us"] = ev_time(lambda: half_n114(gr, x), args.iters)
    m, n, l = hd.mix(gr, x)
    rec["n114_mix_only_us"] = ev_time(lambda: hd.mix(gr, x), args.iters)
    rec["n114_combine_only_us"] = ev_time(lambda: hd.combine(x, m, l, HC), args.iters)
    rec["n114_fold_half_us"] = ev_time(lambda: hd.combine_mix(gr, x, m, l), args.iters)
    RES["timing"].append(rec)
    note(f"[eager M={M}] per half (mix+combine): " + ", ".join(f"{k} {v:.1f}" for k, v in rec.items() if k != "M"))


def graph_bench(M, gen):
    note(f"--- graph M={M}: {args.graph_halves} chained halves over {args.nsets} weight sets")
    if not (hasattr(torch.xpu, "XPUGraph") and hasattr(torch.xpu, "graph")):
        note("[graph] torch.xpu.XPUGraph not available: SKIP")
        return
    grs = [make_gr(gen) for _ in range(args.nsets)]
    x0 = (1.5 * torch.randn(M, D, generator=gen)).to(torch.bfloat16).to(DEV)
    K = args.graph_halves

    def chain_ref(x):
        for i in range(K):
            x = half_ref(grs[i % len(grs)], x)
        return x

    def chain_n107(x):
        for i in range(K):
            x = half_n107(grs[i % len(grs)], x)
        return x

    def chain_n114(x):
        for i in range(K):
            x = half_n114(grs[i % len(grs)], x)
        return x

    def chain_fold(x):
        m, n, l = hd.mix(grs[0], x)
        r = x
        for i in range(1, K):
            m, r, n, l = hd.combine_mix(grs[i % len(grs)], r, m, l)
        return hd.combine(r, m, l, HC)

    paths = [("ref_compiled", chain_ref)] + ([("n107", chain_n107)] if hx is not None else []) + \
        [("n114", chain_n114), ("n114_fold", chain_fold)]
    rec = {"M": M, "halves": K}
    outs = {}
    for name, fn in paths:
        try:
            xin = x0.clone()
            for _ in range(2):
                fn(xin)
            torch.xpu.synchronize()
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(xpu_graph=g):
                out = fn(xin)
            torch.xpu.synchronize()
            xin.copy_((1.5 * torch.randn(M, D, generator=gen)).to(torch.bfloat16))
            g.replay()
            torch.xpu.synchronize()
            got = out.clone()
            if name.startswith("n114"):
                want = fn(xin)
                torch.xpu.synchronize()
                check(f"graph replay == eager ({name})", M, got, want, rule="exact")
                outs[name] = got
            t = ev_time(g.replay, 20, warm=3)
            rec[f"{name}_graph_us_per_half"] = t / K
            rec[f"{name}_graph_ms_per_96"] = t * 96 / K / 1e3
            del g
        except Exception as e:
            tag = "FAIL" if name.startswith("n114") else "INFO (baseline path, not counted)"
            note(f"  [{tag}] graph {name} M={M}: {type(e).__name__}: {e}")
            traceback.print_exc()
            if name.startswith("n114"):
                FAILS.append(f"graph:{name}@M={M}")
    if "n114" in outs and "n114_fold" in outs:
        check("graph fold chain == unfolded chain", M, outs["n114_fold"], outs["n114"], rule="exact")
    RES["graph"].append(rec)
    note(f"[graph M={M}] per half: " + ", ".join(f"{k[:-len('_graph_us_per_half')]} {v:.1f} us" for k, v in rec.items()
                                                  if k.endswith("_graph_us_per_half")))


def patch_test(gen):
    note("--- patch_hc_dec routing (GatedResidual.mix / combine, installed sglang)")
    os.environ["EXL3_HC_DEC_SYCL"] = "1"
    import importlib
    import patch_hc_dec as P
    P = importlib.reload(P)
    try:
        assert P.install(import_now=True), "install failed"
        gr = make_gr(gen)
        x = (1.5 * torch.randn(2, D, generator=gen)).to(torch.bfloat16).to(DEV)
        m, res = gr.mix(x)
        out = gr.combine(m, res)
        m4, n4, l4 = hd.mix(gr, x)
        want = hd.combine(x, m4, l4, HC)
        ok = P.STATS["mix_sycl"] == 1 and P.STATS["combine_sycl"] == 1
        check("patched mix == direct n114", 2, m, m4, rule="exact")
        check("patched combine == direct n114", 2, out, want, rule="exact")
        xb = (1.5 * torch.randn(16, D, generator=gen)).to(torch.bfloat16).to(DEV)
        mb, resb = gr.mix(xb)
        gr.combine(mb, resb)
        ok = ok and P.STATS["mix_orig"] == 1 and P.STATS["combine_orig"] == 1
        note(f"  [{'PASS' if ok else 'FAIL'}] routing STATS {dict(P.STATS)}")
        if not ok:
            FAILS.append("patch routing")
    except Exception as e:
        note(f"  [FAIL] patch test: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILS.append("patch")
    finally:
        P.unpatch()


def sweep(M, gen):
    gr = make_gr(gen)
    x = (1.5 * torch.randn(M, D, generator=gen)).to(torch.bfloat16).to(DEV)
    best = None
    for ks in (1, 2, 4, 8, 16):
        for uc in (8, 16, 32, 64):
            for mode in (0, 1):
                try:
                    t = ev_time(lambda: hd.mix(gr, x, ks=ks, uc=uc, mode=mode), 50)
                except Exception as e:
                    note(f"  sweep ks{ks} uc{uc} mode{mode}: {type(e).__name__}")
                    continue
                RES["sweep"].append(dict(M=M, ks=ks, uc=uc, mode=mode, mix_us=t))
                if best is None or t < best[0]:
                    best = (t, ks, uc, mode)
    note(f"[sweep M={M}] best mix {best[0]:.1f} us at ks={best[1]} uc={best[2]} mode={best[3]}")


def main():
    gen = torch.Generator().manual_seed(args.seed)
    ms = [int(s) for s in args.ms.split(",")]
    note(f"device {torch.xpu.get_device_name(0)}; n114 cfg {hd.CFG}; n107 {'yes' if hx is not None else 'no'}")
    for M in ms:
        try:
            correctness(M, gen)
        except Exception as e:
            note(f"  [FAIL] correctness M={M}: {type(e).__name__}: {e}")
            traceback.print_exc()
            FAILS.append(f"correctness@M={M}")
    for M in ms:
        try:
            timing(M, gen)
        except Exception as e:
            note(f"  [FAIL] timing M={M}: {type(e).__name__}: {e}")
            traceback.print_exc()
            FAILS.append(f"timing@M={M}")
    if not args.no_graph:
        for M in [m for m in ms if m in (1, 2, 4)]:
            graph_bench(M, gen)
    if not args.no_patch:
        patch_test(gen)
    if args.sweep:
        for M in [m for m in ms if m in (1, 4)]:
            sweep(M, gen)
    RES["fails"] = FAILS
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"hc_{RES['time']}.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=1)
    note(f"wrote {path}")
    note("=== ALL PASS" if not FAILS else f"=== FAIL: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
