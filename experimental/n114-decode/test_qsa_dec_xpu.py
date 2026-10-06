"""N114 fused decode QSA: B70 test (correctness, eager + XPU-graph timing, graph replay with changing inputs).

Current path = what graph decode runs today per QSA layer after the indexer's block selection (exl3xpu SYCL
qsa_decode_select, kept by both paths and not timed):
  SGLang torch_expand_qsa_block_indices -> exl3xpu graph-mode _logical_to_physical (req_to_token[row_req_pool_indices])
  -> exl3xpu qsa_sparse_attention_reference (n107-qsa/qsa_xpu_base.py, mounted at /n107qsa by run_xpu.sh).
Fused = torch.ops.n114qsa.decode_attn on the block ids (expand) and on the expanded logical ids (logical mode).

usage (omarchy, from ~/freetoken-exl3):
  bench/xpu_run.sh 0000:84:00.0 n114-qsa kernels/xpu_bmg/n114-decode/run_xpu.sh test_qsa_dec_xpu.py [--quick] [--no-sweep]
Writes results/qsa_<time>.json; exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qsa_dec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bs", default="1,2,4,8")
ap.add_argument("--ctx", default="8192,32768")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--no-sweep", action="store_true")
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--emulate-cpu", action="store_true", help="harness dry run on CPU: the op is qsa_dec.emulate")
args = ap.parse_args()
if args.quick:
    args.bs, args.ctx, args.iters, args.no_sweep = "1,4", "8192", 20, True

DEV = "xpu"
if args.emulate_cpu:   # offline harness check (Mac): no library, no graphs, wall-clock timing
    DEV = "cpu"
    args.no_graph, args.no_sweep, args.iters = True, True, 1
    _emu = qsa_dec.emulate
    qsa_dec.load = lambda: True
    qsa_dec.decode_attention = lambda *a, chunk=None, cpw=None, sg=None, fp8=None, target_wg=None, **k: _emu(*a, **k)
    qsa_dec.geometry = lambda *a, **k: None

    class _Ev:
        def __init__(self, **k): self.t = 0.0
        def record(self): self.t = time.perf_counter()
        def elapsed_time(self, o): return (o.t - self.t) * 1e3

    torch.xpu.Event = _Ev
    torch.xpu.synchronize = lambda *a, **k: None
    torch.xpu.get_device_name = lambda *a, **k: "cpu (emulated op)"
RATIO, TOPK, NB = 4, 2048, 512
SCALE = 1.0 / 16
FAILS: list = []
RES: dict = {"cfg": qsa_dec.CFG, "cases": [], "timing": [], "graph": [], "sweep": []}


def note(s):
    print(s, flush=True)


def load_ref():
    for p in ("/n107qsa/qsa_xpu_base.py", os.path.join(HERE, "..", "n107-qsa", "qsa_xpu_base.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("qsa_xpu_base_ref", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise RuntimeError("n107-qsa/qsa_xpu_base.py not found (run through run_xpu.sh)")


EX = load_ref()
try:
    from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices as EXPAND
    EXPAND_SRC = "sglang"
except Exception as e:  # pragma: no cover
    note(f"sglang expansion not importable ({e}); using the emulation's expansion + argsort")
    EXPAND_SRC = "emulated"

    def EXPAND(blk, qpos, seqlen, ratio, topk):
        cells = qsa_dec.expand_cells(blk, qpos, seqlen, ratio, topk)
        order = torch.arange(cells.shape[1], device=cells.device).unsqueeze(0).expand_as(cells)
        key = torch.where(cells >= 0, order, order + cells.shape[1])
        return cells.gather(1, torch.argsort(key, dim=1, stable=True)).to(torch.int32)


def l2p(logical, t2b, rowlen, rr, rt):
    # exl3xpu qsa_xpu.install()._l2p, graph-mode branch (verbatim logic)
    sequence_ids = t2b.long()
    row_lengths = rowlen.to(torch.int32).index_select(0, sequence_ids)
    valid = (logical >= 0) & (logical < row_lengths.unsqueeze(1))
    safe = logical.clamp(min=0, max=rt.shape[1] - 1).long()
    req = rr.long().index_select(0, sequence_ids)
    slots = rt[req[:, None], safe]
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def current_path(d):
    lg = EXPAND(d["blk"], d["qpos"], d["seqlen"], RATIO, TOPK)
    slots = l2p(lg, d["t2b"], d["rowlen"], d["rr"], d["rt"])
    return EX.qsa_sparse_attention_reference(d["q"], d["k"], d["v"], slots, SCALE)


def fused(d, **cfg):
    return qsa_dec.decode_attention(d["q"], d["k"], d["v"], d["blk"], table=d["rt"], rowlen=d["rowlen"], t2b=d["t2b"],
                                    rreq=d["rr"], qpos=d["qpos"], seqlen=d["seqlen"], expand=True, ratio=RATIO,
                                    token_topk=TOPK, scale=SCALE, **cfg)


def fused_logical(d, logical, **cfg):
    return qsa_dec.decode_attention(d["q"], d["k"], d["v"], logical, table=d["rt"], rowlen=d["rowlen"], t2b=d["t2b"],
                                    rreq=d["rr"], expand=False, scale=SCALE, **cfg)


def make(bs, ctx, g, verify=False, short=False):
    """bs requests of context ctx (verify: 4 rows per request with causal prefixes); fp8 KV pool, req_to_token."""
    R = bs * 4 if verify else bs
    n_slots = (bs + 1) * (ctx + 64) + 1024
    k = torch.randn(n_slots, 2, 256, device=DEV).to(torch.float8_e4m3fn)        # device RNG (torch.manual_seed)
    v = torch.randn(n_slots, 2, 256, device=DEV).to(torch.float8_e4m3fn)
    perm = torch.randperm(n_slots, generator=g).to(torch.int32).to(DEV)          # CPU generator
    rt = perm[: (bs + 1) * (ctx + 64)].view(bs + 1, ctx + 64).contiguous()
    rr = torch.arange(1, bs + 1, device=DEV, dtype=torch.int64)                     # request rows 1..bs
    lens = [max(1, ctx - (37 * i if short else 0)) for i in range(bs)]
    if short:
        lens = [[3, 1, 9, ctx // 3, 200, 7, 5000, 2][i % 8] for i in range(bs)]
    rowlen = torch.tensor(lens, device=DEV, dtype=torch.int64)
    t2b, sl = [], []
    for r in range(R):
        s = r // 4 if verify else r
        t2b.append(s)
        sl.append(max(1, lens[s] - 3 + r % 4) if verify else lens[s])
    t2b = torch.tensor(t2b, device=DEV, dtype=torch.int32)
    seqlen = torch.tensor(sl, device=DEV, dtype=torch.int32)
    qpos = (seqlen.long() - 1)
    blk = torch.full((R, NB), -1, dtype=torch.int32, device=DEV)
    for r in range(R):
        nblk = sl[r] // RATIO
        if nblk:
            sel = torch.randperm(nblk, generator=g)[:NB].to(torch.int32).to(DEV)
            blk[r, : sel.numel()] = sel
    q = (torch.randn(R, 24, 256, device=DEV) * 0.6).to(torch.bfloat16)
    return dict(q=q, k=k, v=v, blk=blk, qpos=qpos, seqlen=seqlen, t2b=t2b, rowlen=rowlen, rr=rr, rt=rt, R=R)


def errs(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.mean())


def check_case(name, d):
    cur = current_path(d)
    fz = fused(d)
    lg = EXPAND(d["blk"], d["qpos"], d["seqlen"], RATIO, TOPK)
    fl = fused_logical(d, lg)
    ref = qsa_dec.emulate(d["q"], d["k"], d["v"], d["blk"], table=d["rt"], rowlen=d["rowlen"], t2b=d["t2b"],
                          rreq=d["rr"], qpos=d["qpos"], seqlen=d["seqlen"], expand=True, ratio=RATIO, token_topk=TOPK,
                          scale=SCALE)
    # fp32 reference without the final bf16 rounding
    m_f, a_f = errs(fz, ref)
    m_l, a_l = errs(fl, ref)
    m_c, a_c = errs(cur, ref)
    m_fc, _ = errs(fz, cur)
    finite = bool(torch.isfinite(fz.float()).all())
    # fused vs current is dominated by the current path's bf16 scores / P (large on rows with few tokens)
    ok = finite and m_f <= 3e-2 and a_f <= 2e-3 and m_l <= 3e-2 and m_fc <= max(4e-2, 2 * m_c + 1e-2)
    note(f"[{'PASS' if ok else 'FAIL'}] {name}: fused vs fp32 max {m_f:.2e} mean {a_f:.2e} | logical {m_l:.2e} | "
         f"current vs fp32 max {m_c:.2e} mean {a_c:.2e} | fused vs current {m_fc:.2e}")
    RES["cases"].append(dict(name=name, ok=ok, fused_max=m_f, fused_mean=a_f, logical_max=m_l, current_max=m_c,
                             current_mean=a_c, fused_vs_current=m_fc))
    if not ok:
        FAILS.append(name)


def time_fn(fn, iters):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) * 1e3 / iters   # us


def capture(fn):
    for _ in range(2):
        fn()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(xpu_graph=graph):
        out = fn()
    return graph, out


def timing(bs, ctx, d):
    lg = EXPAND(d["blk"], d["qpos"], d["seqlen"], RATIO, TOPK)
    row = dict(bs=bs, ctx=ctx, geometry=qsa_dec.geometry(d["R"]))
    row["current_us"] = time_fn(lambda: current_path(d), args.iters)
    row["fused_us"] = time_fn(lambda: fused(d), args.iters)
    row["fused_logical_us"] = time_fn(lambda: fused_logical(d, lg), args.iters)
    if not args.no_graph:
        for key, fn in (("current", lambda: current_path(d)), ("fused", lambda: fused(d))):
            try:
                g, _ = capture(fn)
                row[f"{key}_graph_us"] = time_fn(g.replay, args.iters)
            except Exception as e:
                row[f"{key}_graph_us"] = None
                note(f"  graph capture of {key} failed: {type(e).__name__}: {e}")
                if key == "fused":
                    FAILS.append(f"graph-capture fused bs{bs} ctx{ctx}")
    note(f"[time] bs={bs} ctx={ctx}: current {row['current_us']:.1f} us (graph {row.get('current_graph_us')}) | "
         f"fused {row['fused_us']:.1f} us (graph {row.get('fused_graph_us')}) | fused(logical) "
         f"{row['fused_logical_us']:.1f} us | geometry {row['geometry']}")
    RES["timing"].append(row)


def graph_replay_test(bs, ctx, g):
    """Capture the fused op once, then change every device input in place and replay."""
    d = make(bs, ctx, g)
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}
    graph, out = capture(lambda: fused(static))
    ok_all = True
    for step in range(3):
        nd = make(bs, ctx, g, short=(step == 1))
        for key in ("q", "blk", "qpos", "seqlen", "t2b", "rowlen", "rr", "rt"):
            static[key].copy_(nd[key])
        nd["k"], nd["v"] = static["k"], static["v"]                  # pool stays (as in serving)
        graph.replay()
        torch.xpu.synchronize()
        want = fused(nd)
        cur = current_path(nd)
        exact = bool(torch.equal(out, want))
        m, _ = errs(out, cur)
        ok = exact and m <= 6e-2
        ok_all &= ok
        note(f"[{'PASS' if ok else 'FAIL'}] graph replay bs={bs} ctx={ctx} step {step}{' (short rows)' if step == 1 else ''}:"
             f" replay == eager {exact}, vs current max {m:.2e}")
        RES["graph"].append(dict(bs=bs, ctx=ctx, step=step, exact=exact, vs_current=m, ok=ok))
    if not ok_all:
        FAILS.append(f"graph replay bs{bs}")


def sweep(g):
    for bs in (1, 4):
        d = make(bs, 32768, g)
        ref = fused(d)
        for chunk in (64, 128, 256):
            for cpw in (0, 1, 2):
                for sg in (16, 32):
                    try:
                        o = fused(d, chunk=chunk, cpw=cpw, sg=sg)
                        m, _ = errs(o, ref)
                        us = time_fn(lambda: fused(d, chunk=chunk, cpw=cpw, sg=sg), args.iters)
                    except Exception as e:
                        note(f"  sweep chunk={chunk} cpw={cpw} sg={sg}: {type(e).__name__}: {e}")
                        continue
                    geo = qsa_dec.geometry(d["R"], chunk=chunk, cpw=cpw)
                    RES["sweep"].append(dict(bs=bs, chunk=chunk, cpw=cpw, sg=sg, us=us, diff=m, geometry=geo))
                    note(f"  sweep bs={bs} chunk={chunk:3d} cpw={cpw} sg={sg}: {us:7.1f} us  geo {geo}  diff {m:.1e}")
        best = min((r for r in RES["sweep"] if r["bs"] == bs), key=lambda r: r["us"], default=None)
        if best:
            note(f"[sweep] bs={bs} best: chunk={best['chunk']} cpw={best['cpw']} sg={best['sg']} {best['us']:.1f} us")


def main():
    t0 = time.time()
    if not qsa_dec.load():
        note(f"FAIL: library not loadable: {qsa_dec.error()}")
        return 2
    note(f"device {torch.xpu.get_device_name(0)}; cfg {qsa_dec.CFG}; expansion from {EXPAND_SRC}")
    torch.manual_seed(11)
    g = torch.Generator().manual_seed(11)
    bss = [int(x) for x in args.bs.split(",")]
    ctxs = [int(x) for x in args.ctx.split(",")]
    # correctness
    for ctx in ctxs:
        for bs in bss:
            check_case(f"decode bs={bs} ctx={ctx}", make(bs, ctx, g))
    check_case("edge rows bs=8 (seq 1/2/3/7/9, mixed)", make(8, 8192, g, short=True))
    check_case("verify bs=2 x 4 rows ctx=8192", make(2, 8192, g, verify=True))
    # bf16 KV smoke
    d = make(2, 8192, g)
    d["k"], d["v"] = d["k"].to(torch.bfloat16), d["v"].to(torch.bfloat16)
    check_case("bf16 KV bs=2 ctx=8192", d)
    # timing
    for ctx in ctxs:
        for bs in bss:
            timing(bs, ctx, make(bs, ctx, g))
    if not args.no_graph:
        for bs in (1, 4):
            try:
                graph_replay_test(bs, 8192, g)
            except Exception as e:
                note(f"[FAIL] graph replay bs={bs}: {type(e).__name__}: {e}")
                FAILS.append(f"graph replay bs{bs}")
    if not args.no_sweep:
        sweep(g)
    RES["fails"] = FAILS
    RES["seconds"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"qsa_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(RES, f, indent=1)
    note(f"wrote {out} ({RES['seconds']:.0f} s)")
    note("=== ALL PASS" if not FAILS else f"=== FAIL: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
