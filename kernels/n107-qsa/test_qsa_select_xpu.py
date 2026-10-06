#!/usr/bin/env python3
"""N108 QSA prefill block selection on the B70: current path (SGLang torch_qsa_mqa_prefill fp32 einsum + relu-sum,
row-chunked like QSAIndexer.select_prefill_tokens, + exl3xpu qsa_fast_topk torch.topk) vs the SYCL port
(torch.ops.n108qsa.prefill_select: fused scores + exact radix top-k, Strata's rule: ties -> lowest index).

  python3 test_qsa_select_xpu.py --ctx 8192 32768 --rows 8192 --out /n107qsa/results/n108_sel.jsonl

Reported per case: ms per layer of both paths, identical selected sets (fraction of rows), mean Jaccard of the sets,
score agreement (SYCL scores vs torch logits), and the exactness of the SYCL top-k against torch.topk applied to the
SYCL scores themselves (must select identical sets except for exact ties, which the rules order differently).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import qsa_ref as R  # noqa: E402
import qsa_sycl as SY  # noqa: E402

LOGITS_BUDGET = 128 * 1024 * 1024   # SGLang _QSA_PREFILL_LOGITS_BUDGET_BYTES


def torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale=None):
    """Verbatim SGLang qsa/mqa.py::torch_qsa_mqa_prefill (the XPU path)."""
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (columns < row_ends.to(q.device).reshape(-1, 1))
    return logits.masked_fill(~valid, -float("inf"))


def row_chunk(rows, keys, heads):
    """SGLang _qsa_prefill_row_chunk_size."""
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    max_rows = max(block_q, LOGITS_BUDGET // (keys * 4))
    max_rows = max(block_q, max_rows // block_q * block_q)
    return min(rows, max_rows)


def current_select(q, k3, rs, re, topk):
    rows = q.shape[0]
    out = torch.empty((rows, topk), dtype=torch.int32, device=q.device)
    cs = row_chunk(rows, k3.shape[0], q.shape[1])
    for r0 in range(0, rows, cs):
        r1 = min(rows, r0 + cs)
        logits = torch_qsa_mqa_prefill(q[r0:r1], k3, rs[r0:r1], re[r0:r1])
        out[r0:r1] = R.qsa_fast_topk_xpu(logits, rs[r0:r1], re[r0:r1], topk)
    return out


def sets(bi):
    """[R, topk] block indices (any order, -1 padded) -> sorted with -1 last (comparable row by row)."""
    big = torch.iinfo(torch.int32).max
    x = torch.where(bi >= 0, bi, torch.full_like(bi, big))
    return torch.sort(x, dim=1).values


def compare(a, b):
    sa, sb = sets(a), sets(b)
    same_rows = (sa == sb).all(dim=1)
    va, vb = a >= 0, b >= 0
    na, nb = va.sum(1), vb.sum(1)
    # |A & B| via membership of b's entries in a (both sorted)
    big = torch.iinfo(torch.int32).max
    pos = torch.searchsorted(sa.contiguous(), sb.contiguous())
    hit = (torch.gather(sa, 1, pos.clamp_max(sa.shape[1] - 1)) == sb) & (sb != big)
    inter = hit.sum(1)
    union = na + nb - inter
    jac = torch.where(union > 0, inter.float() / union.clamp_min(1).float(), torch.ones_like(inter, dtype=torch.float32))
    return dict(identical_rows=float(same_rows.float().mean()), mean_jaccard=float(jac.mean()),
                min_jaccard=float(jac.min()), cells_differing=int((na - inter).sum()))


def bench(fn, iters, warmup=1):
    out = fn()
    torch.xpu.synchronize()
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return out, statistics.median(ts)


def patch_test(fh, dev):
    """SGLang's real QSAIndexer.select_prefill_tokens (with exl3xpu's qsa_fast_topk, as the server runs it) vs the
    EXL3_QSA_SELECT=sycl replacement installed by patch_qsa, on a 2-sequence chunked batch (row_starts > 0)."""
    import types
    os.environ["EXL3_QSA_DEVICES"] = "xpu"
    from sglang.srt.layers.attention.qsa import qsa_indexer as qi
    qi.qsa_fast_topk = R.qsa_fast_topk_xpu            # what exl3xpu.qsa_xpu.install() sets
    import patch_qsa
    orig = qi.QSAIndexer.select_prefill_tokens
    msg = patch_qsa._install_select_sycl()
    new = qi.QSAIndexer.select_prefill_tokens
    self = types.SimpleNamespace(block_topk=R.BLOCK_TOPK, compress_ratio=R.RATIO, token_topk=R.BUDGET)
    g = torch.Generator().manual_seed(9)
    lens, ext = [20000, 9000], [3000, 5000]
    keys, q, rs, re, qp, sl = [], [], [], [], [], []
    off = 0
    for L, e in zip(lens, ext):
        C = L // R.RATIO
        keys.append(torch.randn((C, 1, R.INDEX_DIM), generator=g))
        q.append(torch.randn((e, R.INDEX_HEADS, R.INDEX_DIM), generator=g))
        pos = torch.arange(L - e, L)
        qp.append(pos)
        rs.append(torch.full((e,), off, dtype=torch.int32))
        re.append((off + torch.minimum((pos + 1) // R.RATIO, torch.tensor(C))).to(torch.int32))
        sl.append(torch.full((e,), L, dtype=torch.int32))
        off += C
    keys = torch.cat(keys).to(torch.bfloat16).to(dev)
    q = torch.cat(q).to(torch.bfloat16).to(dev)
    rs, re, qp, sl = (torch.cat(x).to(dev) for x in (rs, re, qp, sl))
    a = orig(self, q, keys, rs, re, qp, sl)
    b = new(self, q, keys, rs, re, qp, sl)
    c = compare(a, b)
    ok = c["identical_rows"] == 1.0 and a.shape == b.shape and a.dtype == b.dtype
    emit(dict(kind="patch_select", install=msg, rows=int(q.shape[0]), shape=list(b.shape), **c, ok=ok), fh)
    qi.QSAIndexer.select_prefill_tokens = orig
    return ok


def emit(rec, fh):
    print(json.dumps(rec), flush=True)
    if fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=R.BLOCK_TOPK)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rpw", type=int, nargs="+", default=[8])
    ap.add_argument("--kscale", type=float, default=1.0)
    ap.add_argument("--patch-test", action="store_true", help="real SGLang select_prefill_tokens vs the patched one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("xpu")
    fh = open(args.out, "a") if args.out else None
    ok_lib = SY.load()
    emit(dict(kind="env", torch=torch.__version__, xpu=torch.xpu.get_device_name(0), sycl_lib=SY.lib_path(),
              sycl_ok=ok_lib, err=SY.error()), fh)
    fails = []
    if not ok_lib or not hasattr(torch.ops.n108qsa, "prefill_select"):
        emit(dict(kind="summary", ok=False, fails=["library"]), fh)
        sys.exit(1)
    if args.patch_test and not patch_test(fh, dev):
        fails.append("patched select_prefill_tokens differs")
    for S in args.ctx:
        Rr = min(args.rows, S)
        g = torch.Generator().manual_seed(S)
        C = S // R.RATIO
        q = torch.randn((Rr, R.INDEX_HEADS, R.INDEX_DIM), generator=g).to(torch.bfloat16).to(dev)
        k = (torch.randn((C, R.INDEX_DIM), generator=g) * args.kscale).to(torch.bfloat16).to(dev)
        k3 = k.view(C, 1, R.INDEX_DIM)
        qpos = torch.arange(S - Rr, S, device=dev)
        re = torch.minimum((qpos + 1) // R.RATIO, torch.tensor(C, device=dev)).to(torch.int32)
        rs = torch.zeros_like(re)
        tag = f"S={S} R={Rr} C={C} topk={args.topk}"
        cur, cur_ms = bench(lambda: current_select(q, k3, rs, re, args.topk), args.iters)
        emit(dict(tag=tag, kind="time", variant="current_torch", ms=round(cur_ms, 3)), fh)
        for rpw in args.rpw:
            new, new_ms = bench(lambda: torch.ops.n108qsa.prefill_select(q, k, rs, re, args.topk, math.sqrt(R.INDEX_DIM),
                                                                         rpw, 128), args.iters)
            c = compare(cur, new)
            emit(dict(tag=tag, kind="result", variant=f"sycl_select_rpw{rpw}", ms=round(new_ms, 3),
                      speedup=round(cur_ms / new_ms, 2), per_8k_chunk_s_saved_12_layers=round(12 * (cur_ms - new_ms) * 1e-3
                                                                                              * 8192 / Rr, 3),
                      vs_current=c), fh)
        # score agreement on a row sample + exactness of the radix top-k on the SYCL scores
        idx = torch.linspace(0, Rr - 1, min(Rr, 512)).long().to(dev)
        sc = torch.ops.n108qsa.prefill_scores(q[idx].contiguous(), k, rs[idx].contiguous(), re[idx].contiguous(),
                                              math.sqrt(R.INDEX_DIM), 8)
        cols = torch.arange(C, device=dev)[None, :]
        valid = cols < (re[idx] - rs[idx])[:, None]
        ref = torch_qsa_mqa_prefill(q[idx], k3, rs[idx], re[idx])
        d = (sc - ref).abs().masked_fill(~valid, 0)
        rel = float((d / ref.abs().clamp_min(1e-6)).masked_fill(~valid, 0).max())
        bitexact = float(((sc == ref) | ~valid).all(dim=1).float().mean())
        tk_on_sycl = R.qsa_fast_topk_xpu(sc.masked_fill(~valid, -float("inf")), rs[idx], re[idx], args.topk)
        new_rows = torch.ops.n108qsa.prefill_select(q[idx].contiguous(), k, rs[idx].contiguous(), re[idx].contiguous(),
                                                    args.topk, math.sqrt(R.INDEX_DIM), 8, 128)
        exact = compare(tk_on_sycl, new_rows)
        ties = int(sum(int((sc[i][valid[i]].unique().numel() != int(valid[i].sum()))) for i in range(sc.shape[0])))
        emit(dict(tag=tag, kind="scores", rows=int(idx.numel()), max_abs=float(d.max()), max_rel=rel,
                  rows_bitexact=bitexact, rows_with_tied_scores=ties, sycl_topk_vs_torch_topk_on_sycl_scores=exact), fh)
        if exact["identical_rows"] < 1.0 and ties == 0:
            fails.append(f"{tag}: radix top-k differs from torch.topk on identical scores")
        del q, k, cur
        torch.xpu.empty_cache()
    emit(dict(kind="summary", ok=not fails, fails=fails), fh)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
