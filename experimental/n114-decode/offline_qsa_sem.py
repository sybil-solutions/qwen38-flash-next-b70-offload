"""N114 QSA offline semantic check (Mac / CPU, no SYCL): the fused op's semantics (qsa_dec.expand_cells + mapping +
qsa_dec.emulate) against the path graph decode runs today:
  SGLang torch_expand_qsa_block_indices (verbatim source from the image-era copy sg2/qsa/kernel.py, and the newer D206
  copy) -> exl3xpu _logical_to_physical (graph mode: req_to_token[row_req_pool_indices]) / the original
  (token_slot_table) -> exl3xpu qsa_sparse_attention_reference (n107-qsa/qsa_xpu_base.py, verbatim module).
Checks: identical multisets of physical slots per row (both l2p routes, decode and verify-style rows, edge rows), and
output error of both against an fp64 reference.
usage: python3 offline_qsa_sem.py [--cases N]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import qsa_dec  # noqa: E402

SRC = {
    "sg2": os.path.join(ROOT, "integration/src_ref/sg2/qsa/kernel.py"),
    "d206": os.path.join(ROOT, "runs/2026-10-02-D206-b70-v41-runtime/source/sglang-bb9a820f09f97480bfc6d07564fb2b691b8857a3"
                               "/python/sglang/srt/layers/attention/qsa/kernel.py"),
}


def load_fn(path, name):
    tree = ast.parse(open(path).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[node], type_ignores=[])
    ns = {"torch": torch, "Optional": None}
    exec(compile(mod, path, "exec"), ns)
    return ns[name]


def load_exl3xpu():
    p = os.path.join(ROOT, "kernels/xpu_bmg/n107-qsa/qsa_xpu_base.py")
    spec = importlib.util.spec_from_file_location("qsa_xpu_base_offline", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def l2p_exl3xpu_graph(logical, t2b, seqlens, rr, rt):
    # verbatim logic of exl3xpu qsa_xpu.install()._l2p (graph-mode branch)
    sequence_ids = t2b.long()
    row_lengths = seqlens.to(torch.int32).index_select(0, sequence_ids)
    valid = (logical >= 0) & (logical < row_lengths.unsqueeze(1))
    safe = logical.clamp(min=0, max=rt.shape[1] - 1).long()
    req = rr.long().index_select(0, sequence_ids)
    slots = rt[req[:, None], safe]
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def l2p_original(logical, t2b, seqlens, table):
    # verbatim logic of QwenSparseAttnBackend._logical_to_physical
    sequence_ids = t2b.long()
    row_lengths = seqlens.to(torch.int32).index_select(0, sequence_ids)
    valid = (logical >= 0) & (logical < row_lengths.unsqueeze(1))
    safe = logical.clamp(min=0, max=table.shape[1] - 1).long()
    slots = table[sequence_ids[:, None], safe]
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def scenario(g, R, verify, edges, NS=5000, NT=6, TW=3300, ratio=4, topk=2048):
    nb = topk // ratio
    lens = [3000, 2600, 3, 1, 0, 1500, 7, 2051]
    nseq = max(1, R // 4) if verify else R
    rowlen = torch.tensor([lens[s % 8] if edges else int(torch.randint(2100, 3200, (1,), generator=g)) for s in range(nseq)])
    rr = torch.tensor([(s * 5 + 1) % NT for s in range(nseq)])
    rt = torch.stack([torch.randperm(NS, generator=g)[:TW] for _ in range(NT)]).to(torch.int32)
    t2b, sl, qp, blk = [], [], [], []
    for r in range(R):
        s = min(nseq - 1, r // 4) if verify else r
        L = int(rowlen[s])
        x = max(0, L - 3 + r % 4) if verify else L
        t2b.append(s); sl.append(x); qp.append(x - 1)
        nblk = x // ratio
        perm = torch.randperm(max(nblk, 1), generator=g)[:nb] if nblk else torch.empty(0, dtype=torch.long)
        row = torch.full((nb,), -1, dtype=torch.long)
        row[:perm.numel()] = perm
        holes = torch.rand(nb, generator=g) < 0.03
        row[holes] = -1
        if edges and r == 1 and nblk:
            row[5] = nblk + 3
        blk.append(row)
    return dict(blk=torch.stack(blk).to(torch.int32), qpos=torch.tensor(qp), seqlen=torch.tensor(sl, dtype=torch.int32),
                t2b=torch.tensor(t2b, dtype=torch.int32), rowlen=rowlen, rr=rr, rt=rt, ratio=ratio, topk=topk, NS=NS)


def fp64_attention(q, k, v, slots, scale):
    R, Hq, D = q.shape
    Hk = k.shape[1]
    out = torch.zeros(R, Hq, D, dtype=torch.float64)
    for r in range(R):
        s = slots[r][slots[r] >= 0].long()
        if s.numel() == 0:
            continue
        kk = k.index_select(0, s).double().repeat_interleave(Hq // Hk, 1)
        vv = v.index_select(0, s).double().repeat_interleave(Hq // Hk, 1)
        p = torch.softmax(torch.einsum("hd,nhd->hn", q[r].double(), kk) * scale, -1)
        out[r] = torch.einsum("hn,nhd->hd", p, vv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()
    expands = {k: load_fn(p, "torch_expand_qsa_block_indices") for k, p in SRC.items() if os.path.exists(p)}
    ex = load_exl3xpu()
    g = torch.Generator().manual_seed(3)
    fails = 0
    worst = {"fused": 0.0, "current": 0.0}
    for R in (1, 2, 4, 8):
        for verify, edges in ((False, False), (False, True), (True, False)):
            if verify and R < 4:
                continue
            for _ in range(args.cases):
                sc = scenario(g, R, verify, edges)
                NS = sc["NS"]
                k8 = (torch.randn(NS, 2, 256, generator=g) * 1.0).to(torch.float8_e4m3fn)
                v8 = (torch.randn(NS, 2, 256, generator=g) * 1.0).to(torch.float8_e4m3fn)
                kf, vf = k8.float(), v8.float()
                q = (torch.randn(R, 24, 256, generator=g) * 0.6).to(torch.bfloat16)
                scale = 1.0 / 16
                # current path, both expansion versions
                logical = None
                for name, fn in expands.items():
                    lg = fn(sc["blk"], sc["qpos"], sc["seqlen"], sc["ratio"], sc["topk"])
                    if logical is None:
                        logical = lg
                    elif not torch.equal(lg, logical):
                        print(f"[FAIL] expansion drift {name}")
                        fails += 1
                cur_slots = l2p_exl3xpu_graph(logical, sc["t2b"], sc["rowlen"], sc["rr"], sc["rt"])
                table_seq = sc["rt"][sc["rr"].long()]                       # token_slot_table-style rows
                cur_slots2 = l2p_original(logical, sc["t2b"], sc["rowlen"], table_seq)
                # fused semantics
                cells = qsa_dec.expand_cells(sc["blk"], sc["qpos"], sc["seqlen"], sc["ratio"], sc["topk"])
                seq = sc["t2b"].long()
                tlen = sc["rowlen"].long().index_select(0, seq).unsqueeze(1)
                valid = (cells >= 0) & (cells < tlen)
                col = cells.clamp(0, sc["rt"].shape[1] - 1)
                fz = torch.where(valid, sc["rt"].long()[sc["rr"].long().index_select(0, seq).unsqueeze(1), col], -1)
                same = all(torch.equal(torch.sort(cur_slots[r][cur_slots[r] >= 0].long()).values,
                                       torch.sort(fz[r][fz[r] >= 0]).values) for r in range(R))
                same2 = torch.equal(cur_slots, cur_slots2)
                # outputs
                truth = fp64_attention(q, kf, vf, cur_slots, scale)
                cur = ex.qsa_sparse_attention_reference(q, k8, v8, cur_slots, scale)
                fz_out = qsa_dec.emulate(q, k8.view(torch.uint8).view(torch.float8_e4m3fn), v8, sc["blk"],
                                         table=sc["rt"], rowlen=sc["rowlen"], t2b=sc["t2b"], rreq=sc["rr"],
                                         qpos=sc["qpos"], seqlen=sc["seqlen"], expand=True, ratio=sc["ratio"],
                                         token_topk=sc["topk"], scale=scale)
                fz_log = qsa_dec.emulate(q, k8, v8, logical, table=table_seq, rowlen=sc["rowlen"], t2b=sc["t2b"],
                                         expand=False, scale=scale)
                e_cur = float((cur.double() - truth).abs().max())
                e_fz = float((fz_out.double() - truth).abs().max())
                e_log = float((fz_log.double() - truth).abs().max())
                empty = [r for r in range(R) if not bool((cur_slots[r] >= 0).any())]
                zeros = all(bool((fz_out[r] == 0).all()) and bool((cur[r] == 0).all()) for r in empty)
                worst["fused"] = max(worst["fused"], e_fz, e_log)
                worst["current"] = max(worst["current"], e_cur)
                ok = same and same2 and zeros and e_fz <= 8e-3 and e_log <= 8e-3
                fails += not ok
                tag = "verify" if verify else "edges" if edges else "decode"
                print(f"[{'PASS' if ok else 'FAIL'}] R={R} {tag:6s} slot-sets equal {same} l2p routes equal {same2} "
                      f"empty rows {len(empty)} zero {zeros} | max_abs vs fp64: current {e_cur:.3e} fused {e_fz:.3e} "
                      f"fused(logical) {e_log:.3e}")
    print(f"worst max_abs vs fp64: current path {worst['current']:.3e}, fused semantics {worst['fused']:.3e}")
    print("ALL PASS" if not fails else f"SOME FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
