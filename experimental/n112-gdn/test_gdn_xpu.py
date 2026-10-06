"""N112 standalone harness on the B70 (run through run_xpu.sh): correctness of the SYCL GDN prompt recurrence against
SGLang's Triton chunk_gated_delta_rule and an fp64 token-serial reference, then per-layer timing at T=8192.

usage: python3 test_gdn_xpu.py [--quick] [--no-time] [--cfgs 4,32,128;8,16,128;...]
Writes results/n112_<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gdn_sycl  # noqa: E402

DEV = "xpu"
HK, H, K, V = 16, 48, 128, 128
QKV = 2 * HK * K + H * V   # 10240 conv channels
SCALE = K ** -0.5
RES = {"tests": [], "timing": {}}


def log(*a):
    print(*a, flush=True)


def sync():
    torch.xpu.synchronize()


# ------------------------------------------------------------------ inputs
def make_inputs(T, seed, slots=8, state_dtype=torch.float32, state_scale=0.3):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mixed = (torch.randn(T, QKV, generator=gen) * 0.7).to(torch.bfloat16)
    mixed = torch.nn.functional.silu(mixed.float()).to(torch.bfloat16).to(DEV)   # like the conv output
    q = mixed[:, : HK * K].view(1, T, HK, K)
    k = mixed[:, HK * K: 2 * HK * K].view(1, T, HK, K)
    v = mixed[:, 2 * HK * K:].view(1, T, H, V)
    a = (torch.randn(T, H, generator=gen)).to(torch.bfloat16).to(DEV)
    b = (torch.randn(T, H, generator=gen)).to(torch.bfloat16).to(DEV)
    A_log = torch.log(torch.empty(H).uniform_(0.5, 16.0, generator=gen)).to(DEV)
    dt_bias = (torch.randn(H, generator=gen) * 0.5).to(DEV)
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    pool = (torch.randn(slots, H, V, K, generator=gen) * state_scale).to(state_dtype).to(DEV)
    return dict(mixed=mixed, q=q, k=k, v=v, g=g, beta=beta, pool=pool)


# ------------------------------------------------------------------ implementations
def run_triton(q, k, v, g, beta, pool, idx, cu):
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    o, _, h = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=pool, initial_state_indices=idx,
                                     cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True)
    return o


def run_sycl(q, k, v, g, beta, pool, idx, cu, **cfg):
    o, _, _ = gdn_sycl.chunk_gated_delta_rule_sycl(q, k, v, g, beta, None, pool, idx, cu, True, **cfg)
    return o


def ref_fp64(q, k, v, g, beta, pool, idx, cu, dev=None):
    """token-serial fp64 reference; returns (o [1,T,H,V] fp64 cpu, {slot: final state fp64})."""
    dev = dev or REF_DEV
    T = q.shape[1]
    qd = q[0].to(dev, torch.float64)
    kd = k[0].to(dev, torch.float64)
    qd = qd / torch.sqrt((qd * qd).sum(-1, keepdim=True) + 1e-6)
    kd = kd / torch.sqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
    rep = H // HK
    qd = qd.repeat_interleave(rep, dim=1)   # [T, H, K]
    kd = kd.repeat_interleave(rep, dim=1)
    vd = v[0].to(dev, torch.float64)
    gd = g.reshape(T, H).to(dev, torch.float64).exp()
    bd = beta.reshape(T, H).to(dev, torch.float64)
    o = torch.zeros(T, H, V, dtype=torch.float64, device=dev)
    finals = {}
    cu_l = cu.tolist()
    idx_l = idx.tolist()
    for n in range(len(idx_l)):
        slot = idx_l[n]
        S = pool[slot].to(dev, torch.float64).clone() if slot >= 0 else torch.zeros(H, V, K, dtype=torch.float64, device=dev)
        for t in range(cu_l[n], cu_l[n + 1]):
            S.mul_(gd[t].view(H, 1, 1))
            kv = torch.einsum("hvk,hk->hv", S, kd[t])
            d = bd[t].view(H, 1) * (vd[t] - kv)
            S.add_(d.unsqueeze(-1) * kd[t].unsqueeze(1))
            o[t] = SCALE * torch.einsum("hvk,hk->hv", S, qd[t])
        if slot >= 0:
            finals[slot] = S
    return o.unsqueeze(0).cpu(), {s: x.cpu() for s, x in finals.items()}


def err(x, ref):
    x = x.double().cpu()
    ref = ref.double().cpu()
    d = (x - ref)
    return dict(rel=float(d.norm() / max(ref.norm(), 1e-30)), max_abs=float(d.abs().max()),
                ref_absmax=float(ref.abs().max()))


def state_err(pool, finals):
    out = {}
    for s, ref in finals.items():
        out[s] = err(pool[s], ref)
    worst = max((e["rel"] for e in out.values()), default=0.0)
    return worst, out


def record(name, ok, **kw):
    log(f"[{'PASS' if ok else 'FAIL'}] {name} " + " ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, dict)))
    RES["tests"].append(dict(name=name, ok=bool(ok), **kw))


# ------------------------------------------------------------------ correctness
def check_case(name, T, cu_list, idx_list, seed, state_dtype=torch.float32, do_ref=True, cfg=None):
    cfg = cfg or {}
    inp = make_inputs(T, seed, state_dtype=state_dtype)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)
    idx = torch.tensor(idx_list, dtype=torch.int32, device=DEV)
    pool0 = inp["pool"]
    p_tr = pool0.clone()
    p_sy = pool0.clone()
    o_tr = run_triton(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], p_tr, idx, cu)
    o_sy = run_sycl(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], p_sy, idx, cu, **cfg)
    sync()
    # untouched slots must stay bit-identical
    used = {s for s in idx_list if s >= 0}
    untouched_ok = all(torch.equal(p_sy[s], pool0[s]) for s in range(pool0.shape[0]) if s not in used)
    res = dict(T=T, nseq=len(idx_list), state_dtype=str(state_dtype), cfg=cfg)
    res["sycl_vs_triton_o"] = err(o_sy, o_tr)
    sw = max(err(p_sy[s].float(), p_tr[s].float())["rel"] for s in used) if used else 0.0
    res["sycl_vs_triton_state_rel"] = sw
    ok = untouched_ok and math.isfinite(res["sycl_vs_triton_o"]["rel"])
    if do_ref:
        o_ref, finals = ref_fp64(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], pool0.float(), idx.cpu(), cu.cpu())
        res["sycl_vs_ref_o"] = err(o_sy, o_ref)
        res["triton_vs_ref_o"] = err(o_tr, o_ref)
        res["sycl_vs_ref_state_rel"] = state_err(p_sy.float(), finals)[0]
        res["triton_vs_ref_state_rel"] = state_err(p_tr.float(), finals)[0]
        # SYCL must be at least about as accurate as Triton (both vs fp64), and accurate in absolute terms
        tol_o = max(1.5 * res["triton_vs_ref_o"]["rel"], 5e-3)
        tol_s = max(1.5 * res["triton_vs_ref_state_rel"], 5e-3 if state_dtype == torch.float32 else 2e-2)
        ok = ok and res["sycl_vs_ref_o"]["rel"] <= tol_o and res["sycl_vs_ref_state_rel"] <= tol_s
    else:
        ok = ok and res["sycl_vs_triton_o"]["rel"] < 2e-2 and sw < 2e-2
    res["untouched_slots_ok"] = untouched_ok
    record(name, ok, **{k_: (round(v_["rel"], 6) if isinstance(v_, dict) and "rel" in v_ else v_) for k_, v_ in res.items()
                        if k_ not in ("cfg",)}, detail=res)
    return ok


def check_continuation(T=16384, seed=7, cfg=None):
    """process T tokens as 2 chunks (state carried in the pool) vs one shot; SYCL and Triton."""
    cfg = cfg or {}
    inp = make_inputs(T, seed)
    half = T // 2
    idx = torch.tensor([2], dtype=torch.int32, device=DEV)
    out = {}
    for impl in ("sycl", "triton"):
        fn = run_sycl if impl == "sycl" else run_triton
        kw = cfg if impl == "sycl" else {}
        p1 = inp["pool"].clone()
        o1 = fn(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], p1, idx,
                torch.tensor([0, T], dtype=torch.int32, device=DEV), **kw)
        p2 = inp["pool"].clone()
        parts = []
        for c0 in (0, half):
            sl = slice(c0, c0 + half)
            qc, kc, vc = inp["q"][:, sl], inp["k"][:, sl], inp["v"][:, sl]
            gc, bc = inp["g"][:, sl], inp["beta"][:, sl]
            parts.append(fn(qc, kc, vc, gc, bc, p2, idx, torch.tensor([0, half], dtype=torch.int32, device=DEV), **kw))
        o2 = torch.cat(parts, dim=1)
        sync()
        out[impl] = dict(o_chunked_vs_oneshot=err(o2, o1), state_chunked_vs_oneshot=err(p2[2], p1[2]),
                         o1=o1, p1=p1[2].clone())
    o_ref, finals = ref_fp64(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["pool"].float(), idx.cpu(),
                             torch.tensor([0, T], dtype=torch.int32))
    res = {}
    for impl in ("sycl", "triton"):
        res[impl] = dict(o_chunked_vs_oneshot=out[impl]["o_chunked_vs_oneshot"],
                         state_chunked_vs_oneshot=out[impl]["state_chunked_vs_oneshot"],
                         o_vs_ref=err(out[impl]["o1"], o_ref), state_vs_ref=err(out[impl]["p1"], finals[2]))
    s = res["sycl"]
    # token-serial fp32 with the state carried in an fp32 pool: chunking must not change anything (bit-identical
    # math in the same order); allow tiny slack anyway
    ok = (s["o_chunked_vs_oneshot"]["max_abs"] <= 1e-6 + 0 * 1 and s["state_chunked_vs_oneshot"]["max_abs"] <= 1e-6
          and s["o_vs_ref"]["rel"] <= max(1.5 * res["triton"]["o_vs_ref"]["rel"], 5e-3)
          and s["state_vs_ref"]["rel"] <= max(1.5 * res["triton"]["state_vs_ref"]["rel"], 5e-3))
    record(f"continuation T={T} 2x{half}", ok,
           sycl_o_chunk_vs_one=s["o_chunked_vs_oneshot"]["max_abs"], sycl_state_chunk_vs_one=s["state_chunked_vs_oneshot"]["max_abs"],
           sycl_o_ref=round(s["o_vs_ref"]["rel"], 6), triton_o_ref=round(res["triton"]["o_vs_ref"]["rel"], 6),
           sycl_state_ref=round(s["state_vs_ref"]["rel"], 6), triton_state_ref=round(res["triton"]["state_vs_ref"]["rel"], 6),
           triton_o_chunk_vs_one=res["triton"]["o_chunked_vs_oneshot"]["rel"], detail=res)
    return ok


# ------------------------------------------------------------------ timing
def bench(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return dict(ms_med=1e3 * ts[len(ts) // 2], ms_min=1e3 * ts[0])


def timing(T=8192, cfgs=None):
    inp = make_inputs(T, 11)
    q, k, v, g, beta, pool = inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["pool"]
    tim = {}

    def seqs(n):
        cu = torch.tensor([i * (T // n) for i in range(n)] + [T], dtype=torch.int32, device=DEV)
        idx = torch.arange(n, dtype=torch.int32, device=DEV)
        return cu, idx

    for n in (1, 2, 4):
        cu, idx = seqs(n)
        p = pool.clone()
        tim[f"triton_core_T{T}_n{n}"] = bench(lambda: run_triton(q, k, v, g, beta, p, idx, cu))
        for c in cfgs:
            p2 = pool.clone()
            tim[f"sycl_core_T{T}_n{n}_cfg{c['rg']},{c['cb']},{c['grf']}"] = bench(
                lambda: run_sycl(q, k, v, g, beta, p2, idx, cu, **c))
    # Triton chain breakdown (single sequence), the same calls ChunkGatedDeltaRuleFunction makes
    try:
        from sglang.kernels.ops.attention.fla import chunk as C
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
        cu, idx = seqs(1)
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        tim["tr_contig_copies_qkv"] = bench(lambda: (q.contiguous(), k.contiguous(), v.contiguous()))
        tim["tr_l2norm_qk"] = bench(lambda: (l2norm_fwd(qc), l2norm_fwd(kc)))
        qn, kn = l2norm_fwd(qc), l2norm_fwd(kc)
        ci = prepare_chunk_indices(cu, 64)
        tim["tr_prepare_chunk_indices"] = bench(lambda: prepare_chunk_indices(cu, 64))
        tim["tr_cumsum"] = bench(lambda: C.chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu, chunk_indices=ci))
        gc = C.chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu, chunk_indices=ci)
        tim["tr_intra_kkt_solve_wu"] = bench(lambda: C.chunk_gated_delta_rule_fwd_intra(
            k=kn, v=vc, g=gc, beta=beta, cu_seqlens=cu, chunk_indices=ci))
        w, u, A = C.chunk_gated_delta_rule_fwd_intra(k=kn, v=vc, g=gc, beta=beta, cu_seqlens=cu, chunk_indices=ci)
        p = pool.clone()
        tim["tr_fwd_h"] = bench(lambda: C.chunk_gated_delta_rule_fwd_h(
            k=kn, w=w, u=u, g=gc, initial_state=p, initial_state_indices=idx, cu_seqlens=cu, chunk_indices=ci))
        h, v_new = C.chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=gc, initial_state=p, initial_state_indices=idx,
                                                  cu_seqlens=cu, chunk_indices=ci)
        tim["tr_fwd_o"] = bench(lambda: C.chunk_fwd_o(q=qn, k=kn, v=v_new, h=h, g=gc, scale=SCALE, cu_seqlens=cu))
    except Exception as e:
        tim["breakdown_error"] = f"{type(e).__name__}: {e}"
    # rest of the GDN layer chain (unchanged by N112), for the per-layer total
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
        cu, idx = seqs(1)
        x = inp["mixed"].clone()
        w4 = (torch.randn(QKV, 4) * 0.3).to(torch.bfloat16).to(DEV)
        conv_states = torch.zeros(4, QKV, 3, dtype=torch.bfloat16, device=DEV)
        has_init = torch.ones(1, dtype=torch.bool, device=DEV)
        tim["conv1d"] = bench(lambda: causal_conv1d_fn(x.transpose(0, 1), w4, None, activation="silu",
                                                       conv_states=conv_states, has_initial_state=has_init,
                                                       cache_indices=idx, query_start_loc=cu, seq_lens_cpu=[T]))
    except Exception as e:
        tim["conv1d_error"] = f"{type(e).__name__}: {e}"
    try:
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        a = torch.randn(T, H, device=DEV).to(torch.bfloat16)
        A_log = torch.zeros(H, device=DEV)
        dtb = torch.zeros(H, device=DEV)
        tim["gating"] = bench(lambda: fused_gdn_gating(A_log, a, a, dtb))
    except Exception as e:
        tim["gating_error"] = f"{type(e).__name__}: {e}"
    try:
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        nrm = RMSNorm(V, eps=1e-6, group_size=None, norm_before_gate=True, device=DEV, dtype=torch.bfloat16)
        x2 = torch.randn(T * H, V, device=DEV).to(torch.bfloat16)
        z2 = torch.randn(T * H, V, device=DEV).to(torch.bfloat16)
        tim["rmsnorm_gated"] = bench(lambda: nrm(x2, z2))
    except Exception as e:
        tim["rmsnorm_gated_error"] = f"{type(e).__name__}: {e}"
    for kk, vv in tim.items():
        log(f"  {kk}: {vv}")
    RES["timing"][f"T{T}"] = tim


def main():
    global REF_DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--cfgs", default="4,32,128;4,32,256;8,16,128;4,16,128;2,64,256")
    args = ap.parse_args()
    t_start = time.time()
    assert gdn_sycl.load(), gdn_sycl.error()
    log("device", torch.xpu.get_device_name(0), "torch", torch.__version__)
    REF_DEV = "cpu"
    try:
        x = torch.ones(4, dtype=torch.float64, device=DEV)
        if float((x * 2).sum()) == 8.0:
            REF_DEV = DEV
    except Exception:
        pass
    log("fp64 reference on", REF_DEV)
    cfgs = [dict(zip(("rg", "cb", "grf"), map(int, c.split(",")))) for c in args.cfgs.split(";") if c]
    ok = True
    if not args.no_check:
        ok &= check_case("single T=64 init", 64, [0, 64], [3], 1)
        ok &= check_case("single T=1024 init", 1024, [0, 1024], [5], 2)
        ok &= check_case("single T=8192 init", 8192, [0, 8192], [1], 3)
        ok &= check_case("varlen 6 seqs (len 1, empty, pad -1)", 4096, [0, 37, 1061, 1062, 1062, 4000, 4096],
                         [1, 4, 6, -1, 0, 7], 4)
        ok &= check_case("bf16 state T=1024", 1024, [0, 1024], [2], 5, state_dtype=torch.bfloat16)
        for c in cfgs[1:]:
            ok &= check_case(f"cfg {c} varlen", 2048, [0, 100, 2048], [3, 4], 6, cfg=c, do_ref=False)
        if not args.quick:
            ok &= check_continuation(16384, 7)
    log("ALL PASS" if ok else "SOME FAIL", f"({time.time() - t_start:.0f} s)")
    RES["all_pass"] = bool(ok)
    if not args.no_time:
        timing(8192, cfgs)
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    path = os.path.join(_HERE, "results", f"n112_{time.strftime('%Y%m%d-%H%M%S')}.json")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not isinstance(v, torch.Tensor)}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    with open(path, "w") as f:
        json.dump(clean(RES), f, indent=1)
    log("wrote", path, f"total {time.time() - t_start:.0f} s")


REF_DEV = "cpu"
if __name__ == "__main__":
    main()
