"""N114 GDN decode on the B70: n114 fused kernels (gdn_dec.py) vs SGLang's current XPU decode path.

Current path (Qwen3_5GatedDeltaNet.forward, decode, num_v/num_k = 3 on XPU): fix_query_key_value_ordering split,
b/a .contiguous(), cat(q, k, v), causal_conv1d_update (Triton), TritonGDNKernel.packed_decode (Triton), z reshape,
RMSNormGated (Triton) -- the image's own functions.  Shapes of Qwen3.8-Flash-Next: H 48, Hk 16, K = V = 128,
C = 10240, in_proj_qkvz width 16384, in_proj_ba width 96, ssm fp32.

Checks (exit 0 only if all pass):
  * per step: conv out / conv state / ssm state / y vs the current path at bs 1, 2, 4, 8 (8 with a padded -1 row),
    both conv-state layouts, 6 chained decode steps (each path on its own state);
  * round_prod 1 vs 0: bit-equality of the conv output against the real Triton conv (which rounding the XPU backend
    uses for bf16 x bf16);
  * ba fusion (projection inside the kernel) vs current path + F.linear;
  * XPU graph capture of 36 layers x (conv + rec_norm), replay with new inputs == eager;
  * patch_gdn_dec.install() against the installed SGLang (source anchors present).
Timing (us per layer call; torch.xpu.Event): eager and graph-replayed (36 layers captured back to back) for the
current path and n114 (+ the rg / grf sweep), bs 1 / 2 / 4 / 8.  Writes results/gdn_<time>.json.

usage (omarchy, from ~/freetoken-exl3):
  bench/xpu_run.sh 0000:84:00.0 n114-dec kernels/xpu_bmg/n114-decode/run_xpu.sh test_gdn_dec_xpu.py [--quick]
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
import gdn_dec  # noqa: E402

H, HK, K, V = 48, 16, 128, 128
C = 2 * HK * K + H * V
QKVZ = C + H * V
D = 2560
W = 4
EPS = 1e-6
SCALE = K ** -0.5
FAILS = []
RES = {"checks": [], "timing": []}


def note(msg):
    print(msg, flush=True)


def check(name, ok, **kv):
    RES["checks"].append(dict(name=name, ok=bool(ok), **kv))
    note(f"  {'PASS' if ok else 'FAIL'} {name} " + " ".join(f"{k}={v}" for k, v in kv.items()))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------- SGLang reference
def sglang_parts():
    from sglang.srt.layers.attention.linear import gdn_backend as GB
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    try:
        from sglang.srt.models.qwen3_5 import RMSNormGated
    except Exception:
        try:
            from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
        except Exception:
            from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
    return GB.causal_conv1d_update, TritonGDNKernel(), RMSNormGated


def make_norm(RMSNormGated, act, dev, g):
    n = RMSNormGated(V, eps=EPS, group_size=None, norm_before_gate=True, device=dev, dtype=torch.bfloat16,
                     activation=act)
    with torch.no_grad():
        n.weight.copy_((1 + 0.2 * torch.randn(V, generator=g)).to(torch.bfloat16))
    return n


def current_path(parts, norm, qkvz, ba, conv_state, conv_w, ssm, idx, A_log, dt_bias):
    conv_upd, kern, _ = parts
    kd, vd = HK * K, H * V
    query, key, value, z = qkvz.split([kd, kd, vd, vd], dim=-1)
    b, a = ba.split([H, H], dim=-1)
    value = value.reshape(value.size(0), -1, V)
    z = z.reshape(z.size(0), -1, V)
    b = b.contiguous()
    a = a.contiguous()
    query, key, value = map(lambda x: x.reshape(x.shape[0], -1), (query, key, value))
    mixed = torch.cat((query, key, value), dim=-1)
    mixed = conv_upd(mixed, conv_state, conv_w, None, "silu", conv_state_indices=idx)
    core = kern.packed_decode(mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=SCALE, ssm_states=ssm,
                              cache_indices=idx, num_v_heads=H, head_v_dim=V)
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core = norm(core, z)
    return core.reshape(z_shape).reshape(qkvz.shape[0], -1), mixed


def n114_path(norm, qkvz, ba, conv_state, conv_w, ssm, idx, A_log, dt_bias, act, **kw):
    return gdn_dec.gdn_decode(qkvz, ba, conv_state, conv_w, ssm, idx, A_log, dt_bias, norm.weight, EPS, H=H,
                              Hk=HK, scale=SCALE, act=act, **kw)


# ------------------------------------------------------------------------------------------- inputs
def make_inputs(B, slots, dev, g, layout="dimcontig", pad=False):
    qkvz = torch.randn(B, QKVZ, generator=g).to(torch.bfloat16).to(dev)
    ba = (1.5 * torch.randn(B, 2 * H, generator=g)).to(torch.bfloat16).to(dev)
    conv_w = (0.5 * torch.randn(C, W, generator=g)).to(torch.bfloat16).to(dev)
    if layout == "dimcontig":
        cs = torch.randn(slots, W - 1, C, generator=g).to(torch.bfloat16).to(dev).transpose(1, 2)
    else:
        cs = torch.randn(slots, C, W - 1, generator=g).to(torch.bfloat16).to(dev)
    ssm = (0.05 * torch.randn(slots, H, V, K, generator=g)).to(dev)
    perm = torch.randperm(slots, generator=g)[:B].to(torch.int32)
    if pad and B > 1:
        perm[B // 2] = -1
    idx = perm.to(dev)
    A_log = torch.log(1 + 15 * torch.rand(H, generator=g)).to(dev)
    dt_bias = torch.randn(H, generator=g).to(dev)
    return qkvz, ba, cs, conv_w, ssm, idx, A_log, dt_bias


def bit_eq(a, b):
    return float((a == b).float().mean()) if a.numel() else 1.0


# ------------------------------------------------------------------------------------------- correctness
def correctness(parts, dev, quick):
    note("--- correctness vs the current SGLang path")
    g = torch.Generator().manual_seed(1)
    bss = (1, 4) if quick else (1, 2, 4, 8)
    for act in ("sigmoid",) if quick else ("sigmoid", "swish"):
        norm = make_norm(parts[2], act, dev, g)
        for layout in ("dimcontig", "contig"):
            for B in bss:
                qkvz, ba, cs, cw, ssm, idx, A_log, dtb = make_inputs(B, 16, dev, g, layout, pad=(B == 8))
                cs_r, ssm_r = cs.clone(), ssm.clone()
                cs_n, ssm_n = cs.clone(), ssm.clone()
                v = idx >= 0
                yeqs, ymaxs, ceqs = [], [], []
                cst_ok = True
                smax = 0.0
                for step in range(6):
                    if step:
                        qkvz = torch.randn(B, QKVZ, generator=g).to(torch.bfloat16).to(dev)
                        ba = (1.5 * torch.randn(B, 2 * H, generator=g)).to(torch.bfloat16).to(dev)
                    y_r, conv_r = current_path(parts, norm, qkvz, ba, cs_r, cw, ssm_r, idx, A_log, dtb)
                    y_n = n114_path(norm, qkvz, ba, cs_n, cw, ssm_n, idx, A_log, dtb, act)
                    conv_n = torch.ops.n114gdn.conv(qkvz, cw, cs.clone(), idx, C, True, gdn_dec.ROUND_PROD) \
                        if step == 0 else None
                    torch.xpu.synchronize()
                    yeqs.append(bit_eq(y_n[v], y_r[v]))
                    ymaxs.append(float((y_n[v].float() - y_r[v].float()).abs().max()))
                    if conv_n is not None:
                        ceqs.append(bit_eq(conv_n[v], conv_r[v]))
                    cst_ok &= bool(torch.equal(cs_n, cs_r))
                    smax = max(smax, float((ssm_n - ssm_r).abs().max()))
                yref = float(y_r[v].float().abs().max()) if v.any() else 1.0
                ok = cst_ok and min(yeqs) >= 0.97 and max(ymaxs) <= 0.04 * yref and smax <= 1e-3 and \
                    (not ceqs or ceqs[0] >= 0.999)
                pad_ok = bool((y_n[~v] == 0).all()) if (~v).any() else True
                check(f"gdn {act} {layout} B={B} 6 steps", ok and pad_ok, conv_bitEq=f"{ceqs[0]:.5f}",
                      conv_state_equal=cst_ok, y_bitEq_min=f"{min(yeqs):.5f}", y_maxabs=f"{max(ymaxs):.3e}",
                      y_absmax=f"{yref:.3e}", ssm_maxabs=f"{smax:.3e}", pad_rows_zero=pad_ok)
    # which conv rounding matches the XPU Triton conv
    g = torch.Generator().manual_seed(2)
    qkvz, ba, cs, cw, ssm, idx, A_log, dtb = make_inputs(8, 16, dev, g)
    conv_upd = parts[0]
    kd, vd = HK * K, H * V
    mixed = qkvz[:, :2 * kd + vd].contiguous()
    ref = conv_upd(mixed, cs.clone(), cw, None, "silu", conv_state_indices=idx)
    e1 = bit_eq(torch.ops.n114gdn.conv(qkvz, cw, cs.clone(), idx, C, True, True), ref)
    e0 = bit_eq(torch.ops.n114gdn.conv(qkvz, cw, cs.clone(), idx, C, True, False), ref)
    RES["conv_round_prod_bitEq"] = {"1": e1, "0": e0}
    check("conv rounding (best of round_prod 1/0 matches Triton)", max(e1, e0) >= 0.999,
          round_prod1=f"{e1:.5f}", round_prod0=f"{e0:.5f}", env_default=int(gdn_dec.ROUND_PROD))
    # ba fusion
    g = torch.Generator().manual_seed(3)
    norm = make_norm(parts[2], "sigmoid", dev, g)
    for B in (1, 4):
        qkvz, ba, cs, cw, ssm, idx, A_log, dtb = make_inputs(B, 16, dev, g)
        x = torch.randn(B, D, generator=g).to(torch.bfloat16).to(dev)
        wba = (0.03 * torch.randn(2 * H, D, generator=g)).to(torch.bfloat16).to(dev)
        ba_l = F.linear(x, wba)
        cs_r, ssm_r = cs.clone(), ssm.clone()
        y_r, _ = current_path(parts, norm, qkvz, ba_l, cs_r, cw, ssm_r, idx, A_log, dtb)
        y_n = n114_path(norm, qkvz, None, cs.clone(), cw, ssm.clone(), idx, A_log, dtb, "sigmoid", x=x, w_ba=wba)
        torch.xpu.synchronize()
        e = bit_eq(y_n, y_r)
        m = float((y_n.float() - y_r.float()).abs().max())
        check(f"ba-fused B={B}", e >= 0.95 and m <= 0.05 * float(y_r.float().abs().max()), y_bitEq=f"{e:.5f}",
              y_maxabs=f"{m:.3e}")


# ------------------------------------------------------------------------------------------- graph
def graph_check(parts, dev):
    note("--- XPU graph capture: 36 layers x n114 (conv + rec_norm), replay with new inputs == eager")
    if not (hasattr(torch.xpu, "XPUGraph") and hasattr(torch.xpu, "graph")):
        check("graph available", False)
        return
    g = torch.Generator().manual_seed(4)
    norm = make_norm(parts[2], "sigmoid", dev, g)
    for B in (1, 4):
        L = 36
        layers = [make_inputs(B, 4, dev, g) for _ in range(L)]
        snap = [(cs.clone(), ssm.clone()) for (_, _, cs, _, ssm, _, _, _) in layers]

        def step():
            return [n114_path(norm, *lay[:2], lay[2], lay[3], lay[4], lay[5], lay[6], lay[7], "sigmoid") for lay in layers]

        try:
            step()   # warm-up (JIT) outside capture
            torch.xpu.synchronize()
            for (_, _, cs, _, ssm, _, _, _), (c0, s0) in zip(layers, snap):
                cs.copy_(c0)
                ssm.copy_(s0)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(xpu_graph=graph):
                outs = step()
            torch.xpu.synchronize()
            # capture must not have run the kernels on the real buffers... reset anyway, set new inputs
            for (qkvz, ba, cs, _, ssm, idx, _, _), (c0, s0) in zip(layers, snap):
                cs.copy_(c0)
                ssm.copy_(s0)
                qkvz.copy_(torch.randn(qkvz.shape, generator=g).to(torch.bfloat16))
                ba.copy_((1.5 * torch.randn(ba.shape, generator=g)).to(torch.bfloat16))
            ins = [(l[0].clone(), l[1].clone()) for l in layers]
            graph.replay()
            torch.xpu.synchronize()
            got = [o.clone() for o in outs]
            st_g = [(l[2].clone(), l[4].clone()) for l in layers]
            for l, (c0, s0), (q0, b0) in zip(layers, snap, ins):
                l[2].copy_(c0)
                l[4].copy_(s0)
            want = step()
            torch.xpu.synchronize()
            ok = all(torch.equal(a, b) for a, b in zip(got, want))
            ok_st = all(torch.equal(sg[0], l[2]) and torch.equal(sg[1], l[4]) for sg, l in zip(st_g, layers))
            check(f"graph replay == eager B={B} (36 layers)", ok and ok_st, y_equal=ok, states_equal=ok_st)
        except Exception as e:
            traceback.print_exc()
            check(f"graph capture B={B}", False, error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------------------------------- timing
def ev_time(fn, iters):
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters   # us


def timing(parts, dev, quick):
    note("--- timing (us per layer call)")
    g = torch.Generator().manual_seed(5)
    norm = make_norm(parts[2], "sigmoid", dev, g)
    bss = (1, 2, 4) if quick else (1, 2, 4, 8)
    L = 36
    for B in bss:
        layers = [make_inputs(B, 4, dev, g) for _ in range(L)]
        x = torch.randn(B, D, generator=g).to(torch.bfloat16).to(dev)
        wba = (0.03 * torch.randn(2 * H, D, generator=g)).to(torch.bfloat16).to(dev)
        variants = {
            "current": lambda l: current_path(parts, norm, l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]),
            "current+ba_gemm": lambda l: current_path(parts, norm, l[0], F.linear(x, wba), l[2], l[3], l[4], l[5],
                                                      l[6], l[7]),
            "n114": lambda l: n114_path(norm, l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], "sigmoid"),
            "n114_ba_fused": lambda l: n114_path(norm, l[0], None, l[2], l[3], l[4], l[5], l[6], l[7], "sigmoid",
                                                 x=x, w_ba=wba),
        }
        if not quick:
            for rg in (2, 4, 8):
                for grf in (128, 256):
                    if (rg, grf) == (gdn_dec.CFG["rg"], gdn_dec.CFG["grf"]):
                        continue
                    variants[f"n114_rg{rg}_grf{grf}"] = (lambda l, rg=rg, grf=grf: n114_path(
                        norm, l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], "sigmoid", rg=rg, grf=grf))
        for name, fn in variants.items():
            try:
                def all_layers():
                    for l in layers:
                        fn(l)
                all_layers()
                torch.xpu.synchronize()
                eager = ev_time(all_layers, 3 if quick else 10) / L
                graph_us = None
                if hasattr(torch.xpu, "XPUGraph"):
                    gr = torch.xpu.XPUGraph()
                    with torch.xpu.graph(xpu_graph=gr):
                        all_layers()
                    torch.xpu.synchronize()
                    gr.replay()
                    torch.xpu.synchronize()
                    graph_us = ev_time(gr.replay, 20 if quick else 50) / L
                    del gr
                RES["timing"].append(dict(B=B, variant=name, eager_us=eager, graph_us=graph_us))
                note(f"  B={B} {name:22s} eager {eager:8.1f} us/layer   graph {graph_us if graph_us is None else round(graph_us, 1)} us/layer")
            except Exception as e:
                traceback.print_exc()
                note(f"  B={B} {name}: ERROR {type(e).__name__}: {e}")
        del layers
        torch.xpu.empty_cache()


def install_check():
    note("--- patch_gdn_dec.install() against the installed SGLang")
    os.environ["EXL3_GDN_DEC_SYCL"] = "1"
    try:
        import importlib
        import patch_gdn_dec
        importlib.reload(patch_gdn_dec)
        ok = patch_gdn_dec.install()
        check("patch_gdn_dec.install() (source anchors present, library loads)", ok)
    except Exception as e:
        traceback.print_exc()
        check("patch_gdn_dec.install()", False, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    dev = torch.device("xpu")
    note(f"n114 gdn test: {torch.xpu.get_device_name(0)} torch {torch.__version__} lib {gdn_dec.lib_path()} "
         f"cfg {gdn_dec.CFG} round_prod {gdn_dec.ROUND_PROD}")
    if not gdn_dec.load():
        note(f"FAIL library: {gdn_dec.error()}")
        return 2
    parts = sglang_parts()
    t0 = time.time()
    for fn in (lambda: correctness(parts, dev, args.quick), lambda: graph_check(parts, dev)):
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            FAILS.append(f"{type(e).__name__}: {e}")
    if not args.no_timing:
        timing(parts, dev, args.quick)
    install_check()
    RES["fails"] = FAILS
    RES["elapsed_s"] = time.time() - t0
    out = os.path.join(HERE, "results", f"gdn_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(RES, open(out, "w"), indent=1)
    note(f"=== {'ALL PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}  ({out}, {RES['elapsed_s']:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
