"""N114 offline semantics check (Mac / CPU, Triton interpreter): SGLang's own GDN decode kernels vs the n114 kernel
semantics (the formulas csrc/gdn_dec.sycl implements and csrc/cpu_test_gdn.cpp checks the SYCL code against).

Runs the verbatim Triton kernels of the installed-image-era SGLang source tree (causal_conv1d_update,
fused_recurrent_gated_delta_rule_packed_decode, _layer_norm_fwd_1pass_kernel; extracted from the .py files, launched
with TRITON_INTERPRET=1) on SGLang-ordered inputs (fix_query_key_value_ordering split + cat + .contiguous()), and the
n114 semantics on the RAW projections (q|k|v|z columns of in_proj_qkvz, b|a of in_proj_ba, conv state read in place).
Reports bit-equality / max abs of conv out, conv state, ssm state, y.

usage: TRITON_INTERPRET=1 <python with triton+torch> offline_gdn_sem.py --src <sglang python dir> [--bs 1 2 4]
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


def extract(path, names):
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            out.append("\n".join(lines[start:node.end_lineno]))
    assert len(out) == len(names), (path, names, len(out))
    return "\n\n".join(out)


def load_kernels(srcdir):
    ns = {"triton": triton, "tl": tl, "torch": torch, "Optional": None}
    exec("@triton.jit\ndef exp(x):\n    return tl.exp(x)\n", ns)
    k = os.path.join(srcdir, "sglang/kernels/ops")
    conv_src = extract(os.path.join(k, "mamba/causal_conv1d_triton.py"), ["_causal_conv1d_update_kernel"])
    # the Triton interpreter multiplies raw bf16 bit patterns (garbage); spell the compiled bf16 x bf16 -> bf16
    # multiply out (fp32 product rounded to bf16).  Whether the XPU backend really rounds the product is checked on
    # the GPU (test_gdn_dec_xpu.py: round_prod 1 vs 0 bit-equality against the real kernel).
    n = conv_src.count("acc += matrix_x * matrix_w")
    assert n >= 1, "conv kernel text changed"
    conv_src = conv_src.replace(
        "acc += matrix_x * matrix_w",
        "acc += (matrix_x.to(tl.float32) * matrix_w.to(tl.float32)).to(tl.bfloat16).to(tl.float32)")
    exec(conv_src, ns)
    exec(extract(os.path.join(k, "attention/fla/fused_recurrent.py"),
                 ["fused_recurrent_gated_delta_rule_packed_decode_kernel"]), ns)
    exec(extract(os.path.join(k, "attention/fla/layernorm_gated.py"), ["_layer_norm_fwd_1pass_kernel"]), ns)
    return ns


def sglang_ref(ns, qkvz, ba, conv_state, conv_w, ssm, idx, A_log, dt_bias, norm_w, eps, H, Hk, act):
    """the original decode tail, kernels launched like their SGLang wrappers (decode, seqlen 1)."""
    B = qkvz.shape[0]
    K = V = 128
    kd, vd = Hk * K, H * V
    # Qwen3_5GatedDeltaNet.fix_query_key_value_ordering (verbatim logic)
    query, key, value, z = qkvz.split([kd, kd, vd, vd], dim=-1)
    b, a = ba.split([H, H], dim=-1)
    value = value.reshape(value.size(0), -1, V)
    z = z.reshape(z.size(0), -1, V)
    b = b.contiguous()
    a = a.contiguous()
    query, key, value = map(lambda x: x.reshape(x.shape[0], -1), (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1)
    # causal_conv1d_update(x, conv_state, w, None, 'silu', conv_state_indices=idx)
    x = mixed_qkv.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = conv_w.shape[1]
    out = torch.empty_like(x)
    ns["_causal_conv1d_update_kernel"][(batch, triton.cdiv(dim, 256))](
        x, conv_w, None, conv_state, None, idx, None, x, None, None, None, None, out,
        batch, dim, seqlen, width - 1, conv_state.size(0),
        *x.stride(), *conv_w.stride(), *conv_state.stride(), idx.stride(0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        *out.stride(), -1,
        HAS_BIAS=False, KERNEL_WIDTH=width, SILU_ACTIVATION=True, IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False, NP2_STATELEN=triton.next_power_of_2(width - 1), NP2_SEQLEN=1, USE_PAD_SLOT=True,
        BLOCK_N=256, SAVE_INTERMEDIATE=False, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False)
    conv = out.squeeze(-1)
    # packed decode
    o = conv.new_empty(B, 1, H, V)
    BV = 32
    ns["fused_recurrent_gated_delta_rule_packed_decode_kernel"][(triton.cdiv(V, BV), B * H)](
        conv, a, b, A_log, dt_bias, o, ssm, ssm, idx, K ** -0.5,
        conv.stride(0), a.stride(0), b.stride(0), ssm.stride(0), ssm.stride(0), idx.stride(0),
        H=Hk, HV=H, K=K, V=V, BK=128, BV=BV, SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True)
    core = o.transpose(0, 1).reshape(-1, V)
    zz = z.reshape(-1, V)
    # RMSNormGated (norm_before_gate, group = full 128)
    M, N = core.shape
    y = torch.empty_like(core)
    rstd = torch.empty(M, dtype=torch.float32)
    ns["_layer_norm_fwd_1pass_kernel"][(triton.cdiv(M, 4), 1)](
        core, y, norm_w, None, zz, None, rstd, core.stride(0), y.stride(0), zz.stride(0), 0, 0, M, N, eps,
        BLOCK_N=128, ROWS_PER_BLOCK=4, HAS_BIAS=False, HAS_Z=True, Z_IS_3D=False, Z_HEADS=1, NORM_BEFORE_GATE=True,
        IS_RMS_NORM=True, ACTIVATION=act)
    return conv, y.reshape(B, -1), o


def _trunc_bf(x):
    """fp32 -> bf16 by truncation (what the Triton 3.7 interpreter's casts do), as an fp32 tensor."""
    return (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32)


def n114_sem(qkvz, ba, conv_state, conv_w, ssm, idx, A_log, dt_bias, norm_w, eps, H, Hk, act, round_prod=True,
             trunc=False):
    """the n114 kernel math (fp32, same rounding points), vectorised.  trunc=True: the bf16 roundings truncate (only
    to compare against the interpreter, whose casts truncate; the GPU kernels and compiled Triton round to nearest)."""
    B = qkvz.shape[0]
    rb = _trunc_bf if trunc else (lambda x: x.to(torch.bfloat16).float())
    K = V = 128
    C = 2 * Hk * K + H * V
    bf = torch.bfloat16
    conv = torch.zeros(B, C, dtype=bf)
    y = torch.zeros(B, H * V, dtype=bf)
    for r in range(B):
        s = int(idx[r])
        if s < 0:
            continue
        win = torch.cat([conv_state[s].float(), qkvz[r, :C].float().unsqueeze(1)], dim=1)   # [C, W]
        p = win * conv_w.float()
        if round_prod:
            p = rb(p)
        acc = torch.zeros(C)
        for j in range(conv_w.shape[1]):
            acc = acc + p[:, j]
        conv_state[s] = win[:, 1:].to(bf)
        acc = acc / (1.0 + torch.exp(-acc))
        conv[r] = rb(acc).to(bf)
        cv = conv[r].float()
        for h in range(H):
            hk = h // (H // Hk)
            q = cv[hk * K:(hk + 1) * K]
            k = cv[(Hk + hk) * K:(Hk + hk + 1) * K]
            v = cv[2 * Hk * K + h * V:2 * Hk * K + (h + 1) * V]
            q = q / torch.sqrt((q * q).sum() + 1e-6) * K ** -0.5
            k = k / torch.sqrt((k * k).sum() + 1e-6)
            bv = ba[r, h].float()
            av = ba[r, H + h].float()
            xg = av + dt_bias[h].float()
            sp = torch.log(1.0 + torch.exp(xg)) if xg <= 20 else xg
            g = -torch.exp(A_log[h].float()) * sp
            beta = rb(torch.sigmoid(bv))
            S = ssm[s, h].float() * torch.exp(g)          # [V, K]
            kv = S @ k
            d = (v - kv) * beta
            S = S + d[:, None] * k[None, :]
            o = rb(S @ q)
            ssm[s, h] = S.to(ssm.dtype)
            rstd = torch.rsqrt((o * o).sum() / V + eps)
            z = qkvz[r, C + h * V:C + (h + 1) * V].float()
            gate = torch.sigmoid(z) if act == "sigmoid" else z * torch.sigmoid(z)
            y[r, h * V:(h + 1) * V] = rb((o * rstd) * norm_w.float() * gate).to(bf)
    return conv, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--Hk", type=int, default=4)
    args = ap.parse_args()
    ns = load_kernels(args.src)
    torch.manual_seed(0)
    H, Hk, K, V = args.H, args.Hk, 128, 128
    C = 2 * Hk * K + H * V
    slots, W = 6, 4
    bf = torch.bfloat16
    fails = 0
    for act in ("sigmoid", "swish"):
        for B in args.bs:
            idx = torch.tensor([(3 + 2 * i) % slots for i in range(B)], dtype=torch.int32)
            if B >= 4:
                idx[1] = -1
            qkvz = torch.randn(B, C + H * V).to(bf)
            ba = (1.5 * torch.randn(B, 2 * H)).to(bf)
            conv_w = (0.5 * torch.randn(C, W)).to(bf)
            # SGLang conv state view (slots, C, 3) with the channel dim contiguous
            cs0 = torch.randn(slots, W - 1, C).to(bf).transpose(1, 2)
            ssm0 = 0.05 * torch.randn(slots, H, V, K)
            A_log = torch.log(1 + 15 * torch.rand(H))
            dt_bias = torch.randn(H).to(bf)
            norm_w = (1 + 0.2 * torch.randn(V)).to(bf)
            eps = 1e-6
            cs_a, ssm_a = cs0.clone(), ssm0.clone()
            cs_b = cs0.clone()
            ssm_b = ssm0.clone()
            conv_r, y_r, _ = sglang_ref(ns, qkvz, ba, cs_a, conv_w, ssm_a, idx, A_log, dt_bias, norm_w, eps, H, Hk, act)
            for rp, tr in ((True, True), (True, False), (False, False)):
                cs_b, ssm_b = cs0.clone(), ssm0.clone()
                conv_n, y_n = n114_sem(qkvz, ba, cs_b, conv_w, ssm_b, idx, A_log, dt_bias, norm_w, eps, H, Hk, act, rp,
                                       trunc=tr)
                valid = idx >= 0
                ceq = float((conv_n[valid] == conv_r[valid]).float().mean())
                yeq = float((y_n[valid] == y_r[valid]).float().mean())
                ymax = float((y_n[valid].float() - y_r[valid].float()).abs().max())
                cst = bool(torch.equal(cs_b, cs_a))
                smax = float((ssm_b - ssm_a).abs().max())
                ok = ceq > 0.999 and cst and yeq > 0.99 and smax < 1e-5
                if tr:
                    fails += not ok
                print(f"[{'PASS' if ok else ('FAIL' if tr else 'info')}] act={act:7s} B={B} round_prod={int(rp)} "
                      f"trunc={int(tr)} "
                      f"conv bitEq={ceq:.5f} conv_state equal={cst} y bitEq={yeq:.5f} y max_abs={ymax:.3e} "
                      f"ssm max_abs={smax:.3e}", flush=True)
    print("ALL PASS" if not fails else "SOME FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
