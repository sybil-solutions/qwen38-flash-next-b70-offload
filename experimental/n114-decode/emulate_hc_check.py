#!/usr/bin/env python3
"""N114-hc offline check (CPU, no SYCL): the kernels' arithmetic (hc_dec.emulate_*, the same rounding points as
csrc/hc_dec.sycl) vs SGLang's GatedResidual reference math, copied verbatim from layers/hyperconnection.py:
GroupedGemmaRMSNorm eager, _mix_compute / _combine_compute eager and torch.compile (inductor = what XPU runs today).

    python3 emulate_hc_check.py [--ms 1,2,4] [--no-compile]
Prints bit-equality rates; the eager-vs-compiled spread of the reference itself is the noise floor.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_dec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="1,2,4")
ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

HC, HS, LR, EPS = 4, 2560, 320, 1e-6


# ---- verbatim reference (hyperconnection.py) ----
def norm_ref(x, weight, group_size, eps):
    input_dtype = x.dtype
    x_float = x.float()
    x_grouped = x_float.reshape(*x_float.shape[:-1], x_float.shape[-1] // group_size, group_size)
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return (x_norm * (1.0 + weight.float())).to(input_dtype)


def _mix_compute(hyper_input_normed, input_mix_weight_down, input_mix_weight_up, hc, hs):
    input_mix_weight = F.silu(F.linear(hyper_input_normed, input_mix_weight_down) / hc)
    input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
    input_mix_weight = torch.sigmoid(input_mix_weight)
    input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
    output = (input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))).mean(dim=-2)
    return output


def _combine_compute(block_output, residual, normed_residual, block_inject_weight, hc, hs):
    R = residual.unflatten(-1, (hc, hs))
    block_inject_weight_out = 2 * torch.sigmoid(F.linear(normed_residual, block_inject_weight) / hc)
    injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(-1)
    return (R + injection).flatten(-2)


def eq(a, b):
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    d = (a.float() - b.float()).abs()
    return f"bit_eq {100.0 * (a.view(torch.int16) == b.view(torch.int16)).float().mean().item():8.4f}% max_abs {d.max().item():.3e}"


torch.manual_seed(args.seed)
mixc = combc = None
if not args.no_compile:
    try:
        mixc, combc = torch.compile(_mix_compute), torch.compile(_combine_compute)
    except Exception as e:  # pragma: no cover
        print("torch.compile unavailable:", e)
D = HC * HS
w = (0.2 * torch.randn(D)).to(torch.bfloat16)
wd = (0.02 * torch.randn(LR, D)).to(torch.bfloat16)
wu = (0.06 * torch.randn(D, LR)).to(torch.bfloat16)
wi = (0.02 * torch.randn(HC, D)).to(torch.bfloat16)
for M in [int(s) for s in args.ms.split(",")]:
    x = (1.5 * torch.randn(M, D)).to(torch.bfloat16)
    bo = (0.8 * torch.randn(M, HS)).to(torch.bfloat16)
    n_ref = norm_ref(x, w, HS, EPS)
    mixed_e, n_e, l_e, _ = hc_dec.emulate_mix(x, w, wd, wu, wi, EPS, HC)
    print(f"M={M}")
    print(f"  norm   emu vs eager                 {eq(n_e, n_ref)}")
    mix_eager = _mix_compute(n_ref, wd, wu, HC, HS).to(torch.bfloat16)
    mixed_from_ref, _, _, _ = hc_dec.emulate_mix(x, w, wd, wu, wi, EPS, HC)
    print(f"  mix    emu vs eager                 {eq(mixed_e, mix_eager)}")
    comb_eager = _combine_compute(bo, x, n_ref, wi, HC, HS).to(torch.bfloat16)
    l_ref = F.linear(n_ref, wi).float()
    comb_e = hc_dec.emulate_combine(x, bo, l_ref, HC)
    print(f"  comb   emu(l=F.linear) vs eager     {eq(comb_e, comb_eager)}")
    print(f"  l      emu vs F.linear bf16         {eq(l_e, l_ref)}")
    if mixc is not None:
        mix_comp = mixc(n_ref, wd, wu, HC, HS).to(torch.bfloat16)
        comb_comp = combc(bo, x, n_ref, wi, HC, HS).to(torch.bfloat16)
        print(f"  mix    emu vs compiled              {eq(mixed_e, mix_comp)}")
        print(f"  mix    eager vs compiled (floor)    {eq(mix_eager, mix_comp)}")
        print(f"  comb   emu vs compiled              {eq(comb_e, comb_comp)}")
        print(f"  comb   eager vs compiled (floor)    {eq(comb_eager, comb_comp)}")
    # fold == combine then mix
    r1 = hc_dec.emulate_combine(x, bo, l_e, HC)
    m_f, n_f, l_f, r_f = hc_dec.emulate_mix(x, w, wd, wu, wi, EPS, HC, prev=(bo, l_e))
    m_u, n_u, l_u, _ = hc_dec.emulate_mix(r1, w, wd, wu, wi, EPS, HC)
    print(f"  fold   combine_mix vs combine+mix   r {eq(r_f, r1)} | mix {eq(m_f, m_u)}")
