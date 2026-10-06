"""N114-hc: SYCL hyper-connection decode kernels (Strata fused_gr_read(_multi) port, MIT; see csrc/hc_dec.sycl).

GatedResidual.mix / combine for M = 1..8 token rows (decode / verify) with SGLang's bf16 rounding points; the weights
(Wd 320x10240, Wu 10240x320, Wi 4x10240 bf16) are read once for all rows. Three kernels per mix (pre-norm [+ the
folded previous combine], split-K down, up + epilogue), one per standalone combine. XPU-graph capturable.

Library: build/hc_dec.so (csrc/build_sycl.sh hc) or $EXL3_HC_DEC_LIB. Ops: torch.ops.n114hc.{mix,combine_mix,combine}.
Launch config (env, read at import): EXL3_HC_DEC_CFG = ks,uc,mode  (default 0,32,0)
    ks    K splits of the down projection (0 = auto: 8 for M <= 2, else 4)
    uc    output columns per up work-group (8 | 16 | 32 | 64; work-group = uc * hc)
    mode  up kernel: 0 = one lane per Wu row, 1 = one sub-group per Wu row
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_loaded = False
_err: Optional[str] = None
MAX_M = 8


def _cfg_from_env():
    d = [0, 32, 0]
    try:
        vals = [int(x) for x in os.environ.get("EXL3_HC_DEC_CFG", "").split(",") if x.strip()]
    except ValueError:
        vals = []
    vals = (vals + d[len(vals):])[:3]
    if vals[0] < 0 or vals[1] not in (8, 16, 32, 64) or vals[2] not in (0, 1):
        vals = d
    return dict(ks=vals[0], uc=vals[1], mode=vals[2])


CFG = _cfg_from_env()


def lib_path() -> str:
    return os.environ.get("EXL3_HC_DEC_LIB") or os.path.join(_HERE, "build", "hc_dec.so")


def load() -> bool:
    global _loaded, _err
    if _loaded:
        return True
    if _err is not None:
        return False
    try:
        torch.ops.load_library(lib_path())
        _loaded = True
    except Exception as e:  # pragma: no cover
        _err = f"{type(e).__name__}: {e}"
    return _loaded


def error() -> Optional[str]:
    return _err


def ks_for(M: int, D: int, ks: Optional[int] = None) -> int:
    k = CFG["ks"] if ks is None else ks
    if k <= 0:
        k = 8 if M <= 2 else 4
    while k > 1 and D % (k * 8):
        k //= 2
    return max(1, k)


def _al16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _rows_ok(t: torch.Tensor, M: int, cols: int) -> bool:
    return (t.dim() == 2 and t.shape[0] == M and t.shape[1] == cols and t.dtype == torch.bfloat16
            and t.stride(1) == 1 and t.stride(0) % 8 == 0 and _al16(t))


def _w_ok(t: Optional[torch.Tensor], rows: int, cols: int, dev) -> bool:
    return (t is not None and t.dtype == torch.bfloat16 and tuple(t.shape) == (rows, cols) and t.is_contiguous()
            and _al16(t) and t.device == dev)


def module_why_not(gr) -> Optional[str]:
    """Static checks of a GatedResidual instance (None = the kernels support its layout)."""
    cfg = getattr(gr, "config", None)
    if cfg is None or not getattr(cfg, "hc_per_branch_norm", False):
        return "not hc_per_branch_norm"
    norm = getattr(gr, "hc_norm", None)
    hc, hs = int(getattr(gr, "hc_count", 0)), int(getattr(gr, "hidden_size", 0))
    if norm is None or hc <= 0 or hs <= 0 or getattr(norm, "group_size", None) != hs:
        return "norm layout"
    w = norm.weight
    D = hc * hs
    if (w.dim() != 1 or w.numel() != D or not w.is_contiguous()
            or w.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
        return "norm weight"
    down, up = getattr(gr, "input_mix_weight_down", None), getattr(gr, "input_mix_weight_up", None)
    if down is None or up is None:
        return "no mix weights"
    lr = down.weight.shape[0]
    if lr % 8 or lr < 8 or lr > 1024 or hs % 8 or hs // 8 > 1024 or hc > 8:
        return f"geometry hc{hc} hs{hs} lr{lr}"
    if hs % CFG["uc"] or (CFG["uc"] * hc) % 16 or CFG["uc"] * hc > 1024:
        return "uc config"
    dev = w.device
    # Wd / Wu / Wi: bf16 only (SGLang's nn.Linear(dtype=bf16) converts the checkpoint's F16 on load); anything
    # else (fp16 / fp32 / quantized) stays on the original path
    if not _w_ok(down.weight, lr, D, dev) or not _w_ok(up.weight, D, lr, dev):
        return "mix weight dtype/layout"
    inj = getattr(gr, "block_inject_weight", None)
    if inj is not None and not _w_ok(inj.weight, hc, D, dev):
        return "inject weight dtype/layout"
    return None


def why_not(gr, x: torch.Tensor) -> Optional[str]:
    """None when n114 handles gr.mix(x)."""
    if x.device.type != "xpu":
        return "device"
    if x.dim() != 2 or not (1 <= x.shape[0] <= MAX_M):
        return "rows"
    r = module_why_not(gr)
    if r is not None:
        return r
    D = gr.hc_count * gr.hidden_size
    if not _rows_ok(x, x.shape[0], D):
        return "x layout"
    if gr.hc_norm.weight.device != x.device:
        return "device mismatch"
    return None


def combine_why_not(r: torch.Tensor, bo: torch.Tensor, l: torch.Tensor, hc: int, hs: int) -> Optional[str]:
    if r.device.type != "xpu":
        return "device"
    M = r.shape[0] if r.dim() == 2 else -1
    if not (1 <= M <= MAX_M):
        return "rows"
    if not _rows_ok(r, M, hc * hs) or not _rows_ok(bo, M, hs) or bo.device != r.device:
        return "layout"
    if l is None or l.dtype != torch.float32 or l.numel() != M * hc or not l.is_contiguous():
        return "l"
    return None


def _wi(gr) -> Optional[torch.Tensor]:
    inj = getattr(gr, "block_inject_weight", None)
    return inj.weight if inj is not None else None


def mix(gr, x: torch.Tensor, *, ks: Optional[int] = None, uc: Optional[int] = None,
        mode: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mixed [M, hs], n [M, hc*hs], l [M, hc] fp32 (empty without inject weight)) of gr.mix(x)."""
    hc, hs = int(gr.hc_count), int(gr.hidden_size)
    return torch.ops.n114hc.mix(x, gr.hc_norm.weight, gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight,
                                _wi(gr), float(gr.hc_norm.variance_epsilon), hc, ks_for(x.shape[0], hc * hs, ks),
                                int(CFG["uc"] if uc is None else uc), int(CFG["mode"] if mode is None else mode))


def combine_mix(gr, r: torch.Tensor, bo: torch.Tensor, l: torch.Tensor, *, ks: Optional[int] = None,
                uc: Optional[int] = None, mode: Optional[int] = None):
    """r_new = combine(r, bo, l) of the PREVIOUS half, then gr.mix(r_new). Returns (mixed, r_new, n, l_new)."""
    hc, hs = int(gr.hc_count), int(gr.hidden_size)
    return torch.ops.n114hc.combine_mix(r, bo, l, gr.hc_norm.weight, gr.input_mix_weight_down.weight,
                                        gr.input_mix_weight_up.weight, _wi(gr), float(gr.hc_norm.variance_epsilon),
                                        hc, ks_for(r.shape[0], hc * hs, ks), int(CFG["uc"] if uc is None else uc),
                                        int(CFG["mode"] if mode is None else mode))


def combine(r: torch.Tensor, bo: torch.Tensor, l: torch.Tensor, hc: int) -> torch.Tensor:
    """bf16(r + bo * bf16(2 sigmoid(l / hc))) per branch: GatedResidual.combine given l = bf16(n . Wi)."""
    return torch.ops.n114hc.combine(r, bo, l, int(hc))


# ---------------------------------------------------------------------------------------------------------------------
# pure-torch emulation of the kernels' arithmetic (CPU or XPU; used by the offline check and as a debugging aid)

def _rb(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def emulate_mix(x, norm_w, wd, wu, wi, eps, hc, prev=None):
    """prev = (bo, l_prev) folds the previous combine. Returns (mixed, n, l, r_used) as bf16 / fp32 tensors."""
    M, D = x.shape
    hs = D // hc
    r = x.float()
    if prev is not None:
        bo, lp = prev
        g = _rb(2.0 * torch.sigmoid(lp.float() / hc))                     # [M, hc]
        p = bo.float().unsqueeze(1) * g.unsqueeze(-1)                       # [M, hc, hs] (product rounded to fp32)
        r = _rb(r.view(M, hc, hs) + p).view(M, D)
    rg = r.view(M, hc, hs)
    inv = torch.tensor(1.0 / hs, dtype=torch.float32)
    var = (rg * rg).sum(-1, keepdim=True) * inv
    rs = torch.rsqrt(var + eps)
    n = _rb((rg * rs) * (1.0 + norm_w.float().view(hc, hs))).view(M, D)
    t = _rb(n.double() @ wd.double().t()).float()
    xx = t / hc
    a = _rb(xx * torch.sigmoid(xx))
    u = _rb(a.double() @ wu.double().t()).float().view(M, hc, hs)
    mixed = ((torch.sigmoid(u) * n.view(M, hc, hs)).sum(1) * (1.0 / hc)).to(torch.bfloat16)
    l = _rb(n.double() @ wi.double().t()).float() if wi is not None else None
    return mixed, n.to(torch.bfloat16), l, r.to(torch.bfloat16)


def emulate_combine(r, bo, l, hc):
    M, D = r.shape
    hs = D // hc
    g = _rb(2.0 * torch.sigmoid(l.float() / hc))
    p = bo.float().unsqueeze(1) * g.unsqueeze(-1)
    return (r.float().view(M, hc, hs) + p).to(torch.bfloat16).view(M, D)
