"""N114: fused SYCL GDN decode on Intel XPU (Strata fused_gdn_* port, MIT; see csrc/gdn_dec.sycl).

One decode step of a Qwen3.5 / Qwen3.8-Flash-Next GatedDeltaNet layer from the raw input projections:

    y = gdn_decode(qkvz, ba, ...)   # == RMSNormGated(packed_decode(causal_conv1d_update(cat(q, k, v)), a, b), z)

  * qkvz [B, 2*key_dim + 2*value_dim] bf16: in_proj_qkvz output, columns q | k | v | z (the XPU split order of
    Qwen3_5GatedDeltaNet.fix_query_key_value_ordering, num_v_heads / num_k_heads not in the fused-split ratios);
  * ba [B, 2*H] bf16: in_proj_ba output, columns b | a   (or None + x / w_ba: the projection runs in the kernel);
  * conv_state (slots, C, L) bf16 and ssm_state [slots, H, 128, 128] fp32 | bf16 are updated in place for slot >= 0.
Returns y [B, H*128] bf16 (the out_proj input).  Two kernels, no host sync, XPU-graph capturable.

Library: build/gdn_dec.so (csrc/build_sycl.sh gdn) or $EXL3_GDN_DEC_LIB.  Ops: torch.ops.n114gdn.{conv,rec_norm}.
Config (env, read at import):
    EXL3_GDN_DEC_CFG = rg,grf        (default 4,128)  rg row groups per value column (2 | 4 | 8), grf 128 | 256
    EXL3_GDN_DEC_ROUND_PROD = 0|1    (default 1) conv products rounded to bf16 like Triton's bf16 x bf16 multiply
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_loaded = False
_err: Optional[str] = None
_empty = {}


def _cfg_from_env():
    vals = [int(x) for x in os.environ.get("EXL3_GDN_DEC_CFG", "4,128").split(",") if x.strip()]
    d = [4, 128]
    vals = (vals + d[len(vals):])[:2]
    if vals[0] not in (2, 4, 8) or vals[1] not in (128, 256):
        vals = d
    return dict(rg=vals[0], grf=vals[1])


CFG = _cfg_from_env()
ROUND_PROD = os.environ.get("EXL3_GDN_DEC_ROUND_PROD", "1") != "0"


def lib_path() -> str:
    return os.environ.get("EXL3_GDN_DEC_LIB") or os.path.join(_HERE, "build", "gdn_dec.so")


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


def _none(dev) -> torch.Tensor:
    t = _empty.get(dev)
    if t is None:
        t = _empty[dev] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    return t


ACT = {"sigmoid": 0, "swish": 1, "silu": 1}


def why_not(qkvz, ba, conv_state, conv_w, ssm_state, idx, *, H, Hk, K=128, V=128, act="sigmoid",
            norm_w=None, x=None, w_ba=None) -> Optional[str]:
    """None when the kernels handle these inputs, else the reason."""
    if K != 128 or V != 128:
        return f"head dims {K}/{V}"
    if H <= 0 or Hk <= 0 or H % Hk:
        return "H % Hk"
    if act not in ACT:
        return f"act {act}"
    C = 2 * Hk * K + H * V
    if qkvz.device.type != "xpu" or qkvz.dtype != torch.bfloat16 or qkvz.ndim != 2 or qkvz.stride(1) != 1:
        return "qkvz dtype/layout"
    if qkvz.shape[1] < C + H * V:
        return f"qkvz width {qkvz.shape[1]} < {C + H * V}"
    if w_ba is None:
        if ba is None or ba.dtype != torch.bfloat16 or ba.ndim != 2 or ba.stride(1) != 1 or ba.shape[1] < 2 * H \
                or ba.shape[0] != qkvz.shape[0]:
            return "ba"
    else:
        if x is None or x.dtype != torch.bfloat16 or x.ndim != 2 or x.stride(1) != 1 or x.shape[0] != qkvz.shape[0] \
                or w_ba.dtype != torch.bfloat16 or tuple(w_ba.shape) != (2 * H, x.shape[1]) or not w_ba.is_contiguous():
            return "x / w_ba"
    if conv_w.dtype != torch.bfloat16 or conv_w.ndim != 2 or conv_w.shape[0] != C or not (2 <= conv_w.shape[1] <= 4):
        return f"conv_w {tuple(conv_w.shape)} {conv_w.dtype}"
    if conv_state.dtype != torch.bfloat16 or conv_state.ndim != 3 or conv_state.shape[1] != C \
            or conv_state.shape[2] < conv_w.shape[1] - 1:
        return f"conv_state {tuple(conv_state.shape)} {conv_state.dtype}"
    if ssm_state.ndim != 4 or tuple(ssm_state.shape[1:]) != (H, V, K) \
            or ssm_state.dtype not in (torch.float32, torch.bfloat16) \
            or ssm_state.stride(3) != 1 or ssm_state.stride(2) != K or ssm_state.stride(1) != V * K:
        return f"ssm_state {tuple(ssm_state.shape)} {ssm_state.dtype}"
    if idx.ndim != 1 or idx.numel() != qkvz.shape[0] or idx.dtype not in (torch.int32, torch.int64) \
            or not idx.is_contiguous():
        return "idx"
    if norm_w is not None and (norm_w.numel() != V or not norm_w.is_contiguous()):
        return "norm_w"
    if any(t.device != qkvz.device for t in (conv_state, conv_w, ssm_state, idx)):
        return "device"
    return None


def gdn_decode(qkvz, ba, conv_state, conv_w, ssm_state, idx, A_log, dt_bias, norm_w, eps, *, H, Hk, scale,
               act="sigmoid", K=128, V=128, x=None, w_ba=None, rg=None, grf=None, round_prod=None):
    """Fused decode (see module doc).  act='none' returns the bf16 recurrence output (no norm) for tests."""
    if not load():
        raise RuntimeError(f"n114gdn library not loadable from {lib_path()}: {_err}")
    C = 2 * Hk * K + H * V
    rp = ROUND_PROD if round_prod is None else bool(round_prod)
    conv = torch.ops.n114gdn.conv(qkvz, conv_w, conv_state, idx, C, True, rp)
    z = qkvz[:, C:C + H * V]
    e = _none(qkvz.device)
    if w_ba is None:
        b = ba[:, :H]
        a = ba[:, H:2 * H]
        bx = wba = e
    else:
        a = b = e
        bx, wba = x, w_ba
    actc = 2 if act == "none" else ACT[act]
    return torch.ops.n114gdn.rec_norm(conv, a, b, bx, wba, A_log, dt_bias, ssm_state, idx, z,
                                      norm_w if norm_w is not None else e, float(eps), actc, float(scale), int(H),
                                      int(Hk), int(CFG["rg"] if rg is None else rg),
                                      int(CFG["grf"] if grf is None else grf))
