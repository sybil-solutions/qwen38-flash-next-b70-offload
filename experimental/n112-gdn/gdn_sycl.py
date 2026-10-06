"""N112: SYCL token-serial GDN prompt recurrence on Intel XPU (Strata port, MIT; see csrc/gdn_sycl.sycl).

Drop-in for SGLang's `chunk_gated_delta_rule(q, k, v, g, beta, initial_state=pool, initial_state_indices=slots,
cu_seqlens=query_start_loc, head_first=False, use_qk_l2norm_in_kernel=True, inplace_update=True)` as called by
TritonGDNKernel.extend: returns (o [1, T, H, V] bf16, None, None) and writes each sequence's final state back to
pool[slot] (slot >= 0) in place. The per-chunk states `h` are not produced (only SGLang's mamba radix-track path
reads them; patch_gdn.py keeps Triton for those batches).

Library: build/gdn_sycl.so (csrc/build_sycl.sh) or $EXL3_GDN_SYCL_LIB. Op: torch.ops.n112gdn.prefill.
Config (env, read at import): EXL3_GDN_SYCL_CFG = rg,cb,grf (default 4,32,128): rg row groups per column
(2 | 4 | 8), cb value columns per work-group, grf 128 | 256.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_loaded = False
_err: Optional[str] = None
_CFGS = {(4, 32), (4, 16), (8, 16), (8, 32), (2, 64), (2, 32)}


def _cfg_from_env():
    vals = [int(x) for x in os.environ.get("EXL3_GDN_SYCL_CFG", "4,32,128").split(",") if x.strip()]
    d = [4, 32, 128]
    vals = (vals + d[len(vals):])[:3]
    if (vals[0], vals[1]) not in _CFGS or vals[2] not in (128, 256):
        vals = d
    return dict(rg=vals[0], cb=vals[1], grf=vals[2])


CFG = _cfg_from_env()


def lib_path() -> str:
    return os.environ.get("EXL3_GDN_SYCL_LIB") or os.path.join(_HERE, "build", "gdn_sycl.so")


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


def _rows_ok(t: torch.Tensor) -> bool:
    # [1, T, heads, 128] bf16 view: inner stride 1, token / head strides multiples of 8 elements, 16-byte base
    return (t.stride(-1) == 1 and t.stride(1) % 8 == 0 and t.stride(2) % 8 == 0 and t.data_ptr() % 16 == 0)


def why_not(q, k, v, g, beta, state, indices, cu_seqlens, use_l2=True, inplace_update=True) -> Optional[str]:
    """None when the SYCL kernel handles these inputs, else the reason (patch_gdn.py counts them)."""
    if not inplace_update:
        return "inplace_update=False"
    if not use_l2:
        return "no l2"
    if state is None or indices is None or cu_seqlens is None:
        return "no state/indices/cu_seqlens"
    if q.device.type != "xpu":
        return "device"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return "dtype"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return "rank/batch"
    T, Hk, K = q.shape[1:]
    H, V = v.shape[2], v.shape[3]
    if K != 128 or V != 128 or tuple(k.shape[1:]) != (T, Hk, K) or v.shape[1] != T or Hk == 0 or H % Hk:
        return f"shape q{tuple(q.shape)} v{tuple(v.shape)}"
    if not (_rows_ok(q) and _rows_ok(k)) or v.stride(-1) != 1:
        return "q/k/v layout"
    if g is None or beta is None or g.shape[-2:] != (T, H) or beta.shape[-2:] != (T, H) or g.numel() != T * H \
            or beta.numel() != T * H:
        return "g/beta shape"
    if state.ndim != 4 or tuple(state.shape[1:]) != (H, V, K) or state.dtype not in (torch.float32, torch.bfloat16):
        return f"state {tuple(state.shape)} {state.dtype}"
    if state.stride(3) != 1 or state.stride(2) != K or state.stride(1) != V * K or state.device != q.device:
        return "state layout"
    if indices.ndim != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() != indices.numel() + 1:
        return "indices/cu_seqlens"
    if T >= 2 ** 31 - 1:
        return "T"
    return None


def chunk_gated_delta_rule_sycl(q, k, v, g, beta, scale=None, initial_state=None, initial_state_indices=None,
                                cu_seqlens=None, use_qk_l2norm_in_kernel=True, *, rg=None, cb=None, grf=None):
    if not load():
        raise RuntimeError(f"n112gdn library not loadable from {lib_path()}: {_err}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    T, H = v.shape[1], v.shape[2]
    g2 = g.reshape(T, H)
    b2 = beta.reshape(T, H)
    if g2.dtype != torch.float32 or g2.stride(1) != 1:
        g2 = g2.float().contiguous()
    if b2.dtype != torch.float32 or b2.stride(1) != 1:
        b2 = b2.float().contiguous()
    idx = initial_state_indices
    if idx.dtype != torch.int32 or not idx.is_contiguous():
        idx = idx.to(torch.int32).contiguous()
    cu = cu_seqlens
    if cu.dtype != torch.int32 or not cu.is_contiguous():
        cu = cu.to(torch.int32).contiguous()
    c = CFG
    o = torch.ops.n112gdn.prefill(
        q[0], k[0], v[0], g2, b2, initial_state, idx, cu, float(scale), bool(use_qk_l2norm_in_kernel), 1e-6,
        int(c["rg"] if rg is None else rg), int(c["cb"] if cb is None else cb), int(c["grf"] if grf is None else grf))
    return o.unsqueeze(0), None, None
