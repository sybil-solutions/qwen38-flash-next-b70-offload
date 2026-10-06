"""N108: SYCL per-query QSA prompt attention on Intel XPU (Strata port, MIT; see csrc/qsa_row_sycl.sycl).

Drop-in for exl3xpu `qsa_sparse_attention(q, k_cache, v_cache, token_slots, softmax_scale)` with SGLang
`qsa_sparse_attention_reference` semantics (softmax over valid slots only, no valid slot -> 0, GQA h // 12,
scale = softmax_scale or D ** -0.5). K/V are read in place from the paged pool through the physical slots: no gather
copy, no torch.unique union, no host sync. fp32 scores / softmax / accumulation, bf16 output.

Library: build/qsa_row_sycl.so (csrc/build_sycl.sh) or $EXL3_QSA_SYCL_LIB. Ops: torch.ops.n108qsa.row_attn,
torch.ops.n108qsa.fp8_decode_test.

Config (env, read at import; per-call keyword arguments override):
    EXL3_QSA_SYCL_CFG   kernel,cpw,sg[,grf[,fp8]]   default "2,0,16,128,1"
        kernel 1 = strata (Strata's attn_chunk_kernel: 64-cell chunks, sub-group reductions)
               2 = bcast  (lane-per-cell Q.K with broadcast q, 256-cell chunks)
        cpw    chunks per work-group; 0 = all (one work-group per (row, kv head), no merge); 1 = Strata split-K
        sg     sub-group size 16 | 32;  grf 128 | 256 (large register file);  fp8 0 = exact int decode, 1 = via fp16
    EXL3_QSA_SYCL_SCRATCH_MB   split-K partials budget (default 256; rows are processed in batches)
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_loaded = False
_err: Optional[str] = None


def _cfg_from_env():
    vals = [int(x) for x in os.environ.get("EXL3_QSA_SYCL_CFG", "2,0,16,128,1").split(",") if x.strip()]
    d = [2, 0, 16, 128, 1]
    vals = (vals + d[len(vals):])[:5]
    return dict(kernel=vals[0], cpw=vals[1], sg=vals[2], grf=vals[3], fp8_mode=vals[4])


CFG = _cfg_from_env()
SCRATCH_MB = int(os.environ.get("EXL3_QSA_SYCL_SCRATCH_MB", "256"))


def lib_path() -> str:
    return os.environ.get("EXL3_QSA_SYCL_LIB") or os.path.join(_HERE, "build", "qsa_row_sycl.so")


def load() -> bool:
    """Load the op library once; False (and error()) when it is missing or does not load."""
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


_KV_OK = (torch.bfloat16, torch.float16) + ((torch.float8_e4m3fn,) if hasattr(torch, "float8_e4m3fn") else ())


def supports(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             token_slots: Optional[torch.Tensor] = None) -> bool:
    """Shapes/dtypes/layouts the kernel handles; everything else must take the previous implementation."""
    if q.device.type != "xpu" or q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        return False
    if q.dtype != torch.bfloat16 or k_cache.dtype not in _KV_OK or v_cache.dtype != k_cache.dtype:
        return False
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    if D != 256 or k_cache.shape[2] != 256 or tuple(v_cache.shape[1:]) != (Hk, 256) or Hk <= 0 or Hq != 12 * Hk:
        return False
    if v_cache.shape[0] != k_cache.shape[0]:
        return False
    if q.stride(2) != 1 or k_cache.stride(2) != 1 or v_cache.stride(2) != 1:
        return False
    al = 16 // k_cache.element_size()
    for t in (k_cache, v_cache):
        if t.stride(0) % al or t.stride(1) % al or t.data_ptr() % 16:
            return False
    if token_slots is not None:
        if token_slots.ndim != 2 or token_slots.shape[0] != R or token_slots.dtype not in (torch.int32, torch.int64):
            return False
        if token_slots.device != q.device:
            return False
    if R * Hk > 2 ** 31 - 1:
        return False
    return True


def qsa_sparse_attention_sycl(q, k_cache, v_cache, token_slots, softmax_scale=None, *, kernel=None, cpw=None,
                              sg=None, grf=None, fp8_mode=None, scratch_mb=None):
    if not load():
        raise RuntimeError(f"n108qsa library not loadable from {lib_path()}: {_err}")
    scale = softmax_scale or q.shape[-1] ** -0.5
    c = CFG
    slots = token_slots
    if slots.dtype != torch.int32:
        slots = slots.to(torch.int32)
    if slots.stride(1) != 1:
        slots = slots.contiguous()
    return torch.ops.n108qsa.row_attn(
        q, k_cache, v_cache, slots, float(scale),
        int(c["kernel"] if kernel is None else kernel), int(c["cpw"] if cpw is None else cpw),
        int(c["sg"] if sg is None else sg), int(c["grf"] if grf is None else grf),
        int(c["fp8_mode"] if fp8_mode is None else fp8_mode), int(SCRATCH_MB if scratch_mb is None else scratch_mb))


def supports_select(q, keys, row_starts, row_ends, topk) -> bool:
    """Indexer prefill selection inputs the SYCL select kernels handle (bf16 q [R, 4, 128], bf16 keys [C, (1,) 128])."""
    if q.device.type != "xpu" or q.ndim != 3 or tuple(q.shape[1:]) != (4, 128) or q.dtype != torch.bfloat16:
        return False
    if keys.dtype != torch.bfloat16 or keys.device != q.device or keys.shape[-1] != 128:
        return False
    if not (keys.ndim == 2 or (keys.ndim == 3 and keys.shape[1] == 1)):
        return False
    if q.stride(2) != 1 or row_starts.numel() != q.shape[0] or row_ends.numel() != q.shape[0] or topk <= 0:
        return False
    return hasattr(torch.ops, "n108qsa") and hasattr(torch.ops.n108qsa, "prefill_select")


def prefill_select(q, keys, row_starts, row_ends, topk, score_scale=None, rpw=8, scratch_mb=128):
    """Block indices [R, topk] int32 relative to row_starts (ascending, -1 padded) = the selected set of
    qsa_fast_topk(torch_qsa_mqa_prefill(q, keys, row_starts, row_ends), row_starts, row_ends, topk); ties at the
    threshold -> lowest index."""
    if not load():
        raise RuntimeError(f"n108qsa library not loadable: {_err}")
    k2 = keys.reshape(keys.shape[0], keys.shape[-1])
    if k2.stride(1) != 1 or k2.stride(0) % 8 or k2.data_ptr() % 16:
        k2 = k2.contiguous()
    rs = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    re = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    return torch.ops.n108qsa.prefill_select(q, k2, rs, re, int(topk), float(score_scale or q.shape[-1] ** 0.5),
                                            int(rpw), int(scratch_mb))


def fp8_decode_test(codes: torch.Tensor) -> torch.Tensor:
    """[2, n] fp32: the device's exact-int and via-fp16 decodes of uint8 e4m3fn codes."""
    if not load():
        raise RuntimeError(f"n108qsa library not loadable: {_err}")
    return torch.ops.n108qsa.fp8_decode_test(codes)
