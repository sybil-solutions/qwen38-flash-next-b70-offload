"""N114: fused decode / verify QSA attention on Intel XPU (Strata port, MIT; see csrc/qsa_dec.sycl).

One op per QSA layer and step replaces SGLang's non-CUDA paged path (torch block expansion -> logical->physical slot
mapping -> sparse GQA reference attention, ~40 small kernels) with 2 kernels (chunk + merge), reads every per-step
value (positions, sequence lengths, block ids, request rows) from device tensors and is XPU-graph capturable.

Library: build/qsa_dec.so (csrc/build_sycl.sh qsa) or $EXL3_QSA_DEC_LIB. Op: torch.ops.n114qsa.decode_attn.
Config (env, read at import): EXL3_QSA_DEC_CFG = chunk,cpw,sg,fp8,target_wg (default 64,0,16,1,160):
  chunk 64 | 128 | 256 cells per chunk; cpw chunks per work-group (0 = auto: about target_wg work-groups);
  sg 16 | 32; fp8 0 exact | 1 via fp16 (both exact decodes).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_loaded = False
_err: Optional[str] = None

HD = 256
G = 12


def _cfg_from_env():
    d = [64, 0, 16, 1, 160]
    try:
        vals = [int(x) for x in os.environ.get("EXL3_QSA_DEC_CFG", "").split(",") if x.strip()]
    except ValueError:
        vals = []
    vals = (vals + d[len(vals):])[:5]
    if vals[0] not in (64, 128, 256) or vals[2] not in (16, 32) or vals[3] not in (0, 1):
        vals = d
    return dict(chunk=vals[0], cpw=vals[1], sg=vals[2], fp8=vals[3], target_wg=vals[4])


CFG = _cfg_from_env()


def lib_path() -> str:
    return os.environ.get("EXL3_QSA_DEC_LIB") or os.path.join(_HERE, "build", "qsa_dec.so")


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


_KV_DTYPES = (torch.bfloat16, torch.float16, getattr(torch, "float8_e4m3fn", None))
_INT = (torch.int32, torch.int64)


def _vec_ok(t) -> bool:
    return t is not None and t.dtype in _INT and (t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1))


def why_not(q, k_cache, v_cache, idx, table, rowlen, t2b=None, rreq=None, qpos=None, seqlen=None,
            expand=False, ratio=4, token_topk=2048) -> Optional[str]:
    """None when the fused kernel handles these inputs, else the reason (patch_qsa_dec.py counts them)."""
    if q.device.type != "xpu":
        return "device"
    if q.dtype != torch.bfloat16:
        return "q dtype"
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        return "rank"
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    if D != HD or k_cache.shape[2] != HD or v_cache.shape[1:] != k_cache.shape[1:] or Hq != G * Hk:
        return f"shape q{tuple(q.shape)} k{tuple(k_cache.shape)}"
    if k_cache.dtype not in _KV_DTYPES or v_cache.dtype != k_cache.dtype:
        return f"kv dtype {k_cache.dtype}"
    if q.stride(2) != 1 or k_cache.stride(2) != 1 or v_cache.stride(2) != 1:
        return "inner stride"
    al = 16 // k_cache.element_size()
    for t in (k_cache, v_cache):
        if t.stride(0) % al or t.stride(1) % al or t.data_ptr() % 16:
            return "kv alignment"
    if idx is None or idx.ndim != 2 or idx.shape[0] != R or idx.dtype not in _INT:
        return "idx"
    if table is None or table.ndim != 2 or table.dtype not in _INT or table.stride(1) != 1:
        return "table"
    if not _vec_ok(rowlen):
        return "rowlen"
    if t2b is not None and (not _vec_ok(t2b) or t2b.shape[0] != R):
        return "t2b"
    if rreq is not None and not _vec_ok(rreq):
        return "rreq"
    if expand:
        if not (_vec_ok(qpos) and _vec_ok(seqlen)) or qpos.shape[0] != R or seqlen.shape[0] != R:
            return "qpos/seqlen"
        if idx.shape[1] * ratio < token_topk:
            return "block width"
    for t in (idx, table, rowlen, t2b, rreq, qpos, seqlen):
        if t is not None and t.device != q.device:
            return "device mix"
    return None


def decode_attention(q, k_cache, v_cache, idx, *, table, rowlen, t2b=None, rreq=None, qpos=None, seqlen=None,
                     expand=False, ratio=4, token_topk=2048, scale=None, chunk=None, cpw=None, sg=None, fp8=None,
                     target_wg=None) -> torch.Tensor:
    """Sparse GQA over the (expanded, mapped) cells of every row; returns [R, Hq, 256] bf16.

    expand=True: idx [R, W] int32 compressed block ids (-1 invalid), expanded as torch_expand_qsa_block_indices
    (block * ratio + 0..ratio-1 masked to < seqlen[row], first token_topk cells, then the ratio - 1 pending-tail
    tokens of qpos[row]). expand=False: idx [R, W] logical token ids. Cell -> slot: table[row(seq), logical] where
    seq = t2b[row] (identity if None), row(seq) = rreq[seq] (seq if None), valid iff 0 <= logical < rowlen[seq].
    """
    if not load():
        raise RuntimeError(f"n114qsa library not loadable from {lib_path()}: {_err}")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if idx.dtype != torch.int32 or idx.stride(-1) != 1:
        idx = idx.to(torch.int32).contiguous()
    c = CFG

    def v1(t):
        return t.reshape(-1) if t is not None and t.ndim == 2 else t

    return torch.ops.n114qsa.decode_attn(
        q, k_cache, v_cache, idx, v1(qpos), v1(seqlen), v1(t2b), v1(rowlen), v1(rreq), table, float(scale),
        int(ratio), int(token_topk), bool(expand), int(c["chunk"] if chunk is None else chunk),
        int(c["cpw"] if cpw is None else cpw), int(c["sg"] if sg is None else sg),
        int(c["fp8"] if fp8 is None else fp8), int(c["target_wg"] if target_wg is None else target_wg))


def geometry(rows, hk=2, cells=2051, chunk=None, cpw=None, target_wg=None):
    c = CFG
    return list(torch.ops.n114qsa.geometry(int(rows), int(hk), int(cells), int(c["chunk"] if chunk is None else chunk),
                                           int(c["cpw"] if cpw is None else cpw),
                                           int(c["target_wg"] if target_wg is None else target_wg)))


# ---------------------------------------------------------------------------------------------------------------
# Pure-torch emulation of the op's semantics (CPU or XPU; used by the offline check and as a test oracle).
def expand_cells(idx, qpos, seqlen, ratio, token_topk):
    """[R, token_topk + ratio - 1] logical cells in the kernel's cell order (blocks then tail), -1 = masked."""
    R = idx.shape[0]
    dev = idx.device
    b = idx.long()
    off = torch.arange(ratio, device=dev)
    e = torch.where(b.unsqueeze(-1) >= 0, b.unsqueeze(-1) * ratio + off, torch.full_like(b.unsqueeze(-1).expand(-1, -1, ratio), -1))
    e = e.reshape(R, -1)[:, :token_topk]
    sl = seqlen.long().reshape(R, 1)
    e = torch.where((e >= 0) & (e < sl), e, torch.full_like(e, -1))
    vis = qpos.long().reshape(R, 1) + 1
    ts = torch.div(vis, ratio, rounding_mode="floor") * ratio
    o = torch.arange(ratio - 1, device=dev).unsqueeze(0)
    tail = ts + o
    tail = torch.where((o < vis - ts) & (tail < sl), tail, torch.full_like(tail, -1))
    return torch.cat([e, tail], dim=1)


def emulate(q, k_cache, v_cache, idx, *, table, rowlen, t2b=None, rreq=None, qpos=None, seqlen=None, expand=False,
            ratio=4, token_topk=2048, scale=None):
    """fp32 emulation of decode_attention (scores/softmax/accumulation in fp32, bf16 output)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    cells = expand_cells(idx, qpos, seqlen, ratio, token_topk) if expand else idx.long()
    seq = t2b.long().reshape(-1) if t2b is not None else torch.arange(R, device=q.device)
    trow = rreq.long().reshape(-1).index_select(0, seq) if rreq is not None else seq
    tlen = rowlen.long().reshape(-1).index_select(0, seq).unsqueeze(1)
    valid = (cells >= 0) & (cells < tlen)
    col = cells.clamp(0, table.shape[1] - 1)
    slots = table.long()[trow.unsqueeze(1), col]
    valid &= slots >= 0
    out = torch.zeros(R, Hq, D, dtype=torch.float32, device=q.device)
    for r in range(R):
        s = slots[r][valid[r]]
        if s.numel() == 0:
            continue
        kk = k_cache.index_select(0, s).float().repeat_interleave(Hq // Hk, dim=1)   # [n, Hq, D]
        vv = v_cache.index_select(0, s).float().repeat_interleave(Hq // Hk, dim=1)
        sc = torch.einsum("hd,nhd->hn", q[r].float(), kk) * scale
        p = torch.softmax(sc, dim=-1)
        out[r] = torch.einsum("hn,nhd->hd", p, vv)
    return out.to(torch.bfloat16)
