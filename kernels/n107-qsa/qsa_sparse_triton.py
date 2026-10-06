"""N107 QSA sparse attention for prefill on Intel XPU (Arc Pro B70, Xe2) in Triton.

Replaces exl3xpu `qsa_sparse_attention_union` (torch.unique over ~16.8M slot ids + host syncs + dense masked fp32
attention over the union, O(rows x union)) by kernels whose work is O(rows x 2051) and that never sync the host.

Semantics = SGLang `qsa_sparse_attention_reference` (the exact reference; the union path is an approximation of it
with bf16 scores):
  out[r, h] = sum_j softmax_j(scale * q[r, h] . K[slot_rj, h // G]) V[slot_rj, h // G]  over valid slots (slot >= 0)
  rows with no valid slot -> 0;  G = Hq / Hk (repeat_interleave GQA mapping);  scale = softmax_scale or D ** -0.5.
Precision: bf16 operands, fp32 scores (bf16 x bf16 products are exact in fp32), fp32 online softmax and accumulation,
P rounded to bf16 for the PV product (same as the union path), output in q.dtype.

Kernels
  _qsa_row_fwd    per-row (primary): program = (query row, kv head); the G=12 q heads of that kv head form the
                  M=16 tile; loops over the row's own 2051 slots in BLOCK_N chunks, K/V gathered straight from the
                  (fp8) pool by physical slot. Tiles with no valid slot are skipped (rows near the sequence start).
  _qsa_bitmap +   block-union (alternative): rows grouped in tiles of BM (64) rows of one sequence; a bitmap of each
  _qsa_tile_fwd   row's selected logical positions + a per-(tile, KV block) "any row selected" flag are built on
                  device (atomic_or, no unique, no host sync); program = (tile, q head) walks the tile's causal KV
                  blocks, skips unflagged blocks, loads each kept block once for 64 rows (contiguous logical positions
                  -> physical via token_slot_table) and masks every (row, key) by its selection bit.
KV dtype: bf16/fp16 pools, or fp8 e4m3fn pools decoded in-kernel (KV_MODE=2: integer bit decode on a uint8 view,
always compiles; KV_MODE=1: Triton fp8e4nv -> bf16 cast). The QSA backend writes the pool without k/v scales, so the
decode is a plain cast (as in the union path).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
_FP8 = tuple(t for t in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)) if t is not None)


# ----------------------------------------------------------------------------------------------------------------
# helpers

@triton.jit
def _e4m3fn_to_f32(x):
    """uint8 e4m3fn codes -> fp32 values (normal: 2^(e-7)(1+m/8); subnormal: m 2^-9). NaN code 0x7f decodes to 480."""
    u = x.to(tl.uint32)
    s = (u >> 7) & 1
    e = (u >> 3) & 15
    m = u & 7
    normal = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    sub = tl.where(s != 0, -sub, sub)
    return tl.where(e == 0, sub, normal)


@triton.jit
def _load_kv(ptrs, mask, KV_MODE: tl.constexpr, OUT_DT: tl.constexpr):
    if KV_MODE == 2:
        x = tl.load(ptrs, mask=mask, other=0)
        return _e4m3fn_to_f32(x).to(OUT_DT)
    else:
        x = tl.load(ptrs, mask=mask, other=0.0)
        return x.to(OUT_DT)


# ----------------------------------------------------------------------------------------------------------------
# per-row kernel

@triton.jit
def _qsa_row_fwd(Q, K, V, O, IDX, T, qk_scale,
                 s_qm, s_qh, s_km, s_kh, s_vm, s_vh, s_om, s_oh, s_im, s_in,
                 G: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
                 KV_MODE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    OUT_DT: tl.constexpr = Q.dtype.element_ty
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < G
    heads = kvh * G + offs_h
    q = tl.load(Q + row * s_qm + heads[:, None] * s_qh + offs_d[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    idx_row = IDX + row * s_im
    k_base = K + kvh * s_kh
    v_base = V + kvh * s_vh
    for start in range(0, T, BLOCK_N):
        cols = start + offs_n
        slot = tl.load(idx_row + cols * s_in, mask=cols < T, other=-1)
        valid = slot >= 0
        n_valid = tl.sum(valid.to(tl.int32), axis=0)
        if n_valid > 0:
            slot64 = tl.where(valid, slot, 0).to(tl.int64)
            k = _load_kv(k_base + slot64[:, None] * s_km + offs_d[None, :], valid[:, None], KV_MODE, OUT_DT)
            s = tl.dot(q, tl.trans(k)) * qk_scale                          # [BLOCK_H, BLOCK_N] fp32
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_kv(v_base + slot64[:, None] * s_vm + offs_d[None, :], valid[:, None], KV_MODE, OUT_DT)
            acc = acc * alpha[:, None] + tl.dot(p.to(OUT_DT), v)
            m_i = m_new
    has = l_i > 0
    out = acc / tl.where(has, l_i, 1.0)[:, None]
    out = tl.where(has[:, None], out, 0.0)
    tl.store(O + row * s_om + heads[:, None] * s_oh + offs_d[None, :], out.to(OUT_DT), mask=hmask[:, None])


# ----------------------------------------------------------------------------------------------------------------
# block-union kernels

@triton.jit
def _qsa_bitmap(IDX, s_im, s_in, T, BITS, W, FLAGS, NB, row0,
                BM: tl.constexpr, BN: tl.constexpr, BLOCK_T: tl.constexpr):
    r = tl.program_id(0)                       # row within the sequence
    cb = tl.program_id(1)
    cols = cb * BLOCK_T + tl.arange(0, BLOCK_T)
    grow = (row0 + r).to(tl.int64)
    idx = tl.load(IDX + grow * s_im + cols * s_in, mask=cols < T, other=-1).to(tl.int32)
    valid = idx >= 0
    safe = tl.where(valid, idx, 0)
    one = tl.full([BLOCK_T], 1, tl.int32)
    tl.atomic_or(BITS + grow * W + safe // 32, one << (safe % 32), mask=valid)
    tl.store(FLAGS + (r // BM) * NB + safe // BN, tl.full([BLOCK_T], 1, tl.int8), mask=valid)


@triton.jit
def _qsa_tile_fwd(Q, K, V, O, BITS, FLAGS, TABLE, row0, n_rows, prefix, kv_len, W, NB, qk_scale,
                  s_qm, s_qh, s_km, s_kh, s_vm, s_vh, s_om, s_oh,
                  G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, KV_MODE: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    OUT_DT: tl.constexpr = Q.dtype.element_ty
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    lrow = t * BM + offs_m
    rmask = lrow < n_rows
    grow = (row0 + lrow).to(tl.int64)
    q = tl.load(Q + grow[:, None] * s_qm + h * s_qh + offs_d[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum(prefix + tl.minimum((t + 1) * BM, n_rows), kv_len)     # causal end of the tile's last row
    k_base = K + kvh * s_kh
    v_base = V + kvh * s_vh
    for j in range(0, (hi + BN - 1) // BN):
        f = tl.load(FLAGS + t * NB + j)
        if f != 0:
            pos = j * BN + offs_n
            pmask = pos < kv_len
            slot = tl.load(TABLE + pos, mask=pmask, other=0).to(tl.int64)
            k = _load_kv(k_base + slot[:, None] * s_km + offs_d[None, :], pmask[:, None], KV_MODE, OUT_DT)
            s = tl.dot(q, tl.trans(k)) * qk_scale                          # [BM, BN]
            wd = tl.load(BITS + grow[:, None] * W + (pos // 32)[None, :],
                         mask=rmask[:, None] & pmask[None, :], other=0)
            sel = ((wd >> (pos % 32)[None, :]) & 1) != 0
            s = tl.where(sel, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_kv(v_base + slot[:, None] * s_vm + offs_d[None, :], pmask[:, None], KV_MODE, OUT_DT)
            acc = acc * alpha[:, None] + tl.dot(p.to(OUT_DT), v)
            m_i = m_new
    has = l_i > 0
    out = acc / tl.where(has, l_i, 1.0)[:, None]
    out = tl.where(has[:, None], out, 0.0)
    tl.store(O + grow[:, None] * s_om + h * s_oh + offs_d[None, :], out.to(OUT_DT), mask=rmask[:, None])


# ----------------------------------------------------------------------------------------------------------------
# host wrappers (no host<->device sync: all sizes come from tensor shapes or host-side metadata)

def _env_cfg(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    v = os.environ.get(name)
    if not v:
        return default
    vals = tuple(int(x) for x in v.split(",") if x.strip())
    return vals + default[len(vals):]


ROW_CFG = _env_cfg("EXL3_QSA_TRITON_CFG", (64, 8, 0))           # BLOCK_N, num_warps, num_stages (0 = default)
TILE_CFG = _env_cfg("EXL3_QSA_TILE_CFG", (64, 64, 8, 0))         # BM, BN, num_warps, num_stages
FP8_MODE = os.environ.get("EXL3_QSA_TRITON_FP8", "bits")         # bits | cast
GRF_MODE = os.environ.get("EXL3_QSA_GRF", "")                    # '' (default) | large | auto  (Intel backend)
TPW = int(os.environ.get("EXL3_QSA_TPW", "0"))                   # threads_per_warp (Intel backend); 0 = default


def _launch_opts(num_warps: int, num_stages: int, threads_per_warp: Optional[int] = None) -> dict:
    o = {"num_warps": num_warps}
    if num_stages:
        o["num_stages"] = num_stages
    if GRF_MODE:
        o["grf_mode"] = GRF_MODE
    tpw = TPW if threads_per_warp is None else threads_per_warp
    if tpw:
        o["threads_per_warp"] = tpw
    return o


def supports(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             token_slots: Optional[torch.Tensor] = None, fp8_mode: Optional[str] = None) -> bool:
    fp8_mode = fp8_mode or FP8_MODE
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    if D not in (64, 128, 256) or k_cache.shape[2] != D or tuple(v_cache.shape[1:]) != (Hk, D):
        return False
    if Hk <= 0 or Hq % Hk != 0 or Hq // Hk > 64:
        return False
    kd = k_cache.dtype
    if kd != v_cache.dtype:
        return False
    if kd in _FP8:
        if fp8_mode == "bits" and kd != torch.float8_e4m3fn:
            return False
    elif kd not in (torch.bfloat16, torch.float16):
        return False
    if q.stride(2) != 1 or k_cache.stride(2) != 1 or v_cache.stride(2) != 1:
        return False
    if token_slots is not None:
        if token_slots.ndim != 2 or token_slots.shape[0] != R:
            return False
        if token_slots.dtype not in (torch.int32, torch.int64):
            return False
    return q.device == k_cache.device == v_cache.device


def _kv_args(k_cache, v_cache, fp8_mode):
    if k_cache.dtype in _FP8:
        if fp8_mode == "bits":
            return k_cache.view(torch.uint8), v_cache.view(torch.uint8), 2
        return k_cache, v_cache, 1
    return k_cache, v_cache, 0


def qsa_sparse_attention_triton(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                token_slots: torch.Tensor, softmax_scale: Optional[float] = None, *,
                                block_n: Optional[int] = None, num_warps: Optional[int] = None,
                                num_stages: Optional[int] = None, fp8_mode: Optional[str] = None,
                                threads_per_warp: Optional[int] = None,
                                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Drop-in for exl3xpu/SGLang qsa_sparse_attention(q, k_cache, v_cache, token_slots, softmax_scale).

    q [R, Hq, D] bf16/fp16; k_cache/v_cache [N, Hk, D] (bf16/fp16/fp8 e4m3fn); token_slots [R, T] int32/int64 physical
    slots, -1 = not selected. Returns [R, Hq, D] in q.dtype."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    fp8_mode = fp8_mode or FP8_MODE
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    T = token_slots.shape[1]
    if out is None:
        out = torch.empty_like(q)
    if R == 0:
        return out
    bn = block_n or ROW_CFG[0]
    nw = num_warps or ROW_CFG[1]
    ns = ROW_CFG[2] if num_stages is None else num_stages
    kp, vp, kv_mode = _kv_args(k_cache, v_cache, fp8_mode)
    _qsa_row_fwd[(R, Hk)](
        q, kp, vp, out, token_slots, T, float(scale) * LOG2E,
        q.stride(0), q.stride(1), kp.stride(0), kp.stride(1), vp.stride(0), vp.stride(1),
        out.stride(0), out.stride(1), token_slots.stride(0), token_slots.stride(1),
        G=G, BLOCK_H=max(16, triton.next_power_of_2(G)), BLOCK_N=bn, D=D, KV_MODE=kv_mode,
        **_launch_opts(nw, ns, threads_per_warp))
    return out


def qsa_tile_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, logical_idx: torch.Tensor,
                       token_slot_table: torch.Tensor, seqs: Sequence[Tuple[int, int, int, int, int]],
                       softmax_scale: Optional[float] = None, *, BM: Optional[int] = None, BN: Optional[int] = None,
                       num_warps: Optional[int] = None, num_stages: Optional[int] = None,
                       fp8_mode: Optional[str] = None, threads_per_warp: Optional[int] = None) -> torch.Tensor:
    """Block-union variant. logical_idx [R, T]: per-row selected LOGICAL positions (the indexer's topk_indices,
    -1 invalid); token_slot_table [B, >= kv_len] logical -> physical slot; seqs: host tuples
    (seq_id, row0, n_rows, prefix_len, kv_len) covering every row (rows of one sequence are contiguous)."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    fp8_mode = fp8_mode or FP8_MODE
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    T = logical_idx.shape[1]
    bm = BM or TILE_CFG[0]
    bn = BN or TILE_CFG[1]
    nw = num_warps or TILE_CFG[2]
    ns = TILE_CFG[3] if num_stages is None else num_stages
    assert bn % 32 == 0 or 32 % bn == 0
    out = torch.empty_like(q)
    if R == 0:
        return out
    kv_max = max(int(s[4]) for s in seqs)
    W = (kv_max + 31) // 32
    bits = torch.zeros((R, W), dtype=torch.int32, device=q.device)
    kp, vp, kv_mode = _kv_args(k_cache, v_cache, fp8_mode)
    block_t = 1024
    for seq_id, row0, n_rows, prefix, kv_len in seqs:
        if n_rows == 0:
            continue
        n_tiles = (n_rows + bm - 1) // bm
        NB = (kv_len + bn - 1) // bn
        flags = torch.zeros((n_tiles, NB), dtype=torch.int8, device=q.device)
        _qsa_bitmap[(n_rows, triton.cdiv(T, block_t))](
            logical_idx, logical_idx.stride(0), logical_idx.stride(1), T, bits, W, flags, NB, row0,
            BM=bm, BN=bn, BLOCK_T=block_t, num_warps=4)
        table = token_slot_table[seq_id]
        _qsa_tile_fwd[(n_tiles, Hq)](
            q, kp, vp, out, bits, flags, table, row0, n_rows, prefix, kv_len, W, NB, float(scale) * LOG2E,
            q.stride(0), q.stride(1), kp.stride(0), kp.stride(1), vp.stride(0), vp.stride(1),
            out.stride(0), out.stride(1),
            G=G, BM=bm, BN=bn, D=D, KV_MODE=kv_mode, **_launch_opts(nw, ns, threads_per_warp))
    return out


def compact_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, token_slot_table: torch.Tensor,
               seq_lens_host: Sequence[int], dtype=torch.bfloat16):
    """Per-sequence logical-order copy of the visible K/V (dequantised): [sum(L), Hk, D] each. Host lengths only."""
    ks, vs = [], []
    for i, L in enumerate(seq_lens_host):
        if L <= 0:
            continue
        idx = token_slot_table[i, :L].long()
        ks.append(k_cache.index_select(0, idx).to(dtype))
        vs.append(v_cache.index_select(0, idx).to(dtype))
    if not ks:
        e = k_cache.new_empty((0,) + tuple(k_cache.shape[1:]), dtype=dtype)
        return e, e.clone()
    return torch.cat(ks), torch.cat(vs)


def qsa_compact_row_attention(q, k_cache, v_cache, logical_idx, token_slot_table, token_to_batch_idx,
                              sequence_lengths, seq_lens_host, softmax_scale=None, **kw):
    """Per-row kernel over a bf16 logical-order copy of each sequence's K/V (no fp8 decode in the inner loop,
    compact L2 footprint). Row offsets are computed on device from sequence_lengths (no H2D of host lists)."""
    kc, vc = compact_kv(k_cache, v_cache, token_slot_table, seq_lens_host, dtype=q.dtype)
    lens = sequence_lengths.to(torch.long)
    cu = torch.cumsum(lens, 0) - lens
    base = cu.index_select(0, token_to_batch_idx.long())
    cidx = torch.where(logical_idx >= 0, logical_idx.long() + base[:, None], torch.full_like(logical_idx, -1,
                                                                                               dtype=torch.long))
    return qsa_sparse_attention_triton(q, kc, vc, cidx.to(torch.int32), softmax_scale, **kw)


__all__ = ["qsa_sparse_attention_triton", "qsa_tile_attention", "qsa_compact_row_attention", "compact_kv",
           "supports", "ROW_CFG", "TILE_CFG", "FP8_MODE"]
