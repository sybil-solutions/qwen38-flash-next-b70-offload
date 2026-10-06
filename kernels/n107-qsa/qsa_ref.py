"""N107 QSA: pure-torch references, kernel emulators and input builders (no Triton, device-agnostic).

Everything here runs on CPU, XPU or CUDA. Three groups:

* references (copied from the code that runs today, so tests do not need SGLang / exl3xpu importable):
    - ref_sglang_rows      SGLang qsa/kernel.py::qsa_sparse_attention_reference (per-row Python loop, fp32)
    - ref_fp32             the same semantics vectorised in row chunks (fp32 scores, fp32 softmax, fp32 PV)
    - union_exl3xpu        exl3xpu qsa_xpu.py::qsa_sparse_attention_union (the current B70 prefill path)
    - torch_expand_qsa_block_indices / qsa_fast_topk_xpu  (SGLang torch expander, exl3xpu top-k)
* emulators of the Triton kernels in qsa_sparse_triton.py, step for step (same tiles, masks, exp2, -inf guards,
  bf16 rounding of P, fp8 bit decode). They validate the algorithm on a machine without Triton.
* make_case(): realistic prefill inputs (one sequence, R query rows at the end of an S-token context, paged
  fp8/bf16 KV pool with shuffled pages, structured top-512 block selection expanded to 2051 slots per row).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

LOG2E = 1.4426950408889634

# Qwen3.8-Flash-Next full-attention / QSA facts (config.json; SGLang qwen3_5.py, qsa/config.py)
HQ, HK, HEAD_DIM = 24, 2, 256
INDEX_HEADS, INDEX_DIM = 4, 128
BUDGET, RATIO = 2048, 4
BLOCK_TOPK = BUDGET // RATIO            # 512 compressed blocks per row
FINAL_TOPK = BUDGET + RATIO - 1         # 2051 token slots per row (2048 expanded + up to 3 pending-tail tokens)


# ----------------------------------------------------------------------------------------------------------------
# references

def ref_sglang_rows(q, k_cache, v_cache, token_slots, softmax_scale=None, rows=None):
    """Verbatim body of SGLang qsa_sparse_attention_reference, optionally on a subset of rows."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in (range(q.shape[0]) if rows is None else rows):
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype))
    return torch.stack(outputs)


def ref_fp32(q, k_cache, v_cache, token_slots, softmax_scale=None, chunk=32):
    """SGLang reference semantics, vectorised: per row softmax over its valid slots in fp32; no valid slot -> 0."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    out = torch.empty_like(q)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        n = r1 - r0
        sl = token_slots[r0:r1]
        valid = sl >= 0
        idx = sl.clamp_min(0).long().reshape(-1)
        kk = k_cache.index_select(0, idx).view(n, -1, Hk, D).float()
        vv = v_cache.index_select(0, idx).view(n, -1, Hk, D).float()
        qq = q[r0:r1].view(n, Hk, G, D).float()
        s = torch.einsum("rhgd,rshd->rhgs", qq, kk) * scale
        s = s.masked_fill(~valid[:, None, None, :], -float("inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        o = torch.einsum("rhgs,rshd->rhgd", p, vv)
        out[r0:r1] = o.reshape(n, Hq, D).to(q.dtype)
    return out


def union_exl3xpu(q, k_cache, v_cache, token_slots, softmax_scale=None, dense_bytes=128 << 20):
    """Verbatim exl3xpu qsa_sparse_attention_union (EXL3_QSA_DENSE_BYTES as in n104_serve.sh = 128 MiB)."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    valid = token_slots >= 0
    uni = torch.unique(token_slots[valid])
    nu = int(uni.numel())
    out = torch.zeros_like(q)
    if nu == 0:
        return out
    col = torch.where(valid, torch.searchsorted(uni, token_slots.clamp_min(0).to(uni.dtype)),
                      torch.full_like(token_slots, nu, dtype=torch.long))
    kk = k_cache.index_select(0, uni.long()).to(torch.bfloat16).permute(1, 2, 0).contiguous()
    vv = v_cache.index_select(0, uni.long()).to(torch.bfloat16).transpose(0, 1).contiguous()
    B = max(8, min(R, dense_bytes // max(1, Hq * nu * 6)))
    for r0 in range(0, R, B):
        r1 = min(R, r0 + B)
        n = r1 - r0
        mask = torch.zeros((n, nu + 1), dtype=torch.bool, device=q.device)
        mask.scatter_(1, col[r0:r1].long(), True)
        mask = mask[:, :nu]
        qq = q[r0:r1].to(torch.bfloat16).view(n, Hk, G, D).permute(1, 0, 2, 3).reshape(Hk, n * G, D)
        sc = torch.matmul(qq, kk).view(Hk, n, G, nu).float() * scale
        sc.masked_fill_(~mask[None, :, None, :], -float("inf"))
        p = torch.softmax(sc, dim=-1)
        p = torch.nan_to_num_(p, nan=0.0).to(torch.bfloat16).view(Hk, n * G, nu)
        o = torch.matmul(p, vv).view(Hk, n, G, D).permute(1, 0, 2, 3).reshape(n, Hq, D)
        out[r0:r1] = o.to(q.dtype)
    return out


def qsa_fast_topk_xpu(logits, row_starts, row_ends, topk):
    """Verbatim exl3xpu qsa_xpu.qsa_fast_topk (valid entries first, -1 after)."""
    R, C = logits.shape
    starts = row_starts.to(device=logits.device, dtype=torch.long).reshape(-1, 1)
    lengths = (row_ends.to(device=logits.device, dtype=torch.long).reshape(-1, 1) - starts)
    cols = torch.arange(C, device=logits.device).unsqueeze(0)
    valid = (cols >= starts) & (cols < starts + lengths)
    masked = logits.float().masked_fill(~valid, -float("inf"))
    k = min(topk, C)
    idx = torch.topk(masked, k, dim=1).indices
    rel = idx - starts
    keep = torch.arange(k, device=logits.device).unsqueeze(0) < lengths
    out = torch.where(keep, rel, torch.full_like(rel, -1)).to(torch.int32)
    if k < topk:
        out = torch.cat([out, out.new_full((R, topk - k), -1)], dim=1)
    return out


def torch_expand_qsa_block_indices(block_indices, query_positions, sequence_lengths, compress_ratio, token_topk):
    """Verbatim SGLang qsa/kernel.py::torch_expand_qsa_block_indices (the XPU path today)."""
    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    rows = block_indices.shape[0]
    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)).reshape(
        rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where((expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)), expanded,
                           torch.full_like(expanded, -1))
    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = torch.div(visible_tokens, compress_ratio, rounding_mode="floor") * compress_ratio
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (tail < sequence_lengths.unsqueeze(1))
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))
    result = torch.cat([expanded, tail], dim=1)
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


# ----------------------------------------------------------------------------------------------------------------
# fp8 e4m3fn bit decode (same integer formula as the Triton kernel's KV_MODE=2)

def decode_e4m3fn_bits(u8: torch.Tensor) -> torch.Tensor:
    u = u8.to(torch.int64)
    s = (u >> 7) & 1
    e = (u >> 3) & 15
    m = u & 7
    bits = (s << 31) | ((e + 120) << 23) | (m << 20)
    bits = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).to(torch.int32)
    normal = bits.view(torch.float32)
    sub = m.to(torch.float32) * 0.001953125
    sub = torch.where(s != 0, -sub, sub)
    return torch.where(e == 0, sub, normal)


def kv_to_f32(x: torch.Tensor, kv_mode: str = "native") -> torch.Tensor:
    if x.dtype == torch.float8_e4m3fn and kv_mode == "bits":
        return decode_e4m3fn_bits(x.view(torch.uint8))
    return x.float()


# ----------------------------------------------------------------------------------------------------------------
# emulators (mirror qsa_sparse_triton.py)

def emulate_row_kernel(q, k_cache, v_cache, token_slots, softmax_scale=None, block_n=64, kv_mode="bits"):
    """_qsa_row_fwd: program (row, kv head), BLOCK_H = G heads, BLOCK_N slots per step, online softmax in exp2."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    qk_scale = scale * LOG2E
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    T = token_slots.shape[1]
    qq = q.view(R, Hk, G, D).float()                           # bf16 values, exact in fp32
    m = torch.full((R, Hk, G), -float("inf"), device=q.device)
    l = torch.zeros((R, Hk, G), device=q.device)
    acc = torch.zeros((R, Hk, G, D), device=q.device)
    for start in range(0, T, block_n):
        sl = token_slots[:, start:start + block_n]
        valid = sl >= 0                                         # [R, n]
        idx = torch.where(valid, sl, torch.zeros_like(sl)).long()
        vm = valid[..., None, None]
        kk = torch.where(vm, kv_to_f32(k_cache[idx], kv_mode).to(q.dtype).float(), 0.0)   # masked load, other=0
        s = torch.einsum("rhgd,rnhd->rhgn", qq, kk) * qk_scale
        s = s.masked_fill(~valid[:, None, None, :], -float("inf"))
        m_new = torch.maximum(m, s.amax(-1))
        m_safe = torch.where(m_new == -float("inf"), torch.zeros_like(m_new), m_new)
        alpha = torch.exp2(m - m_safe)
        p = torch.exp2(s - m_safe[..., None])
        l = l * alpha + p.sum(-1)
        vv = torch.where(vm, kv_to_f32(v_cache[idx], kv_mode).to(q.dtype).float(), 0.0)
        acc = acc * alpha[..., None] + torch.einsum("rhgn,rnhd->rhgd", p.to(q.dtype).float(), vv)
        m = m_new
    out = torch.where((l > 0)[..., None], acc / torch.where(l > 0, l, torch.ones_like(l))[..., None],
                      torch.zeros_like(acc))
    return out.reshape(R, Hq, D).to(q.dtype)


def pack_bits(sel: torch.Tensor) -> torch.Tensor:
    """bool [R, L] -> int32 words [R, ceil(L/32)], bit b of word w = sel[:, 32w + b] (two's complement wrap)."""
    R, L = sel.shape
    W = (L + 31) // 32
    pad = torch.zeros((R, W * 32), dtype=torch.int64, device=sel.device)
    pad[:, :L] = sel.to(torch.int64)
    words = (pad.view(R, W, 32) << torch.arange(32, device=sel.device)).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
    return words.to(torch.int32)


def emulate_tile_kernel(q, k_cache, v_cache, logical_idx, token_slot_table, seqs, softmax_scale=None,
                        BM=64, BN=64, kv_mode="bits"):
    """_qsa_bitmap + _qsa_tile_fwd: per sequence, tiles of BM query rows x one q head; KV blocks of BN logical
    positions, skipped unless some row of the tile selected one of their tokens; per-(row, key) selection bit."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    qk_scale = scale * LOG2E
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    out = torch.empty_like(q)
    L_max = max(s[4] for s in seqs)
    W = (L_max + 31) // 32
    for seq_id, row0, n_rows, prefix, kv_len in seqs:
        li = logical_idx[row0:row0 + n_rows]
        valid = li >= 0
        sel = torch.zeros((n_rows, W * 32), dtype=torch.bool, device=q.device)
        rr = torch.arange(n_rows, device=q.device)[:, None].expand_as(li)
        sel[rr[valid], li[valid].long()] = True
        words = pack_bits(sel)                                  # what _qsa_bitmap builds with atomic_or
        n_tiles = (n_rows + BM - 1) // BM
        NB = (kv_len + BN - 1) // BN
        flags = torch.zeros((n_tiles, NB), dtype=torch.bool, device=q.device)
        tile_of = (rr[valid] // BM)
        flags[tile_of, li[valid].long() // BN] = True
        table = token_slot_table[seq_id]
        for t in range(n_tiles):
            lr = torch.arange(t * BM, min((t + 1) * BM, n_rows), device=q.device)
            qt = q[row0 + lr].float()                           # [m, Hq, D]
            m = torch.full((lr.numel(), Hq), -float("inf"), device=q.device)
            l = torch.zeros_like(m)
            acc = torch.zeros((lr.numel(), Hq, D), device=q.device)
            hi = min(prefix + min((t + 1) * BM, n_rows), kv_len)
            for j in range((hi + BN - 1) // BN):
                if not bool(flags[t, j]):
                    continue
                pos = torch.arange(j * BN, (j + 1) * BN, device=q.device)
                pmask = pos < kv_len
                slot = torch.where(pmask, table[pos.clamp_max(table.shape[0] - 1)], torch.zeros_like(pos)).long()
                pm = pmask[:, None, None]
                kk = torch.where(pm, kv_to_f32(k_cache[slot], kv_mode).to(q.dtype).float(), 0.0)   # [BN, Hk, D]
                vv = torch.where(pm, kv_to_f32(v_cache[slot], kv_mode).to(q.dtype).float(), 0.0)
                kq = kk.repeat_interleave(G, dim=1)             # [BN, Hq, D]
                vq = vv.repeat_interleave(G, dim=1)
                s = torch.einsum("mhd,nhd->mhn", qt, kq) * qk_scale
                wd = words[lr][:, (pos.clamp_max(W * 32 - 1) // 32)]           # [m, BN]
                bit = ((wd >> (pos % 32)[None, :]) & 1) != 0
                bit = bit & pmask[None, :]
                s = s.masked_fill(~bit[:, None, :], -float("inf"))
                m_new = torch.maximum(m, s.amax(-1))
                m_safe = torch.where(m_new == -float("inf"), torch.zeros_like(m_new), m_new)
                alpha = torch.exp2(m - m_safe)
                p = torch.exp2(s - m_safe[..., None])
                l = l * alpha + p.sum(-1)
                acc = acc * alpha[..., None] + torch.einsum("mhn,nhd->mhd", p.to(q.dtype).float(), vq)
                m = m_new
            o = torch.where((l > 0)[..., None], acc / torch.where(l > 0, l, torch.ones_like(l))[..., None],
                            torch.zeros_like(acc))
            out[row0 + lr] = o.to(q.dtype)
    return out


# ----------------------------------------------------------------------------------------------------------------
# inputs

def make_case(S: int, R: int, *, device="cpu", kv_dtype=torch.float8_e4m3fn, sel="structured", page=64,
              Hq=HQ, Hk=HK, D=HEAD_DIM, budget=BUDGET, ratio=RATIO, seed=0, extra_pages=16,
              logits_chunk=1024) -> dict:
    """One sequence of S tokens whose last R tokens are this prefill chunk (prefix = S - R).

    KV pool: [pages * page, Hk, D] with this sequence's pages scattered over the pool (logical -> physical through
    token_slot_table, like SGLang's req_to_token). Selection per row: top-(budget/ratio) compressed blocks among the
    row's visible complete blocks, by
      structured: shared per-block salience + row noise + recency bonus (neighbouring rows overlap, like real heads)
      uniform:    independent random scores per row (worst case for any union/tile scheme)
    expanded with the SGLang torch expander (+ pending tail), then mapped to physical slots (-1 = invalid)."""
    assert 0 < R <= S
    g = torch.Generator(device="cpu").manual_seed(seed)
    prefix = S - R
    n_pages = (S + page - 1) // page
    pool_pages = n_pages + extra_pages
    perm = torch.randperm(pool_pages - 1, generator=g)[:n_pages] + 1          # page 0 reserved (SGLang dummy slot)
    pos = torch.arange(S)
    table = (perm[pos // page] * page + pos % page).to(torch.int32)            # [S]
    N = pool_pages * page
    k = torch.randn((N, Hk, D), generator=g)
    v = torch.randn((N, Hk, D), generator=g)
    q = torch.randn((R, Hq, D), generator=g)
    k_cache = k.to(kv_dtype).to(device)
    v_cache = v.to(kv_dtype).to(device)
    q = q.to(torch.bfloat16).to(device)
    table = table.to(device)
    block_topk = budget // ratio
    nb = S // ratio
    qpos = torch.arange(prefix, S, device=device)
    row_ends = torch.minimum((qpos + 1) // ratio, torch.tensor(nb, device=device))
    row_starts = torch.zeros_like(row_ends)
    seq_lens = torch.full((R,), S, device=device, dtype=torch.long)
    salience = torch.randn((max(nb, 1),), generator=g).to(device)
    blocks = []
    for r0 in range(0, R, logits_chunk):
        r1 = min(R, r0 + logits_chunk)
        if nb == 0:
            blocks.append(torch.full((r1 - r0, block_topk), -1, dtype=torch.int32, device=device))
            continue
        gg = torch.Generator(device="cpu").manual_seed(seed * 1000003 + r0)
        noise = torch.randn((r1 - r0, nb), generator=gg).to(device)
        if sel == "uniform":
            logits = noise
        else:
            b = torch.arange(nb, device=device)[None, :]
            dist = (row_ends[r0:r1, None] - 1 - b).clamp_min(0).float()
            logits = 1.0 * salience[None, :] + 0.7 * noise + 2.0 * torch.exp(-dist / 64.0)
        logits = logits.clamp_min(0)                  # relu-sum logits are >= 0, like torch_qsa_mqa_prefill
        blocks.append(qsa_fast_topk_xpu(logits, row_starts[r0:r1], row_ends[r0:r1], block_topk))
    block_idx = torch.cat(blocks)
    logical = torch_expand_qsa_block_indices(block_idx, qpos, seq_lens, ratio, budget)            # [R, 2051] int32
    valid = logical >= 0
    phys = torch.where(valid, table[logical.clamp_min(0).long()], torch.full_like(logical, -1)).to(torch.int32)
    return dict(q=q, k_cache=k_cache, v_cache=v_cache, logical=logical, slots=phys, block_idx=block_idx,
                table=table.view(1, -1), qpos=qpos, seq_lens=seq_lens, prefix=prefix, S=S, R=R,
                scale=D ** -0.5, seqs=[(0, 0, R, prefix, S)])


def selection_stats(logical: torch.Tensor, S: int, BM=64, BN=64) -> dict:
    """Union size over all rows and per BM-row tile (fraction of the tile's causal KV blocks touched)."""
    R = logical.shape[0]
    valid = logical >= 0
    n_valid = valid.sum(1).float()
    uni = torch.unique(logical[valid]).numel()
    tiles, frac = 0, 0.0
    for t0 in range(0, R, BM):
        li = logical[t0:t0 + BM]
        blk = torch.unique(li[li >= 0].long() // BN)
        tiles += 1
        frac += blk.numel() / max(1, (int(li.max()) // BN + 1) if li.numel() else 1)
    return dict(rows=R, mean_valid=float(n_valid.mean()), union_tokens=int(uni), union_frac=uni / max(1, S),
                tile_block_frac=frac / max(1, tiles))
