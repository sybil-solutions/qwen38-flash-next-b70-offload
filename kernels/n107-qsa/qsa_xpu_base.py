"""QSA (Qwen sparse attention, Qwen3.8-Flash-Next full-attention layers) on XPU.

SGLang's non-CUDA fallbacks for the QSA block top-k (`qsa_fast_topk`) and the sparse GQA attention
(`qsa_sparse_attention_reference`) loop over query rows in Python with `int(tensor[row])` host reads: ~60 tok/s prefill
on the B70 and not capturable in an XPU graph ("wait method cannot be used for an event associated with a command
graph"). Same semantics here, vectorised and fixed-shape:
  * top-k: mask columns outside [start, start + length) to -inf, one torch.topk over the row block, indices relative to
    start, entries past `length` = -1 (the CUDA operator's fixed-width relative output; order may differ, the expander
    sorts valid entries anyway).
  * sparse attention: gather K/V of the selected slots per row block (EXL3_QSA_ROWS rows at a time), masked softmax
    (rows without any valid slot -> 0, like the reference), GQA by head grouping.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)
_ROWS = int(os.environ.get("EXL3_QSA_ROWS", "256"))


def qsa_fast_topk(logits: torch.Tensor, row_starts: torch.Tensor, row_ends: torch.Tensor, topk: int) -> torch.Tensor:
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


def qsa_sparse_attention_reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                   token_slots: torch.Tensor, softmax_scale: Optional[float] = None) -> torch.Tensor:
    scale = softmax_scale or q.shape[-1] ** -0.5
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    out = torch.empty_like(q)
    for r0 in range(0, R, _ROWS):
        r1 = min(R, r0 + _ROWS)
        slots = token_slots[r0:r1]
        valid = slots >= 0                                                   # [r, S]
        idx = slots.clamp_min(0).long()
        kk = k_cache.index_select(0, idx.reshape(-1)).view(r1 - r0, -1, Hk, D).to(torch.bfloat16)
        vv = v_cache.index_select(0, idx.reshape(-1)).view(r1 - r0, -1, Hk, D).to(torch.bfloat16)
        qq = q[r0:r1].view(r1 - r0, Hk, G, D).to(torch.bfloat16)
        s = torch.einsum("rhgd,rshd->rhgs", qq, kk).float() * scale
        s = s.masked_fill(~valid[:, None, None, :], -float("inf"))
        p = torch.softmax(s, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        o = torch.einsum("rhgs,rshd->rhgd", p.to(torch.bfloat16), vv)
        out[r0:r1] = o.reshape(r1 - r0, Hq, D).to(q.dtype)
    return out


_DENSE_MIN_ROWS = int(os.environ.get("EXL3_QSA_DENSE_MIN_ROWS", "64"))
_DENSE_SCORE_BYTES = int(os.environ.get("EXL3_QSA_DENSE_BYTES", str(384 << 20)))


def qsa_sparse_attention_union(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                               token_slots: torch.Tensor, softmax_scale: Optional[float] = None) -> torch.Tensor:
    """Exactly the sparse attention, computed as dense attention over the UNION of the selected slots with every
    non-selected (row, key) pair masked to -inf: K/V gathered once per call instead of once per query row, bf16 GEMMs.
    Prefill only (data-dependent union size -> host sync, not capturable)."""
    scale = softmax_scale or q.shape[-1] ** -0.5
    R, Hq, D = q.shape
    Hk = k_cache.shape[1]
    G = Hq // Hk
    valid = token_slots >= 0
    uni = torch.unique(token_slots[valid])                          # sorted physical slots
    nu = int(uni.numel())
    out = torch.zeros_like(q)
    if nu == 0:
        return out
    # invalid (-1) entries go to a spare column nu (dropped): writing them anywhere real could clear a selected key
    col = torch.where(valid, torch.searchsorted(uni, token_slots.clamp_min(0).to(uni.dtype)),
                      torch.full_like(token_slots, nu, dtype=torch.long))
    kk = k_cache.index_select(0, uni.long()).to(torch.bfloat16).permute(1, 2, 0).contiguous()   # [Hk, D, nu]
    vv = v_cache.index_select(0, uni.long()).to(torch.bfloat16).transpose(0, 1).contiguous()   # [Hk, nu, D]
    B = max(8, min(R, _DENSE_SCORE_BYTES // max(1, Hq * nu * 6)))
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


def qsa_sparse_attention(q, k_cache, v_cache, token_slots, softmax_scale=None):
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    capturing = hasattr(torch.xpu, "is_current_stream_capturing") and torch.xpu.is_current_stream_capturing()
    if q.shape[0] >= _DENSE_MIN_ROWS and not capturing and os.environ.get("EXL3_QSA_UNION", "1") == "1":
        return qsa_sparse_attention_union(q, k_cache, v_cache, token_slots, softmax_scale)
    return qsa_sparse_attention_reference(q, k_cache, v_cache, token_slots, softmax_scale)


def install() -> None:
    import importlib
    patched = []
    for mod_name in ("sglang.srt.layers.attention.qsa.kernel", "sglang.srt.layers.attention.qsa.qsa_indexer",
                     "sglang.srt.layers.attention.qsa.metadata", "sglang.srt.layers.attention.qwen_sparse_attn_backend",
                     "sglang.srt.layers.attention.qsa"):
        try:
            m = importlib.import_module(mod_name)
        except Exception as e:  # pragma: no cover
            logger.info("exl3xpu: QSA shim: %s not importable (%s)", mod_name, e)
            continue
        for name, fn in (("qsa_fast_topk", qsa_fast_topk), ("qsa_sparse_attention", qsa_sparse_attention),
                         ("qsa_sparse_attention_reference", qsa_sparse_attention_reference)):
            if hasattr(m, name):
                setattr(m, name, fn)
                patched.append(f"{mod_name.rsplit('.', 1)[-1]}.{name}")
    logger.info("exl3xpu: QSA XPU shim: vectorised top-k + sparse attention (%s)", ", ".join(patched))
    if os.environ.get("EXL3_QSA_GRAPH_KERNELS", "1") == "1":
        # XPU graph replay: SGLang refreshes the QSA graph metadata with Triton kernels on CUDA only and otherwise uses
        # a host "slow-path" fallback, with which graph-replayed decode degenerates on the B70 (eager is correct).
        # The Triton kernels are device-agnostic: allow them on XPU.
        try:
            gm = importlib.import_module("sglang.srt.layers.attention.qsa.graph_metadata")
            qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")

            def supports_graph_metadata_kernels(pool, device):
                from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
                return torch.device(device).type in ("cuda", "xpu") and isinstance(pool, QSATokenToKVPool)
            gm.supports_graph_metadata_kernels = supports_graph_metadata_kernels
            cls = qb.QwenSparseAttnBackend if hasattr(qb, "QwenSparseAttnBackend") else None
            for name in dir(qb):
                c = getattr(qb, name)
                if isinstance(c, type) and hasattr(c, "_can_replay_with_gpu_kernels"):
                    def _can(self, metadata, seq_lens):
                        if self.req_to_token is None:
                            return False
                        return seq_lens.device.type in ("cuda", "xpu") and supports_graph_metadata_kernels(
                            metadata.indexer_metadata.token_to_kv_pool, seq_lens.device)
                    c._can_replay_with_gpu_kernels = _can
                    logger.info("exl3xpu: QSA graph-metadata Triton kernels enabled on XPU (%s)", name)
                    if os.environ.get("EXL3_QSA_TRACE", "0") == "1":
                        import sys as _sys
                        for meth in ("_replay_cuda_graph_metadata_gpu", "_update_qsa_cuda_graph_metadata",
                                     "_replay_cuda_graph_metadata"):
                            if hasattr(c, meth):
                                orig = getattr(c, meth)

                                def wrap(self, *a, _o=orig, _m=meth, **k):
                                    n = getattr(self, "_exl3_trace_n", {})
                                    n[_m] = n.get(_m, 0) + 1
                                    self._exl3_trace_n = n
                                    if n[_m] <= 3:
                                        print(f"EXL3_QSA_TRACE {_m} call {n[_m]}", file=_sys.stderr, flush=True)
                                    return _o(self, *a, **k)
                                setattr(c, meth, wrap)
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: QSA graph-metadata patch failed (%s)", e)
    if os.environ.get("EXL3_QSA_TRITON_EXTEND", "0") == "1":   # REJECTED on B70: 2.5-6.5 s per 4k chunk (torch 2.8)
        # Prefill: SGLang's CUDA branch of forward_extend runs Triton sparse-GQA kernels (device-agnostic); the torch
        # fallback gathers ~2k keys per query row (~250 ms per layer per 4k chunk on the B70). Take the Triton branch
        # on XPU too: re-define forward_extend with its `q.is_cuda` tests accepting XPU tensors.
        try:
            import inspect, re, textwrap
            qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
            if not getattr(torch.cuda, "_exl3_devname", False):
                _orig_name = torch.cuda.get_device_name
                torch.cuda.get_device_name = lambda *a, **k: (torch.xpu.get_device_name(0) if not torch.cuda.is_available()
                                                              else _orig_name(*a, **k))
                torch.cuda._exl3_devname = True
            for name in dir(qb):
                c = getattr(qb, name)
                if isinstance(c, type) and "forward_extend" in c.__dict__ and not getattr(c.forward_extend, "_exl3_xpu", False):
                    src = textwrap.dedent(inspect.getsource(c.forward_extend))
                    new_src = re.sub(r"\bq\.is_cuda\b", '(q.device.type in ("cuda", "xpu"))', src)
                    if new_src == src:
                        continue
                    ns: dict = {}
                    exec(compile(new_src, f"<exl3xpu:{name}.forward_extend>", "exec"), qb.__dict__, ns)
                    ns["forward_extend"]._exl3_xpu = True
                    c.forward_extend = ns["forward_extend"]
                    logger.info("exl3xpu: QSA %s.forward_extend uses the Triton sparse-GQA kernels on XPU", name)
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: QSA Triton extend patch failed (%s)", e)
    if os.environ.get("EXL3_QSA_DECODE_SELECT", "1") == "1":
        # decode block selection: fused SYCL kernel bounded by the real context (the torch fallback scores and top-ks
        # the whole graph-sized page table, 65,536 compressed keys, every step)
        try:
            qi = importlib.import_module("sglang.srt.layers.attention.qsa.qsa_indexer")
            from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
            from .moe_offload import ops as _ops
            cls = qi.QSAIndexer
            orig_sel = cls.select_decode_tokens

            def select_decode_tokens(self, q, compressed_cache, compressed_page_table, compressed_lengths,
                                     max_model_len, query_positions, sequence_lengths):
                if q.dtype not in (torch.bfloat16, torch.float16) or compressed_cache.dtype != q.dtype:
                    return orig_sel(self, q, compressed_cache, compressed_page_table, compressed_lengths,
                                    max_model_len, query_positions, sequence_lengths)
                B = q.shape[0]
                scratch = torch.empty((B, int(max_model_len)), dtype=torch.float32, device=q.device)
                out = torch.empty((B, self.block_topk), dtype=torch.int32, device=q.device)
                _ops().qsa_decode_select(q.contiguous(), compressed_cache.contiguous(),
                                         compressed_page_table.to(torch.int32).contiguous(),
                                         compressed_lengths.to(torch.int32).contiguous(), scratch, out, self.block_topk,
                                         int(getattr(self, "index_n_heads", 0) or 0))
                return expand_qsa_block_indices(out, query_positions, sequence_lengths,
                                                compress_ratio=self.compress_ratio, token_topk=self.token_topk)
            cls.select_decode_tokens = select_decode_tokens
            logger.info("exl3xpu: QSA decode block selection: fused SYCL kernel")
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: QSA decode select patch failed (%s)", e)
    # Graph replay on non-CUDA devices: the backend's torch path maps logical -> physical KV slots through
    # metadata.token_slot_table, which in graph mode is a DUMMY buffer (_graph_dummy_token_slot_table, only the CUDA
    # kernels are fed through the graph page tables) and is never refreshed at replay -> full-attention layers read
    # the wrong KV (graph decode degraded from the first step, eager correct). Map through req_to_token with the
    # refreshed row_req_pool_indices instead.
    try:
        qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
        for name in dir(qb):
            c = getattr(qb, name)
            if isinstance(c, type) and "_logical_to_physical" in c.__dict__:
                orig_l2p = c.__dict__["_logical_to_physical"]
                orig_fn = orig_l2p.__func__ if isinstance(orig_l2p, staticmethod) else orig_l2p

                def _l2p(self, logical_indices, metadata, _orig=orig_fn):
                    rr = getattr(metadata, "row_req_pool_indices", None)
                    rt = getattr(self, "req_to_token", None)
                    if not getattr(metadata, "is_cuda_graph", False) or rr is None or rt is None:
                        return _orig(logical_indices, metadata)
                    sequence_ids = metadata.token_to_batch_idx.long()
                    row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(0, sequence_ids)
                    valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
                    safe = logical_indices.clamp(min=0, max=rt.shape[1] - 1).long()
                    req = rr.long().index_select(0, sequence_ids)
                    slots = rt[req[:, None], safe]
                    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)
                c._logical_to_physical = _l2p
                logger.info("exl3xpu: QSA %s._logical_to_physical: graph mode maps through req_to_token", name)
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: QSA logical->physical patch failed (%s)", e)
