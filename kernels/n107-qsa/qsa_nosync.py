"""N107 QSA: host-sync-free replacements for the prefill path of SGLang's QSA indexer (results bit-identical).

Per QSA layer and prefill forward, SGLang 0.5.20 on XPU syncs the host 3x before the attention itself:
  1. QSAIndexer.apply_rope (q path, project_qk)          int(positions.max().item())  -> rotary cache guard
  2. QSAIndexer.apply_rope (compressed keys, update_key_state_and_compress -> normalize_compressed_keys)  same
  3. QSAIndexerMetadata.get_prefill_mqa_inputs           self.sequence_lengths.tolist()
(the attention's union path adds unique/nonzero/numel; qsa_sparse_triton.py removes those).

Fixes, all with host data SGLang already has (ForwardBatch.seq_lens_cpu, a CPU tensor):
  1+2. `_ensure_cos_sin_cache_length(n)` only grows the cache when n >= its length (SGLang rotary_embedding/base.py:
       "Ensure cos_sin_cache length > needed_max_pos"; rows are appended, existing rows never change). For a plain
       extend batch without multimodal inputs every position is < max(seq_lens), so calling it with the host bound
       max(seq_lens_cpu) - 1 >= positions.max() is a no-op whenever the original call is a no-op, and otherwise only
       appends rows the batch never reads. Identical outputs, no sync. Multimodal / speculative / decode batches keep
       the original code path.
  3.   the per-sequence lengths are read from seq_lens_cpu[:n] instead of a D2H .tolist().
The host context is set by a wrapper around QSAIndexer.forward_cuda (the XPU dispatch target via the exl3xpu
forward_xpu shim) and cleared after it, so no other caller sees it. EXL3_QSA_NOSYNC_CHECK=1 re-enables the device
reads and asserts the host values match (bring-up only: it syncs).

The replacement bodies are verbatim copies of SGLang's with only the marked lines changed; dependencies are injected
so the functions can be unit-tested without SGLang (test_qsa_xpu.py --nosync).
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
from typing import Callable, List, Optional

import torch

logger = logging.getLogger(__name__)
CHECK = os.environ.get("EXL3_QSA_NOSYNC_CHECK", "0") == "1"

_tls = threading.local()


class HostCtx:
    __slots__ = ("seq_lens", "max_pos")

    def __init__(self, seq_lens: Optional[List[int]], max_pos: Optional[int]):
        self.seq_lens = seq_lens
        self.max_pos = max_pos


def current() -> Optional[HostCtx]:
    return getattr(_tls, "ctx", None)


def host_ctx_from_forward_batch(forward_batch) -> Optional[HostCtx]:
    """Host bounds for a plain (non-speculative) extend/mixed batch; None for anything else."""
    try:
        mode = forward_batch.forward_mode
        if mode.is_decode() or not mode.is_extend():
            return None
        for name in ("is_target_verify", "is_draft_extend", "is_draft_extend_v2"):
            f = getattr(mode, name, None)
            if f is not None and f():
                return None
        sl = getattr(forward_batch, "seq_lens_cpu", None)
        if sl is None:
            return None
        n = int(forward_batch.seq_lens.shape[0])            # host shape, no sync
        if isinstance(sl, torch.Tensor):
            if sl.device.type != "cpu":
                return None
            lens = [int(x) for x in sl[:n].tolist()]           # CPU tensor -> host list, no device involved
        else:
            lens = [int(x) for x in list(sl)[:n]]
        if len(lens) != n or n == 0:
            return None
        has_mm = False
        cm = getattr(forward_batch, "contains_mm_inputs", None)
        if cm is not None:
            has_mm = bool(cm())
        max_pos = None if has_mm else max(lens) - 1
        return HostCtx(lens, max_pos)
    except Exception:  # pragma: no cover - never let the fast path break a forward
        return None


def wrap_forward_cuda(orig: Callable) -> Callable:
    def forward_cuda(self, hidden_states, positions, forward_batch, indexer_metadata, *a, **k):
        prev = getattr(_tls, "ctx", None)
        _tls.ctx = host_ctx_from_forward_batch(forward_batch)
        try:
            return orig(self, hidden_states, positions, forward_batch, indexer_metadata, *a, **k)
        finally:
            _tls.ctx = prev
    forward_cuda._n107 = True
    forward_cuda.__wrapped__ = orig
    return forward_cuda


_GUARD_OK = {}


def guard_is_grow_only(rotary_emb) -> bool:
    """True when the rotary class' _ensure_cos_sin_cache_length is SGLang's grow-only guard (checked once per class)."""
    cls = type(rotary_emb)
    ok = _GUARD_OK.get(cls)
    if ok is None:
        try:
            src = inspect.getsource(cls._ensure_cos_sin_cache_length)
            ok = ("cur_len" in src and "needed_max_pos < cur_len" in src and "torch.cat" in src)
        except Exception:
            ok = False
        if not ok:
            logger.warning("n107 qsa_nosync: %s._ensure_cos_sin_cache_length is not the known grow-only guard; "
                           "keeping the synchronous .item() path for it", cls.__name__)
        _GUARD_OK[cls] = ok
    return ok


def make_apply_rope(get_is_capture_mode: Callable[[], bool], apply_rotary_emb: Callable) -> Callable:
    """QSAIndexer.apply_rope with the cache-guard argument taken from the host context."""

    def apply_rope(self, positions: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = positions.shape[-1] if positions.ndim == 2 else positions.numel()
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if not get_is_capture_mode() and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length"):
            # ---- n107: host bound instead of int(positions.max().item()) ----
            ctx = current()
            if ctx is not None and ctx.max_pos is not None and guard_is_grow_only(self.rotary_emb):
                if CHECK:
                    dev_max = int(positions.max().item())
                    assert dev_max <= ctx.max_pos, f"n107 nosync: positions.max() {dev_max} > host bound {ctx.max_pos}"
                self.rotary_emb._ensure_cos_sin_cache_length(ctx.max_pos)
            else:
                self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
            # ---- end n107 ----

        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[:, :half_rotary_dim]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[:, :half_rotary_dim]
        rotated = apply_rotary_emb(tensor[..., :rotary_dim], cos, sin, self.rotary_emb.is_neox_style)
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    apply_rope._n107 = True
    return apply_rope


def make_get_prefill_mqa_inputs(build_qsa_row_ranges: Callable) -> Callable:
    """QSAIndexerMetadata.get_prefill_mqa_inputs with host sequence lengths."""

    def get_prefill_mqa_inputs(self, layer_id: int, positions: torch.Tensor):
        pool = self.token_to_kv_pool
        ratio = self.compress_ratio
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        parts = []
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        # ---- n107: host lengths instead of sequence_lengths.tolist() ----
        ctx = current()
        n = int(sequence_lengths.shape[0])
        if ctx is not None and ctx.seq_lens is not None and len(ctx.seq_lens) == n:
            sequence_lengths_list = ctx.seq_lens
            if CHECK:
                dev = sequence_lengths.tolist()
                assert dev == sequence_lengths_list, f"n107 nosync: seq_lens device {dev} != host {sequence_lengths_list}"
        else:
            sequence_lengths_list = sequence_lengths.tolist()
        # ---- end n107 ----
        for sequence_id in range(len(sequence_lengths_list)):
            complete_blocks = int(sequence_lengths_list[sequence_id]) // ratio
            if complete_blocks == 0:
                continue
            compressed_locs = (self.token_slot_table[sequence_id, : complete_blocks * ratio : ratio].long() // ratio)
            parts.append(compressed_buffer.index_select(0, compressed_locs))
        compressed_keys = (
            torch.cat(parts, dim=0) if parts
            else compressed_buffer.new_empty((0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim))
        )
        num_valid_tokens = self.token_to_batch_idx.numel()
        if positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA prefill positions are shorter than the request mapping: "
                f"positions={positions.numel()}, mapping={num_valid_tokens}"
            )
        positions = positions[:num_valid_tokens]
        row_starts, row_ends, _ = build_qsa_row_ranges(
            sequence_lengths,
            positions.to(sequence_lengths.device),
            self.token_to_batch_idx.to(sequence_lengths.device),
            self.compress_ratio,
        )
        return compressed_keys, row_starts, row_ends, sequence_lengths

    get_prefill_mqa_inputs._n107 = True
    return get_prefill_mqa_inputs


def install() -> List[str]:
    """Patch SGLang's QSAIndexer / QSAIndexerMetadata in place. Returns the list of patched names."""
    import importlib
    done = []
    qi = importlib.import_module("sglang.srt.layers.attention.qsa.qsa_indexer")
    md = importlib.import_module("sglang.srt.layers.attention.qsa.metadata")
    cls = qi.QSAIndexer
    if not getattr(cls.forward_cuda, "_n107", False):
        cls.forward_cuda = wrap_forward_cuda(cls.forward_cuda)
        done.append("QSAIndexer.forward_cuda(host ctx)")
    if not getattr(cls.apply_rope, "_n107", False):
        cls.apply_rope = make_apply_rope(qi.get_is_capture_mode, qi.apply_rotary_emb)
        done.append("QSAIndexer.apply_rope")
    M = md.QSAIndexerMetadata
    if not getattr(M.get_prefill_mqa_inputs, "_n107", False):
        M.get_prefill_mqa_inputs = make_get_prefill_mqa_inputs(md.build_qsa_row_ranges)
        done.append("QSAIndexerMetadata.get_prefill_mqa_inputs")
    return done
