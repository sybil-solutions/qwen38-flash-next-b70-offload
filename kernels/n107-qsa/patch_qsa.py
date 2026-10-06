"""N107 QSA prefill patch for SGLang 0.5.20 + exl3xpu on Intel XPU. Env-guarded, default off.

    EXL3_QSA_TRITON=1          master switch (default 0: nothing is patched)
    EXL3_QSA_IMPL=sycl_row     N108 alternative master switch: qsa_sparse_attention -> SYCL per-query kernel
                               (qsa_sycl.py, Strata port; build/qsa_row_sycl.so), union/reference for everything
                               else. Takes precedence over EXL3_QSA_TRITON for the attention; in this mode the
                               indexer patches below are OFF unless EXL3_QSA_NOSYNC=1 / EXL3_QSA_TRITON_EXPAND=1 are set
                               explicitly. Knobs: EXL3_QSA_SYCL_CFG (see qsa_sycl.py), EXL3_QSA_SYCL_MIN_ROWS (16),
                               EXL3_QSA_TRITON_CHECK=N also applies (first N calls compared with the union path)
    EXL3_QSA_SELECT=sycl       N108: QSAIndexer.select_prefill_tokens -> fused SYCL indexer scores + exact radix
                               top-k (same selected sets as fp32 einsum + torch.topk; ties -> lowest index), then the
                               usual expander. Independent of the attention switches (installs on its own too).
    EXL3_QSA_TRITON_VARIANT    row (default) | tile | compact | auto
                                 row:     drop-in for qsa_sparse_attention (physical slots, fp8 decoded in-kernel)
                                 tile:    block-union kernel (64-row tiles, bitmap masks) via forward_extend
                                 compact: row kernel over a per-layer bf16 logical-order copy of each sequence's K/V
                                 auto:    tile while the batch's longest sequence <= EXL3_QSA_TILE_MAX_CTX, else row
    EXL3_QSA_TILE_MAX_CTX      context threshold for auto (default 0 = always row); set from test_qsa_xpu.py results
    EXL3_QSA_TRITON_FP8        bits (default, integer e4m3fn decode) | cast (Triton fp8e4nv -> bf16)
    EXL3_QSA_TRITON_CFG        row kernel  BLOCK_N,num_warps[,num_stages]       (default 64,8)
    EXL3_QSA_TILE_CFG          tile kernel BM,BN,num_warps[,num_stages]         (default 64,64,8)
    EXL3_QSA_GRF               '' | large | auto  (Intel register-file mode compile option)
    EXL3_QSA_TRITON_MIN_ROWS   rows below this use the previous implementation (default 16)
    EXL3_QSA_NOSYNC            1 (default) host-sync removal in the indexer (qsa_nosync.py)
    EXL3_QSA_NOSYNC_CHECK      1: verify the host bounds against the device (syncs; bring-up only)
    EXL3_QSA_TRITON_EXPAND     1 (default) SGLang's Triton block-index expander on XPU (exact for the valid-first
                               top-k layout exl3xpu's qsa_fast_topk produces) instead of the torch argsort expander
    EXL3_QSA_TRITON_CHECK      N: for the first N calls also run the previous implementation and log the max abs
                               difference (syncs; bring-up only)
    EXL3_QSA_DUMP_DIR          dump the logical top-k indices of the first EXL3_QSA_DUMP_N (2) prefill calls with
                               >= 1024 rows to this directory (test_qsa_xpu.py --dump replays them)

Only XPU tensors take the new paths, never inside graph capture; any unsupported shape/dtype, or a Triton compile
error on first use, falls back to the implementation that was installed before (exl3xpu's union/reference).

Installation: exl3xpu's plugin calls exl3xpu.qsa_xpu.install() at activation; qsa_xpu_overlay.py (mounted over
exl3xpu/qsa_xpu.py, with the image's original mounted as exl3xpu/qsa_xpu_base.py) calls install() below right
after the original install, so this patch wraps the functions exl3xpu installed. See README.md.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import torch

logger = logging.getLogger("n107.qsa")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_installed = False
# device types that take the new paths (tests set EXL3_QSA_DEVICES=cpu to drive the patch with TRITON_INTERPRET=1)
_DEVS = tuple(x for x in os.environ.get("EXL3_QSA_DEVICES", "xpu").split(",") if x)


def _env(name, default):
    return os.environ.get(name, default)


def _capturing() -> bool:
    f = getattr(torch.xpu, "is_current_stream_capturing", None) if hasattr(torch, "xpu") else None
    try:
        return bool(f()) if f is not None else False
    except Exception:
        return False


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"N107QSA {msg}", file=sys.stderr, flush=True)


class _Dispatch:
    """qsa_sparse_attention replacement: Triton per-row kernel (or, with fn=, the N108 SYCL kernel) on XPU, previous
    function otherwise."""

    def __init__(self, prev, kmod, fn=None, name="row", min_rows_env="EXL3_QSA_TRITON_MIN_ROWS"):
        self.prev = prev
        self.K = kmod
        self.fn = fn if fn is not None else kmod.qsa_sparse_attention_triton
        self.name = name
        self.min_rows = int(_env(min_rows_env, "16"))
        self.check_left = int(_env("EXL3_QSA_TRITON_CHECK", "0"))
        self.broken = False
        self.calls = 0

    def __call__(self, q, k_cache, v_cache, token_slots, softmax_scale=None):
        if (self.broken or q.device.type not in _DEVS or q.shape[0] < self.min_rows or _capturing()
                or not self.K.supports(q, k_cache, v_cache, token_slots)):
            return self.prev(q, k_cache, v_cache, token_slots, softmax_scale)
        try:
            out = self.fn(q, k_cache, v_cache, token_slots, softmax_scale)
        except Exception as e:  # compile/launch failure: permanent fallback
            self.broken = True
            _log(f"{self.name} kernel failed ({type(e).__name__}: {e}); falling back to {getattr(self.prev, '__name__', self.prev)}")
            return self.prev(q, k_cache, v_cache, token_slots, softmax_scale)
        self.calls += 1
        if self.check_left > 0:
            self.check_left -= 1
            ref = self.prev(q, k_cache, v_cache, token_slots, softmax_scale)
            d = (out.float() - ref.float()).abs()
            _log(f"check rows={q.shape[0]} max_abs={float(d.max()):.3e} mean_abs={float(d.mean()):.3e} "
                 f"ref_absmax={float(ref.float().abs().max()):.3e}")
        return out


def _host_seqs(forward_batch, n_rows):
    """[(seq_id, row0, n_rows, prefix, kv_len)] from host metadata, or None when unavailable/inconsistent."""
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    sl = getattr(forward_batch, "seq_lens_cpu", None)
    if ext is None or sl is None:
        return None
    ext = [int(x) for x in ext]
    sl = [int(x) for x in (sl.tolist() if isinstance(sl, torch.Tensor) else sl)][: len(ext)]
    if len(sl) != len(ext) or sum(ext) != n_rows:
        return None
    out, row0 = [], 0
    for i, (e, L) in enumerate(zip(ext, sl)):
        out.append((i, row0, e, L - e, L))
        row0 += e
    return out


def _wrap_forward_extend(cls, K, variant, dispatch):
    orig = cls.forward_extend
    dump_dir = _env("EXL3_QSA_DUMP_DIR", "")
    dump_left = [int(_env("EXL3_QSA_DUMP_N", "2"))] if dump_dir else [0]
    state = {"broken": False}
    tile_max_ctx = int(_env("EXL3_QSA_TILE_MAX_CTX", "0"))

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, topk_indices=None, **kwargs):
        if (topk_indices is None or q.device.type not in _DEVS or _capturing()
                or self._is_speculative_paged_mode(forward_batch.forward_mode)):
            return orig(self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                        topk_indices=topk_indices, **kwargs)
        n = topk_indices.shape[0]
        seqs = _host_seqs(forward_batch, n)
        if dump_left[0] > 0 and seqs is not None and n >= 1024:
            dump_left[0] -= 1
            try:
                os.makedirs(dump_dir, exist_ok=True)
                md = self._resolve_metadata(forward_batch)
                path = os.path.join(dump_dir, f"qsa_topk_L{layer.layer_id}_{int(time.time() * 1e3)}.pt")
                torch.save(dict(logical=topk_indices.cpu(), seqs=seqs, layer=layer.layer_id,
                                token_to_batch_idx=md.token_to_batch_idx.cpu()), path)
                _log(f"dumped {path}")
            except Exception as e:  # pragma: no cover
                _log(f"dump failed: {e}")
        use = variant
        if variant == "auto":
            use = "tile" if seqs is not None and max(s[4] for s in seqs) <= tile_max_ctx else "row"
        if use == "row" or seqs is None or state["broken"] or n < dispatch.min_rows:
            return orig(self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                        topk_indices=topk_indices, **kwargs)
        # replicate the original prologue (save K/V, reshape, trim DP padding rows)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        q3 = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        num_output_rows = q3.shape[0]
        if n > num_output_rows:
            raise ValueError(f"QSA top-k rows exceed query rows: topk={n}, query={num_output_rows}")
        q3 = q3[:n]
        metadata = self._resolve_metadata(forward_batch)
        pool = self.token_to_kv_pool
        kb = pool.get_key_buffer(layer.layer_id)
        vb = pool.get_value_buffer(layer.layer_id)
        out = None
        if K.supports(q3, kb, vb, topk_indices):
            try:
                if use == "tile":
                    out = K.qsa_tile_attention(q3, kb, vb, topk_indices, metadata.token_slot_table, seqs,
                                               layer.scaling)
                else:  # compact
                    out = K.qsa_compact_row_attention(q3, kb, vb, topk_indices, metadata.token_slot_table,
                                                      metadata.token_to_batch_idx, metadata.sequence_lengths,
                                                      [s[4] for s in seqs], layer.scaling)
            except Exception as e:
                state["broken"] = True
                _log(f"{use} kernel failed ({type(e).__name__}: {e}); falling back")
                out = None
        if out is None:
            slots = self._logical_to_physical(topk_indices, metadata)
            out = dispatch.prev(q3, kb, vb, slots, layer.scaling)
        return self._pad_extend_output(out, num_output_rows)

    forward_extend._n107 = True
    forward_extend.__wrapped__ = orig
    cls.forward_extend = forward_extend


def _install_sycl_row() -> None:
    """EXL3_QSA_IMPL=sycl_row: only the attention entry point changes (plus the indexer patches when explicitly set)."""
    import importlib
    import qsa_sycl as S
    if not S.load():
        _log(f"EXL3_QSA_IMPL=sycl_row: {S.lib_path()} not loadable ({S.error()}); server keeps the previous path")
        return
    done = []
    qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    prev = qb.qsa_sparse_attention
    dispatch = _Dispatch(prev, S, fn=S.qsa_sparse_attention_sycl, name="sycl_row", min_rows_env="EXL3_QSA_SYCL_MIN_ROWS")
    qb.qsa_sparse_attention = dispatch
    done.append(f"qwen_sparse_attn_backend.qsa_sparse_attention -> sycl_row (prev={getattr(prev, '__module__', '?')}."
                f"{getattr(prev, '__name__', '?')})")
    for mod_name in ("exl3xpu.qsa_xpu", "exl3xpu.qsa_xpu_base"):
        m = sys.modules.get(mod_name)
        if m is not None and hasattr(m, "qsa_sparse_attention"):
            m.qsa_sparse_attention = dispatch
    _install_indexer_patches(done, default="0")
    _log(f"installed impl=sycl_row cfg={S.CFG} min_rows={dispatch.min_rows} lib={S.lib_path()}: " + "; ".join(done))


def _install_indexer_patches(done, default="1") -> None:
    import importlib
    if _env("EXL3_QSA_NOSYNC", default) == "1":
        try:
            import qsa_nosync
            done += qsa_nosync.install()
        except Exception as e:
            _log(f"nosync patch failed: {e!r}")
    if _env("EXL3_QSA_TRITON_EXPAND", default) == "1":
        try:
            qi = importlib.import_module("sglang.srt.layers.attention.qsa.qsa_indexer")
            km = importlib.import_module("sglang.srt.layers.attention.qsa.kernel")
            orig_expand = qi.expand_qsa_block_indices
            if not getattr(orig_expand, "_n107", False):
                def expand_qsa_block_indices(block_indices, query_positions, sequence_lengths, compress_ratio,
                                             token_topk):
                    if block_indices.device.type in _DEVS and block_indices.ndim == 2 and \
                            block_indices.shape[1] == (token_topk + compress_ratio - 1) // compress_ratio and \
                            query_positions.numel() == block_indices.shape[0] == sequence_lengths.numel():
                        return km.triton_expand_qsa_block_indices(
                            block_indices.contiguous(),
                            query_positions.to(device=block_indices.device).contiguous(),
                            sequence_lengths.to(device=block_indices.device).contiguous(),
                            compress_ratio, token_topk)
                    return orig_expand(block_indices, query_positions, sequence_lengths, compress_ratio, token_topk)
                expand_qsa_block_indices._n107 = True
                qi.expand_qsa_block_indices = expand_qsa_block_indices
                done.append("qsa_indexer.expand_qsa_block_indices (Triton on XPU)")
        except Exception as e:
            _log(f"expand patch failed: {e!r}")


def _install_select_sycl() -> str:
    """EXL3_QSA_SELECT=sycl: replace QSAIndexer.select_prefill_tokens on XPU (fallback: the original method)."""
    import importlib
    import qsa_sycl as S
    if not S.load():
        return f"select: library not loadable ({S.error()})"
    qi = importlib.import_module("sglang.srt.layers.attention.qsa.qsa_indexer")
    cls = qi.QSAIndexer
    orig = cls.select_prefill_tokens
    if getattr(orig, "_n108", False):
        return "select: already installed"
    state = {"broken": False, "check": int(_env("EXL3_QSA_SELECT_CHECK", "0"))}

    def select_prefill_tokens(self, q, compressed_keys, row_starts, row_ends, query_positions,
                              sequence_lengths_for_rows):
        if (state["broken"] or q.device.type not in _DEVS or _capturing() or q.shape[0] == 0
                or not S.supports_select(q, compressed_keys, row_starts, row_ends, self.block_topk)):
            return orig(self, q, compressed_keys, row_starts, row_ends, query_positions, sequence_lengths_for_rows)
        try:
            if compressed_keys.shape[0] == 0:
                bi = torch.full((q.shape[0], self.block_topk), -1, dtype=torch.int32, device=q.device)
            else:
                bi = S.prefill_select(q, compressed_keys, row_starts, row_ends, self.block_topk)
        except Exception as e:
            state["broken"] = True
            _log(f"sycl select failed ({type(e).__name__}: {e}); falling back to the torch path")
            return orig(self, q, compressed_keys, row_starts, row_ends, query_positions, sequence_lengths_for_rows)
        out = qi.expand_qsa_block_indices(bi, query_positions, sequence_lengths_for_rows,
                                          compress_ratio=self.compress_ratio, token_topk=self.token_topk)
        if state["check"] > 0:   # bring-up: same selected token sets as the original (syncs)
            state["check"] -= 1
            ref = orig(self, q, compressed_keys, row_starts, row_ends, query_positions, sequence_lengths_for_rows)
            big = torch.iinfo(torch.int32).max
            a = torch.sort(torch.where(out >= 0, out, torch.full_like(out, big)), 1).values
            b = torch.sort(torch.where(ref >= 0, ref, torch.full_like(ref, big)), 1).values
            _log(f"select check rows={q.shape[0]} identical_rows={float((a == b).all(1).float().mean()):.6f}")
        return out
    select_prefill_tokens._n108 = True
    select_prefill_tokens.__wrapped__ = orig
    cls.select_prefill_tokens = select_prefill_tokens
    return "QSAIndexer.select_prefill_tokens -> sycl select"


def install(kmod=None) -> None:
    """kmod: kernel module override (tests); default qsa_sparse_triton."""
    global _installed
    impl = _env("EXL3_QSA_IMPL", "")
    if not _installed and _env("EXL3_QSA_SELECT", "") == "sycl" and not getattr(install, "_sel_done", False):
        install._sel_done = True
        try:
            _log(_install_select_sycl())
        except Exception as e:
            _log(f"sycl select install failed ({e!r}); indexer unchanged")
    if _installed or (_env("EXL3_QSA_TRITON", "0") != "1" and impl != "sycl_row"):
        if impl and impl != "sycl_row" and not _installed:
            _log(f"unknown EXL3_QSA_IMPL={impl}: nothing installed")
        return
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()) and "cpu" not in _DEVS:
        _log("EXL3_QSA_TRITON=1 / EXL3_QSA_IMPL but no XPU: not installed")
        return
    _installed = True
    if impl == "sycl_row":
        try:
            _install_sycl_row()
        except Exception as e:  # never break server start-up
            _log(f"sycl_row install failed ({e!r}); server keeps the previous path")
        return
    import importlib
    if kmod is None:
        import qsa_sparse_triton as kmod
    K = kmod
    variant = _env("EXL3_QSA_TRITON_VARIANT", "row")
    if variant not in ("row", "tile", "compact", "auto"):
        _log(f"unknown EXL3_QSA_TRITON_VARIANT={variant}; using row")
        variant = "row"
    done = []
    qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    prev = qb.qsa_sparse_attention
    dispatch = _Dispatch(prev, K)
    qb.qsa_sparse_attention = dispatch
    done.append(f"qwen_sparse_attn_backend.qsa_sparse_attention (prev={getattr(prev, '__module__', '?')}."
                f"{getattr(prev, '__name__', '?')})")
    for mod_name in ("exl3xpu.qsa_xpu", "exl3xpu.qsa_xpu_base"):
        m = sys.modules.get(mod_name)
        if m is not None and hasattr(m, "qsa_sparse_attention"):
            m.qsa_sparse_attention = dispatch
    if variant != "row" or _env("EXL3_QSA_DUMP_DIR", ""):
        for name in dir(qb):
            c = getattr(qb, name)
            if isinstance(c, type) and "forward_extend" in c.__dict__ and "_logical_to_physical" in c.__dict__ \
                    and not getattr(c.forward_extend, "_n107", False):
                _wrap_forward_extend(c, K, variant, dispatch)
                done.append(f"{name}.forward_extend ({variant})")
    _install_indexer_patches(done, default="1")
    _log(f"installed variant={variant} fp8={K.FP8_MODE} row_cfg={K.ROW_CFG} tile_cfg={K.TILE_CFG}: " + "; ".join(done))
