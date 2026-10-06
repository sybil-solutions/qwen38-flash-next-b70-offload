"""N114 decode / verify QSA patch for SGLang on Intel XPU: the paged sparse-attention path of every QSA layer ->
one fused SYCL op (qsa_dec.py, Strata port, MIT) that also does the block expansion and the logical->physical slot
mapping. Env-guarded, default off; graph-capturable (decode runs inside XPU graph capture/replay).

    EXL3_QSA_DEC_SYCL=1        master switch (default 0: nothing is patched)
    EXL3_QSA_DEC_DEFER=1       decode batches (forward_mode.is_decode() and no spec_info): the indexer's
                               select_decode_tokens returns the [rows, 512] block ids and skips the torch expansion
                               (~20 kernels per layer); the fused op expands on the device (default 1)
    EXL3_QSA_DEC_MAX_ROWS=64   calls with more query rows keep the previous path
    EXL3_QSA_DEC_CFG           chunk,cpw,sg,fp8,target_wg (qsa_dec.py; default 64,0,16,1,160)
    EXL3_QSA_DEC_CHECK=N       bring-up: for the first N fused calls outside graph capture also run the previous path
                               and log the max abs difference (adds work, no host sync in the fused path itself)
    EXL3_QSA_DEC_DIR           directory of this file (plugin hook; default /n114)

What is replaced (looked up and wrapped at install time, i.e. AFTER exl3xpu's qsa_xpu.install() and the optional
n107-qsa patch_qsa.install(), so the previous callables are exactly what serves today):

  QwenSparseAttnBackend._forward_paged_attention(q, layer, forward_batch, topk_indices)  [every class of
  sglang.srt.layers.attention.qwen_sparse_attn_backend that defines it]. On XPU its non-CUDA branch is
      _expand_block_indices (no-op for expanded indices) -> _logical_to_physical -> qsa_sparse_attention
  (exl3xpu: torch reference in graph capture / for < 64 rows, ~15 kernels; its _logical_to_physical maps through
  req_to_token[row_req_pool_indices] in graph mode, token_slot_table otherwise). The fused op reproduces both
  mappings (it detects which _logical_to_physical is installed and reads the same tensors) and the reference
  semantics (softmax over valid slots, empty row -> 0, GQA h // 12) with fp32 scores (the reference rounds scores and
  P to bf16). Mode-independent: decode, target_verify and draft decode rows all take it when supported.

  QSAIndexer.select_decode_tokens (the installed one: exl3xpu's SYCL qsa_decode_select + torch expansion) only when
  EXL3_QSA_DEC_DEFER=1 and the batch is plain decode without speculative info (so no MTP shared-index capture sees
  block ids): the expansion callable it uses (closure cell or module global `expand_qsa_block_indices`) is swapped for
  the duration of the call by one that records its arguments and returns the block ids unchanged. The block-id tensor
  carries the recorded call (`_n114_qsa`); whenever the fused op does not take a call, the previous attention path
  gets the indices expanded by the ORIGINAL expansion with the recorded arguments, i.e. bit-identical to today.
  The batch mode comes from a thread-local set by wrappers around QSAIndexer's forward entry points.

Fallbacks: non-XPU, unsupported shape/dtype/layout (qsa_dec.why_not), an unknown _logical_to_physical, rows >
MAX_ROWS, or any exception (then the fused path is disabled for the process, one warning). STATS counts calls per
path and reason.

Serve-stack hook (N104 plugin): after exl3xpu's qsa_xpu.install() (the n107-qsa overlay's install() is the natural
place: right after `_base.install()` / `patch_qsa.install()`), or from sglang_plugin.activate() after exl3xpu:
    if os.environ.get("EXL3_QSA_DEC_SYCL", "0") == "1":
        import sys; sys.path.insert(0, os.environ.get("EXL3_QSA_DEC_DIR", "/n114"))
        import patch_qsa_dec; patch_qsa_dec.install()
install() is idempotent; if exl3xpu replaces select_decode_tokens later, the indexer entry wrapper re-wraps it.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("EXL3_QSA_DEC_SYCL", "0") == "1"
DEFER = os.environ.get("EXL3_QSA_DEC_DEFER", "1") == "1"
MAX_ROWS = int(os.environ.get("EXL3_QSA_DEC_MAX_ROWS", "64"))
STATS: Counter = Counter()
_state = {"broken": False, "check": int(os.environ.get("EXL3_QSA_DEC_CHECK", "0")), "first": True,
          "logged": set(), "defer_off": None}
_tls = threading.local()
_installed = False
_EXPAND = "expand_qsa_block_indices"
_ENTRY = ("forward_xpu", "forward_cuda", "forward_native", "_forward_impl", "forward")
# model-side entry (fb = 3rd argument) that runs the indexer and the MTP shared-index capture; covers indexers whose
# platform forward was bound at construction time
_MODEL_ENTRY = ("_compute_qsa_topk_indices",)
_MODEL_MODULES = ("sglang.srt.models.qwen4_exp", "sglang.srt.models.qwen3_5", "sglang.srt.models.qwen4_exp_mtp")
_INDEXER = [None]
_DEVICES = {"xpu"}          # device types the fused path takes (the offline mock test adds "cpu")


def _log(msg: str) -> None:
    print(f"N114QSA {msg}", file=sys.stderr, flush=True)


def _log_once(key: str, msg: str) -> None:
    if key not in _state["logged"]:
        _state["logged"].add(key)
        _log(msg)


def _capturing() -> bool:
    import torch
    f = getattr(torch.xpu, "is_current_stream_capturing", None) if hasattr(torch, "xpu") else None
    try:
        return bool(f()) if f is not None else False
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ indexer side
class _Deferred:
    __slots__ = ("args", "kwargs", "expand")

    def __init__(self, args, kwargs, expand):
        self.args, self.kwargs, self.expand = args, kwargs, expand

    def field(self, i, name, default=None):
        if len(self.args) > i:
            return self.args[i]
        return self.kwargs.get(name, default)

    def qpos(self):
        return self.field(1, "query_positions")

    def seqlen(self):
        return self.field(2, "sequence_lengths")

    def ratio(self):
        return int(self.field(3, "compress_ratio"))

    def topk(self):
        return int(self.field(4, "token_topk"))

    def run_original(self):
        return self.expand(*self.args, **self.kwargs)


def _swap_target(fn):
    """(get, set) for the expansion callable `fn` uses: a closure cell (exl3xpu) or a module global (SGLang)."""
    fn = getattr(fn, "__func__", fn)
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    if _EXPAND in code.co_freevars and fn.__closure__ is not None:
        cell = fn.__closure__[code.co_freevars.index(_EXPAND)]
        return (lambda: cell.cell_contents), (lambda v: setattr(cell, "cell_contents", v))
    g = getattr(fn, "__globals__", None)
    if g is not None and _EXPAND in code.co_names and _EXPAND in g:
        return (lambda: g[_EXPAND]), (lambda v: g.__setitem__(_EXPAND, v))
    return None


def _defer_ok(forward_batch) -> bool:
    if not (ENABLED and DEFER) or _state["broken"] or _state["defer_off"]:
        return False
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        return fm is not None and bool(fm.is_decode()) and getattr(forward_batch, "spec_info", None) is None
    except Exception:
        return False


def _wrap_select(cls) -> bool:
    cur = cls.__dict__.get("select_decode_tokens")
    if cur is None:
        return False
    if getattr(cur, "_n114", False):
        return True
    swap = _swap_target(cur)
    if swap is None:
        _log_once("noswap", "select_decode_tokens: expansion callable not found; indices stay expanded (no defer)")
        return False
    orig = cur

    def select_decode_tokens(self, *args, **kwargs):
        ctx = getattr(_tls, "defer", False)
        q = args[0] if args else kwargs.get("q")
        if not ctx or q is None or getattr(q.device, "type", None) not in _DEVICES:
            return orig(self, *args, **kwargs)
        get, put = swap
        real = get()
        rec = {}

        def record(*a, **k):
            rec["d"] = _Deferred(a, k, real)
            rec["ret"] = a[0] if a else k.get("block_indices")
            return rec["ret"]

        put(record)
        try:
            out = orig(self, *args, **kwargs)
        finally:
            put(real)
        if "d" not in rec:
            STATS["defer:not_called"] += 1
            return out
        if out is not rec["ret"]:
            # the selection post-processed the expanded tensor: not the known shape of this function -> stop deferring
            _state["defer_off"] = "post-processed expansion"
            _log("WARNING select_decode_tokens post-processes the expansion; defer disabled, recomputing")
            return rec["d"].run_original()
        try:
            out._n114_qsa = rec["d"]
        except Exception:
            STATS["defer:no_attr"] += 1
            return rec["d"].run_original()
        STATS["defer"] += 1
        return out

    select_decode_tokens._n114 = True
    select_decode_tokens.__wrapped__ = orig
    cls.select_decode_tokens = select_decode_tokens
    return True


def _wrap_entries(cls, names=_ENTRY) -> int:
    """Wrap the entry points whose 3rd argument is the forward batch: they set the thread-local defer flag."""
    n = 0
    for name in names:
        cur = cls.__dict__.get(name)
        if cur is None or getattr(cur, "_n114", False) or not callable(cur):
            continue
        orig = cur

        def entry(self, *args, _orig=orig, **kwargs):
            fb = args[2] if len(args) > 2 else kwargs.get("forward_batch")
            prev = getattr(_tls, "defer", False)
            _tls.defer = _defer_ok(fb)
            if _tls.defer and _INDEXER[0] is not None and not getattr(
                    _INDEXER[0].__dict__.get("select_decode_tokens"), "_n114", False):
                _wrap_select(_INDEXER[0])          # something replaced it after install(): wrap the new one
            try:
                return _orig(self, *args, **kwargs)
            finally:
                _tls.defer = prev

        entry._n114 = True
        entry.__wrapped__ = orig
        setattr(cls, name, entry)
        n += 1
    return n


# ---------------------------------------------------------------------------------------------- attention side
def _l2p_route(backend, meta):
    """(table, rreq) reproducing the installed _logical_to_physical, or None when it is not a known one."""
    fn = getattr(type(backend), "_logical_to_physical", None)
    fn = getattr(fn, "__func__", fn)
    name = getattr(fn, "__name__", "")
    if name == "_l2p":                                    # exl3xpu: graph mode through req_to_token
        rr = getattr(meta, "row_req_pool_indices", None)
        rt = getattr(backend, "req_to_token", None)
        if getattr(meta, "is_cuda_graph", False) and rr is not None and rt is not None:
            return rt, rr
        tst = getattr(meta, "token_slot_table", None)     # its other branch is the original mapping
        return (tst, None) if tst is not None else None
    if name == "_logical_to_physical" and str(getattr(fn, "__module__", "")).startswith("sglang."):
        tst = getattr(meta, "token_slot_table", None)
        return (tst, None) if tst is not None else None
    return None


def _fused(backend, q, layer, forward_batch, idx, deferred):
    import qsa_dec
    R = q.shape[0]
    if q.ndim != 3 or R == 0 or R > MAX_ROWS:
        return None, "rows"
    pool = backend.token_to_kv_pool
    kb = pool.get_key_buffer(layer.layer_id)
    vb = pool.get_value_buffer(layer.layer_id)
    rm = getattr(backend, "_resolve_metadata", None)
    meta = rm(forward_batch) if rm is not None else getattr(backend, "forward_metadata", None)
    if meta is None:
        return None, "metadata"
    route = _l2p_route(backend, meta)
    if route is None:
        return None, "l2p"
    table, rreq = route
    t2b = getattr(meta, "token_to_batch_idx", None)
    rowlen = getattr(meta, "sequence_lengths", None)
    if t2b is None or rowlen is None:
        return None, "metadata"
    kw = dict(table=table, rowlen=rowlen, t2b=t2b, rreq=rreq)
    if deferred is not None:
        kw.update(expand=True, qpos=deferred.qpos(), seqlen=deferred.seqlen(), ratio=deferred.ratio(),
                  token_topk=deferred.topk())
    else:
        ub = getattr(backend, "_uses_block_indices", None)
        if ub is not None and ub(idx):
            return None, "block ids (not deferred here)"
        kw.update(expand=False)
    why = qsa_dec.why_not(q, kb, vb, idx, **kw)
    if why is not None:
        return None, why
    out = qsa_dec.decode_attention(q, kb, vb, idx, scale=getattr(layer, "scaling", None) or None, **kw)
    return out.reshape(R, -1), None


def _wrap_attention(cls) -> bool:
    cur = cls.__dict__.get("_forward_paged_attention")
    if cur is None or getattr(cur, "_n114", False):
        return cur is not None
    orig = cur

    def _forward_paged_attention(self, q, layer, forward_batch, topk_indices, *args, **kwargs):
        deferred = getattr(topk_indices, "_n114_qsa", None)

        def previous():
            ti = deferred.run_original() if deferred is not None else topk_indices
            return orig(self, q, layer, forward_batch, ti, *args, **kwargs)

        if _state["broken"] or args or kwargs or q.device.type not in _DEVICES:
            STATS["prev:" + ("broken" if _state["broken"] else "other")] += 1
            return previous()
        try:
            out, reason = _fused(self, q, layer, forward_batch, topk_indices, deferred)
        except Exception as e:
            _state["broken"] = True
            STATS["error"] += 1
            _log(f"WARNING fused decode QSA disabled after error, previous path from now on: {type(e).__name__}: {e}")
            return previous()
        if out is None:
            STATS["prev:" + reason] += 1
            _log_once("r:" + reason, f"previous QSA path kept ({reason}); first such call rows={q.shape[0]}")
            return previous()
        STATS["fused" + (":expand" if deferred is not None else ":logical")] += 1
        if _state["first"]:
            _state["first"] = False
            _log(f"first fused call: rows={q.shape[0]} idx{tuple(topk_indices.shape)} {topk_indices.dtype} "
                 f"expand={deferred is not None} capture={_capturing()}")
        if _state["check"] > 0 and not _capturing():
            _state["check"] -= 1
            try:
                ref = previous()
                d = (out.float() - ref.float()).abs()
                _log(f"check rows={q.shape[0]} max_abs={float(d.max()):.3e} mean_abs={float(d.mean()):.3e} "
                     f"ref_absmax={float(ref.float().abs().max()):.3e}")
            except Exception as e:  # pragma: no cover
                _log(f"check failed: {type(e).__name__}: {e}")
        return out

    _forward_paged_attention._n114 = True
    _forward_paged_attention.__wrapped__ = orig
    cls._forward_paged_attention = _forward_paged_attention
    return True


def install() -> bool:
    global _installed
    if not ENABLED or _installed:
        return _installed
    import qsa_dec
    if not qsa_dec.load():
        _log(f"WARNING library not loadable ({qsa_dec.error()}); previous QSA decode path stays")
        return False
    qb = importlib.import_module("sglang.srt.layers.attention.qwen_sparse_attn_backend")
    attn = [name for name in dir(qb) if isinstance(getattr(qb, name), type)
            and "_forward_paged_attention" in getattr(qb, name).__dict__ and _wrap_attention(getattr(qb, name))]
    if not attn:
        _log("WARNING no _forward_paged_attention found; nothing patched")
        return False
    sel = entries = 0
    if DEFER:
        try:
            qi = importlib.import_module("sglang.srt.layers.attention.qsa.qsa_indexer")
            cls = qi.QSAIndexer
            _INDEXER[0] = cls
            sel = int(_wrap_select(cls))
            if sel:
                entries = _wrap_entries(cls)
                for mn in _MODEL_MODULES:
                    try:
                        mm = importlib.import_module(mn)
                    except Exception:
                        continue
                    for nm in dir(mm):
                        c = getattr(mm, nm)
                        if isinstance(c, type) and c.__module__ == mn:
                            entries += _wrap_entries(c, _MODEL_ENTRY)
        except Exception as e:  # pragma: no cover
            _log(f"WARNING indexer defer hook not installed ({type(e).__name__}: {e}); indices stay expanded")
    _installed = True
    _log(f"installed: {','.join(attn)}._forward_paged_attention -> fused SYCL ({qsa_dec.lib_path()}, cfg {qsa_dec.CFG}, "
         f"max_rows {MAX_ROWS}); defer {'on' if sel and entries else 'off'} (entries {entries}); check {_state['check']}")
    return True


if ENABLED and os.environ.get("EXL3_QSA_DEC_AUTOINSTALL", "0") == "1":  # pragma: no cover
    install()
