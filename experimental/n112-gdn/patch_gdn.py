"""N112 GDN prefill patch for SGLang 0.5.20 on Intel XPU: TritonGDNKernel.extend -> SYCL token-serial recurrence
(gdn_sycl.py, Strata port, MIT). Env-guarded, default off.

    EXL3_GDN_SYCL=1          master switch (default 0: nothing is patched)
    EXL3_GDN_SYCL_CFG        rg,cb,grf launch config (gdn_sycl.py; default 4,32,128)
    EXL3_GDN_SYCL_MIN_T      extends with fewer tokens use Triton (default 1)
    EXL3_GDN_SYCL_CHECK=N    bring-up: for the first N SYCL calls also run Triton on a copy of the state pool and log
                             the max abs difference of o and of the final states (syncs)
    EXL3_GDN_DIR             directory of this file (plugin hook; default /n112gdn)

What is replaced: only the extend (prefill / chunked-prefill) call TritonGDNKernel.extend ->
chunk_gated_delta_rule(q, k, v, g, beta, initial_state=pool, initial_state_indices=slots, cu_seqlens,
use_qk_l2norm_in_kernel=True, inplace_update=True). Decode, target-verify and the conv / gating / norm kernels are
untouched. Same outputs (o [1, T, H, V] bf16) and the same in-place final-state write-back to pool[slot]; the
per-chunk states `h` are not produced, so batches whose mamba radix tracking needs them
(forward_metadata.track_ssm_h_src non-empty) keep Triton, as do inplace_update=False (multi-item scoring), graph
capture, non-XPU tensors and any shape / dtype / layout gdn_sycl.why_not() rejects. A kernel error disables the SYCL
path for the rest of the process (one warning) and Triton is used from then on. STATS counts calls per path.

Use: exl3xpu plugin hook in sglang_plugin.activate():
    if os.environ.get("EXL3_GDN_SYCL", "0") == "1":
        import sys; sys.path.insert(0, os.environ.get("EXL3_GDN_DIR", "/n112gdn"))
        import patch_gdn; patch_gdn.install()
"""
from __future__ import annotations

import os
import sys
import threading
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("EXL3_GDN_SYCL", "0") == "1"
MIN_T = int(os.environ.get("EXL3_GDN_SYCL_MIN_T", "1"))
STATS: Counter = Counter()
_state = {"broken": False, "check": int(os.environ.get("EXL3_GDN_SYCL_CHECK", "0")), "first": True, "logged": set()}
_tls = threading.local()
_installed = False


def _log(msg: str) -> None:
    print(f"N112GDN {msg}", file=sys.stderr, flush=True)


def _capturing() -> bool:
    import torch
    f = getattr(torch.xpu, "is_current_stream_capturing", None) if hasattr(torch, "xpu") else None
    try:
        return bool(f()) if f is not None else False
    except Exception:
        return False


def _check(K, orig, self, q, k, v, g, beta, ssm_states, cache_indices, query_start_loc, out, kw):
    import torch
    _state["check"] -= 1
    try:
        pool_ref = ssm_states.clone()
        # the SYCL call already updated ssm_states in place; recompute Triton from the pre-call states is impossible
        # here, so the check path runs Triton FIRST on a copy taken before the SYCL call (see _extend)
        ref_o = kw["ref_o"]
        ref_pool = kw["ref_pool"]
        idx = cache_indices[cache_indices >= 0].long()
        do = (out[0].float() - ref_o.float()).abs()
        ds = (ssm_states[idx].float() - ref_pool[idx].float()).abs()
        _log(f"check T={q.shape[1]} nseq={cache_indices.numel()} o max_abs={float(do.max()):.3e} "
             f"o_ref_absmax={float(ref_o.float().abs().max()):.3e} state max_abs={float(ds.max()):.3e} "
             f"state_ref_absmax={float(ref_pool[idx].float().abs().max()):.3e}")
        del pool_ref
    except Exception as e:  # pragma: no cover
        _log(f"check failed: {type(e).__name__}: {e}")


def install(import_now: bool = True) -> bool:
    """Patch TritonGDNKernel.extend (and wrap GDNAttnBackend.forward_extend for the radix-track flag)."""
    global _installed
    if not ENABLED or _installed:
        return _installed
    import gdn_sycl
    if not gdn_sycl.load():
        _log(f"WARNING library not loadable ({gdn_sycl.error()}); Triton GDN prefill stays")
        return False
    from sglang.srt.layers.attention.linear.kernels import gdn_triton as M
    from sglang.srt.layers.attention.linear import gdn_backend as B

    orig_extend = M.TritonGDNKernel.extend

    def extend(self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, inplace_update=True, **kwargs):
        reason = None
        if _state["broken"]:
            reason = "broken"
        elif getattr(_tls, "need_h", False):
            reason = "track_h"
        elif q.device.type != "xpu" or _capturing():
            reason = "device/capture"
        elif q.ndim == 4 and q.shape[1] < MIN_T:
            reason = "min_t"
        else:
            reason = gdn_sycl.why_not(q, k, v, g, beta, ssm_states, cache_indices, query_start_loc, True, inplace_update)
        if reason is None:
            ref = None
            if _state["check"] > 0:
                pool_copy = ssm_states.clone()
                ref_o = orig_extend(self, q, k, v, g, beta, ssm_states=pool_copy, cache_indices=cache_indices,
                                    query_start_loc=query_start_loc, inplace_update=inplace_update, **kwargs)[0]
                ref = (ref_o, pool_copy)
            try:
                out = gdn_sycl.chunk_gated_delta_rule_sycl(q, k, v, g, beta, None, ssm_states, cache_indices,
                                                           query_start_loc, True)
            except Exception as e:
                _state["broken"] = True
                _log(f"WARNING SYCL GDN prefill disabled after error, Triton from now on: {type(e).__name__}: {e}")
                STATS["error"] += 1
                return orig_extend(self, q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,
                                   query_start_loc=query_start_loc, inplace_update=inplace_update, **kwargs)
            STATS["sycl"] += 1
            if _state["first"]:
                _state["first"] = False
                _log(f"first SYCL extend: q{tuple(q.shape)} stride{tuple(q.stride())} v{tuple(v.shape)} "
                     f"stride{tuple(v.stride())} g{tuple(g.shape)} {g.dtype} beta {beta.dtype} state{tuple(ssm_states.shape)} "
                     f"{ssm_states.dtype} idx {cache_indices.dtype}{tuple(cache_indices.shape)} cu {query_start_loc.dtype} "
                     f"cfg {gdn_sycl.CFG}")
            if ref is not None:
                _check(None, orig_extend, self, q, k, v, g, beta, ssm_states, cache_indices, query_start_loc, out,
                       {"ref_o": ref[0], "ref_pool": ref[1]})
            return out
        STATS["triton:" + reason] += 1
        if reason not in _state["logged"]:
            _state["logged"].add(reason)
            _log(f"Triton GDN extend kept ({reason}); first such call T={q.shape[1] if q.ndim == 4 else '?'}")
        return orig_extend(self, q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,
                           query_start_loc=query_start_loc, inplace_update=inplace_update, **kwargs)

    extend.__wrapped__ = orig_extend
    M.TritonGDNKernel.extend = extend

    orig_fe = B.GDNAttnBackend.forward_extend

    def forward_extend(self, *args, **kwargs):
        fm = getattr(self, "forward_metadata", None)
        need = False
        try:
            if fm is not None and getattr(fm, "has_mamba_track_mask", False):
                src = getattr(fm, "track_ssm_h_src", None)
                need = src is None or src.numel() > 0
        except Exception:
            need = True
        prev = getattr(_tls, "need_h", False)
        _tls.need_h = need
        try:
            return orig_fe(self, *args, **kwargs)
        finally:
            _tls.need_h = prev

    forward_extend.__wrapped__ = orig_fe
    B.GDNAttnBackend.forward_extend = forward_extend
    _installed = True
    _log(f"installed: TritonGDNKernel.extend -> SYCL ({gdn_sycl.lib_path()}, cfg {gdn_sycl.CFG}, min_t {MIN_T}, "
         f"check {_state['check']})")
    return True


if ENABLED and os.environ.get("EXL3_GDN_SYCL_AUTOINSTALL", "0") == "1":  # pragma: no cover
    install()
