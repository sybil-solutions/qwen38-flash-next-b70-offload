"""N114 GDN decode patch for SGLang 0.5.20 on Intel XPU: the GatedDeltaNet decode tail -> 2 fused SYCL kernels
(gdn_dec.py, Strata fused_gdn_* port, MIT). Env-guarded, default off.

    EXL3_GDN_DEC_SYCL=1      master switch (default 0: nothing is patched)
    EXL3_GDN_DEC_BA=1        also fold the in_proj_ba projection into the recurrence kernel when in_proj_ba is a plain
                             unquantized bf16 Linear (default 0)
    EXL3_GDN_DEC_CFG         rg,grf launch config (gdn_dec.py; default 4,128)
    EXL3_GDN_DEC_CHECK=N     bring-up: for the first N eager (non-captured) fused calls also run SGLang's own kernels
                             (causal_conv1d_update + packed decode + RMSNormGated) on compact copies of the used state
                             slots and log max |diff| of y, conv state and ssm state (syncs; not during graph capture)
    EXL3_GDN_DEC_DIR         directory of this file (plugin hook; default /n114)

What is replaced, decode batches only (forward_mode.is_decode(); target-verify / extend / idle keep the original):
Qwen3_5GatedDeltaNet.forward's  split -> b/a .contiguous() -> cat(q,k,v) -> RadixLinearAttention -> GDNAttnBackend
.forward_decode (causal_conv1d_update + Triton packed decode) -> z reshape -> RMSNormGated  becomes
conv kernel + recurrence/norm kernel, then the unchanged out_proj.  Wiring (no SGLang source edits):
  * the patched model forward computes in_proj_qkvz / in_proj_ba exactly as before, passes the TRUE q|k|v, a and b as
    strided views through the normal self.attn(...) call (so every layer of the attention-backend stack sees the
    same semantics), and leaves a thread-local context;
  * the patched GDNAttnBackend.forward_decode finds that context, reads the layer's conv / ssm cache and slot indices
    exactly as the original, runs the fused kernels (the gated norm included), calls the original
    _track_mamba_state_decode and returns y; the model forward then skips its own norm and runs out_proj;
  * if the backend cannot take the fused path (replay-SSM ring, non-Triton decode kernel, unexpected dtype / layout,
    a kernel error) it calls the original forward_decode with contiguous copies (= the original inputs bit for bit)
    and the model forward applies the original norm tail: exact fallback.
Installed only when the installed SGLang sources contain the anchor lines this wiring depends on (else a warning and
nothing patched).  A kernel exception disables the fused path for the process (one warning).  STATS counts calls.

Use: exl3xpu plugin hook in sglang_plugin.activate():
    if os.environ.get("EXL3_GDN_DEC_SYCL", "0") == "1":
        import sys; sys.path.insert(0, os.environ.get("EXL3_GDN_DEC_DIR", "/n114"))
        import patch_gdn_dec; patch_gdn_dec.install()
"""
from __future__ import annotations

import inspect
import os
import sys
import threading
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("EXL3_GDN_DEC_SYCL", "0") == "1"
BA_FUSE = os.environ.get("EXL3_GDN_DEC_BA", "0") == "1"
STATS: Counter = Counter()
_state = {"broken": False, "check": int(os.environ.get("EXL3_GDN_DEC_CHECK", "0")), "first": True,
          "logged": set(), "track_nargs": None}
_tls = threading.local()
_installed = False

MODEL_ANCHORS = ("def fix_query_key_value_ordering", "self.norm(core_attn_out, z)", "self.out_proj(core_attn_out)",
                 "self._forward_input_proj(", "mixed_qkv = torch.cat((query, key, value), dim=-1)")
BACKEND_ANCHORS = ("mamba2_layer_cache", "mamba_cache_indices", "causal_conv1d_update", "packed_decode",
                   "_track_mamba_state_decode")


def _log(msg: str) -> None:
    print(f"N114GDN {msg}", file=sys.stderr, flush=True)


def _note(reason: str) -> None:
    STATS["orig:" + reason] += 1
    if reason not in _state["logged"]:
        _state["logged"].add(reason)
        _log(f"original GDN decode kept ({reason})")


def _capturing() -> bool:
    import torch
    f = getattr(torch.xpu, "is_current_stream_capturing", None) if hasattr(torch, "xpu") else None
    try:
        return bool(f()) if f is not None else False
    except Exception:
        return False


def _model_reason(self, hidden_states, forward_batch, M):
    import torch
    if _state["broken"]:
        return "broken"
    fm = getattr(forward_batch, "forward_mode", None)
    if fm is None or not fm.is_decode():
        return "not decode"
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.device.type != "xpu" \
            or hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        return "input"
    try:
        from sglang.srt.runtime_context import get_exec
        if getattr(M, "_is_xpu", True) and get_exec().mamba.linear_attn_backend == "intel_xpu":
            return "intel_xpu backend"
    except Exception:
        pass
    if getattr(M, "_gdn_decode_fused_proj_conv", False):
        return "fused proj conv (cuda)"
    ratios = getattr(M, "_GDN_FUSED_QKVZBA_RATIOS", None)
    if ratios is None or (self.num_v_heads // self.num_k_heads) in ratios:
        return "head ratio uses the fused split layout"
    if getattr(self, "head_k_dim", 0) != 128 or getattr(self, "head_v_dim", 0) != 128:
        return "head dims"
    n = getattr(self, "norm", None)
    if n is None or getattr(n, "group_size", None) is not None or not getattr(n, "norm_before_gate", True) \
            or getattr(n, "bias", None) is not None or getattr(n, "activation", "swish") not in ("sigmoid", "swish", "silu"):
        return "norm config"
    return None


def _ba_weight(self):
    """in_proj_ba weight when the projection can run inside the recurrence kernel, else None."""
    import torch
    m = getattr(self, "in_proj_ba", None)
    if m is None or getattr(m, "bias", None) is not None:
        return None
    qm = getattr(m, "quant_method", None)
    if qm is not None and type(qm).__name__ != "UnquantizedLinearMethod":
        return None
    w = getattr(m, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.ndim != 2 or not w.is_contiguous():
        return None
    return w


def install() -> bool:
    global _installed
    if not ENABLED or _installed:
        return _installed
    import gdn_dec
    gdn_dec_mod = gdn_dec
    if not gdn_dec.load():
        _log(f"WARNING library not loadable ({gdn_dec.error()}); GDN decode unchanged")
        return False
    import torch
    from sglang.srt.models import qwen3_5 as M
    from sglang.srt.layers.attention.linear import gdn_backend as B

    cls = M.Qwen3_5GatedDeltaNet
    try:
        msrc = inspect.getsource(cls)
        bsrc = inspect.getsource(B.GDNAttnBackend.forward_decode)
    except Exception as e:
        _log(f"WARNING cannot read SGLang sources ({e}); not installed")
        return False
    miss = [a for a in MODEL_ANCHORS if a not in msrc] + [a for a in BACKEND_ANCHORS if a not in bsrc]
    if miss:
        _log(f"WARNING SGLang source anchors missing {miss}; not installed")
        return False
    try:
        _state["track_nargs"] = len(inspect.signature(B.GDNAttnBackend._track_mamba_state_decode).parameters)
    except Exception:
        _state["track_nargs"] = None

    orig_forward = cls.forward
    orig_fd = B.GDNAttnBackend.forward_decode

    # ------------------------------------------------------------------------------------------- model forward
    def forward(self, hidden_states, forward_batch, *args, **kwargs):
        reason = None if not (args or kwargs) else "extra args"
        if reason is None:
            reason = _model_reason(self, hidden_states, forward_batch, M)
        if reason is not None:
            _note(reason)
            return orig_forward(self, hidden_states, forward_batch, *args, **kwargs)
        tp = getattr(self, "attn_tp_size", 1) or 1
        k_tp, v_tp, nv_tp = self.key_dim // tp, self.value_dim // tp, self.num_v_heads // tp
        w_ba = _ba_weight(self) if BA_FUSE else None
        if w_ba is not None and tuple(w_ba.shape) == (2 * nv_tp, hidden_states.shape[1]):
            qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba = None
        else:
            w_ba = None
            qkvz, ba = self._forward_input_proj(hidden_states)
            if not isinstance(ba, torch.Tensor):
                _note("input proj tuple")
                return orig_forward(self, hidden_states, forward_batch)
        if not isinstance(qkvz, torch.Tensor) or qkvz.ndim != 2 or qkvz.shape[1] != 2 * k_tp + 2 * v_tp \
                or qkvz.stride(1) != 1:
            _note("qkvz layout")
            return orig_forward(self, hidden_states, forward_batch)
        if ba is None:
            ba_view = None
            b_view = a_view = torch.empty((qkvz.shape[0], nv_tp), dtype=qkvz.dtype, device=qkvz.device)  # placeholder
        else:
            ba_view = ba
            b_view, a_view = ba[:, :nv_tp], ba[:, nv_tp:2 * nv_tp]
        mixed = qkvz[:, :2 * k_tp + v_tp]
        ctx = {"gdn": self, "qkvz": qkvz, "ba": ba_view, "x": hidden_states, "w_ba": w_ba, "y": None,
               "nv": nv_tp, "nk": k_tp // self.head_k_dim}
        prev = getattr(_tls, "ctx", None)
        _tls.ctx = ctx
        try:
            res = self.attn(forward_batch, mixed_qkv=mixed, a=a_view, b=b_view)
        finally:
            _tls.ctx = prev
        if ctx["y"] is not None:
            core = ctx["y"]
        else:   # backend took the original kernels: the original norm tail
            z = qkvz[:, 2 * k_tp + v_tp:].reshape(qkvz.shape[0], -1, self.head_v_dim)
            core_attn_out = res[0] if isinstance(res, tuple) else res
            z_shape_og = z.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            if core_attn_out.shape != z.shape:
                pad = z.new_zeros(z.shape)
                pad[: core_attn_out.shape[0], :] = core_attn_out
                core_attn_out = pad
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output, _ = self.out_proj(core)
        return output

    forward.__wrapped__ = orig_forward

    # ------------------------------------------------------------------------------------------- backend
    def _orig_call(self, ctx, args, kwargs):
        """original forward_decode on contiguous copies of the true inputs (bit-identical to the unpatched path)."""
        gdn = ctx["gdn"]
        qkvz = ctx["qkvz"]
        tp = getattr(gdn, "attn_tp_size", 1) or 1
        k_tp, v_tp, nv = gdn.key_dim // tp, gdn.value_dim // tp, ctx["nv"]
        ba = ctx["ba"]
        if ba is None:
            ba, _ = gdn.in_proj_ba(ctx["x"])
        repl = {"mixed_qkv": qkvz[:, :2 * k_tp + v_tp].contiguous(), "a": ba[:, nv:2 * nv].contiguous(),
                "b": ba[:, :nv].contiguous()}
        names = ["layer", "forward_batch", "mixed_qkv", "a", "b"]
        args = list(args)
        for i, nm in enumerate(names):
            if nm in repl and i < len(args):
                args[i] = repl.pop(nm)
        kwargs = dict(kwargs)
        kwargs.update(repl)
        return orig_fd(self, *args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        ctx = getattr(_tls, "ctx", None)
        if ctx is None:
            return orig_fd(self, *args, **kwargs)
        layer = kwargs.get("layer", args[0] if len(args) > 0 else None)
        forward_batch = kwargs.get("forward_batch", args[1] if len(args) > 1 else None)
        gdn = ctx["gdn"]
        reason = None
        if layer is None or layer is not getattr(gdn, "attn", None):
            reason = "layer mismatch"
        fm = getattr(self, "forward_metadata", None)
        if reason is None and fm is None:
            reason = "no metadata"
        if reason is None:
            for nm in ("replayssm_write_pos", "replayssm_force_flush"):
                if getattr(fm, nm, None) is not None:
                    reason = "replayssm"
        kd = getattr(self, "kernel_dispatcher", None)
        if reason is None and (kd is None or not getattr(kd, "supports_packed_decode", False)
                               or type(getattr(kd, "decode_kernel", None)).__name__ != "TritonGDNKernel"):
            reason = "decode kernel"
        if reason is None:
            try:
                cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
                conv_states = cache.conv[0]
                ssm_states = cache.temporal
                idx = fm.mamba_cache_indices
                if getattr(cache, "replayssm_d", None) is not None:
                    reason = "replayssm cache"
            except Exception as e:
                reason = f"cache ({type(e).__name__})"
        if reason is None and (getattr(layer, "bias", None) is not None
                               or getattr(layer, "activation", "silu") not in ("silu", "swish")):
            reason = "conv bias/activation"
        if reason is None:
            H, Hk = int(layer.num_v_heads), int(layer.num_k_heads)
            norm = gdn.norm
            act = getattr(norm, "activation", "swish")
            reason = gdn_dec_mod.why_not(ctx["qkvz"], ctx["ba"], conv_states, layer.conv_weights, ssm_states, idx,
                                         H=H, Hk=Hk, K=int(layer.head_k_dim), V=int(layer.head_v_dim), act=act,
                                         norm_w=norm.weight, x=ctx["x"], w_ba=ctx["w_ba"])
            if reason is None and (layer.q_dim != Hk * 128 or layer.k_dim != Hk * 128 or layer.v_dim != H * 128):
                reason = "q/k/v dims"
        if reason is not None:
            _note(reason)
            return _orig_call(self, ctx, args, kwargs)
        chk = None
        if _state["check"] > 0 and not _capturing():
            chk = _check_prepare(ctx, layer, conv_states, ssm_states, idx)
        try:
            y = gdn_dec_mod.gdn_decode(ctx["qkvz"], ctx["ba"], conv_states, layer.conv_weights, ssm_states, idx,
                                       layer.A_log, layer.dt_bias, norm.weight, norm.eps, H=H, Hk=Hk,
                                       scale=float(layer.head_k_dim) ** -0.5, act=act, x=ctx["x"], w_ba=ctx["w_ba"])
        except Exception as e:
            _state["broken"] = True
            STATS["error"] += 1
            _log(f"WARNING fused GDN decode disabled after error, original kernels from now on: {type(e).__name__}: {e}")
            return _orig_call(self, ctx, args, kwargs)
        STATS["sycl"] += 1
        if _state["first"]:
            _state["first"] = False
            _log(f"first fused decode: layer {layer.layer_id} B={ctx['qkvz'].shape[0]} qkvz{tuple(ctx['qkvz'].shape)} "
                 f"stride{tuple(ctx['qkvz'].stride())} conv_state{tuple(conv_states.shape)} stride{tuple(conv_states.stride())} "
                 f"ssm{tuple(ssm_states.shape)} {ssm_states.dtype} idx {idx.dtype} act {act} ba_fused {ctx['w_ba'] is not None} "
                 f"cfg {gdn_dec_mod.CFG} capturing {_capturing()}")
        if chk is not None:
            _check_finish(chk, y, conv_states, ssm_states)
        nargs = _state["track_nargs"]
        if nargs is not None and nargs >= 6:
            self._track_mamba_state_decode(forward_batch, conv_states, ssm_states, idx, layer.layer_id)
        else:
            self._track_mamba_state_decode(forward_batch, conv_states, ssm_states, idx)
        ctx["y"] = y
        return y

    forward_decode.__wrapped__ = orig_fd

    # ------------------------------------------------------------------------------------------- bring-up check
    def _check_prepare(ctx, layer, conv_states, ssm_states, idx):
        """compact copies of the used slots + the original kernels' result on them (before the fused call)."""
        _state["check"] -= 1
        try:
            gdn = ctx["gdn"]
            valid = idx >= 0
            sel = idx[valid].long()
            remap = torch.full_like(idx, -1)
            remap[valid] = torch.arange(int(sel.numel()), device=idx.device, dtype=idx.dtype)
            conv_c = conv_states.index_select(0, sel).clone()
            ssm_c = ssm_states.index_select(0, sel).clone()
            tp = getattr(gdn, "attn_tp_size", 1) or 1
            k_tp, v_tp, nv = gdn.key_dim // tp, gdn.value_dim // tp, ctx["nv"]
            qkvz = ctx["qkvz"]
            ba = ctx["ba"]
            if ba is None:
                ba, _ = gdn.in_proj_ba(ctx["x"])
            mixed = qkvz[:, :2 * k_tp + v_tp].contiguous()
            a, b = ba[:, nv:2 * nv].contiguous(), ba[:, :nv].contiguous()
            mixed = B.causal_conv1d_update(mixed, conv_c, layer.conv_weights, layer.bias, layer.activation,
                                           conv_state_indices=remap)
            o = self.kernel_dispatcher.packed_decode(mixed_qkv=mixed, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
                                                     scale=layer.head_k_dim ** -0.5, ssm_states=ssm_c,
                                                     cache_indices=remap, num_v_heads=layer.num_v_heads,
                                                     head_v_dim=layer.head_v_dim)
            z = qkvz[:, 2 * k_tp + v_tp:].reshape(-1, gdn.head_v_dim)
            yref = gdn.norm(o.reshape(-1, o.shape[-1]), z).reshape(qkvz.shape[0], -1)
            return dict(sel=sel, valid=valid, conv=conv_c, ssm=ssm_c, y=yref)
        except Exception as e:  # pragma: no cover
            _log(f"check prepare failed: {type(e).__name__}: {e}")
            return None

    def _check_finish(chk, y, conv_states, ssm_states):
        try:
            v = chk["valid"]
            dy = (y[v].float() - chk["y"][v].float()).abs()
            dc = (conv_states.index_select(0, chk["sel"]).float() - chk["conv"].float()).abs()
            ds = (ssm_states.index_select(0, chk["sel"]).float() - chk["ssm"].float()).abs()
            eq = float((y[v] == chk["y"][v]).float().mean()) if dy.numel() else 1.0
            _log(f"check B={y.shape[0]} y max_abs={float(dy.max()) if dy.numel() else 0:.3e} y_bitEq={eq:.4f} "
                 f"y_ref_absmax={float(chk['y'][v].float().abs().max()) if dy.numel() else 0:.3e} "
                 f"conv_state max_abs={float(dc.max()) if dc.numel() else 0:.3e} "
                 f"ssm max_abs={float(ds.max()) if ds.numel() else 0:.3e} "
                 f"ssm_ref_absmax={float(chk['ssm'].float().abs().max()) if ds.numel() else 0:.3e}")
        except Exception as e:  # pragma: no cover
            _log(f"check failed: {type(e).__name__}: {e}")

    cls.forward = forward
    B.GDNAttnBackend.forward_decode = forward_decode
    _installed = True
    _log(f"installed: Qwen3_5GatedDeltaNet decode -> n114gdn conv + rec_norm ({gdn_dec.lib_path()}, cfg {gdn_dec.CFG}, "
         f"round_prod {gdn_dec.ROUND_PROD}, ba_fuse {BA_FUSE}, check {_state['check']})")
    return True


if ENABLED and os.environ.get("EXL3_GDN_DEC_AUTOINSTALL", "0") == "1":  # pragma: no cover
    install()
