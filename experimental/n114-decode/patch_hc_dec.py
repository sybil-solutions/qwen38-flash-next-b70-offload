"""N114-hc patch for SGLang (image 24c872759256) on Intel XPU: GatedResidual.mix / combine of decode-sized batches
(1 <= M <= EXL3_HC_DEC_MAX_M rows, XPU, bf16, hc_per_branch_norm) -> the SYCL kernels in hc_dec.py (Strata
fused_gr_read port, MIT). Env-guarded, default off: nothing is patched unless EXL3_HC_DEC_SYCL=1.

    EXL3_HC_DEC_SYCL=1      master switch (default 0)
    EXL3_HC_DEC_FOLD=1      also fold the attention half's combine into the MLP half's mix inside
                            Qwen4ExpLayerExtensionMixin._prepare_qwen4_exp_mlp (one kernel less per layer; only if the
                            method's source has the expected anchors, else not patched) (default 0)
    EXL3_HC_DEC_MAX_M       largest row count routed to the kernels (default 8, max 8)
    EXL3_HC_DEC_CFG         ks,uc,mode launch config (hc_dec.py)
    EXL3_HC_DEC_DIR         directory of this file (plugin hook; default /n114)

Routing: mix() of an eligible call returns (mixed, res) where res is a 2-tuple subclass (hyper_input, normed) that
also carries l = bf16(normed . Wi) (computed by the same launch) and the owning module; combine() uses the SYCL
combine only for such a res from the same module, so a residual from the original path (or another module) always
gets the original combine. Everything else (CUDA, CPU, prefill-sized batches, other layouts / dtypes, M == 0) calls
the callables that were installed when install() ran - e.g. n107-hc's Triton path when EXL3_HC_XPU=1. A kernel error
disables n114-hc for the rest of the process (one warning). No host syncs: safe inside XPU graph capture.
STATS counts calls per path.

Use: exl3xpu plugin hook in sglang_plugin.activate() (after the n107-hc hook):
    if os.environ.get("EXL3_HC_DEC_SYCL", "0") == "1":
        import sys; sys.path.insert(0, os.environ.get("EXL3_HC_DEC_DIR", "/n114"))
        import patch_hc_dec; patch_hc_dec.install()
"""
from __future__ import annotations

import inspect
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("EXL3_HC_DEC_SYCL", "0") == "1"
FOLD = os.environ.get("EXL3_HC_DEC_FOLD", "0") == "1"
MAX_M = max(1, min(8, int(os.environ.get("EXL3_HC_DEC_MAX_M", "8"))))
TARGET = "sglang.srt.layers.hyperconnection"
FOLD_TARGET = "sglang.srt.models.qwen4_exp"
FOLD_ANCHORS = ("attn_tp_all_reduce(hidden_states)",
                "self.attn_hyper_connection.combine(hidden_states, residual)",
                "self.mlp_hyper_connection.mix(hidden_states)")

STATS: Counter = Counter()
_state = {"broken": False, "logged": set(), "installed": False, "fold": "off", "mods": {}}


def _log(msg: str) -> None:
    print(f"N114HC {msg}", file=sys.stderr, flush=True)


class HCRes(tuple):
    """(hyper_input, hyper_input_normed) plus the n114 extras (l, owner); unpacks like the original 2-tuple."""

    def __new__(cls, r, n, l, owner):
        self = super().__new__(cls, (r, n))
        self.l = l          # [M, hc] fp32 = bf16(n . Wi) from the same launch, or None (no inject weight)
        self.owner = owner  # the GatedResidual whose Wi produced l
        return self


def _hd():
    import hc_dec
    return hc_dec


def _disable(err: BaseException) -> None:
    _state["broken"] = True
    STATS["error"] += 1
    _log(f"WARNING n114-hc disabled after error, original path from now on: {type(err).__name__}: {err}")


def _mod_ok(gr) -> bool:
    key = id(gr)
    v = _state["mods"].get(key)
    if v is None:
        r = _hd().module_why_not(gr)
        v = r is None
        _state["mods"][key] = v
        if not v and r not in _state["logged"]:
            _state["logged"].add(r)
            _log(f"module kept on the original path ({r})")
    return v


def _eligible(gr, x) -> bool:
    if _state["broken"] or x.device.type != "xpu" or x.dim() != 2 or not (1 <= x.shape[0] <= MAX_M):
        return False
    if not _mod_ok(gr):
        return False
    return _hd().why_not(gr, x) is None


def _wrap_mix(orig):
    def mix(self, hyper_input):
        if _eligible(self, hyper_input):
            try:
                mixed, n, l = _hd().mix(self, hyper_input)
            except Exception as e:
                _disable(e)
            else:
                STATS["mix_sycl"] += 1
                if "first_mix" not in _state["logged"]:
                    _state["logged"].add("first_mix")
                    _log(f"first SYCL mix: x{tuple(hyper_input.shape)} cfg {_hd().CFG}")
                return mixed, HCRes(hyper_input, n, l if l.numel() else None, self)
        STATS["mix_orig"] += 1
        return orig(self, hyper_input)

    mix._n114 = True
    mix.__wrapped__ = orig
    return mix


def _wrap_combine(orig):
    def combine(self, block_output, residuals):
        if isinstance(residuals, HCRes) and not _state["broken"]:
            l, owner = residuals.l, residuals.owner
            r = residuals[0]
            hd = _hd()
            if (owner is self and l is not None and block_output.dim() == 2
                    and hd.combine_why_not(r, block_output, l, self.hc_count, self.hidden_size) is None):
                try:
                    out = hd.combine(r, block_output, l, self.hc_count)
                except Exception as ex:
                    _disable(ex)
                else:
                    STATS["combine_sycl"] += 1
                    return out
        STATS["combine_orig"] += 1
        return orig(self, block_output, residuals)

    combine._n114 = True
    combine.__wrapped__ = orig
    return combine


def _install_fold() -> None:
    if not FOLD or _state["fold"] != "off":
        return
    mod = sys.modules.get(FOLD_TARGET)
    if mod is None:
        return   # retried at the next mix() call (the model module is imported before any forward)
    cls = getattr(mod, "Qwen4ExpLayerExtensionMixin", None)
    orig = getattr(cls, "_prepare_qwen4_exp_mlp", None) if cls is not None else None
    if orig is None or getattr(orig, "_n114", False):
        _state["fold"] = "absent" if orig is None else "on"
        return
    try:
        src = inspect.getsource(orig)
    except Exception:
        src = ""
    missing = [a for a in FOLD_ANCHORS if a not in src]
    if missing or not hasattr(mod, "attn_tp_all_reduce"):
        _state["fold"] = "anchors"
        _log(f"fold NOT installed: _prepare_qwen4_exp_mlp differs from the expected source (missing {missing})")
        return
    all_reduce = mod.attn_tp_all_reduce

    def _prepare_qwen4_exp_mlp(self, hidden_states, residual, forward_batch, *args, **kwargs):
        if (isinstance(residual, HCRes) and not args and not kwargs and not _state["broken"]
                and not forward_batch.forward_mode.is_idle()):
            ahc, mhc = self.attn_hyper_connection, self.mlp_hyper_connection
            l, owner = residual.l, residual.owner
            r = residual[0]
            hd = _hd()
            if (owner is ahc and l is not None and _mod_ok(mhc) and ahc.hc_count == mhc.hc_count
                    and ahc.hidden_size == mhc.hidden_size and hidden_states.dim() == 2):
                bo = all_reduce(hidden_states)
                if hd.combine_why_not(r, bo, l, ahc.hc_count, ahc.hidden_size) is None:
                    try:
                        mixed, r_new, n, l_new = hd.combine_mix(mhc, r, bo, l)
                    except Exception as ex:
                        _disable(ex)
                    else:
                        STATS["fold_sycl"] += 1
                        return mixed, HCRes(r_new, n, l_new if l_new.numel() else None, mhc)
                # not foldable after the all-reduce: finish exactly like the original
                hidden_states = ahc.combine(bo, residual)
                STATS["fold_orig"] += 1
                return mhc.mix(hidden_states)
        STATS["fold_orig"] += 1
        return orig(self, hidden_states, residual, forward_batch, *args, **kwargs)

    _prepare_qwen4_exp_mlp._n114 = True
    _prepare_qwen4_exp_mlp.__wrapped__ = orig
    cls._prepare_qwen4_exp_mlp = _prepare_qwen4_exp_mlp
    _state["fold"] = "on"
    _log("fold installed: attn combine + mlp mix in one n114 launch sequence")


def install(import_now: bool = True) -> bool:
    if not ENABLED or _state["installed"]:
        return _state["installed"]
    hd = _hd()
    if not hd.load():
        _log(f"WARNING library not loadable ({hd.error()}); original hyper-connection path stays")
        return False
    import importlib
    H = sys.modules.get(TARGET) or (importlib.import_module(TARGET) if import_now else None)
    if H is None:
        return False
    GR = H.GatedResidual
    if not getattr(GR.mix, "_n114", False):
        GR.mix = _wrap_mix(GR.mix)
    if not getattr(GR.combine, "_n114", False):
        GR.combine = _wrap_combine(GR.combine)
    if FOLD:
        orig_mix = GR.mix

        def mix_then_fold(self, hyper_input, _m=orig_mix):
            if _state["fold"] == "off":
                _install_fold()
            return _m(self, hyper_input)

        mix_then_fold._n114 = True
        mix_then_fold.__wrapped__ = orig_mix
        GR.mix = mix_then_fold
        _install_fold()
    _state["installed"] = True
    _log(f"installed: GatedResidual.mix/combine -> SYCL for XPU rows <= {MAX_M} ({hd.lib_path()}, cfg {hd.CFG}, "
         f"fold {'requested' if FOLD else 'off'})")
    return True


def unpatch() -> None:
    H = sys.modules.get(TARGET)
    if H is not None:
        for name in ("mix", "combine"):
            f = getattr(H.GatedResidual, name)
            while getattr(f, "_n114", False):
                f = f.__wrapped__
            setattr(H.GatedResidual, name, f)
    M = sys.modules.get(FOLD_TARGET)
    cls = getattr(M, "Qwen4ExpLayerExtensionMixin", None) if M is not None else None
    if cls is not None and getattr(cls._prepare_qwen4_exp_mlp, "_n114", False):
        cls._prepare_qwen4_exp_mlp = cls._prepare_qwen4_exp_mlp.__wrapped__
    _state.update(installed=False, fold="off")


if ENABLED and os.environ.get("EXL3_HC_DEC_AUTOINSTALL", "0") == "1":  # pragma: no cover
    install()
