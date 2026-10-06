"""n107-hc monkeypatch: route SGLang's hyper-connection fallbacks to the Triton kernels in hc_xpu.py for XPU tensors.

Off unless EXL3_HC_XPU=1. CUDA paths are untouched: the patched GroupedGemmaRMSNorm.forward and the per-instance
GatedResidual._mix_compute / _combine_compute only take the Triton path for XPU tensors whose shapes/dtypes/layout
hc_xpu.*_supported() accepts; everything else (CUDA, CPU, fp32, non-contiguous, empty) calls the original code. In
GatedResidual.mix/combine the CUDA JIT branches come first and are unchanged; the patched callables only replace what
runs in their final `else` (today: torch.compile'd _mix_compute / _combine_compute, which are then never compiled on XPU).

Use (pick one):
  * exl3xpu plugin hook (preferred): in sglang_plugin.activate():
        if os.environ.get("EXL3_HC_XPU", "0") == "1":
            import sys; sys.path.insert(0, os.environ.get("EXL3_HC_DIR", "/hc"))
            import patch_hc; patch_hc.install(import_now=True)
  * plain import before the model is built (sitecustomize, a launcher wrapper, ...): `import patch_hc` -- with
    EXL3_HC_XPU=1 it patches immediately if sglang.srt.layers.hyperconnection is already imported, otherwise it
    installs an import hook that patches the module right after it is first imported.
  * already-built models: patch_hc.patch_model(model) re-wires existing GatedResidual instances.

Env:
  EXL3_HC_XPU=1            master switch (default 0)
  EXL3_HC_NORM=1           Triton grouped Gemma RMSNorm (default 1)
  EXL3_HC_MIX=epilogue     epilogue | fused | fused_silu | off   (default epilogue: GEMMs stay in oneDNN)
  EXL3_HC_COMBINE=torch    torch | fused | off                   (default torch: 10240->hc GEMM stays in oneDNN)
  EXL3_HC_ROUND_G=1        round g = 2*sigmoid(l/hc) to bf16 like inductor does (default 1)
  EXL3_HC_VERBOSE=1        print the first use of each kernel (default 1) ; patch_hc.STATS counts calls per path
  launch configs: see hc_xpu._env_cfg (EXL3_HC_NORM_CFG, EXL3_HC_EPI_CFG, EXL3_HC_UP_CFG, EXL3_HC_COMB_CFG,
  EXL3_HC_COMBF_CFG, EXL3_HC_TPW, EXL3_HC_UP_BLOCKPTR)
A kernel that raises (compile error, unsupported backend feature) is disabled for the rest of the process with one
warning and the original path is used from then on.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

TARGET = "sglang.srt.layers.hyperconnection"
ENABLED = os.environ.get("EXL3_HC_XPU", "0") == "1"
MIX_MODE = os.environ.get("EXL3_HC_MIX", "epilogue")
COMB_MODE = os.environ.get("EXL3_HC_COMBINE", "torch")
ROUND_G = os.environ.get("EXL3_HC_ROUND_G", "1") == "1"
VERBOSE = os.environ.get("EXL3_HC_VERBOSE", "1") == "1"

STATS: Counter = Counter()
_state = {
    "norm": os.environ.get("EXL3_HC_NORM", "1") == "1",
    "mix": MIX_MODE != "off",
    "combine": COMB_MODE != "off",
}
_seen = set()
_applied = False
_hc = None


def _log(msg: str) -> None:
    print(f"n107-hc: {msg}", file=sys.stderr, flush=True)


def _kernels():
    global _hc
    if _hc is None:
        import hc_xpu
        _hc = hc_xpu
    return _hc


def _first(kind: str, shape) -> None:
    if VERBOSE and kind not in _seen:
        _seen.add(kind)
        _log(f"{kind} on XPU via Triton, first shape {tuple(shape)}")


def _disable(kind: str, err: BaseException) -> None:
    _state[kind] = False
    _log(f"WARNING {kind} kernel disabled after error, original path from now on: {type(err).__name__}: {err}")


# ---------------------------------------------------------------------------------------------------------------------

def _make_norm_forward(orig):
    def forward(self, x):
        if _state["norm"]:
            hc = _kernels()
            if hc.norm_supported(x, self.weight, self.group_size):
                try:
                    y = hc.gemma_rmsnorm_grouped(x, self.weight, self.group_size, self.variance_epsilon)
                    STATS["norm_xpu"] += 1
                    _first("norm", x.shape)
                    return y
                except Exception as e:  # pragma: no cover - device specific
                    _disable("norm", e)
        STATS["norm_orig"] += 1
        return orig(self, x)

    forward._n107 = True
    forward._orig = orig
    return forward


def _wrap_mix(orig):
    if getattr(orig, "_n107", False):
        return orig

    def _mix_compute(hyper_input_normed, input_mix_weight_down, input_mix_weight_up, hc, hs):
        if _state["mix"]:
            k = _kernels()
            if k.mix_supported(hyper_input_normed, input_mix_weight_down, input_mix_weight_up, hc, hs):
                try:
                    o = k.hc_mix(hyper_input_normed, input_mix_weight_down, input_mix_weight_up, hc, hs,
                                 mode=MIX_MODE)
                    STATS["mix_xpu"] += 1
                    _first(f"mix[{MIX_MODE}]", hyper_input_normed.shape)
                    return o
                except Exception as e:  # pragma: no cover
                    _disable("mix", e)
        STATS["mix_orig"] += 1
        return orig(hyper_input_normed, input_mix_weight_down, input_mix_weight_up, hc, hs)

    _mix_compute._n107 = True
    _mix_compute._orig = orig
    return _mix_compute


def _wrap_combine(orig):
    if getattr(orig, "_n107", False):
        return orig

    def _combine_compute(block_output, residual, normed_residual, block_inject_weight, hc, hs):
        if _state["combine"]:
            k = _kernels()
            if k.combine_supported(block_output, residual, normed_residual, block_inject_weight, hc, hs):
                try:
                    o = k.hc_combine(block_output, residual, normed_residual, block_inject_weight, hc, hs,
                                     inject=COMB_MODE, round_g=ROUND_G)
                    STATS["combine_xpu"] += 1
                    _first(f"combine[{COMB_MODE}]", residual.shape)
                    return o
                except Exception as e:  # pragma: no cover
                    _disable("combine", e)
        STATS["combine_orig"] += 1
        return orig(block_output, residual, normed_residual, block_inject_weight, hc, hs)

    _combine_compute._n107 = True
    _combine_compute._orig = orig
    return _combine_compute


def _patch_instance(obj) -> None:
    f = getattr(obj, "_mix_compute", None)
    if f is not None:
        obj._mix_compute = _wrap_mix(f)
    f = getattr(obj, "_combine_compute", None)
    if f is not None:
        obj._combine_compute = _wrap_combine(f)


def _apply(mod) -> None:
    global _applied
    if _applied or getattr(mod, "_n107_hc", False):
        _applied = True
        return
    Norm = mod.GroupedGemmaRMSNorm
    if not getattr(Norm.forward, "_n107", False):
        Norm.forward = _make_norm_forward(Norm.forward)
    GR = mod.GatedResidual
    orig_init = GR.__init__
    if not getattr(orig_init, "_n107", False):
        def __init__(self, *a, **k):
            orig_init(self, *a, **k)
            _patch_instance(self)
        __init__._n107 = True
        __init__._orig = orig_init
        GR.__init__ = __init__
    mod._n107_hc = True
    _applied = True
    try:
        have = _kernels().HAVE_TRITON
    except Exception as e:  # pragma: no cover
        have = f"no ({e})"
    _log(f"installed (norm={_state['norm']}, mix={MIX_MODE}, combine={COMB_MODE}, round_g={ROUND_G}, "
         f"triton={have})")


def patch_model(model) -> int:
    """Re-wire GatedResidual instances that were built before install(). Returns how many were patched."""
    mod = sys.modules.get(TARGET)
    if mod is None:
        return 0
    n = 0
    for m in model.modules():
        if isinstance(m, mod.GatedResidual):
            _patch_instance(m)
            n += 1
    return n


def unpatch() -> None:
    """Restore the class methods (instances already built keep their wrappers; set EXL3_HC_* off to bypass them)."""
    global _applied
    mod = sys.modules.get(TARGET)
    if mod is None:
        return
    f = mod.GroupedGemmaRMSNorm.forward
    if getattr(f, "_n107", False):
        mod.GroupedGemmaRMSNorm.forward = f._orig
    f = mod.GatedResidual.__init__
    if getattr(f, "_n107", False):
        mod.GatedResidual.__init__ = f._orig
    mod._n107_hc = False
    _applied = False


# ---------------------------------------------------------------------------------------------------------------------
# post-import hook (for use before sglang is imported)

class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _apply(module)
        except Exception as e:  # pragma: no cover
            _log(f"WARNING patch failed: {type(e).__name__}: {e}")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _Loader(spec.loader)
                sys.meta_path[:] = [x for x in sys.meta_path if x is not self]
                return spec
        return None


def install(import_now: bool = False, force: bool = False) -> bool:
    """Patch (or arm the import hook). Returns True if the patch is active or armed."""
    if not (ENABLED or force):
        return False
    mod = sys.modules.get(TARGET)
    if mod is None and import_now:
        import importlib
        mod = importlib.import_module(TARGET)
    if mod is not None:
        _apply(mod)
        return True
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


if ENABLED:
    install()
