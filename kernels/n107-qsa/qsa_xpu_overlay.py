"""Mounted over exl3xpu/qsa_xpu.py (read-only bind mount); the image's original file is mounted next to it as
exl3xpu/qsa_xpu_base.py (runs/N104-nvtier/src/exl3xpu/qsa_xpu.py is byte-identical to the image's copy).

Behaviour with EXL3_QSA_TRITON unset/0 is exactly the original module: same functions, same install().
With EXL3_QSA_TRITON=1 or EXL3_QSA_IMPL=sycl_row (N108), install() additionally runs the N107 patch (patch_qsa.py from EXL3_N107_QSA_DIR,
default /n107qsa) after the original install, so it wraps what exl3xpu installed.
"""
from __future__ import annotations

import os
import sys
import types

from . import qsa_xpu_base as _base
from .qsa_xpu_base import (  # noqa: F401  (public names of the original module)
    qsa_fast_topk,
    qsa_sparse_attention,
    qsa_sparse_attention_reference,
    qsa_sparse_attention_union,
)


class _Forward(types.ModuleType):
    """Attribute writes to the original's public functions also land in qsa_xpu_base, where they are looked up
    (keeps e.g. the N106 profiler's wrap of qsa_sparse_attention_union working through the overlay)."""

    def __setattr__(self, name, value):
        if name in ("qsa_fast_topk", "qsa_sparse_attention", "qsa_sparse_attention_reference",
                    "qsa_sparse_attention_union"):
            setattr(_base, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Forward


def install() -> None:
    _base.install()
    if (os.environ.get("EXL3_QSA_TRITON", "0") != "1" and not os.environ.get("EXL3_QSA_IMPL")
            and not os.environ.get("EXL3_QSA_SELECT")):
        return
    d = os.environ.get("EXL3_N107_QSA_DIR", "/n107qsa")
    try:
        if d not in sys.path:
            sys.path.insert(0, d)
        import patch_qsa
        patch_qsa.install()
    except Exception as e:  # pragma: no cover
        print(f"N107QSA overlay: patch not installed ({e!r})", file=sys.stderr, flush=True)
