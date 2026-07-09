"""Lazy advanced backend resolution."""

from __future__ import annotations

import importlib

_ADVANCED_MODULES = {
    "warp_3dgs": "gswarp._internal.backends.warp.backend_3dgs_advanced",
    "warp_3dgs_flow": "gswarp._internal.backends.warp.backend_3dgs_flow_advanced",
}


def try_resolve_advanced_backend(method):
    module_name = _ADVANCED_MODULES.get(method.backend_family)
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    if not getattr(module, "BACKEND_AVAILABLE", False):
        return None
    return getattr(module, "BACKEND", None) or module


__all__ = ["try_resolve_advanced_backend"]
