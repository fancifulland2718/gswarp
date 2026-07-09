"""Backend selection for gswarp methods."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from gswarp._internal.methods.spec import MethodSpec

_STABLE_BACKENDS = {
    "warp_3dgs": "gswarp._internal.backends.warp.backend_3dgs",
    "warp_3dgs_flow": "gswarp._internal.backends.warp.backend_3dgs_flow",
}


def _load_stable_backend(method: MethodSpec) -> ModuleType:
    try:
        module_name = _STABLE_BACKENDS[method.backend_family]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported backend family: {method.backend_family!r}") from exc
    return importlib.import_module(module_name)


def resolve_backend(method: MethodSpec) -> ModuleType:
    mode = os.environ.get("GSWARP_WARP_BACKEND", "auto").strip().lower()
    if mode not in {"auto", "stable", "advanced"}:
        raise RuntimeError("GSWARP_WARP_BACKEND must be one of 'auto', 'stable', or 'advanced'")

    if mode == "stable":
        return _load_stable_backend(method)

    from gswarp._internal.backends.warp.advanced import try_resolve_advanced_backend

    advanced = try_resolve_advanced_backend(method)
    if advanced is not None:
        return advanced
    if mode == "advanced":
        raise RuntimeError(f"Advanced Warp backend unavailable for method {method.name!r}")
    return _load_stable_backend(method)


__all__ = ["resolve_backend"]
