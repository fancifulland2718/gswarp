"""Lazy advanced backend resolution."""

from __future__ import annotations

import importlib
from typing import Any

from .capabilities import WarpFeatureSet, detect_warp_features

_ADVANCED_MODULES = {
    "warp_3dgs": "gswarp._internal.backends.warp.backend_3dgs_advanced",
    "warp_3dgs_flow": "gswarp._internal.backends.warp.backend_3dgs_flow_advanced",
}


def validate_advanced_backend_module(
    module,
    features: WarpFeatureSet | None = None,
) -> tuple[Any | None, str | None]:
    if not getattr(module, "BACKEND_AVAILABLE", False):
        return None, "backend is not implemented"

    features = detect_warp_features() if features is None else features
    minimum = tuple(getattr(module, "MIN_WARP_VERSION", (0, 0, 0)))
    if features.version < minimum:
        return None, (
            f"requires warp-lang >= {'.'.join(map(str, minimum))}; "
            f"detected {'.'.join(map(str, features.version))}"
        )

    required = frozenset(getattr(module, "REQUIRED_WARP_CAPABILITIES", ()))
    missing = features.missing(required)
    if missing:
        return None, f"missing Warp capabilities: {sorted(missing)!r}"

    backend = getattr(module, "BACKEND", None) or module
    return backend, None


def resolve_advanced_backend(method) -> tuple[Any | None, str | None]:
    module_name = _ADVANCED_MODULES.get(method.backend_family)
    if module_name is None:
        return None, f"no advanced backend is registered for {method.backend_family!r}"
    module = importlib.import_module(module_name)
    return validate_advanced_backend_module(module)


def try_resolve_advanced_backend(method):
    backend, _reason = resolve_advanced_backend(method)
    return backend


__all__ = [
    "resolve_advanced_backend",
    "try_resolve_advanced_backend",
    "validate_advanced_backend_module",
]
