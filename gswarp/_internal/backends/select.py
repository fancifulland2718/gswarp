"""Method-plan resolution for stable and advanced Warp backends."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from gswarp._internal.api.outputs import pack_flow_outputs, pack_standard_outputs
from gswarp._internal.methods.registry import get_method
from gswarp._internal.methods.spec import MethodPlan, MethodSpec, MethodStages


_STABLE_BACKENDS = {
    "warp_3dgs": "gswarp._internal.backends.warp.backend_3dgs",
    "warp_3dgs_flow": "gswarp._internal.backends.warp.backend_3dgs_flow",
}

_STAGE_PROFILES = {
    "warp_3dgs": {
        "output_mode": "standard_meta",
        "primitive": "gaussian_3d",
        "projection": "perspective_3dgs",
        "filtering": "none",
        "appearance": "sh_or_rgb",
        "pre_adapter": None,
        "flow": False,
        "empty_forward_name": "empty_forward_stage",
        "preprocess_name": "preprocess_stage",
        "features_name": "feature_stage",
        "render_name": "render_stage",
        "build_state_name": "build_state_stage",
        "backward_name": "rasterize_gaussians_backward_typed",
    },
    "warp_3dgs_flow": {
        "output_mode": "flow_flat",
        "primitive": "gaussian_3d",
        "projection": "perspective_3dgs",
        "filtering": "none",
        "appearance": "sh_or_rgb",
        "pre_adapter": None,
        "flow": True,
        "empty_forward_name": "empty_forward_stage",
        "preprocess_name": "preprocess_stage",
        "features_name": "feature_stage",
        "render_name": "render_stage",
        "build_state_name": "build_state_stage",
        "backward_name": "rasterize_gaussians_flow_backward_typed",
    },
}


def _standard_output_adapter(outputs: tuple[Any, ...], meta_type: type[Any]):
    return pack_standard_outputs(outputs, meta_type)


def _flow_output_adapter(outputs: tuple[Any, ...], _meta_type: type[Any] | None = None):
    return pack_flow_outputs(outputs)


def _load_stable_backend(method: MethodSpec) -> ModuleType:
    try:
        module_name = _STABLE_BACKENDS[method.backend_family]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported backend family: {method.backend_family!r}") from exc
    return importlib.import_module(module_name)


def _backend_mode() -> str:
    mode = os.environ.get("GSWARP_WARP_BACKEND", "auto").strip().lower()
    if mode not in {"auto", "stable", "advanced"}:
        raise RuntimeError("GSWARP_WARP_BACKEND must be one of 'auto', 'stable', or 'advanced'")
    return mode


def _validate_profile(spec: MethodSpec) -> dict[str, Any]:
    try:
        profile = _STAGE_PROFILES[spec.backend_family]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported backend family: {spec.backend_family!r}") from exc

    for field in ("output_mode", "primitive", "projection", "filtering", "appearance", "pre_adapter"):
        if getattr(spec, field) != profile[field]:
            raise RuntimeError(
                f"Method {spec.name!r} requests {field}={getattr(spec, field)!r}, "
                f"but {spec.backend_family!r} has no matching stage implementation"
            )
    return profile


def _required_capabilities(spec: MethodSpec) -> frozenset[str]:
    required = spec.required_capabilities
    if spec.requires_advanced_warp:
        required = required | frozenset({"advanced_warp"})
    return required


def build_method_plan(
    spec: MethodSpec,
    backend: Any,
    *,
    stage_overrides: Mapping[str, Callable[..., Any]] | None = None,
) -> MethodPlan:
    """Compose one immutable plan from an implemented backend protocol.

    ``stage_overrides`` is internal and test-oriented: future method builders can
    replace one compatible stage while retaining the remaining bindings.
    """
    profile = _validate_profile(spec)
    capabilities = frozenset(getattr(backend, "BACKEND_CAPABILITIES", ()))
    missing = _required_capabilities(spec) - capabilities
    if missing:
        raise RuntimeError(
            f"Backend for method {spec.name!r} lacks required capabilities: {sorted(missing)!r}"
        )

    defaults = {
        "empty_forward": getattr(backend, profile["empty_forward_name"]),
        "preprocess": getattr(backend, profile["preprocess_name"]),
        "features": getattr(backend, profile["features_name"]),
        "binning": getattr(backend, "_build_binning_state"),
        "render": getattr(backend, profile["render_name"]),
        "build_state": getattr(backend, profile["build_state_name"]),
        "backward": getattr(backend, profile["backward_name"]),
        "mark_visible": getattr(backend, "mark_visible"),
    }
    if stage_overrides is not None:
        unknown = set(stage_overrides) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown method-stage overrides: {sorted(unknown)!r}")
        defaults.update(stage_overrides)

    return MethodPlan(
        spec=spec,
        backend=backend,
        stages=MethodStages(**defaults),
        output_adapter=_flow_output_adapter if profile["flow"] else _standard_output_adapter,
        state_schema=getattr(backend, "ForwardState"),
        capabilities=capabilities,
        flow=profile["flow"],
    )


@lru_cache(maxsize=None)
def _resolve_registered_plan(method_name: str, mode: str) -> MethodPlan:
    method = get_method(method_name)
    stable = _load_stable_backend(method)
    if mode == "stable":
        return build_method_plan(method, stable)

    from gswarp._internal.backends.warp.advanced import resolve_advanced_backend

    advanced, unavailable_reason = resolve_advanced_backend(method)
    if advanced is not None:
        try:
            return build_method_plan(method, advanced)
        except RuntimeError as exc:
            unavailable_reason = str(exc)
            if mode == "advanced":
                raise
    if mode == "advanced":
        raise RuntimeError(
            f"Advanced Warp backend unavailable for method {method.name!r}: "
            f"{unavailable_reason or 'unknown reason'}"
        )
    return build_method_plan(method, stable)


def resolve_backend(method: str | MethodSpec) -> MethodPlan:
    """Resolve a registered method to one cached immutable execution plan."""
    registered = get_method(method)
    return _resolve_registered_plan(registered.name, _backend_mode())


def clear_method_plan_cache() -> None:
    _resolve_registered_plan.cache_clear()


__all__ = ["build_method_plan", "clear_method_plan_cache", "resolve_backend"]
