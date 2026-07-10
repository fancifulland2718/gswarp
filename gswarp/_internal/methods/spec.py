"""Method specification records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str
    backend_family: str
    output_mode: str
    primitive: str = "gaussian_3d"
    projection: str = "perspective_3dgs"
    filtering: str = "none"
    appearance: str = "sh_or_rgb"
    pre_adapter: str | None = None
    requires_advanced_warp: bool = False
    required_capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MethodStages:
    """Immutable function bindings for one rasterization method."""

    preprocess: Callable[..., Any]
    binning: Callable[..., Any]
    render: Callable[..., Any]
    forward: Callable[..., Any]
    backward: Callable[..., Any]
    mark_visible: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class MethodPlan:
    """Resolved method composition shared by a complete frontend call."""

    spec: MethodSpec
    backend: Any
    stages: MethodStages
    output_adapter: Callable[..., Any]
    state_schema: type[Any]
    capabilities: frozenset[str]
    flow: bool


__all__ = ["MethodSpec", "MethodStages", "MethodPlan"]
