"""Method specification records."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["MethodSpec"]
