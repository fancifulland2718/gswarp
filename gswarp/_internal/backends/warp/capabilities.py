"""Warp runtime feature detection used by optional advanced backends."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import ModuleType

import warp as wp


def parse_warp_version(value: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return 0, 0, 0
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


@dataclass(frozen=True, slots=True)
class WarpFeatureSet:
    version: tuple[int, int, int]
    capabilities: frozenset[str]

    def missing(self, required) -> frozenset[str]:
        return frozenset(required) - self.capabilities


def detect_warp_features(module: ModuleType = wp) -> WarpFeatureSet:
    version = parse_warp_version(str(getattr(module, "__version__", "0.0.0")))
    capabilities = {"stable_warp"}

    if version >= (1, 11, 0):
        if hasattr(module, "launch_tiled") and hasattr(module, "tile_sum"):
            capabilities.add("tile_axis_reduce")
        if hasattr(module, "grad"):
            capabilities.add("inline_function_grad")
    if version >= (1, 12, 0):
        if hasattr(module, "div_approx") and hasattr(module, "inverse_approx"):
            capabilities.add("approx_math")
        if hasattr(module, "Texture2D"):
            capabilities.add("texture_sampling")
    if hasattr(module, "capture_if") and hasattr(module, "is_conditional_graph_supported"):
        capabilities.add("conditional_graph")
    if hasattr(module, "compile_aot_module") and hasattr(module, "load_aot_module"):
        capabilities.add("aot_module")

    return WarpFeatureSet(version=version, capabilities=frozenset(capabilities))


__all__ = ["WarpFeatureSet", "detect_warp_features", "parse_warp_version"]
