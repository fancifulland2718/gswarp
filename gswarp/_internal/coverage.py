"""Tile-coverage policy shared by methods and raster backends."""

from __future__ import annotations


TILE_COVERAGE_MODES = ("auto", "snugbox", "accutile_sweep", "conic_rect")

FOOTPRINT_EXACT_SCREEN_CONIC = "exact_screen_conic"
FOOTPRINT_APPROXIMATE_SCREEN_CONIC = "approximate_screen_conic"
FOOTPRINT_AXIS_ALIGNED = "axis_aligned_footprint"
FOOTPRINT_CUSTOM = "custom_footprint"
FOOTPRINT_CAPABILITIES = (
    FOOTPRINT_EXACT_SCREEN_CONIC,
    FOOTPRINT_APPROXIMATE_SCREEN_CONIC,
    FOOTPRINT_AXIS_ALIGNED,
    FOOTPRINT_CUSTOM,
)

_TILE_COVERAGE_MODE_IDS = {
    "snugbox": 0,
    "accutile_sweep": 1,
    "conic_rect": 2,
    "auto": 3,
}


def normalize_tile_coverage_mode(mode: str) -> str:
    if mode not in TILE_COVERAGE_MODES:
        choices = ", ".join(repr(value) for value in TILE_COVERAGE_MODES)
        raise ValueError(f"mode must be one of {choices}")
    return mode


def resolve_tile_coverage_mode(mode: str, footprint_capability: str) -> str:
    """Resolve method-level safety without choosing workload geometry."""
    mode = normalize_tile_coverage_mode(mode)
    if footprint_capability not in FOOTPRINT_CAPABILITIES:
        raise ValueError(f"unknown footprint capability: {footprint_capability!r}")
    if footprint_capability == FOOTPRINT_EXACT_SCREEN_CONIC:
        if mode == "auto":
            return "accutile_sweep"
        return mode
    if mode == "auto":
        return "snugbox"
    if mode != "snugbox":
        raise ValueError(
            f"tile coverage mode {mode!r} requires an exact_screen_conic footprint; "
            f"method declares {footprint_capability!r}"
        )
    return mode


def tile_coverage_mode_id(mode: str) -> int:
    return _TILE_COVERAGE_MODE_IDS[normalize_tile_coverage_mode(mode)]


__all__ = [
    "FOOTPRINT_APPROXIMATE_SCREEN_CONIC",
    "FOOTPRINT_AXIS_ALIGNED",
    "FOOTPRINT_CAPABILITIES",
    "FOOTPRINT_CUSTOM",
    "FOOTPRINT_EXACT_SCREEN_CONIC",
    "TILE_COVERAGE_MODES",
    "normalize_tile_coverage_mode",
    "resolve_tile_coverage_mode",
    "tile_coverage_mode_id",
]
