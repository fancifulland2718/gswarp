"""Tile-coverage contracts shared by methods and raster backends."""

from __future__ import annotations

from dataclasses import dataclass


TILE_COVERAGE_MODES = ("auto", "snugbox", "accutile_sweep", "conic_rect")
CONCRETE_TILE_COVERAGE_MODES = ("snugbox", "accutile_sweep", "conic_rect")

FOOTPRINT_SCREEN_CONIC = "screen_conic"
FOOTPRINT_AXIS_ALIGNED = "axis_aligned"
FOOTPRINT_RAY_SPLAT = "ray_splat"
FOOTPRINT_CUSTOM = "custom"
FOOTPRINTS = (
    FOOTPRINT_SCREEN_CONIC,
    FOOTPRINT_AXIS_ALIGNED,
    FOOTPRINT_RAY_SPLAT,
    FOOTPRINT_CUSTOM,
)

SUPPORT_CUDA_3SIGMA_CLAMPED = "cuda_3sigma_clamped"
SUPPORT_ALPHA_CUTOFF = "alpha_cutoff"
SUPPORT_CUSTOM = "custom"
SUPPORT_POLICIES = (
    SUPPORT_CUDA_3SIGMA_CLAMPED,
    SUPPORT_ALPHA_CUTOFF,
    SUPPORT_CUSTOM,
)

SAMPLE_PIXEL_CENTERS = "pixel_centers"
SAMPLE_CONTINUOUS = "continuous"
SAMPLE_CUSTOM = "custom"
SAMPLE_DOMAINS = (SAMPLE_PIXEL_CENTERS, SAMPLE_CONTINUOUS, SAMPLE_CUSTOM)

EXACT_POLICY_CUDA_COMPAT_CONIC = "cuda_compat_screen_conic"


@dataclass(frozen=True, slots=True)
class CoverageContract:
    """Method-owned footprint semantics used to select safe tile coverage."""

    footprint: str
    support: str
    sample_domain: str
    exact_policy: str | None = None

    def __post_init__(self) -> None:
        if self.footprint not in FOOTPRINTS:
            raise ValueError(f"unknown coverage footprint: {self.footprint!r}")
        if self.support not in SUPPORT_POLICIES:
            raise ValueError(f"unknown coverage support policy: {self.support!r}")
        if self.sample_domain not in SAMPLE_DOMAINS:
            raise ValueError(f"unknown coverage sample domain: {self.sample_domain!r}")
        if self.exact_policy not in (None, EXACT_POLICY_CUDA_COMPAT_CONIC):
            raise ValueError(f"unknown exact coverage policy: {self.exact_policy!r}")
        if self.exact_policy == EXACT_POLICY_CUDA_COMPAT_CONIC and (
            self.footprint != FOOTPRINT_SCREEN_CONIC
            or self.support != SUPPORT_CUDA_3SIGMA_CLAMPED
            or self.sample_domain != SAMPLE_PIXEL_CENTERS
        ):
            raise ValueError(
                "cuda-compatible exact coverage requires a screen conic, "
                "CUDA 3-sigma support, and pixel-center sampling"
            )


BASELINE_3DGS_COVERAGE = CoverageContract(
    footprint=FOOTPRINT_SCREEN_CONIC,
    support=SUPPORT_CUDA_3SIGMA_CLAMPED,
    sample_domain=SAMPLE_PIXEL_CENTERS,
    exact_policy=EXACT_POLICY_CUDA_COMPAT_CONIC,
)
CONSERVATIVE_COVERAGE = CoverageContract(
    footprint=FOOTPRINT_CUSTOM,
    support=SUPPORT_CUSTOM,
    sample_domain=SAMPLE_CUSTOM,
)

_CONCRETE_MODE_IDS = {
    "snugbox": 0,
    "accutile_sweep": 1,
    "conic_rect": 2,
}


def normalize_tile_coverage_mode(mode: str) -> str:
    if mode not in TILE_COVERAGE_MODES:
        choices = ", ".join(repr(value) for value in TILE_COVERAGE_MODES)
        raise ValueError(f"mode must be one of {choices}")
    return mode


def resolve_tile_coverage_mode(mode: str, contract: CoverageContract) -> str:
    """Resolve a public policy to one concrete kernel mode."""

    mode = normalize_tile_coverage_mode(mode)
    supports_cuda_exact = (
        contract.exact_policy == EXACT_POLICY_CUDA_COMPAT_CONIC
    )
    if supports_cuda_exact:
        return "accutile_sweep" if mode == "auto" else mode
    if mode == "auto":
        return "snugbox"
    if mode != "snugbox":
        raise ValueError(
            f"tile coverage mode {mode!r} requires exact policy "
            f"{EXACT_POLICY_CUDA_COMPAT_CONIC!r}; method declares {contract!r}"
        )
    return mode


def tile_coverage_mode_id(mode: str) -> int:
    mode = normalize_tile_coverage_mode(mode)
    if mode == "auto":
        raise ValueError("auto tile coverage must be resolved before kernel dispatch")
    return _CONCRETE_MODE_IDS[mode]


def resolve_tile_coverage_mode_id(
    mode: str, contract: CoverageContract
) -> int:
    """Resolve a method contract on the host and return its concrete kernel id."""
    return tile_coverage_mode_id(resolve_tile_coverage_mode(mode, contract))


__all__ = [
    "BASELINE_3DGS_COVERAGE",
    "CONCRETE_TILE_COVERAGE_MODES",
    "CONSERVATIVE_COVERAGE",
    "CoverageContract",
    "EXACT_POLICY_CUDA_COMPAT_CONIC",
    "FOOTPRINT_AXIS_ALIGNED",
    "FOOTPRINT_CUSTOM",
    "FOOTPRINT_RAY_SPLAT",
    "FOOTPRINT_SCREEN_CONIC",
    "FOOTPRINTS",
    "SAMPLE_DOMAINS",
    "SAMPLE_CONTINUOUS",
    "SAMPLE_CUSTOM",
    "SAMPLE_PIXEL_CENTERS",
    "SUPPORT_ALPHA_CUTOFF",
    "SUPPORT_CUDA_3SIGMA_CLAMPED",
    "SUPPORT_CUSTOM",
    "SUPPORT_POLICIES",
    "TILE_COVERAGE_MODES",
    "normalize_tile_coverage_mode",
    "resolve_tile_coverage_mode",
    "resolve_tile_coverage_mode_id",
    "tile_coverage_mode_id",
]
