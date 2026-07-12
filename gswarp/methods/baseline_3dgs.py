"""Baseline 3DGS method specification."""

from gswarp._internal.methods.spec import MethodSpec
from gswarp._internal.coverage import FOOTPRINT_EXACT_SCREEN_CONIC

METHOD = MethodSpec(
    name="baseline_3dgs",
    backend_family="warp_3dgs",
    output_mode="standard_meta",
    footprint_capability=FOOTPRINT_EXACT_SCREEN_CONIC,
)

__all__ = ["METHOD"]
