"""Baseline 3DGS method specification."""

from gswarp._internal.methods.spec import MethodSpec
from gswarp._internal.coverage import BASELINE_3DGS_COVERAGE

METHOD = MethodSpec(
    name="baseline_3dgs",
    backend_family="warp_3dgs",
    output_mode="standard_meta",
    coverage=BASELINE_3DGS_COVERAGE,
)

__all__ = ["METHOD"]
