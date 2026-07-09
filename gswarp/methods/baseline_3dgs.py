"""Baseline 3DGS method specification."""

from gswarp._internal.methods.spec import MethodSpec

METHOD = MethodSpec(
    name="baseline_3dgs",
    backend_family="warp_3dgs",
    output_mode="standard_meta",
)

__all__ = ["METHOD"]
