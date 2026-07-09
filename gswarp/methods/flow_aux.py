"""Flow auxiliary method specification."""

from gswarp._internal.methods.spec import MethodSpec

METHOD = MethodSpec(
    name="flow_aux",
    backend_family="warp_3dgs_flow",
    output_mode="flow_flat",
)

__all__ = ["METHOD"]
