"""Flow auxiliary method specification."""

from gswarp._internal.methods.spec import MethodSpec
from gswarp._internal.coverage import FOOTPRINT_EXACT_SCREEN_CONIC

METHOD = MethodSpec(
    name="flow_aux",
    backend_family="warp_3dgs_flow",
    output_mode="flow_flat",
    footprint_capability=FOOTPRINT_EXACT_SCREEN_CONIC,
)

__all__ = ["METHOD"]
