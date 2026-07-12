"""Flow auxiliary method specification."""

from gswarp._internal.methods.spec import MethodSpec
from gswarp._internal.coverage import BASELINE_3DGS_COVERAGE

METHOD = MethodSpec(
    name="flow_aux",
    backend_family="warp_3dgs_flow",
    output_mode="flow_flat",
    coverage=BASELINE_3DGS_COVERAGE,
)

__all__ = ["METHOD"]
