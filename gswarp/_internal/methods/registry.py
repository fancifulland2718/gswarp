"""Implemented method registry."""

from __future__ import annotations

from types import MappingProxyType

from gswarp.methods.baseline_3dgs import METHOD as BASELINE_3DGS
from gswarp.methods.generated_3dgs import METHOD as GENERATED_3DGS
from gswarp.methods.flow_aux import METHOD as FLOW_AUX
from gswarp.methods.mip_3dgs import METHOD as MIP_3DGS
from gswarp.methods.twodgs import METHOD as TWODGS
from gswarp._internal.methods.spec import MethodSpec

METHODS = MappingProxyType({
    "baseline_3dgs": BASELINE_3DGS,
    "generated_3dgs": GENERATED_3DGS,
    "flow_aux": FLOW_AUX,
    "mip_3dgs": MIP_3DGS,
    "twodgs": TWODGS,
})


def get_method(method: str | MethodSpec) -> MethodSpec:
    """Return the canonical explicitly registered method specification."""
    if isinstance(method, MethodSpec):
        registered = METHODS.get(method.name)
        if registered != method:
            raise ValueError(f"Method {method.name!r} is not registered with this specification")
        return registered
    try:
        return METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown gswarp method: {method!r}") from exc

__all__ = ["BASELINE_3DGS", "FLOW_AUX", "GENERATED_3DGS", "MIP_3DGS", "TWODGS", "METHODS", "get_method"]
