"""Implemented method registry."""

from __future__ import annotations

from gswarp.methods.baseline_3dgs import METHOD as BASELINE_3DGS
from gswarp.methods.flow_aux import METHOD as FLOW_AUX

METHODS = {
    "baseline_3dgs": BASELINE_3DGS,
    "flow_aux": FLOW_AUX,
}

__all__ = ["BASELINE_3DGS", "FLOW_AUX", "METHODS"]
