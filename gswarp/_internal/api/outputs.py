"""Public output packing helpers."""

from __future__ import annotations


def pack_standard_outputs(outputs, meta_cls):
    color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv = outputs
    meta = meta_cls(
        depth=depth,
        alpha=alpha,
        proj_2D=proj_2D,
        conic_2D=conic_2D,
        conic_2D_inv=conic_2D_inv,
    )
    return color, radii, meta


def pack_flow_outputs(outputs):
    return outputs


__all__ = ["pack_standard_outputs", "pack_flow_outputs"]
