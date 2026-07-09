"""Public input validation helpers."""

from __future__ import annotations

from typing import NamedTuple

import torch


class GaussianInputs(NamedTuple):
    shs: torch.Tensor
    colors_precomp: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    cov3D_precomp: torch.Tensor


def normalize_gaussian_inputs(
    *,
    dc,
    shs,
    colors_precomp,
    scales,
    rotations,
    cov3D_precomp,
) -> GaussianInputs:
    """Normalize public rasterizer optional inputs without touching backend state."""
    if dc is not None and shs is not None:
        shs = torch.cat([dc, shs], dim=1)
    elif dc is not None and shs is None:
        shs = dc

    if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
        raise Exception("Please provide excatly one of either SHs or precomputed colors!")

    if ((scales is None or rotations is None) and cov3D_precomp is None) or (
        (scales is not None or rotations is not None) and cov3D_precomp is not None
    ):
        raise Exception("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!")

    if shs is None:
        shs = torch.Tensor([])
    if colors_precomp is None:
        colors_precomp = torch.Tensor([])
    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    return GaussianInputs(
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )


__all__ = ["GaussianInputs", "normalize_gaussian_inputs"]
