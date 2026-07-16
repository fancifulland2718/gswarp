"""Classic Mip-Splatting-compatible 3D Gaussian rasterizer method."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch.nn as nn

from gswarp._internal.api.validation import normalize_gaussian_inputs, validate_rasterizer_inputs
from gswarp._internal.coverage import BASELINE_3DGS_COVERAGE
from gswarp._internal.frontend import common, mip_autograd
from gswarp._internal.methods.spec import MethodSpec

if TYPE_CHECKING:
    import torch


METHOD = MethodSpec(
    name="mip_3dgs",
    backend_family="warp_mip_3dgs",
    output_mode="standard_meta",
    filtering="mip",
    coverage=BASELINE_3DGS_COVERAGE,
    pre_adapter="mip_3dgs",
)


class MipGaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: "torch.Tensor"
    scale_modifier: float
    viewmatrix: "torch.Tensor"
    projmatrix: "torch.Tensor"
    sh_degree: int
    campos: "torch.Tensor"
    prefiltered: bool
    debug: bool = False
    antialiasing: bool = False
    backward_mode: str | None = None
    binning_sort_mode: str | None = None
    auto_tune: bool = True
    auto_tune_verbose: bool = True
    filter_variance: float = 0.3


def _plan():
    return common.plan_for(METHOD)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    """Render with the explicit screen-space Mip low-pass variance.

    ``antialiasing`` belongs to the baseline 3DGS compatibility API and is not
    an alias for Mip filtering. Configure this method exclusively with
    ``filter_variance``.
    """

    validate_rasterizer_inputs(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    if raster_settings.antialiasing:
        raise ValueError("mip_3dgs uses filter_variance; antialiasing must remain False")
    filter_variance = float(raster_settings.filter_variance)
    if not math.isfinite(filter_variance) or filter_variance < 0.0:
        raise ValueError("filter_variance must be finite and non-negative")
    from gswarp.rasterizer import RasterizerMeta

    plan = _plan()
    outputs = mip_autograd.rasterize_gaussians(
        plan,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    return common.adapt_outputs(plan, outputs, RasterizerMeta)


class MipGaussianRasterizer(nn.Module):
    """Module wrapper with Mip-specific settings and baseline-style inputs."""

    def __init__(self, raster_settings: MipGaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        dc=None,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        inputs = normalize_gaussian_inputs(
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        return rasterize_gaussians(
            means3D,
            means2D,
            inputs.shs,
            inputs.colors_precomp,
            opacities,
            inputs.scales,
            inputs.rotations,
            inputs.cov3D_precomp,
            self.raster_settings,
        )


__all__ = [
    "METHOD",
    "MipGaussianRasterizationSettings",
    "MipGaussianRasterizer",
    "rasterize_gaussians",
]
