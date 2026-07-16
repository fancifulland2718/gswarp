"""Renderer adapter for materialized per-view standard 3D Gaussians.

Scaffold-GS, Deformable-3DGS and 4D-GS-style systems retain their model and
trainer upstream. This module rasterizes their canonical tensor outputs without
detaching PyTorch gradients.
"""
from typing import TYPE_CHECKING

import torch.nn as nn

from gswarp._internal.api.validation import normalize_gaussian_inputs, validate_rasterizer_inputs
from gswarp._internal.coverage import BASELINE_3DGS_COVERAGE
from gswarp._internal.frontend import common, standard_autograd
from gswarp._internal.methods.spec import MethodSpec

if TYPE_CHECKING:
    from gswarp.rasterizer import GaussianRasterizationSettings


METHOD = MethodSpec(
    name="generated_3dgs",
    backend_family="warp_generated_3dgs",
    output_mode="standard_meta",
    coverage=BASELINE_3DGS_COVERAGE,
    pre_adapter="generated_3dgs",
)


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
    """Rasterize canonical tensors while preserving their upstream graph."""

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
    plan = _plan()
    from gswarp.rasterizer import RasterizerMeta
    outputs = standard_autograd.rasterize_gaussians(
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


class GeneratedGaussianRasterizer(nn.Module):
    """Module wrapper for a caller's materialized standard Gaussian tensors."""

    def __init__(self, raster_settings: "GaussianRasterizationSettings"):
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


__all__ = ["GeneratedGaussianRasterizer", "METHOD", "rasterize_gaussians"]
