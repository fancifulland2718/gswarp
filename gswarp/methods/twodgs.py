"""Perspective-correct 2D Gaussian Splatting method API."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn

from gswarp._internal.coverage import (
    CoverageContract,
    FOOTPRINT_RAY_SPLAT,
    SAMPLE_PIXEL_CENTERS,
    SUPPORT_CUSTOM,
)
from gswarp._internal.frontend import common, twod_autograd
from gswarp._internal.methods.spec import MethodSpec


METHOD = MethodSpec(
    name="twodgs",
    backend_family="warp_2dgs",
    output_mode="twodgs",
    primitive="gaussian_2d",
    projection="ray_splat",
    appearance="sh_or_rgb",
    coverage=CoverageContract(
        footprint=FOOTPRINT_RAY_SPLAT,
        support=SUPPORT_CUSTOM,
        sample_domain=SAMPLE_PIXEL_CENTERS,
    ),
    pre_adapter="twodgs",
)


class TwoDGaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool = False
    backward_mode: str | None = None
    binning_sort_mode: str | None = None
    auto_tune: bool = True
    auto_tune_verbose: bool = True
    filter_radius: float = 1.0
    near_plane: float = 0.2
    far_plane: float = 100.0


class TwoDGSResult(NamedTuple):
    color: torch.Tensor
    radii: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    normal: torch.Tensor
    distortion: torch.Tensor
    median_depth: torch.Tensor
    proj_2D: torch.Tensor
    conic_2D: torch.Tensor
    conic_2D_inv: torch.Tensor


def _plan():
    return common.plan_for(METHOD)


def _empty_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=reference.device)


def _validate(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    raster_settings,
) -> None:
    tensors = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "raster_settings.bg": raster_settings.bg,
        "raster_settings.viewmatrix": raster_settings.viewmatrix,
        "raster_settings.projmatrix": raster_settings.projmatrix,
        "raster_settings.campos": raster_settings.campos,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError("2DGS inputs and camera settings must be torch tensors")
    if not means3D.is_cuda or means3D.dtype != torch.float32 or means3D.ndim != 2 or means3D.shape[1] != 3:
        raise ValueError("means3D must be a CUDA float32 tensor with shape (N, 3)")
    count, device = means3D.shape[0], means3D.device
    for name, value, shape in (
        ("means2D", means2D, (count, 3)),
        ("opacities", opacities, (count, 1)),
        ("scales", scales, (count, 2)),
        ("rotations", rotations, (count, 4)),
        ("raster_settings.bg", raster_settings.bg, (3,)),
        ("raster_settings.viewmatrix", raster_settings.viewmatrix, (4, 4)),
        ("raster_settings.projmatrix", raster_settings.projmatrix, (4, 4)),
        ("raster_settings.campos", raster_settings.campos, (3,)),
    ):
        if value.device != device or value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a CUDA float32 tensor with shape {shape}")
    has_sh, has_colors = sh.numel() != 0, colors_precomp.numel() != 0
    if count != 0 and has_sh == has_colors:
        raise ValueError("provide exactly one of sh or colors_precomp")
    if has_colors and (
        colors_precomp.device != device
        or colors_precomp.dtype != torch.float32
        or tuple(colors_precomp.shape) != (count, 3)
    ):
        raise ValueError("colors_precomp must have shape (N, 3) on the surfel device")
    if has_sh:
        if sh.device != device or sh.dtype != torch.float32 or count == 0 or sh.numel() % (count * 3) != 0:
            raise ValueError("sh must contain complete RGB coefficients on the surfel device")
        coefficient_count = sh.numel() // (count * 3)
        if raster_settings.sh_degree < 0 or raster_settings.sh_degree > 3:
            raise ValueError("2DGS supports SH degrees from 0 through 3")
        if coefficient_count < (raster_settings.sh_degree + 1) ** 2:
            raise ValueError("sh does not contain enough coefficients for raster_settings.sh_degree")
    elif raster_settings.sh_degree < 0 or raster_settings.sh_degree > 3:
        raise ValueError("2DGS supports SH degrees from 0 through 3")
    if raster_settings.prefiltered:
        raise ValueError("2DGS does not support baseline prefiltered visibility semantics")
    if raster_settings.image_height <= 0 or raster_settings.image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.scale_modifier,
        raster_settings.filter_radius,
        raster_settings.near_plane,
        raster_settings.far_plane,
    )) or raster_settings.far_plane <= raster_settings.near_plane:
        raise ValueError("2DGS scalar settings must be finite, positive, and use far_plane > near_plane")


def rasterize_surfels(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    raster_settings,
) -> TwoDGSResult:
    """Rasterize 2D oriented Gaussian surfels with perspective-correct rays."""

    _validate(means3D, means2D, sh, colors_precomp, opacities, scales, rotations, raster_settings)
    (
        color,
        radii,
        depth,
        alpha,
        normal,
        distortion,
        median_depth,
        proj_2d,
        conic_2d,
        conic_2d_inv,
    ) = twod_autograd.rasterize_surfels(
        _plan(),
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings,
    )
    return TwoDGSResult(
        color=color,
        radii=radii,
        depth=depth,
        alpha=alpha,
        normal=normal,
        distortion=distortion,
        median_depth=median_depth,
        proj_2D=proj_2d,
        conic_2D=conic_2d,
        conic_2D_inv=conic_2d_inv,
    )


class TwoDGaussianRasterizer(nn.Module):
    """Module wrapper for 2D oriented Gaussian surfels."""

    def __init__(self, raster_settings: TwoDGaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        shs=None,
        colors_precomp=None,
    ) -> TwoDGSResult:
        if (shs is None) == (colors_precomp is None):
            raise ValueError("provide exactly one of shs or colors_precomp")
        return rasterize_surfels(
            means3D,
            means2D,
            _empty_like(means3D) if shs is None else shs,
            _empty_like(means3D) if colors_precomp is None else colors_precomp,
            opacities,
            scales,
            rotations,
            self.raster_settings,
        )


__all__ = [
    "METHOD",
    "TwoDGSResult",
    "TwoDGaussianRasterizationSettings",
    "TwoDGaussianRasterizer",
    "rasterize_surfels",
]
