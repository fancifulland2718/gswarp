from typing import NamedTuple

import torch
import torch.nn as nn

from gswarp._internal.api.validation import normalize_gaussian_inputs, validate_rasterizer_inputs
from gswarp._internal.frontend import common, standard_autograd
from gswarp.methods.baseline_3dgs import METHOD


class RasterizerMeta(NamedTuple):
    """Auxiliary outputs from the non-flow rasterizer forward pass."""

    depth: torch.Tensor
    alpha: torch.Tensor
    proj_2D: torch.Tensor
    conic_2D: torch.Tensor
    conic_2D_inv: torch.Tensor


def _backend():
    return common.backend_for(METHOD)


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


def get_backward_mode():
    return _backend().get_backward_mode()


def set_backward_mode(mode):
    _backend().set_backward_mode(mode)


def get_binning_sort_mode():
    return _backend().get_binning_sort_mode()


def set_binning_sort_mode(mode):
    _backend().set_binning_sort_mode(mode)


def get_tile_coverage_mode():
    return _backend().get_tile_coverage_mode()


def set_tile_coverage_mode(mode):
    _backend().set_tile_coverage_mode(mode)


def get_default_parameter_info():
    return _backend().get_default_parameter_info()


def initialize_runtime_tuning(device=None, verbose=True):
    return _backend().initialize_runtime_tuning(device=device, verbose=verbose)


def get_runtime_tuning_report(device=None):
    return _backend().get_runtime_tuning_report(device=device)


def get_runtime_auto_tuning_config():
    return _backend().get_runtime_auto_tuning_config()


def get_compute_depth():
    return _backend().get_compute_depth()


def set_compute_depth(enabled):
    _backend().set_compute_depth(enabled)


def clear_warp_caches():
    """Release all grow-only Warp buffer caches."""
    _backend().clear_warp_caches()


def get_warp_cache_report():
    """Return bounded Warp cache occupancy and retained tensor bytes."""
    return _backend().get_warp_cache_report()


class GaussianRasterizationSettings(NamedTuple):
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
    antialiasing: bool = False
    backward_mode: str | None = None
    binning_sort_mode: str | None = None
    auto_tune: bool = True
    auto_tune_verbose: bool = True


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            return common.mark_visible(_plan(), positions, self.raster_settings)

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
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    "RasterizerMeta",
    "rasterize_gaussians",
    "initialize_runtime_tuning",
    "get_runtime_tuning_report",
    "clear_warp_caches",
    "get_warp_cache_report",
    "get_backward_mode",
    "set_backward_mode",
    "get_compute_depth",
    "set_compute_depth",
    "get_binning_sort_mode",
    "set_binning_sort_mode",
    "get_tile_coverage_mode",
    "set_tile_coverage_mode",
]
