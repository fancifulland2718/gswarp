from typing import NamedTuple

import torch
import torch.nn as nn

from . import _rasterizer_flow as _warp_backend
from ._stream import ensure_aligned


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
    return _WarpRasterizeGaussians.apply(
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


def get_backward_mode():
    return _warp_backend.get_backward_mode()


def set_backward_mode(mode):
    _warp_backend.set_backward_mode(mode)


def get_binning_sort_mode():
    return _warp_backend.get_binning_sort_mode()


def set_binning_sort_mode(mode):
    _warp_backend.set_binning_sort_mode(mode)


def get_default_parameter_info():
    return _warp_backend.get_default_parameter_info()


def initialize_runtime_tuning(device=None, verbose=True):
    return _warp_backend.initialize_runtime_tuning(device=device, verbose=verbose)


def get_runtime_tuning_report(device=None):
    return _warp_backend.get_runtime_tuning_report(device=device)


def get_runtime_auto_tuning_config():
    return _warp_backend.get_runtime_auto_tuning_config()


def get_compute_depth():
    return _warp_backend.get_compute_depth()


def set_compute_depth(enabled):
    _warp_backend.set_compute_depth(enabled)


def get_compute_flow_aux():
    return _warp_backend.get_compute_flow_aux()


def set_compute_flow_aux(enabled):
    _warp_backend.set_compute_flow_aux(enabled)


def get_flow_topk():
    return _warp_backend.get_flow_topk()


def set_flow_topk(k):
    _warp_backend.set_flow_topk(k)


def clear_warp_caches():
    """Release all grow-only Warp buffer caches.

    Call after ``densify_and_prune`` or when switching scenes to reclaim GPU
    memory held by stale high-water-mark allocations.  The caches will be
    lazily re-populated on the next forward pass.
    """
    _warp_backend.clear_warp_caches()


def _run_with_experimental_settings(raster_settings, fn):
    ensure_aligned()
    previous_backward_mode = _warp_backend.get_backward_mode()
    previous_binning_sort_mode = _warp_backend.get_binning_sort_mode()
    previous_compute_flow_aux = _warp_backend.get_compute_flow_aux()
    previous_auto_tuning = None
    backward_mode = getattr(raster_settings, "backward_mode", None)
    binning_sort_mode = getattr(raster_settings, "binning_sort_mode", None)
    compute_flow_aux = getattr(raster_settings, "compute_flow_aux", None)
    auto_tune = getattr(raster_settings, "auto_tune", True)
    auto_tune_verbose = getattr(raster_settings, "auto_tune_verbose", True)

    try:
        previous_auto_tuning = _warp_backend.get_runtime_auto_tuning_config()
        _warp_backend.set_runtime_auto_tuning(enabled=auto_tune, verbose=auto_tune_verbose)

        if backward_mode is not None:
            if backward_mode != previous_backward_mode:
                _warp_backend.set_backward_mode(backward_mode)

        if binning_sort_mode is not None:
            if binning_sort_mode != previous_binning_sort_mode:
                _warp_backend.set_binning_sort_mode(binning_sort_mode)

        if compute_flow_aux is not None:
            if bool(compute_flow_aux) != previous_compute_flow_aux:
                _warp_backend.set_compute_flow_aux(bool(compute_flow_aux))

        return fn()
    finally:
        if previous_auto_tuning is not None:
            current_auto_tuning = _warp_backend.get_runtime_auto_tuning_config()
            if current_auto_tuning != previous_auto_tuning:
                _warp_backend.set_runtime_auto_tuning(
                    enabled=previous_auto_tuning["enabled"],
                    verbose=previous_auto_tuning["verbose"],
                )
        if _warp_backend.get_compute_flow_aux() != previous_compute_flow_aux:
            _warp_backend.set_compute_flow_aux(previous_compute_flow_aux)
        if _warp_backend.get_binning_sort_mode() != previous_binning_sort_mode:
            _warp_backend.set_binning_sort_mode(previous_binning_sort_mode)
        if _warp_backend.get_backward_mode() != previous_backward_mode:
            _warp_backend.set_backward_mode(previous_backward_mode)


class _WarpRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
        )

        outputs = _run_with_experimental_settings(raster_settings, lambda: _warp_backend.rasterize_gaussians(*args))

        (
            num_rendered,
            color,
            depth,
            alpha,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            proj_2D,
            conic_2D,
            conic_2D_inv,
            gs_per_pixel,
            weight_per_gs_pixel,
            x_mu,
        ) = outputs

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            alpha,
        )
        return color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu

    @staticmethod
    def backward(
        ctx,
        grad_color,
        grad_radii,
        grad_depth,
        grad_alpha,
        grad_proj_2D,
        grad_conic_2D,
        grad_conic_2D_inv,
        dummy_gs_per_pixel,
        dummy_weight_per_gs_pixel,
        grad_x_mu,
    ):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, opacities, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_color,
            grad_depth,
            grad_alpha,
            grad_proj_2D,
            grad_conic_2D,
            grad_conic_2D_inv,
            dummy_gs_per_pixel,
            dummy_weight_per_gs_pixel,
            grad_x_mu,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            alpha,
            raster_settings.enable_flow_grad,
        )

        grads = _run_with_experimental_settings(raster_settings, lambda: _warp_backend.rasterize_gaussians_backward(*args))

        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales,
            grad_rotations,
        ) = grads

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


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
    enable_flow_grad: bool = True
    compute_flow_aux: bool | None = None
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
            raster_settings = self.raster_settings
            visible = _warp_backend.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )

        return visible

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
        raster_settings = self.raster_settings

        # When dc (0th-order SH) is provided separately, concatenate with
        # the remaining SH coefficients so the warp backend receives a
        # single contiguous SH tensor.
        if dc is not None and shs is not None:
            shs = torch.cat([dc, shs], dim=1)
        elif dc is not None and shs is None:
            shs = dc

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
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

        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )


__all__ = [
    # Core rasterizer types
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    # Core functions
    "rasterize_gaussians",
    # Setup & memory management
    "initialize_runtime_tuning",
    "get_runtime_tuning_report",
    "clear_warp_caches",
    # Runtime mode controls
    "get_backward_mode",
    "set_backward_mode",
    "get_compute_depth",
    "set_compute_depth",
    "get_binning_sort_mode",
    "set_binning_sort_mode",
    "get_compute_flow_aux",
    "set_compute_flow_aux",
    "get_flow_topk",
    "set_flow_topk",
]
