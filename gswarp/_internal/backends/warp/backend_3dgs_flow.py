from __future__ import annotations

from typing import Any

import torch

from .constants import NUM_CHANNELS
from . import runtime as _runtime
from .runtime import (
    get_default_parameter_info,
    get_runtime_auto_tuning_config,
    set_runtime_auto_tuning,
    get_compute_depth,
    set_compute_depth,
    initialize_runtime_tuning,
    get_runtime_tuning_report,
    is_available,
    get_backward_mode,
    set_backward_mode,
    get_binning_sort_mode,
    set_binning_sort_mode,
)
from .memory import clear_common_warp_caches, clear_flow_warp_caches
from .packing import _pack_forward_aux_buffers
from .preprocess_ops import preprocess_gaussians, mark_visible, reset_preprocess_tracking
from .binning_ops import _build_binning_state
from .flow_ops import (
    _make_empty_forward_outputs as flow_make_empty_forward_outputs,
    render_gaussians_flow,
    rasterize_gaussians_flow_backward as rasterize_gaussians_backward,
    get_compute_flow_aux,
    set_compute_flow_aux,
    get_flow_topk,
    set_flow_topk,
)


def clear_warp_caches() -> None:
    clear_common_warp_caches()
    clear_flow_warp_caches()
    reset_preprocess_tracking()


def rasterize_gaussians(*args: Any):
        _runtime._require_warp()
        (
            _background,
            means3D,
            _colors,
            _opacity,
            _scales,
            _rotations,
            _scale_modifier,
            _cov3D_precomp,
            _viewmatrix,
            _projmatrix,
            _tan_fovx,
            _tan_fovy,
            image_height,
            image_width,
            _sh,
            _degree,
            _campos,
            _prefiltered,
        ) = args

        if means3D.ndim != 2 or means3D.shape[1] != 3:
            raise ValueError("means3D must have dimensions (num_points, 3)")

        if means3D.shape[0] == 0:
            return flow_make_empty_forward_outputs(means3D, image_height, image_width)

        feature_ptr = None

        if _cov3D_precomp.numel() != 0:
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    cov3D_precomp=_cov3D_precomp,
                    shs=_sh.reshape(means3D.shape[0], -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacity,
                    prefiltered=_prefiltered,
                )
            feature_ptr = _colors.reshape(means3D.shape[0], NUM_CHANNELS).to(dtype=torch.float32) if _colors.numel() != 0 else preprocess_outputs.rgb

        if _cov3D_precomp.numel() == 0 and _scales.numel() != 0 and _rotations.numel() != 0:
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    scales=_scales,
                    rotations=_rotations,
                    scale_modifier=_scale_modifier,
                    shs=_sh.reshape(means3D.shape[0], -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacity,
                    prefiltered=_prefiltered,
                )
            feature_ptr = _colors.reshape(means3D.shape[0], NUM_CHANNELS).to(dtype=torch.float32) if _colors.numel() != 0 else preprocess_outputs.rgb

        binning_state = _build_binning_state(preprocess_outputs, image_height, image_width)
        out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, _n_contrib = render_gaussians_flow(
                preprocess_outputs,
                binning_state,
                feature_ptr,
                _background.to(dtype=torch.float32),
                image_height,
                image_width,
            )

        geom_buffer, binning_buffer, img_buffer = _pack_forward_aux_buffers(preprocess_outputs, binning_state, _n_contrib)
        return (
            binning_state.num_rendered,
            out_color,
            out_depth,
            out_alpha,
            preprocess_outputs.radii,
            geom_buffer,
            binning_buffer,
            img_buffer,
            preprocess_outputs.proj_2d,
            preprocess_outputs.conic_2d,
            preprocess_outputs.conic_2d_inv,
            gs_per_pixel,
            weight_per_gs_pixel,
            x_mu,
        )


__all__ = [
    "clear_warp_caches",
    "get_default_parameter_info",
    "is_available",
    "get_backward_mode",
    "set_backward_mode",
    "get_binning_sort_mode",
    "set_binning_sort_mode",
    "get_runtime_auto_tuning_config",
    "get_runtime_tuning_report",
    "initialize_runtime_tuning",
    "preprocess_gaussians",
    "rasterize_gaussians",
    "mark_visible",
    "rasterize_gaussians_backward",
    "set_runtime_auto_tuning",
    "get_compute_depth",
    "set_compute_depth",
    "get_compute_flow_aux",
    "set_compute_flow_aux",
    "get_flow_topk",
    "set_flow_topk",
]
