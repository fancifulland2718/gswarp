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
from .memory import (
    clear_common_warp_caches,
    clear_flow_warp_caches,
    get_warp_cache_report,
)
from .packing import _pack_forward_aux_buffers
from .preprocess_ops import (
    _make_empty_forward_outputs,
    feature_3dgs_stage,
    mark_visible,
    preprocess_3dgs_stage,
    preprocess_gaussians,
)
from .binning_ops import _build_binning_state
from .render_ops import _render_tiles_warp
from .backward_ops import rasterize_gaussians_backward, rasterize_gaussians_backward_typed
from .state import ForwardResult, ForwardState, RenderStageResult, StandardBackwardInterop

BACKEND_CAPABILITIES = frozenset({"stable_warp", "typed_forward", "typed_backward", "mark_visible"})


def empty_forward_stage(inputs) -> ForwardResult:
    outputs = _make_empty_forward_outputs(
        inputs.means3d, inputs.background, inputs.image_height, inputs.image_width
    )
    return ForwardResult(
        num_rendered=0,
        color=outputs[1],
        depth=outputs[2],
        alpha=outputs[3],
        radii=outputs[4],
        proj_2d=outputs[8],
        conic_2d=outputs[9],
        conic_2d_inv=outputs[10],
        state=None,
    )


def preprocess_stage(inputs):
    return preprocess_3dgs_stage(inputs, capture_backward_interop=True)


def feature_stage(inputs, preprocess_outputs):
    return feature_3dgs_stage(inputs, preprocess_outputs)


def render_stage(inputs, preprocess_outputs, binning_state, features) -> RenderStageResult:
    color, depth, alpha, n_contrib, backward_interop = _render_tiles_warp(
        preprocess_outputs,
        binning_state,
        features,
        inputs.background.to(dtype=torch.float32),
        inputs.image_height,
        inputs.image_width,
        capture_backward_interop=True,
    )
    return RenderStageResult(
        color=color,
        depth=depth,
        alpha=alpha,
        n_contrib=n_contrib,
        backward_interop=backward_interop,
    )


def build_state_stage(preprocess_outputs, binning_state, render_result):
    return ForwardState(
        preprocess_outputs,
        binning_state,
        render_result.n_contrib,
        StandardBackwardInterop(
            render=render_result.backward_interop,
            preprocess=preprocess_outputs.backward_interop,
        ),
    )


def clear_warp_caches() -> None:
    from gswarp._stream import clear_execution_stream_cache
    from gswarp.fused_ssim import clear_fused_ssim_caches

    clear_common_warp_caches()
    clear_flow_warp_caches()
    clear_fused_ssim_caches()
    clear_execution_stream_cache()


def _rasterize_gaussians(*args: Any, pack_compatibility_state: bool):
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
            outputs = _make_empty_forward_outputs(means3D, _background, image_height, image_width)
            if pack_compatibility_state:
                return outputs
            return ForwardResult(
                num_rendered=0,
                color=outputs[1], depth=outputs[2], alpha=outputs[3], radii=outputs[4],
                proj_2d=outputs[8], conic_2d=outputs[9], conic_2d_inv=outputs[10], state=None,
            )

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
                    capture_backward_interop=not pack_compatibility_state,
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
                    capture_backward_interop=not pack_compatibility_state,
                )
            feature_ptr = _colors.reshape(means3D.shape[0], NUM_CHANNELS).to(dtype=torch.float32) if _colors.numel() != 0 else preprocess_outputs.rgb

        binning_state = _build_binning_state(preprocess_outputs, image_height, image_width)
        (
            out_color,
            out_depth,
            out_alpha,
            _n_contrib,
            render_backward_interop,
        ) = _render_tiles_warp(
                preprocess_outputs,
                binning_state,
                feature_ptr,
                _background.to(dtype=torch.float32),
                image_height,
                image_width,
                capture_backward_interop=not pack_compatibility_state,
            )

        if not pack_compatibility_state:
            return ForwardResult(
                num_rendered=binning_state.num_rendered,
                color=out_color, depth=out_depth, alpha=out_alpha, radii=preprocess_outputs.radii,
                proj_2d=preprocess_outputs.proj_2d, conic_2d=preprocess_outputs.conic_2d,
                conic_2d_inv=preprocess_outputs.conic_2d_inv,
                state=ForwardState(
                    preprocess_outputs,
                    binning_state,
                    _n_contrib,
                    StandardBackwardInterop(
                        render=render_backward_interop,
                        preprocess=preprocess_outputs.backward_interop,
                    ),
                ),
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
        )


def rasterize_gaussians(*args: Any):
    """Raw compatibility entry point retaining opaque packed buffers."""
    return _rasterize_gaussians(*args, pack_compatibility_state=True)


def rasterize_gaussians_typed(*args: Any) -> ForwardResult:
    """Normal frontend entry point with typed forward state and no packing copy."""
    return _rasterize_gaussians(*args, pack_compatibility_state=False)


__all__ = [
    "clear_warp_caches",
    "get_warp_cache_report",
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
    "empty_forward_stage",
    "preprocess_stage",
    "feature_stage",
    "render_stage",
    "build_state_stage",
    "rasterize_gaussians",
    "rasterize_gaussians_typed",
    "mark_visible",
    "rasterize_gaussians_backward",
    "rasterize_gaussians_backward_typed",
    "set_runtime_auto_tuning",
    "get_compute_depth",
    "set_compute_depth",
]
