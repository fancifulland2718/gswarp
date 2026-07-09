from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ...._tuning import (
    normalize_device as _normalize_runtime_device,
    query_device_info as _query_runtime_device_info,
    query_sm_properties as _query_sm_properties,
    register_kernel_class as _register_kernel_class,
    get_tuned_block_dim,
    initialize_tuning as _tuning_initialize,
    FAMILY_COMPUTE,
    FAMILY_WARP_SPECIALIZED,
)

from .constants import *
from .state import *
from . import runtime as _runtime
from .memory import *
from .packing import *
from .preprocess_ops import _compute_cov3d_from_scale_rotation_warp, _compute_rgb_from_sh_warp, preprocess_gaussians
from .binning_ops import _build_binning_state
from .backward_ops import (
    _backward_rgb_from_sh_warp,
    _backward_render_tiles_warp,
    _backward_projected_means_warp,
    _backward_cov2d_warp,
    _backward_cov3d_from_scale_rotation_warp,
)
from .backward_kernels import *
from .flow_kernels import *

_COMPUTE_FLOW_AUX = __import__("os").environ.get("GSWARP_COMPUTE_FLOW_AUX", "1") != "0"
_FLOW_TOPK = int(__import__("os").environ.get("GSWARP_FLOW_TOPK", "20"))

def get_compute_flow_aux() -> bool:
    return _COMPUTE_FLOW_AUX

def set_compute_flow_aux(enabled: bool) -> None:
    global _COMPUTE_FLOW_AUX
    _COMPUTE_FLOW_AUX = bool(enabled)

def get_flow_topk() -> int:
    return _FLOW_TOPK

def set_flow_topk(k: int) -> None:
    global _FLOW_TOPK
    if int(k) <= 0:
        raise ValueError("flow_topk must be positive")
    _FLOW_TOPK = int(k)


def _make_empty_forward_outputs(means3D, image_height, image_width):
    point_count = means3D.shape[0]
    out_color = _allocate_scalar_tensor((NUM_CHANNELS, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    out_depth = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    out_alpha = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    radii = _allocate_scalar_tensor((point_count,), torch.int32, means3D.device, fill_value=0)
    proj_2d = _allocate_scalar_tensor((point_count, 2), torch.float32, means3D.device, fill_value=0.0)
    conic_2d = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
    conic_2d_inv = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
    if _COMPUTE_FLOW_AUX:
        K = _FLOW_TOPK
        gs_per_pixel = torch.full(
            (K, image_height, image_width), -1, dtype=torch.int32, device=means3D.device
        )
        weight_per_gs_pixel = torch.zeros(
            (K, image_height, image_width), dtype=torch.float32, device=means3D.device
        )
        x_mu = torch.zeros(
            (2, K, image_height, image_width), dtype=torch.float32, device=means3D.device
        )
    else:
        gs_per_pixel = torch.empty((0,), dtype=torch.int32, device=means3D.device)
        weight_per_gs_pixel = torch.empty((0,), dtype=torch.float32, device=means3D.device)
        x_mu = torch.empty((0,), dtype=torch.float32, device=means3D.device)
    geom_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)
    binning_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)
    img_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)

    return (
        0,
        out_color,
        out_depth,
        out_alpha,
        radii,
        geom_buffer,
        binning_buffer,
        img_buffer,
        proj_2d,
        conic_2d,
        conic_2d_inv,
        gs_per_pixel,
        weight_per_gs_pixel,
        x_mu,
    )


def _render_tiles_warp(preprocess_outputs, binning_state, feature_ptr, background, image_height, image_width):
    device = feature_ptr.device
    total_pixels = image_height * image_width
    out_color = torch.empty((NUM_CHANNELS, image_height, image_width), dtype=torch.float32, device=device)
    out_depth = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    out_alpha = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    n_contrib = torch.empty((total_pixels,), dtype=torch.int32, device=device)

    # Flow auxiliary outputs: top-K successful contributors per pixel.
    write_aux_flag = 1 if _COMPUTE_FLOW_AUX else 0
    if write_aux_flag != 0:
        K = _FLOW_TOPK
        gs_per_pixel = torch.full((K, image_height, image_width), -1, dtype=torch.int32, device=device)
        weight_per_gs_pixel = torch.zeros((K, image_height, image_width), dtype=torch.float32, device=device)
        x_mu = torch.zeros((2, K, image_height, image_width), dtype=torch.float32, device=device)
    else:
        K = 0
        gs_per_pixel = torch.empty((0,), dtype=torch.int32, device=device)
        weight_per_gs_pixel = torch.empty((0,), dtype=torch.float32, device=device)
        x_mu = torch.empty((0,), dtype=torch.float32, device=device)

    if binning_state.num_rendered == 0:
        out_color.zero_()
        out_depth.zero_()
        out_alpha.zero_()
        n_contrib.zero_()
        return out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, n_contrib

    # F6: preprocess outputs are already contiguous from torch.empty(); skip redundant .detach().contiguous()
    points_xy_image = preprocess_outputs.points_xy_image
    conic_opacity = preprocess_outputs.conic_opacity
    depths = preprocess_outputs.depths
    feature_ptr = _prep(feature_ptr)
    background = _prep(background.to(dtype=torch.float32, device=device))
    ranges = binning_state.ranges.reshape(-1)
    point_list = binning_state.point_list

    # Warp requires valid (non-null) array bindings even when the kernel's
    # write_aux branch is dormant; use tiny dummy tensors in that case.
    if write_aux_flag != 0:
        _gs_per_pixel_flat = gs_per_pixel.reshape(-1)
        _weight_per_gs_pixel_flat = weight_per_gs_pixel.reshape(-1)
        _x_mu_flat = x_mu.reshape(-1)
    else:
        _gs_per_pixel_flat = torch.empty((1,), dtype=torch.int32, device=device)
        _weight_per_gs_pixel_flat = torch.empty((1,), dtype=torch.float32, device=device)
        _x_mu_flat = torch.empty((1,), dtype=torch.float32, device=device)

    _wp_ranges = wp.from_torch(ranges, dtype=wp.int32)
    _wp_point_list = wp.from_torch(point_list, dtype=wp.int32)
    _wp_points_xy = wp.from_torch(points_xy_image, dtype=wp.vec2)
    _wp_features = wp.from_torch(feature_ptr.reshape(-1), dtype=wp.float32)
    _wp_depths = wp.from_torch(depths, dtype=wp.float32)
    _wp_conic_opacity = wp.from_torch(conic_opacity, dtype=wp.vec4)
    _wp_bg = wp.from_torch(background.reshape(-1), dtype=wp.float32)
    _wp_out_color = wp.from_torch(out_color.reshape(-1), dtype=wp.float32)
    _wp_out_depth = wp.from_torch(out_depth.reshape(-1), dtype=wp.float32)
    _wp_out_alpha = wp.from_torch(out_alpha.reshape(-1), dtype=wp.float32)
    _wp_n_contrib = wp.from_torch(n_contrib, dtype=wp.int32)
    _wp_gs_per_pixel = wp.from_torch(_gs_per_pixel_flat, dtype=wp.int32)
    _wp_weight_per_gs_pixel = wp.from_torch(_weight_per_gs_pixel_flat, dtype=wp.float32)
    _wp_x_mu = wp.from_torch(_x_mu_flat, dtype=wp.float32)
    _compute_depth_flag = int(1 if _runtime._COMPUTE_DEPTH else 0)
    # T1: Tiled-256 cooperative forward render
    _grid_x_fwd = int(binning_state.grid_x)
    _grid_y_fwd = (image_height + BLOCK_Y - 1) // BLOCK_Y
    _num_tiles_fwd = _grid_x_fwd * _grid_y_fwd
    _dim = _num_tiles_fwd * (BLOCK_X * BLOCK_Y)
    _inp = [_wp_ranges, _wp_point_list, _wp_points_xy, _wp_features, _wp_depths,
            _wp_conic_opacity, _wp_bg, int(image_width), int(image_height),
            _grid_x_fwd, _num_tiles_fwd, _compute_depth_flag,
            write_aux_flag, int(K)]
    _out = [_wp_out_color, _wp_out_depth, _wp_out_alpha, _wp_n_contrib,
            _wp_gs_per_pixel, _wp_weight_per_gs_pixel, _wp_x_mu]
    _key = (str(device), _dim, write_aux_flag, int(K))
    _cmd = _C4_LAUNCH_CACHE_FWD_RENDER_TILED256.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_render_tiles_tiled256_warp_kernel, dim=_dim,
                         inputs=_inp, outputs=_out, device=str(device), record_cmd=True,
                         block_dim=256)
        _C4_LAUNCH_CACHE_FWD_RENDER_TILED256[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, n_contrib


def _rasterize_gaussians_backward_python(*args: Any):
        (
            _background,
            means3D,
            _radii,
            _colors,
            _opacities,
            _scales,
            _rotations,
            _scale_modifier,
            _cov3D_precomp,
            _viewmatrix,
            _projmatrix,
            _tan_fovx,
            _tan_fovy,
            grad_color,
            grad_depth,
            grad_alpha,
            grad_proj_2D,
            grad_conic_2D,
            _grad_conic_2D_inv,
            _dummy_gs_per_pixel,
            _dummy_weight_per_gs_pixel,
            _grad_x_mu,
            _sh,
            _degree,
            _campos,
            _geomBuffer,
            _num_rendered,
            _binningBuffer,
            _imgBuffer,
            _alphas,
            _enable_flow_grad,
        ) = args

        point_count = means3D.shape[0]
        device = means3D.device
        # C3: grad_means2D and grad_means3D allocated later by fused accumulation kernel
        # O3: empty_like for grads that are always overwritten before return
        grad_colors = torch.empty_like(_colors)
        grad_opacities = torch.empty_like(_opacities)
        grad_cov3D = torch.empty_like(_cov3D_precomp)
        # L1: grad_scales/grad_rotations deferred to branch interiors (saves ~7 MB @262K on fused path)
        if point_count == 0:
            grad_means2D = torch.zeros((point_count, 3), dtype=torch.float32, device=device)
            grad_means3D = torch.zeros_like(means3D)
            grad_sh = torch.empty_like(_sh)
            grad_scales = torch.zeros_like(_scales)
            grad_rotations = torch.zeros_like(_rotations)
            return grad_means2D, grad_colors, grad_opacities, grad_means3D, grad_cov3D, grad_sh, grad_scales, grad_rotations

        image_height = grad_color.shape[1]
        image_width = grad_color.shape[2]
        cached_forward_state = _unpack_forward_aux_buffers(_geomBuffer, _binningBuffer, _imgBuffer, _num_rendered, image_height, image_width)

        if cached_forward_state is not None:
            preprocess_outputs, binning_state, n_contrib = cached_forward_state
            # Shallow copy to avoid mutating the cached dict
            preprocess_outputs.radii = _radii
            cov3d_all = preprocess_outputs.cov3d_all
        else:
            # Only the cov3d source differs between precomp and scale/rotation paths
            if _cov3D_precomp.numel() != 0:
                cov3d_all = _cov3D_precomp
            else:
                cov3d_all = _compute_cov3d_from_scale_rotation_warp(_scales, _scale_modifier, _rotations)
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    cov3D_precomp=cov3d_all,
                    shs=_sh.reshape(point_count, -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacities,
                    prefiltered=False,
                )
            # Build binning unconditionally 鈥?num_rendered (Python int) avoids
            # the host sync that bool((radii > 0).any()) would trigger.
            binning_state = _build_binning_state(preprocess_outputs, image_height, image_width)
            # Recover n_contrib from the saved img buffer instead of re-running
            # the full forward render (which allocates unused per-pixel outputs).
            _expected_img_bytes = image_height * image_width * 4  # int32 element_size
            if _imgBuffer.numel() == _expected_img_bytes:
                n_contrib = _imgBuffer.view(torch.int32).reshape(image_height, image_width)
            else:
                n_contrib = None

        # Unified: preprocess_gaussians already stores the correct rgb
        # (colors_precomp when provided, SH-evaluated colors otherwise).
        feature_ptr = preprocess_outputs.rgb
        # Always use the saved alpha tensor from forward.
        render_alpha = _alphas
        background_float = _background.to(dtype=torch.float32)
        # num_rendered is a Python int 鈥?no device sync.
        has_active_points = binning_state.num_rendered != 0

        if has_active_points:
            # Fallback: if n_contrib could not be recovered, re-render.
            if n_contrib is None:
                _, _, _, _, _, _, n_contrib = _render_tiles_warp(
                        preprocess_outputs,
                        binning_state,
                        feature_ptr,
                        background_float,
                        image_height,
                        image_width,
                    )
            render_grad_points, render_grad_depths, render_grad_conic_opacity, render_grad_feature = _backward_render_tiles_warp(
                preprocess_outputs,
                binning_state,
                feature_ptr,
                background_float,
                image_height,
                image_width,
                render_alpha,
                n_contrib.reshape(image_height, image_width),
                grad_color,
                grad_depth,
                grad_alpha,
            )
        else:
            render_grad_points = torch.zeros((point_count, 2), dtype=torch.float32, device=device)
            render_grad_depths = torch.zeros((point_count,), dtype=torch.float32, device=device)
            render_grad_conic_opacity = torch.zeros((point_count, 4), dtype=torch.float32, device=device)
            render_grad_feature = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=device)

        if _enable_flow_grad:
                    grad_proj_2d_active = torch.empty((point_count, 2), dtype=torch.float32, device=device)
                    grad_conic_2d_active = torch.empty((point_count, 3), dtype=torch.float32, device=device)
                    # C4: cache wp.from_torch + Launch object
                    _fg_inp = [
                        wp.from_torch(grad_proj_2D.contiguous(), dtype=wp.vec2),
                        wp.from_torch(render_grad_points.contiguous(), dtype=wp.vec2),
                        wp.from_torch(grad_conic_2D.contiguous(), dtype=wp.vec3),
                        wp.from_torch(render_grad_conic_opacity.contiguous(), dtype=wp.vec4),
                    ]
                    _fg_out = [
                        wp.from_torch(grad_proj_2d_active, dtype=wp.vec2),
                        wp.from_torch(grad_conic_2d_active, dtype=wp.vec3),
                    ]
                    _fg_key = (str(device), point_count)
                    _fg_cmd = _C4_LAUNCH_CACHE_FLOW_GRAD.get(_fg_key)
                    if _fg_cmd is None:
                        _fg_cmd = wp.launch(kernel=_fused_flow_grad_prep_warp_kernel, dim=point_count,
                                            inputs=_fg_inp, outputs=_fg_out, device=str(device), record_cmd=True)
                        _C4_LAUNCH_CACHE_FLOW_GRAD[_fg_key] = _fg_cmd
                    else:
                        for _i, _v in enumerate(_fg_inp + _fg_out):
                            _fg_cmd.set_param_at_index(_i, _v)
                    _fg_cmd.launch()
        else:
                    grad_proj_2d_active = grad_proj_2D
                    grad_conic_2d_active = grad_conic_2D

        focal_x = image_width / (2.0 * _tan_fovx)
        focal_y = image_height / (2.0 * _tan_fovy)

        # C6: Fuse cov2d + cov3d into single kernel when scales/rotations are available
        _use_fused_cov = (_cov3D_precomp.numel() == 0
                          and _scales.numel() != 0
                          and _rotations.numel() != 0)

        # opacity / color grads (independent of preprocess backward)
        if grad_opacities.numel() != 0:
            grad_opacities = render_grad_conic_opacity[:, 3:4]
        if grad_colors.numel() != 0:
            grad_colors = render_grad_feature.reshape_as(_colors)
        # SH backward (must run before E2 fused kernel)
        _has_sh = _sh.numel() != 0
        if _has_sh:
                _grad_mean_sh, grad_sh_local = _backward_rgb_from_sh_warp(
                    means3D,
                    _campos.to(device=device, dtype=torch.float32),
                    _sh.reshape(point_count, -1, NUM_CHANNELS),
                    _degree,
                    preprocess_outputs.clamped,
                    render_grad_feature,
                )
                grad_sh = grad_sh_local.reshape_as(_sh)
        else:
                grad_sh = torch.empty_like(_sh)

        if _use_fused_cov:
                grad_means3D = torch.empty(means3D.shape, dtype=means3D.dtype, device=device)
                grad_means2D = torch.empty((point_count, 3), dtype=torch.float32, device=device)
                grad_scales = torch.empty(_scales.shape, dtype=_scales.dtype, device=device)
                grad_rotations = torch.empty(_rotations.shape, dtype=_rotations.dtype, device=device)
                # L3: when has_sh=False the kernel never reads grad_sh_means 鈥?reuse means3D to avoid zero alloc
                _sh_grad = _grad_mean_sh if _has_sh else means3D
                _proj_flat = _projmatrix.contiguous().reshape(-1)
                _view_flat = _viewmatrix.contiguous().reshape(-1)
                _grad_conic_2d_flat = grad_conic_2d_active.contiguous().reshape(-1)
                _cov3d_flat = cov3d_all.contiguous().reshape(-1)
                _e2_inp = [
                    wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
                    wp.from_torch(preprocess_outputs.radii.contiguous(), dtype=wp.int32),
                    wp.from_torch(_proj_flat, dtype=wp.float32),
                    wp.from_torch(_view_flat, dtype=wp.float32),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                    wp.from_torch(grad_proj_2d_active.contiguous(), dtype=wp.vec2),
                    wp.from_torch(render_grad_depths, dtype=wp.float32),
                    wp.from_torch(_cov3d_flat, dtype=wp.float32),
                    float(_tan_fovx),
                    float(_tan_fovy),
                    float(focal_x),
                    float(focal_y),
                    wp.from_torch(render_grad_conic_opacity, dtype=wp.vec4),
                    wp.from_torch(_grad_conic_2d_flat, dtype=wp.float32),
                    wp.from_torch(_scales.contiguous(), dtype=wp.vec3),
                    wp.from_torch(_rotations.contiguous(), dtype=wp.vec4),
                    float(_scale_modifier),
                    wp.from_torch(_sh_grad, dtype=wp.vec3),
                    int(_has_sh),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                ]
                _e2_out = [
                    wp.from_torch(grad_means3D, dtype=wp.vec3),
                    wp.from_torch(grad_means2D, dtype=wp.vec3),
                    wp.from_torch(grad_scales, dtype=wp.vec3),
                    wp.from_torch(grad_rotations, dtype=wp.vec4),
                ]
                _e2_key = (str(device), point_count, int(_has_sh))
                _e2_cmd = _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS.get(_e2_key)
                if _e2_cmd is None:
                    _e2_cmd = wp.launch(kernel=_fused_backward_preprocess_accumulate_warp_kernel, dim=point_count,
                                        inputs=_e2_inp, outputs=_e2_out, device=str(device), record_cmd=True,
                                        block_dim=get_tuned_block_dim("backward_preprocess", device))
                    _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS[_e2_key] = _e2_cmd
                else:
                    for _i, _v in enumerate(_e2_inp + _e2_out):
                        _e2_cmd.set_param_at_index(_i, _v)
                _e2_cmd.launch()
        else:
                _grad_projected = _backward_projected_means_warp(
                    means3D,
                    preprocess_outputs.radii,
                    _projmatrix,
                    _viewmatrix,
                    render_grad_points,
                    grad_proj_2d_active,
                    render_grad_depths,
                )
                _grad_means_cov, grad_cov_from_cov2d = _backward_cov2d_warp(
                    means3D,
                    preprocess_outputs.radii,
                    cov3d_all,
                    _viewmatrix,
                    _tan_fovx,
                    _tan_fovy,
                    focal_x,
                    focal_y,
                    render_grad_conic_opacity[:, :3],
                    grad_conic_2d_active,
                )

                # C3: Fused accumulation 鈥?replaces 2 torch.zeros + 3 torch.add + 1 slice-assign
                grad_means3D = torch.empty(means3D.shape, dtype=means3D.dtype, device=device)
                grad_means2D = torch.empty((point_count, 3), dtype=torch.float32, device=device)
                _sh_grad_input = _grad_mean_sh if _has_sh else _grad_projected
                # C4: cache wp.from_torch + Launch object
                _acc_inp = [
                    wp.from_torch(_grad_projected, dtype=wp.vec3),
                    wp.from_torch(_grad_means_cov, dtype=wp.vec3),
                    wp.from_torch(_sh_grad_input, dtype=wp.vec3),
                    int(_has_sh),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                ]
                _acc_out = [
                    wp.from_torch(grad_means3D, dtype=wp.vec3),
                    wp.from_torch(grad_means2D, dtype=wp.vec3),
                ]
                _acc_key = (str(device), point_count, int(_has_sh))
                _acc_cmd = _C4_LAUNCH_CACHE_ACCUM.get(_acc_key)
                if _acc_cmd is None:
                    _acc_cmd = wp.launch(kernel=_fused_backward_accumulate_warp_kernel, dim=point_count,
                                         inputs=_acc_inp, outputs=_acc_out, device=str(device), record_cmd=True)
                    _C4_LAUNCH_CACHE_ACCUM[_acc_key] = _acc_cmd
                else:
                    for _i, _v in enumerate(_acc_inp + _acc_out):
                        _acc_cmd.set_param_at_index(_i, _v)
                _acc_cmd.launch()

                # L1: allocate grad_scales/grad_rotations per-branch
                if _cov3D_precomp.numel() != 0:
                    grad_cov3D = grad_cov_from_cov2d
                    grad_scales = torch.zeros_like(_scales)
                    grad_rotations = torch.zeros_like(_rotations)
                elif _scales.numel() != 0 and _rotations.numel() != 0:
                    grad_scales, grad_rotations = _backward_cov3d_from_scale_rotation_warp(
                        _scales,
                        _scale_modifier,
                        _rotations,
                        grad_cov_from_cov2d,
                    )
                else:
                    grad_scales = torch.zeros_like(_scales)
                    grad_rotations = torch.zeros_like(_rotations)

        return grad_means2D, grad_colors, grad_opacities, grad_means3D, grad_cov3D, grad_sh, grad_scales, grad_rotations


def rasterize_gaussians_flow_backward(*args: Any):
    _runtime._require_warp()
    return _rasterize_gaussians_backward_python(*args)


render_gaussians_flow = _render_tiles_warp

__all__ = [
    "render_gaussians_flow",
    "rasterize_gaussians_flow_backward",
    "get_compute_flow_aux",
    "set_compute_flow_aux",
    "get_flow_topk",
    "set_flow_topk",
    "_make_empty_forward_outputs",
]
