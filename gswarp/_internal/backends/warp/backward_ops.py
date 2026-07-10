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
from .render_ops import _render_tiles_warp
from .backward_kernels import *

def _backward_rgb_from_sh_warp(means3D, campos, shs, degree, clamped, grad_color):
    point_count = means3D.shape[0]
    grad_means = torch.empty(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    # S1: grad_sh allocated per-branch to avoid peak overlap
    if point_count == 0 or shs.numel() == 0:
        grad_sh = torch.zeros(shs.shape, dtype=shs.dtype, device=shs.device)
        return grad_means, grad_sh

    coeff_count = shs.shape[1]
    # L2: skip conversion when clamped already int32 (from geom_buffer unpack or SH forward)
    clamped_i32 = clamped.contiguous() if clamped.dtype == torch.int32 else clamped.to(torch.int32).contiguous()
    _dev = str(means3D.device)

    _w_means = wp.from_torch(means3D.contiguous(), dtype=wp.vec3)
    _w_campos = wp.from_torch(campos.contiguous().reshape(-1), dtype=wp.float32)
    _w_grad_means = wp.from_torch(grad_means, dtype=wp.vec3)

    if degree <= 1:
        # degree 0-1: use monolithic kernel (scalar arrays, no deg 2-3 work)
        grad_sh = torch.zeros(shs.shape, dtype=shs.dtype, device=shs.device)
        _w_shs = wp.from_torch(shs.contiguous().reshape(-1), dtype=wp.float32)
        _w_clamped = wp.from_torch(clamped_i32.reshape(-1), dtype=wp.int32)
        _w_grad_color = wp.from_torch(grad_color.contiguous().reshape(-1), dtype=wp.float32)
        _w_grad_sh = wp.from_torch(grad_sh.reshape(-1), dtype=wp.float32)
        _inp = [_w_means, _w_campos, _w_shs, int(degree), int(coeff_count), _w_clamped, _w_grad_color]
        _out = [_w_grad_means, _w_grad_sh]
        _key = (_dev, point_count)
        _cmd = _C4_LAUNCH_CACHE_SH.get(_key)
        if _cmd is None:
            _cmd = wp.launch(kernel=_backward_rgb_from_sh_warp_kernel, dim=point_count,
                             inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
            _C4_LAUNCH_CACHE_SH[_key] = _cmd
        else:
            for _i, _v in enumerate(_inp + _out):
                _cmd.set_param_at_index(_i, _v)
        _cmd.launch()
    else:
        # SoA layout for coalesced GPU access: (P, K, 3) 鈫?(K, P, 3)
        # Adjacent threads access adjacent vec3s 鈫?eliminates 86% excessive sectors
        _w_shs_v3 = wp.from_torch(shs.permute(1, 0, 2).contiguous().reshape(-1, 3), dtype=wp.vec3)
        grad_sh_soa = torch.zeros((coeff_count, point_count, 3), dtype=shs.dtype, device=means3D.device)
        _w_grad_sh_v3 = wp.from_torch(grad_sh_soa.reshape(-1, 3), dtype=wp.vec3)
        # M2: pass grad_color + clamped directly to kernel (avoids _masked_grad allocation)
        _w_grad_color_v3 = wp.from_torch(grad_color.contiguous().reshape(-1, 3), dtype=wp.vec3)
        _w_clamped_i32 = wp.from_torch(clamped_i32.reshape(-1), dtype=wp.int32)
        _key = (_dev, point_count)

        _inp = [_w_means, _w_campos, _w_shs_v3, int(degree), int(coeff_count), int(point_count), _w_grad_color_v3, _w_clamped_i32]
        _out = [_w_grad_means, _w_grad_sh_v3]
        _cmd = _C4_LAUNCH_CACHE_SH_V3.get(_key)
        if _cmd is None:
            _cmd = wp.launch(kernel=_backward_rgb_from_sh_v3_warp_kernel, dim=point_count,
                             inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
            _C4_LAUNCH_CACHE_SH_V3[_key] = _cmd
        else:
            for _i, _v in enumerate(_inp + _out):
                _cmd.set_param_at_index(_i, _v)
        _cmd.launch()
        # Transpose grad_sh from SoA (K, P, 3) back to AoS (P, K, 3)
        grad_sh = grad_sh_soa.permute(1, 0, 2).contiguous()

    return grad_means, grad_sh


def _backward_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations, grad_cov3d):
    if scales.numel() == 0 or rotations.numel() == 0:
        return torch.zeros_like(scales), torch.zeros_like(rotations)

    grad_scales = torch.empty(scales.shape, dtype=scales.dtype, device=scales.device)
    grad_rot = torch.empty(rotations.shape, dtype=rotations.dtype, device=rotations.device)
    N = scales.shape[0]
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(scales.contiguous(), dtype=wp.vec3),
        wp.from_torch(rotations.contiguous(), dtype=wp.vec4),
        float(scale_modifier),
        wp.from_torch(grad_cov3d.contiguous().reshape(-1), dtype=wp.float32),
    ]
    _out = [
        wp.from_torch(grad_scales, dtype=wp.vec3),
        wp.from_torch(grad_rot, dtype=wp.vec4),
    ]
    _key = (str(scales.device), N)
    _cmd = _C4_LAUNCH_CACHE_COV3D.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_cov3d_from_scale_rotation_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(scales.device), record_cmd=True)
        _C4_LAUNCH_CACHE_COV3D[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return grad_scales, grad_rot


def _backward_render_tiles_warp(
    preprocess_outputs,
    binning_state,
    feature_ptr,
    background,
    image_height,
    image_width,
    out_alpha,
    n_contrib,
    grad_color,
    grad_depth,
    grad_alpha,
):
    device = feature_ptr.device
    point_count = feature_ptr.shape[0]

    # C3: Combined allocation 鈥?1 memset instead of 4
    _stride = 2 + 1 + 4 + NUM_CHANNELS  # 10
    _bwd_total = point_count * _stride
    _combined = torch.zeros(_bwd_total, dtype=torch.float32, device=device)
    _off = 0
    grad_points_xy = _combined[_off:_off + point_count * 2].reshape(point_count, 2); _off += point_count * 2
    grad_depths = _combined[_off:_off + point_count]; _off += point_count
    grad_conic_opacity = _combined[_off:_off + point_count * 4].reshape(point_count, 4); _off += point_count * 4
    grad_feature = _combined[_off:_off + point_count * NUM_CHANNELS].reshape(point_count, NUM_CHANNELS)

    if binning_state.num_rendered == 0:
        outputs = (grad_points_xy, grad_depths, grad_conic_opacity, grad_feature)
        return outputs

    # C5: preprocess outputs are already contiguous (torch.empty/zeros); skip .detach().contiguous()
    points_xy_image = preprocess_outputs.points_xy_image
    conic_opacity = preprocess_outputs.conic_opacity
    depths = preprocess_outputs.depths
    feature_ptr = feature_ptr.contiguous()
    background = background.to(dtype=torch.float32, device=device)
    ranges = binning_state.ranges.reshape(-1)
    point_list = binning_state.point_list
    out_alpha = out_alpha.reshape(-1)
    n_contrib = n_contrib.to(dtype=torch.int32, device=device).reshape(-1)

    grad_color = grad_color.contiguous().reshape(-1)
    grad_depth = grad_depth.contiguous().reshape(-1)
    grad_alpha = grad_alpha.contiguous().reshape(-1)

    # W1: Smart dispatch 鈥?warp32 for high density, non-tiled for low density
    _grid_x = int(binning_state.grid_x)
    _grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    _num_tiles = _grid_x * _grid_y
    _num_rendered = int(binning_state.num_rendered)

    _wp_ranges = wp.from_torch(ranges, dtype=wp.int32)
    _wp_point_list = wp.from_torch(point_list, dtype=wp.int32)
    _wp_points_xy = wp.from_torch(points_xy_image, dtype=wp.vec2)
    _wp_features = wp.from_torch(feature_ptr.reshape(-1), dtype=wp.float32)
    _wp_depths = wp.from_torch(depths, dtype=wp.float32)
    _wp_conic_opacity = wp.from_torch(conic_opacity, dtype=wp.vec4)
    _wp_bg = wp.from_torch(background.reshape(-1), dtype=wp.float32)
    _wp_out_alpha = wp.from_torch(out_alpha, dtype=wp.float32)
    _wp_n_contrib = wp.from_torch(n_contrib, dtype=wp.int32)
    _wp_grad_color = wp.from_torch(grad_color, dtype=wp.float32)
    _wp_grad_depth = wp.from_torch(grad_depth, dtype=wp.float32)
    _wp_grad_alpha = wp.from_torch(grad_alpha, dtype=wp.float32)
    _wp_grad_xy = wp.from_torch(grad_points_xy, dtype=wp.vec2)
    _wp_grad_d = wp.from_torch(grad_depths, dtype=wp.float32)
    _wp_grad_co = wp.from_torch(grad_conic_opacity, dtype=wp.vec4)
    _wp_grad_f = wp.from_torch(grad_feature, dtype=wp.vec3)
    _compute_depth_flag = int(1 if _runtime.get_active_compute_depth() else 0)

    _dim = _num_tiles * (BLOCK_X * BLOCK_Y)
    # W1: warp32 backward 鈥?warp-level tile_reduce gives 32脳 fewer atomicAdd
    _inp = [_wp_ranges, _wp_point_list, _wp_points_xy, _wp_features, _wp_depths,
            _wp_conic_opacity, _wp_bg, _wp_out_alpha, _wp_n_contrib,
            _wp_grad_color, _wp_grad_depth, _wp_grad_alpha,
            int(image_width), int(image_height), _grid_x, _num_tiles, _compute_depth_flag]
    _out = [_wp_grad_xy, _wp_grad_d, _wp_grad_co, _wp_grad_f]
    _key = ("w32", str(device), _dim)
    _cmd = _C4_LAUNCH_CACHE_RENDER_BWD.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_render_tiles_warp32_kernel, dim=_dim,
                         inputs=_inp, outputs=_out, device=str(device), record_cmd=True,
                         block_dim=32)
        _C4_LAUNCH_CACHE_RENDER_BWD[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    outputs = (grad_points_xy, grad_depths, grad_conic_opacity, grad_feature)
    return outputs


def _backward_projected_means_warp(means3D, radii, projmatrix, viewmatrix,       grad_mean2d, grad_proj_2d, grad_depths):
    if means3D.numel() == 0:
        return torch.zeros_like(means3D)

    out = torch.empty(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    N = means3D.shape[0]
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
        wp.from_torch(radii.contiguous(), dtype=wp.int32),
        wp.from_torch(projmatrix.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(viewmatrix.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(grad_mean2d.contiguous(), dtype=wp.vec2),
        wp.from_torch(grad_proj_2d.contiguous(), dtype=wp.vec2),
        wp.from_torch(grad_depths.contiguous(), dtype=wp.float32),
    ]
    _out = [wp.from_torch(out, dtype=wp.vec3)]
    _key = (str(means3D.device), N)
    _cmd = _C4_LAUNCH_CACHE_PROJ_MEANS.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_projected_means_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(means3D.device), record_cmd=True)
        _C4_LAUNCH_CACHE_PROJ_MEANS[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return out


def _backward_cov2d_warp(
    means3D,
    radii,
    cov3D,
    viewmatrix,
    tanfovx,
    tanfovy,
    focal_x,
    focal_y,
    grad_conic,
    grad_conic_2d,
    grad_conic_2d_inv,
):
    grad_means = torch.zeros(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    grad_cov = torch.zeros(cov3D.shape, dtype=cov3D.dtype, device=cov3D.device)
    if means3D.numel() == 0:
        return grad_means, grad_cov

    N = means3D.shape[0]
    viewmatrix = viewmatrix.contiguous()
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
        wp.from_torch(radii.contiguous(), dtype=wp.int32),
        wp.from_torch(cov3D.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(viewmatrix.reshape(-1), dtype=wp.float32),
        float(tanfovx),
        float(tanfovy),
        float(focal_x),
        float(focal_y),
        wp.from_torch(grad_conic.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(grad_conic_2d.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(grad_conic_2d_inv.contiguous().reshape(-1), dtype=wp.float32),
    ]
    _out = [
        wp.from_torch(grad_means, dtype=wp.vec3),
        wp.from_torch(grad_cov.reshape(-1), dtype=wp.float32),
    ]
    _key = (str(means3D.device), N)
    _cmd = _C4_LAUNCH_CACHE_COV2D.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_cov2d_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(means3D.device), record_cmd=True)
        _C4_LAUNCH_CACHE_COV2D[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return grad_means, grad_cov


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
            grad_conic_2D_inv,
            _sh,
            _degree,
            _campos,
            _geomBuffer,
            _num_rendered,
            _binningBuffer,
            _imgBuffer,
            _alphas,
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
                _, _, _, n_contrib = _render_tiles_warp(
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


        grad_proj_2d_active = grad_proj_2D
        grad_conic_2d_active = grad_conic_2D
        grad_conic_2d_inv_active = grad_conic_2D_inv

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
                _grad_conic_2d_inv_flat = grad_conic_2d_inv_active.contiguous().reshape(-1)
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
                    wp.from_torch(_grad_conic_2d_inv_flat, dtype=wp.float32),
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
                    grad_conic_2d_inv_active,
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


def rasterize_gaussians_backward(*args: Any):
    _runtime._require_warp()
    return _rasterize_gaussians_backward_python(*args)


__all__ = ["rasterize_gaussians_backward", "_rasterize_gaussians_backward_python"]
