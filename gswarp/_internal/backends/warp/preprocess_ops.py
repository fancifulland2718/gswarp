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
from ...._stream import set_launch_params, torch_launch_array
from ...coverage import (
    BASELINE_3DGS_COVERAGE,
    resolve_tile_coverage_mode_id,
)

from .constants import BLOCK_X, BLOCK_Y, NUM_CHANNELS
from .state import PreprocessBackwardInterop, PreprocessOutputs
from . import runtime as _runtime
from .memory import (
    _C4_LAUNCH_CACHE_FWD_PREPROCESS,
    _C4_LAUNCH_CACHE_FWD_SH,
    _allocate_scalar_tensor,
    _get_project_visible_buffers,
)
from .packing import _prep
from .preprocess_kernels import (
    _cov2d_preprocess_masked_pack_scale_rotation_warp_kernel,
    _cov2d_preprocess_masked_pack_warp_kernel,
    _cov3d_from_scale_rotation_warp_kernel,
    _forward_rgb_from_sh_v3_warp_kernel,
    _fused_project_cov3d_cov2d_preprocess_sr_warp_kernel,
    _project_preprocess_visible_points_cov_warp_kernel,
    _project_preprocess_visible_points_scale_warp_kernel,
    _project_visible_points_warp_kernel,
)

def _make_empty_forward_outputs(means3D, background, image_height, image_width):
    point_count = means3D.shape[0]
    background = background.to(device=means3D.device, dtype=torch.float32).reshape(NUM_CHANNELS, 1, 1)
    out_color = background.expand(NUM_CHANNELS, image_height, image_width).clone()
    out_depth = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    out_alpha = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    radii = _allocate_scalar_tensor((point_count,), torch.int32, means3D.device, fill_value=0)
    proj_2d = _allocate_scalar_tensor((point_count, 2), torch.float32, means3D.device, fill_value=0.0)
    conic_2d = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
    conic_2d_inv = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
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
    )


def preprocess_3dgs_stage(inputs, *, capture_backward_interop: bool):
    common = dict(
        shs=inputs.sh.reshape(inputs.means3d.shape[0], -1, NUM_CHANNELS)
        if inputs.sh.numel() != 0
        else None,
        degree=inputs.degree,
        campos=inputs.campos,
        colors_precomp=inputs.colors if inputs.colors.numel() != 0 else None,
        opacities=inputs.opacities,
        prefiltered=inputs.prefiltered,
        capture_backward_interop=capture_backward_interop,
    )
    if inputs.cov3d_precomp.numel() != 0:
        return preprocess_gaussians(
            inputs.means3d,
            inputs.viewmatrix,
            inputs.projmatrix,
            inputs.image_height,
            inputs.image_width,
            inputs.tan_fovx,
            inputs.tan_fovy,
            cov3D_precomp=inputs.cov3d_precomp,
            **common,
        )
    if inputs.scales.numel() != 0 and inputs.rotations.numel() != 0:
        return preprocess_gaussians(
            inputs.means3d,
            inputs.viewmatrix,
            inputs.projmatrix,
            inputs.image_height,
            inputs.image_width,
            inputs.tan_fovx,
            inputs.tan_fovy,
            scales=inputs.scales,
            rotations=inputs.rotations,
            scale_modifier=inputs.scale_modifier,
            **common,
        )
    raise ValueError("either cov3Ds_precomp or scales/rotations must be provided")


def feature_3dgs_stage(inputs, preprocess_outputs):
    if inputs.colors.numel() != 0:
        return inputs.colors.reshape(inputs.means3d.shape[0], NUM_CHANNELS).to(
            dtype=torch.float32
        )
    return preprocess_outputs.rgb


def _compute_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations):
    if scales.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32, device=scales.device)

    scales = _prep(scales)
    rotations = _prep(rotations)
    out_cov3d = torch.empty((scales.shape[0], 6), dtype=torch.float32, device=scales.device)
    wp.launch(
        kernel=_cov3d_from_scale_rotation_warp_kernel,
        dim=scales.shape[0],
        inputs=[
            wp.from_torch(scales, dtype=wp.vec3),
            wp.from_torch(rotations, dtype=wp.vec4),
            float(scale_modifier),
        ],
        outputs=[wp.from_torch(out_cov3d.reshape(-1), dtype=wp.float32)],
        device=str(scales.device),
    )
    return out_cov3d


def _project_visible_points_warp(means3D, viewmatrix, projmatrix):
    point_count = means3D.shape[0]
    visible_mask, p_proj, p_view_z = _get_project_visible_buffers(means3D.device, point_count)
    if point_count == 0:
        return visible_mask, p_proj, p_view_z

    wp.launch(
        kernel=_project_visible_points_warp_kernel,
        dim=point_count,
        inputs=[
            wp.from_torch(_prep(means3D), dtype=wp.vec3),
            wp.from_torch(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
            wp.from_torch(_prep(projmatrix).reshape(-1), dtype=wp.float32),
        ],
        outputs=[
            wp.from_torch(visible_mask, dtype=wp.int32),
            wp.from_torch(p_proj, dtype=wp.vec3),
            wp.from_torch(p_view_z, dtype=wp.float32),
        ],
        device=str(means3D.device),
    )
    return visible_mask, p_proj, p_view_z


def _project_preprocess_visible_points_warp(
    means3D,
    viewmatrix,
    projmatrix,
    tanfovx,
    tanfovy,
    image_width,
    image_height,
    grid_x,
    grid_y,
    cov3D_precomp=None,
    scales=None,
    scale_modifier=1.0,
):
    point_count = means3D.shape[0]
    visible_mask, p_proj, p_view_z = _get_project_visible_buffers(means3D.device, point_count)
    if point_count == 0:
        return visible_mask, p_proj, p_view_z

    outputs = [
        wp.from_torch(visible_mask, dtype=wp.int32),
        wp.from_torch(p_proj, dtype=wp.vec3),
        wp.from_torch(p_view_z, dtype=wp.float32),
    ]

    if cov3D_precomp is not None and cov3D_precomp.numel() != 0:
        wp.launch(
            kernel=_project_preprocess_visible_points_cov_warp_kernel,
            dim=point_count,
            inputs=[
                wp.from_torch(_prep(means3D), dtype=wp.vec3),
                wp.from_torch(_prep(cov3D_precomp).reshape(-1), dtype=wp.float32),
                wp.from_torch(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
                wp.from_torch(_prep(projmatrix).reshape(-1), dtype=wp.float32),
                float(tanfovx),
                float(tanfovy),
                int(image_width),
                int(image_height),
                int(grid_x),
                int(grid_y),
            ],
            outputs=outputs,
            device=str(means3D.device),
        )
        return visible_mask, p_proj, p_view_z

    if scales is None or scales.numel() == 0:
        raise ValueError("preprocess visibility requires either cov3D_precomp or scales")

    wp.launch(
        kernel=_project_preprocess_visible_points_scale_warp_kernel,
        dim=point_count,
        inputs=[
            wp.from_torch(_prep(means3D), dtype=wp.vec3),
            wp.from_torch(_prep(scales), dtype=wp.vec3),
            float(scale_modifier),
            wp.from_torch(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
            wp.from_torch(_prep(projmatrix).reshape(-1), dtype=wp.float32),
            float(tanfovx),
            float(tanfovy),
            int(image_width),
            int(image_height),
            int(grid_x),
            int(grid_y),
        ],
        outputs=outputs,
        device=str(means3D.device),
    )
    return visible_mask, p_proj, p_view_z


def _compute_rgb_from_sh_warp(
    means3D,
    campos,
    shs,
    degree,
    *,
    capture_backward_interop=False,
):
    point_count = means3D.shape[0]
    if point_count == 0 or shs.numel() == 0:
        rgb = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=means3D.device)
        clamped_int = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.int32, device=means3D.device)
        return rgb, clamped_int, None

    rgb = torch.empty((point_count, NUM_CHANNELS), dtype=torch.float32, device=means3D.device)
    clamped_int = torch.empty((point_count, NUM_CHANNELS), dtype=torch.int32, device=means3D.device)

    coeff_count = shs.shape[1]
    _dev = str(means3D.device)
    # P2b: vec3 SH forward 鈥?3脳 fewer load/store instructions
    _inp = [
        torch_launch_array(_prep(means3D), dtype=wp.vec3),
        torch_launch_array(_prep(campos).reshape(-1), dtype=wp.float32),
        torch_launch_array(_prep(shs).reshape(-1, 3), dtype=wp.vec3),
        int(degree),
        int(coeff_count),
    ]
    _out = [
        torch_launch_array(rgb.reshape(-1, 3), dtype=wp.vec3),
        torch_launch_array(clamped_int.reshape(-1), dtype=wp.int32),
    ]
    _key = (_dev, point_count)
    _cmd = _C4_LAUNCH_CACHE_FWD_SH.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_forward_rgb_from_sh_v3_warp_kernel, dim=point_count,
                         inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
        _C4_LAUNCH_CACHE_FWD_SH[_key] = _cmd
    else:
        set_launch_params(_cmd, _inp + _out)
    _cmd.launch()
    backward_interop = None
    if capture_backward_interop:
        backward_interop = PreprocessBackwardInterop(
            means3d=_inp[0],
            campos=_inp[1],
            clamped=_out[1],
        )
    return rgb, clamped_int, backward_interop


def preprocess_gaussians(
    means3D,
    viewmatrix,
    projmatrix,
    image_height,
    image_width,
    tanfovx,
    tanfovy,
    cov3D_precomp=None,
    scales=None,
    rotations=None,
    scale_modifier=1.0,
    shs=None,
    degree=0,
    campos=None,
    colors_precomp=None,
    opacities=None,
    prefiltered=False,
    capture_backward_interop=False,
):
        _runtime._require_warp()
        coverage_mode = resolve_tile_coverage_mode_id(
            _runtime.get_active_tile_coverage_mode(),
            BASELINE_3DGS_COVERAGE,
        )
        if means3D.ndim != 2 or means3D.shape[1] != 3:
            raise ValueError("means3D must have dimensions (num_points, 3)")
        if prefiltered:
            raise NotImplementedError("prefiltered=True is not supported in the Warp preprocess path yet.")

        has_precomputed_cov = cov3D_precomp is not None and cov3D_precomp.numel() != 0
        has_scale_rotation = scales is not None and rotations is not None and scales.numel() != 0 and rotations.numel() != 0

        if has_precomputed_cov == has_scale_rotation:
            raise ValueError("Provide exactly one of cov3D_precomp or scales/rotations")

        device = means3D.device
        backward_interop = (
            PreprocessBackwardInterop() if capture_backward_interop else None
        )

        cov3d_all = None
        # E1: for scale_rotation Warp path, defer cov3d to fused kernel
        if has_precomputed_cov:
            if cov3D_precomp.ndim != 2 or cov3D_precomp.shape[1] != 6:
                raise ValueError("cov3D_precomp must have dimensions (num_points, 6)")
            if means3D.shape[0] != cov3D_precomp.shape[0]:
                raise ValueError("means3D and cov3D_precomp must have the same number of points")
            cov3d_all = cov3D_precomp
        else:
            if has_scale_rotation:
                # E1: cov3d computed inside fused kernel; allocate output tensor here
                cov3d_all = torch.empty((means3D.shape[0], 6), dtype=torch.float32, device=device)
            else:
                cov3d_all = _compute_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations)
            if means3D.shape[0] != cov3d_all.shape[0]:
                raise ValueError("means3D and computed covariances must have the same number of points")

        point_count = means3D.shape[0]
        proj_2d = torch.empty((point_count, 2), dtype=torch.float32, device=device)
        conic_2d = torch.empty((point_count, 3), dtype=torch.float32, device=device)
        conic_2d_inv = torch.empty((point_count, 3), dtype=torch.float32, device=device)
        radii = torch.empty((point_count,), dtype=torch.int32, device=device)
        depths = torch.empty((point_count,), dtype=torch.float32, device=device)
        points_xy_image = torch.empty((point_count, 2), dtype=torch.float32, device=device)
        tiles_touched = torch.empty((point_count,), dtype=torch.int32, device=device)
        conic_opacity = torch.empty((point_count, 4), dtype=torch.float32, device=device)
        rgb = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=device)
        clamped = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.bool, device=device)
        if opacities is None:
            opacities = torch.zeros((point_count, 1), dtype=torch.float32, device=device)
        else:
            opacities = opacities.reshape(point_count, -1).to(dtype=torch.float32)
        focal_x = image_width / (2.0 * tanfovx)
        focal_y = image_height / (2.0 * tanfovy)
        grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
        if has_scale_rotation:
          # E1: fused project + cov3d + cov2d 鈥?visibility, cov3d, and all
          # preprocess outputs produced in a single kernel launch.
          visible_mask = torch.empty((point_count,), dtype=torch.int32, device=device)
          if point_count > 0:
              _e1_inp = [
                  torch_launch_array(_prep(means3D), dtype=wp.vec3),
                  torch_launch_array(_prep(scales), dtype=wp.vec3),
                  torch_launch_array(_prep(rotations), dtype=wp.vec4),
                  float(scale_modifier),
                  torch_launch_array(_prep(opacities.reshape(-1)), dtype=wp.float32),
                  torch_launch_array(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
                  torch_launch_array(_prep(projmatrix).reshape(-1), dtype=wp.float32),
                  float(tanfovx),
                  float(tanfovy),
                  float(focal_x),
                  float(focal_y),
                  int(image_width),
                  int(image_height),
                  int(grid_x),
                  int(grid_y),
                  int(coverage_mode),
              ]
              _e1_out = [
                  torch_launch_array(cov3d_all.reshape(-1), dtype=wp.float32),
                  torch_launch_array(visible_mask, dtype=wp.int32),
                  torch_launch_array(depths, dtype=wp.float32),
                  torch_launch_array(radii, dtype=wp.int32),
                  torch_launch_array(proj_2d, dtype=wp.vec2),
                  torch_launch_array(conic_2d, dtype=wp.vec3),
                  torch_launch_array(conic_2d_inv, dtype=wp.vec3),
                  torch_launch_array(points_xy_image, dtype=wp.vec2),
                  torch_launch_array(tiles_touched, dtype=wp.int32),
                  torch_launch_array(conic_opacity, dtype=wp.vec4),
              ]
              _e1_key = (str(device), point_count)
              _e1_cmd = _C4_LAUNCH_CACHE_FWD_PREPROCESS.get(_e1_key)
              if _e1_cmd is None:
                  _e1_cmd = wp.launch(
                      kernel=_fused_project_cov3d_cov2d_preprocess_sr_warp_kernel,
                      dim=point_count,
                      inputs=_e1_inp,
                      outputs=_e1_out,
                      device=str(device),
                      record_cmd=True,
                      block_dim=get_tuned_block_dim("preprocess", device),
                  )
                  _C4_LAUNCH_CACHE_FWD_PREPROCESS[_e1_key] = _e1_cmd
              else:
                  set_launch_params(_e1_cmd, _e1_inp + _e1_out)
              _e1_cmd.launch()
              if backward_interop is not None:
                  backward_interop.means3d = _e1_inp[0]
                  backward_interop.scales = _e1_inp[1]
                  backward_interop.rotations = _e1_inp[2]
                  backward_interop.viewmatrix = _e1_inp[5]
                  backward_interop.projmatrix = _e1_inp[6]
                  backward_interop.cov3d = _e1_out[0]
                  backward_interop.radii = _e1_out[3]
          p_proj_all = None
          p_view_z_all = None
        elif has_precomputed_cov:
            visible_mask, p_proj_all, p_view_z_all = _project_preprocess_visible_points_warp(
                means3D,
                viewmatrix,
                projmatrix,
                tanfovx,
                tanfovy,
                image_width,
                image_height,
                grid_x,
                grid_y,
                cov3D_precomp=cov3d_all,
            )
        else:
            visible_mask, p_proj_all, p_view_z_all = _project_visible_points_warp(means3D, viewmatrix, projmatrix)

        visible = visible_mask.to(torch.bool)

        if cov3d_all is None:
            cov3d_all = torch.zeros((point_count, 6), dtype=torch.float32, device=device)

        # F3: E1 fused path processes all points 鈥?skip GPU鈫扖PU sync
        visible_count = point_count if has_scale_rotation else (int(visible_mask.sum().item()) if point_count > 0 else 0)

        if point_count == 0 or visible_count == 0:
            depths.zero_()
            radii.zero_()
            proj_2d.zero_()
            conic_2d.zero_()
            conic_2d_inv.zero_()
            points_xy_image.zero_()
            tiles_touched.zero_()
            conic_opacity.zero_()
            return PreprocessOutputs(
                visible=visible,
                depths=depths,
                radii=radii,
                proj_2d=proj_2d,
                conic_2d=conic_2d,
                conic_2d_inv=conic_2d_inv,
                points_xy_image=points_xy_image,
                tiles_touched=tiles_touched,
                rgb=rgb,
                clamped=clamped,
                conic_opacity=conic_opacity,
                cov3d_all=cov3d_all.to(dtype=torch.float32),
                backward_interop=backward_interop,
            )

        if has_scale_rotation:
            pass  # E1: preprocess already done by fused kernel
        elif has_precomputed_cov:
            wp.launch(
                kernel=_cov2d_preprocess_masked_pack_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(_prep(visible_mask), dtype=wp.int32),
                    wp.from_torch(_prep(means3D), dtype=wp.vec3),
                    wp.from_torch(_prep(cov3d_all).reshape(-1), dtype=wp.float32),
                    wp.from_torch(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
                    wp.from_torch(_prep(p_proj_all), dtype=wp.vec3),
                    wp.from_torch(_prep(p_view_z_all), dtype=wp.float32),
                    wp.from_torch(_prep(opacities.reshape(-1)), dtype=wp.float32),
                    float(tanfovx),
                    float(tanfovy),
                    float(focal_x),
                    float(focal_y),
                    int(image_width),
                    int(image_height),
                    int(grid_x),
                    int(grid_y),
                    int(coverage_mode),
                ],
                outputs=[
                    wp.from_torch(depths, dtype=wp.float32),
                    wp.from_torch(radii, dtype=wp.int32),
                    wp.from_torch(proj_2d, dtype=wp.vec2),
                    wp.from_torch(conic_2d, dtype=wp.vec3),
                    wp.from_torch(conic_2d_inv, dtype=wp.vec3),
                    wp.from_torch(points_xy_image, dtype=wp.vec2),
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    wp.from_torch(conic_opacity, dtype=wp.vec4),
                ],
                device=str(device),
            )
        elif not has_scale_rotation:
            wp.launch(
                kernel=_cov2d_preprocess_masked_pack_scale_rotation_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(_prep(visible_mask), dtype=wp.int32),
                    wp.from_torch(_prep(means3D), dtype=wp.vec3),
                    wp.from_torch(_prep(scales), dtype=wp.vec3),
                    wp.from_torch(_prep(rotations), dtype=wp.vec4),
                    float(scale_modifier),
                    wp.from_torch(_prep(viewmatrix).reshape(-1), dtype=wp.float32),
                    wp.from_torch(_prep(p_proj_all), dtype=wp.vec3),
                    wp.from_torch(_prep(p_view_z_all), dtype=wp.float32),
                    wp.from_torch(_prep(opacities.reshape(-1)), dtype=wp.float32),
                    float(tanfovx),
                    float(tanfovy),
                    float(focal_x),
                    float(focal_y),
                    int(image_width),
                    int(image_height),
                    int(grid_x),
                    int(grid_y),
                    int(coverage_mode),
                ],
                outputs=[
                    wp.from_torch(depths, dtype=wp.float32),
                    wp.from_torch(radii, dtype=wp.int32),
                    wp.from_torch(proj_2d, dtype=wp.vec2),
                    wp.from_torch(conic_2d, dtype=wp.vec3),
                    wp.from_torch(conic_2d_inv, dtype=wp.vec3),
                    wp.from_torch(points_xy_image, dtype=wp.vec2),
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    wp.from_torch(conic_opacity, dtype=wp.vec4),
                ],
                device=str(device),
            )

        if colors_precomp is not None and colors_precomp.numel() != 0:
            rgb = colors_precomp.reshape(point_count, NUM_CHANNELS).to(dtype=torch.float32)
        elif shs is not None and shs.numel() != 0:
            if campos is None:
                raise ValueError("campos is required when computing colors from SH coefficients")
            shs = shs.to(device=device, dtype=torch.float32)
            campos = campos.to(device=device, dtype=torch.float32)
            rgb, clamped, sh_backward_interop = _compute_rgb_from_sh_warp(
                means3D,
                campos,
                shs,
                degree,
                capture_backward_interop=capture_backward_interop,
            )
            if backward_interop is not None and sh_backward_interop is not None:
                backward_interop.means3d = sh_backward_interop.means3d
                backward_interop.campos = sh_backward_interop.campos
                backward_interop.clamped = sh_backward_interop.clamped

        return PreprocessOutputs(
            visible=visible,
            depths=depths,
            radii=radii,
            proj_2d=proj_2d,
            conic_2d=conic_2d,
            conic_2d_inv=conic_2d_inv,
            points_xy_image=points_xy_image,
            tiles_touched=tiles_touched,
            rgb=rgb,
            clamped=clamped,
            conic_opacity=conic_opacity,
            cov3d_all=cov3d_all.to(dtype=torch.float32),
            backward_interop=backward_interop,
        )


def mark_visible(*args: Any):
    _runtime._require_warp()
    means3D, viewmatrix, projmatrix = args
    points = means3D.contiguous()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("means3D must have dimensions (num_points, 3)")

    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    visible, _p_proj, _p_view_z = _project_visible_points_warp(points, viewmatrix, projmatrix)
    return visible.to(torch.bool)


__all__ = [
    "preprocess_gaussians",
    "preprocess_3dgs_stage",
    "feature_3dgs_stage",
    "mark_visible",
    "_make_empty_forward_outputs",
    "_compute_cov3d_from_scale_rotation_warp",
    "_project_visible_points_warp",
    "_compute_rgb_from_sh_warp",
]
