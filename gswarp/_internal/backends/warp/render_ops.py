from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ...._stream import torch_launch_array
from .constants import BLOCK_X, BLOCK_Y, NUM_CHANNELS
from . import runtime as _runtime
from .memory import _C4_LAUNCH_CACHE_FWD_RENDER_TILED256
from .packing import _prep
from .render_kernels import _render_tiles_tiled256_warp_kernel
from .state import RenderBackwardInterop

def _render_tiles_warp(
    preprocess_outputs,
    binning_state,
    feature_ptr,
    background,
    image_height,
    image_width,
    *,
    capture_backward_interop=False,
):
    device = feature_ptr.device
    total_pixels = image_height * image_width
    out_color = torch.empty((NUM_CHANNELS, image_height, image_width), dtype=torch.float32, device=device)
    out_depth = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    out_alpha = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    n_contrib = torch.empty((total_pixels,), dtype=torch.int32, device=device)
    background = _prep(background.to(dtype=torch.float32, device=device))

    if binning_state.num_rendered == 0:
        out_color.copy_(background.reshape(NUM_CHANNELS, 1, 1).expand_as(out_color))
        out_depth.zero_()
        out_alpha.zero_()
        n_contrib.zero_()
        return out_color, out_depth, out_alpha, n_contrib, None

    # F6: preprocess outputs are already contiguous from torch.empty(); skip redundant .detach().contiguous()
    points_xy_image = preprocess_outputs.points_xy_image
    conic_opacity = preprocess_outputs.conic_opacity
    depths = preprocess_outputs.depths
    feature_ptr = _prep(feature_ptr)
    ranges = binning_state.ranges.reshape(-1)
    point_list = binning_state.point_list

    _wp_ranges = torch_launch_array(ranges, dtype=wp.int32)
    _wp_point_list = torch_launch_array(point_list, dtype=wp.int32)
    _wp_points_xy = torch_launch_array(points_xy_image, dtype=wp.vec2)
    _wp_features = torch_launch_array(feature_ptr.reshape(-1), dtype=wp.float32)
    _wp_depths = torch_launch_array(depths, dtype=wp.float32)
    _wp_conic_opacity = torch_launch_array(conic_opacity, dtype=wp.vec4)
    _wp_bg = torch_launch_array(background.reshape(-1), dtype=wp.float32)
    _wp_out_color = torch_launch_array(out_color.reshape(-1), dtype=wp.float32)
    _wp_out_depth = torch_launch_array(out_depth.reshape(-1), dtype=wp.float32)
    _wp_out_alpha = torch_launch_array(out_alpha.reshape(-1), dtype=wp.float32)
    _wp_n_contrib = torch_launch_array(n_contrib, dtype=wp.int32)
    _compute_depth_flag = int(1 if _runtime.get_active_compute_depth() else 0)
    # T1: Tiled-256 cooperative forward render
    _grid_x_fwd = int(binning_state.grid_x)
    _grid_y_fwd = (image_height + BLOCK_Y - 1) // BLOCK_Y
    _num_tiles_fwd = _grid_x_fwd * _grid_y_fwd
    _dim = _num_tiles_fwd * (BLOCK_X * BLOCK_Y)
    _inp = [_wp_ranges, _wp_point_list, _wp_points_xy, _wp_features, _wp_depths,
            _wp_conic_opacity, _wp_bg, int(image_width), int(image_height),
            _grid_x_fwd, _num_tiles_fwd, _compute_depth_flag]
    _out = [_wp_out_color, _wp_out_depth, _wp_out_alpha, _wp_n_contrib]
    _key = (str(device), _dim)
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
    backward_interop = None
    if capture_backward_interop:
        backward_interop = RenderBackwardInterop(
            ranges=_wp_ranges,
            point_list=_wp_point_list,
            points_xy=_wp_points_xy,
            features=_wp_features,
            depths=_wp_depths,
            conic_opacity=_wp_conic_opacity,
            background=_wp_bg,
            out_alpha=_wp_out_alpha,
            n_contrib=_wp_n_contrib,
        )
    return out_color, out_depth, out_alpha, n_contrib, backward_interop


render_gaussians = _render_tiles_warp

__all__ = ["render_gaussians", "_render_tiles_warp"]
