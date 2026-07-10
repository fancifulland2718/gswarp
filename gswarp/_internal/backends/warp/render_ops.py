from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import *
from . import runtime as _runtime
from .memory import *
from .packing import *
from .render_kernels import *

def _render_tiles_warp(preprocess_outputs, binning_state, feature_ptr, background, image_height, image_width):
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
        return out_color, out_depth, out_alpha, n_contrib

    # F6: preprocess outputs are already contiguous from torch.empty(); skip redundant .detach().contiguous()
    points_xy_image = preprocess_outputs.points_xy_image
    conic_opacity = preprocess_outputs.conic_opacity
    depths = preprocess_outputs.depths
    feature_ptr = _prep(feature_ptr)
    ranges = binning_state.ranges.reshape(-1)
    point_list = binning_state.point_list

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
    _compute_depth_flag = int(1 if _runtime._COMPUTE_DEPTH else 0)
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
    return out_color, out_depth, out_alpha, n_contrib


render_gaussians = _render_tiles_warp

__all__ = ["render_gaussians", "_render_tiles_warp"]
