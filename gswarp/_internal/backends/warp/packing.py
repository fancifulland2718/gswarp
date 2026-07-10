from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import (
    BLOCK_X,
    BLOCK_Y,
    FORWARD_GEOM_CLAMP_WIDTH,
    FORWARD_GEOM_STRIDE_BYTES,
    NUM_CHANNELS,
    RENDER_TILE_BATCH,
)
from .state import BinningState, PreprocessOutputs
from .memory import _allocate_scalar_tensor

def _prep(tensor: torch.Tensor) -> torch.Tensor:
    """Detach + ensure contiguous, skipping when unnecessary."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _as_detached_contiguous_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _pack_forward_aux_buffers(preprocess_outputs, binning_state, n_contrib):
    points_xy_image_tensor = _as_detached_contiguous_dtype(preprocess_outputs.points_xy_image, torch.float32)
    depths_tensor = _as_detached_contiguous_dtype(preprocess_outputs.depths, torch.float32)
    conic_opacity_tensor = _as_detached_contiguous_dtype(preprocess_outputs.conic_opacity, torch.float32)
    rgb_tensor = _as_detached_contiguous_dtype(preprocess_outputs.rgb, torch.float32)
    cov3d_tensor = _as_detached_contiguous_dtype(preprocess_outputs.cov3d_all, torch.float32)
    # L2: store clamped as int32 鈥?avoids backward int32 conversion allocation
    clamped_tensor = _as_detached_contiguous_dtype(preprocess_outputs.clamped, torch.int32)
    point_list_tensor = _as_detached_contiguous_dtype(binning_state.point_list, torch.int32)
    ranges_tensor = _as_detached_contiguous_dtype(binning_state.ranges.reshape(-1), torch.int32)
    img_tensor = _as_detached_contiguous_dtype(n_contrib.reshape(-1), torch.int32)

    # F4: torch.cat already returns contiguous tensor; skip redundant .contiguous()
    geom_buffer = torch.cat([
        points_xy_image_tensor.reshape(-1).view(torch.uint8),
        depths_tensor.reshape(-1).view(torch.uint8),
        conic_opacity_tensor.reshape(-1).view(torch.uint8),
        rgb_tensor.reshape(-1).view(torch.uint8),
        cov3d_tensor.reshape(-1).view(torch.uint8),
        clamped_tensor.view(torch.uint8).reshape(-1),
    ])
    binning_buffer = torch.cat([point_list_tensor, ranges_tensor]).view(torch.uint8)
    img_buffer = img_tensor.view(torch.uint8)
    return geom_buffer, binning_buffer, img_buffer


def _unpack_forward_aux_buffers(geom_buffer, binning_buffer, img_buffer, num_rendered, image_height, image_width):
    if geom_buffer.numel() == 0 or binning_buffer.numel() == 0 or img_buffer.numel() == 0:
        return None

    grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
    grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    tile_count = grid_x * grid_y
    point_count = geom_buffer.numel() // FORWARD_GEOM_STRIDE_BYTES
    points_xy_bytes = point_count * 2 * 4
    depths_bytes = point_count * 4
    conic_opacity_bytes = point_count * 4 * 4
    rgb_bytes = point_count * 3 * 4
    cov_bytes = point_count * 6 * 4
    geom_clamp_bytes = point_count * FORWARD_GEOM_CLAMP_WIDTH
    # point_list in binning_buffer includes RENDER_TILE_BATCH padding appended
    # by _build_binning_state, so account for it when validating buffer size.
    _i32_size = 4  # torch.int32 element size
    expected_binning_bytes = (num_rendered + RENDER_TILE_BATCH + tile_count * 2) * _i32_size
    expected_img_bytes = image_height * image_width * _i32_size
    if geom_buffer.numel() != points_xy_bytes + depths_bytes + conic_opacity_bytes + rgb_bytes + cov_bytes + geom_clamp_bytes or binning_buffer.numel() != expected_binning_bytes or img_buffer.numel() != expected_img_bytes:
        return None

    offset = 0
    points_xy_image_tensor = geom_buffer[offset : offset + points_xy_bytes].view(torch.float32).reshape(point_count, 2)
    offset = offset + points_xy_bytes
    depths_tensor = geom_buffer[offset : offset + depths_bytes].view(torch.float32)
    offset = offset + depths_bytes
    conic_opacity_tensor = geom_buffer[offset : offset + conic_opacity_bytes].view(torch.float32).reshape(point_count, 4)
    offset = offset + conic_opacity_bytes
    rgb_tensor = geom_buffer[offset : offset + rgb_bytes].view(torch.float32).reshape(point_count, 3)
    offset = offset + rgb_bytes
    cov3d_tensor = geom_buffer[offset : offset + cov_bytes].view(torch.float32).reshape(point_count, 6)
    offset = offset + cov_bytes
    # L2: clamped stored as int32 鈥?direct view, no backward conversion needed
    clamped_tensor = geom_buffer[offset:].view(torch.int32).reshape(point_count, NUM_CHANNELS)
    binning_tensor = binning_buffer.view(torch.int32)
    img_tensor = img_buffer.view(torch.int32)

    preprocess_outputs = PreprocessOutputs(
        visible=torch.empty((0,), dtype=torch.bool, device=geom_buffer.device),
        depths=depths_tensor,
        radii=torch.empty((point_count,), dtype=torch.int32, device=geom_buffer.device),
        proj_2d=torch.empty((0,), dtype=torch.float32, device=geom_buffer.device),
        conic_2d=torch.empty((0,), dtype=torch.float32, device=geom_buffer.device),
        conic_2d_inv=torch.empty((0,), dtype=torch.float32, device=geom_buffer.device),
        points_xy_image=points_xy_image_tensor,
        tiles_touched=torch.empty((0,), dtype=torch.int32, device=geom_buffer.device),
        rgb=rgb_tensor,
        clamped=clamped_tensor,
        conic_opacity=conic_opacity_tensor,
        cov3d_all=cov3d_tensor,
    )
    binning_state = BinningState(
        grid_x=grid_x,
        grid_y=grid_y,
        point_offsets=torch.empty((0,), dtype=torch.int32, device=geom_buffer.device),
        point_list=binning_tensor[:num_rendered + RENDER_TILE_BATCH],
        point_list_keys=torch.empty((0,), dtype=torch.int64, device=geom_buffer.device),
        ranges=binning_tensor[num_rendered + RENDER_TILE_BATCH:].reshape(tile_count, 2),
        num_rendered=num_rendered,
    )
    n_contrib = img_tensor.reshape(image_height, image_width)
    return preprocess_outputs, binning_state, n_contrib


__all__ = ["_prep", "_as_detached_contiguous_dtype", "_pack_forward_aux_buffers", "_unpack_forward_aux_buffers"]
