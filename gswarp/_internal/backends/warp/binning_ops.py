from __future__ import annotations

import torch
import warp as wp

from ...._stream import set_launch_params
from ...coverage import tile_coverage_mode_id
from .constants import (
    BINNING_SORT_MODES,
    BLOCK_X,
    BLOCK_Y,
    TILE_COVERAGE_AUTO,
    TILE_COVERAGE_CONIC_RECT,
    TORCH_SINGLE_SORT_THRESHOLD,
)
from .state import BinningState
from . import runtime as _runtime
from .memory import (
    _C4_LAUNCH_CACHE_BINNING_DUPLICATE,
    _allocate_scalar_tensor,
    _allocate_warp_scalar_array,
    _can_use_warp_scalar_alloc,
    _gather_i32_by_index,
    _get_depth_order_i32_buffer,
    _get_index_gather_i32_buffer,
    _get_radix_sort_buffers,
    _get_radix_sort_i32_buffers,
    _inclusive_scan_i32,
    _pack_binning_sort_keys,
    _warp_radix_sort_i32_pairs_in_place,
    _warp_radix_sort_pairs_in_place,
)
from .packing import _as_detached_contiguous_dtype
from .binning_kernels import (
    _duplicate_with_keys_from_order_warp_kernel,
    _duplicate_with_keys_warp_kernel,
    _duplicate_with_packed_keys_warp_kernel,
    _identify_tile_ranges_from_packed_keys_warp_kernel,
    _identify_tile_ranges_warp_kernel,
    _prepare_depth_sort_payload_warp_kernel,
    _recount_covered_tiles_warp_kernel,
    _unpack_depth_sort_payload_warp_kernel,
)


def _depth_sort_payload_layout(
    point_count: int, tile_count: int
) -> tuple[int, int] | None:
    if point_count <= 0 or tile_count <= 0:
        return None
    point_bits = (point_count - 1).bit_length()
    count_bits = tile_count.bit_length()
    if point_bits + count_bits > 31:
        return None
    return point_bits, (1 << point_bits) - 1


def _build_binning_state(
    preprocess_outputs,
    image_height,
    image_width,
    sort_mode: str | None = None,
):
        radii = preprocess_outputs.radii
        points_xy_image = preprocess_outputs.points_xy_image
        depths = preprocess_outputs.depths
        tiles_touched = _as_detached_contiguous_dtype(preprocess_outputs.tiles_touched, torch.int32)
        point_count = radii.shape[0]
        device = radii.device
        grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
        tile_count = grid_x * grid_y

        if sort_mode is None:
            sort_mode = _runtime.get_active_binning_sort_mode()
        coverage_mode = tile_coverage_mode_id(_runtime.get_active_tile_coverage_mode())

        if sort_mode not in BINNING_SORT_MODES:
            raise ValueError("sort_mode must be one of 'torch', 'warp_radix', or 'warp_depth_stable_tile'")

        if point_count > 0 and coverage_mode in (
            TILE_COVERAGE_CONIC_RECT,
            TILE_COVERAGE_AUTO,
        ):
            coverage_masks = torch.empty(
                (point_count,), dtype=torch.int64, device=device
            )
            wp.launch(
                kernel=_recount_covered_tiles_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(
                        _as_detached_contiguous_dtype(points_xy_image, torch.float32),
                        dtype=wp.vec2,
                    ),
                    wp.from_torch(
                        _as_detached_contiguous_dtype(radii, torch.int32),
                        dtype=wp.int32,
                    ),
                    wp.from_torch(
                        _as_detached_contiguous_dtype(
                            preprocess_outputs.conic_opacity, torch.float32
                        ),
                        dtype=wp.vec4,
                    ),
                    wp.from_torch(
                        _as_detached_contiguous_dtype(
                            preprocess_outputs.conic_2d_inv, torch.float32
                        ),
                        dtype=wp.vec3,
                    ),
                    int(grid_x),
                    int(grid_y),
                    int(coverage_mode),
                ],
                outputs=[
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    wp.from_torch(coverage_masks, dtype=wp.int64),
                ],
                device=str(device),
            )
        else:
            coverage_masks = torch.zeros((1,), dtype=torch.int64, device=device)

        sorted_point_ids: torch.Tensor | None = None
        if point_count > 0 and sort_mode == "warp_depth_stable_tile":
            # Always sort 鈥?the O6 depth-order-unchanged check requires a GPU鈫扖PU
            # sync (.item()) that costs more than the sort itself during training.
            # Use the shared i32 radix sort buffers for depth sorting, then clone
            # sorted_point_ids before those buffers are reused by the duplicate sort.
            _ds_key_buf, _ds_val_buf, _ds_key_wp, _ds_val_wp = _get_radix_sort_i32_buffers(device, point_count * 2)
            point_depth_keys = _ds_key_buf[:point_count]
            sorted_point_ids = _ds_val_buf[:point_count]
            payload_layout = _depth_sort_payload_layout(
                point_count, tile_count
            )
            use_packed_payload = payload_layout is not None
            point_bits = 0
            point_mask = 0
            if payload_layout is not None:
                point_bits, point_mask = payload_layout
            wp.launch(
                kernel=_prepare_depth_sort_payload_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(depths.view(torch.int32), dtype=wp.int32),
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    int(point_bits),
                    int(use_packed_payload),
                ],
                outputs=[
                    _ds_key_wp
                    if _ds_key_wp is not None
                    else wp.from_torch(point_depth_keys, dtype=wp.int32),
                    _ds_val_wp
                    if _ds_val_wp is not None
                    else wp.from_torch(sorted_point_ids, dtype=wp.int32),
                ],
                device=str(device),
            )
            point_depth_keys, sorted_point_ids = _warp_radix_sort_i32_pairs_in_place(
                _ds_key_buf, _ds_val_buf, point_count, _ds_key_wp, _ds_val_wp
            )
            if use_packed_payload:
                sorted_point_ids, sorted_point_ids_wp = (
                    _get_depth_order_i32_buffer(device, point_count)
                )
                sorted_tiles_touched, sorted_tiles_touched_wp = (
                    _get_index_gather_i32_buffer(device, point_count)
                )
                wp.launch(
                    kernel=_unpack_depth_sort_payload_warp_kernel,
                    dim=point_count,
                    inputs=[
                        _ds_val_wp
                        if _ds_val_wp is not None
                        else wp.from_torch(_ds_val_buf, dtype=wp.int32),
                        int(point_bits),
                        int(point_mask),
                    ],
                    outputs=[
                        sorted_point_ids_wp
                        if sorted_point_ids_wp is not None
                        else wp.from_torch(sorted_point_ids, dtype=wp.int32),
                        sorted_tiles_touched_wp
                        if sorted_tiles_touched_wp is not None
                        else wp.from_torch(sorted_tiles_touched, dtype=wp.int32),
                    ],
                    device=str(device),
                )
            else:
                sorted_point_ids = sorted_point_ids.clone()
                sorted_tiles_touched = _gather_i32_by_index(
                    tiles_touched, sorted_point_ids
                )
            point_offsets = _inclusive_scan_i32(sorted_tiles_touched)
        else:
            point_offsets = _inclusive_scan_i32(tiles_touched) if point_count > 0 else _allocate_scalar_tensor((0,), torch.int32, device)

        # ----- Prepare data for binning while GPU finishes the scan -----
        # Moving data prep before the .item() sync allows the host to do useful
        # work while waiting for the inclusive-scan result.
        ranges_warp = None
        ranges = torch.zeros((tile_count + 1, 2), dtype=torch.int32, device=device)
        if point_count > 0:
            points_xy_image_vec2 = _as_detached_contiguous_dtype(points_xy_image, torch.float32)
            radii_i32 = _as_detached_contiguous_dtype(radii, torch.int32)
            conic_opacity_f32 = _as_detached_contiguous_dtype(preprocess_outputs.conic_opacity, torch.float32)
            conic_opacity_wp = wp.from_torch(conic_opacity_f32, dtype=wp.vec4)
            cov2d_inv_f32 = _as_detached_contiguous_dtype(preprocess_outputs.conic_2d_inv, torch.float32)
            cov2d_inv_wp = wp.from_torch(cov2d_inv_f32, dtype=wp.vec3)
            point_offsets_i32 = _as_detached_contiguous_dtype(point_offsets, torch.int32)

        # ----- Read num_rendered via GPU鈫扖PU sync -----
        if point_count > 0:
            num_rendered = int(point_offsets_i32[-1].item())
        else:
            num_rendered = 0
        if num_rendered < 0:
            raise OverflowError(
                "binning tile-reference count overflowed signed int32; reduce Gaussian footprint or image size"
            )

        if num_rendered == 0:
            return BinningState(
                grid_x=grid_x,
                grid_y=grid_y,
                point_list=_allocate_scalar_tensor((0,), torch.int32, device),
                ranges=_allocate_scalar_tensor((tile_count, 2), torch.int32, device, fill_value=0),
                num_rendered=0,
            )

        if sort_mode == "warp_radix":
            point_list_keys_buffer, point_list_buffer, point_list_keys_wp, point_list_wp = _get_radix_sort_buffers(device, num_rendered * 2)
            point_list_keys = point_list_keys_buffer[:num_rendered]
            point_list = point_list_buffer[:num_rendered]
            depths_f32 = _as_detached_contiguous_dtype(depths, torch.float32)
            wp.launch(
                kernel=_duplicate_with_packed_keys_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                    wp.from_torch(point_offsets_i32, dtype=wp.int32),
                    wp.from_torch(radii_i32, dtype=wp.int32),
                    conic_opacity_wp,
                    cov2d_inv_wp,
                    wp.from_torch(depths_f32, dtype=wp.float32),
                    int(grid_x),
                    int(grid_y),
                    int(coverage_mode),
                    wp.from_torch(coverage_masks, dtype=wp.int64),
                ],
                outputs=[
                    wp.from_torch(point_list_keys, dtype=wp.int64),
                    wp.from_torch(point_list, dtype=wp.int32),
                ],
                device=str(radii.device),
            )

            point_list_keys, point_list = _warp_radix_sort_pairs_in_place(
                point_list_keys_buffer,
                point_list_buffer,
                num_rendered,
                point_list_keys_wp,
                point_list_wp,
            )
        elif sort_mode == "warp_depth_stable_tile":
            tile_id_buffer, point_list_buffer, tile_id_buffer_wp, point_list_buffer_wp = (
                _get_radix_sort_i32_buffers(device, num_rendered * 2)
            )
            sorted_point_ids_i32 = _as_detached_contiguous_dtype(sorted_point_ids, torch.int32)
            _dev = str(radii.device)
            _g1_inp = [
                wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                wp.from_torch(radii_i32, dtype=wp.int32),
                conic_opacity_wp,
                cov2d_inv_wp,
                wp.from_torch(sorted_point_ids_i32, dtype=wp.int32),
                wp.from_torch(point_offsets_i32, dtype=wp.int32),
                int(grid_x),
                int(grid_y),
                int(coverage_mode),
                wp.from_torch(coverage_masks, dtype=wp.int64),
            ]
            _g1_out = [
                tile_id_buffer_wp
                if tile_id_buffer_wp is not None
                else wp.from_torch(tile_id_buffer, dtype=wp.int32),
                point_list_buffer_wp
                if point_list_buffer_wp is not None
                else wp.from_torch(point_list_buffer, dtype=wp.int32),
            ]
            _g1_key = (_dev, point_count)
            _g1_cmd = _C4_LAUNCH_CACHE_BINNING_DUPLICATE.get(_g1_key)
            if _g1_cmd is None:
                _g1_cmd = wp.launch(
                    kernel=_duplicate_with_keys_from_order_warp_kernel,
                    dim=point_count,
                    inputs=_g1_inp, outputs=_g1_out,
                    device=_dev, record_cmd=True)
                _C4_LAUNCH_CACHE_BINNING_DUPLICATE[_g1_key] = _g1_cmd
            else:
                set_launch_params(_g1_cmd, _g1_inp + _g1_out)
            _g1_cmd.launch()
            tile_ids = tile_id_buffer[:num_rendered]
            point_list = point_list_buffer[:num_rendered]
            tile_ids, point_list = _warp_radix_sort_i32_pairs_in_place(
                tile_id_buffer,
                point_list_buffer,
                num_rendered,
                tile_id_buffer_wp,
                point_list_buffer_wp,
            )
        else:
            if _can_use_warp_scalar_alloc(device):
                tile_ids_warp, tile_ids = _allocate_warp_scalar_array(num_rendered, torch.int32, device)
                point_list_warp, point_list = _allocate_warp_scalar_array(num_rendered, torch.int32, device)
            else:
                tile_ids_warp = None
                point_list_warp = None
                tile_ids = torch.empty((num_rendered,), dtype=torch.int32, device=device)
                point_list = torch.empty((num_rendered,), dtype=torch.int32, device=device)
            wp.launch(
                kernel=_duplicate_with_keys_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                    wp.from_torch(point_offsets_i32, dtype=wp.int32),
                    wp.from_torch(radii_i32, dtype=wp.int32),
                    conic_opacity_wp,
                    cov2d_inv_wp,
                    int(grid_x),
                    int(grid_y),
                    int(coverage_mode),
                    wp.from_torch(coverage_masks, dtype=wp.int64),
                ],
                outputs=[
                    tile_ids_warp if tile_ids_warp is not None else wp.from_torch(tile_ids, dtype=wp.int32),
                    point_list_warp if point_list_warp is not None else wp.from_torch(point_list, dtype=wp.int32),
                ],
                device=str(radii.device),
            )

            if num_rendered <= TORCH_SINGLE_SORT_THRESHOLD:
                point_list_keys = _pack_binning_sort_keys(tile_ids, point_list, depths)

                order = torch.argsort(point_list_keys, stable=True)
                point_list = point_list[order]
                point_list_keys = point_list_keys[order]
                tile_ids = torch.bitwise_right_shift(point_list_keys, 32).to(torch.int32)
            else:
                point_depth_keys = _gather_i32_by_index(depths.view(torch.int32), point_list)

                order = torch.argsort(point_depth_keys, stable=True)
                tile_ids = tile_ids[order]
                point_list = point_list[order]

                order = torch.argsort(tile_ids, stable=True)
                point_list = point_list[order]
                tile_ids = tile_ids[order]
                point_list_keys = tile_ids.to(torch.int64)

        if sort_mode == "warp_radix":
            wp.launch(
                kernel=_identify_tile_ranges_from_packed_keys_warp_kernel,
                dim=point_list_keys.shape[0],
                inputs=[
                    wp.from_torch(point_list_keys, dtype=wp.int64),
                    ranges_warp if ranges_warp is not None else wp.from_torch(ranges.reshape(-1), dtype=wp.int32),
                    int(point_list_keys.shape[0]),
                ],
                device=str(radii.device),
            )
        else:
            tile_ids_i32 = _as_detached_contiguous_dtype(tile_ids, torch.int32)
            tile_ids_wp = None
            if sort_mode == "warp_depth_stable_tile":
                tile_ids_wp = tile_id_buffer_wp
            wp.launch(
                kernel=_identify_tile_ranges_warp_kernel,
                dim=tile_ids_i32.shape[0],
                inputs=[
                    tile_ids_wp
                    if tile_ids_wp is not None
                    else wp.from_torch(tile_ids_i32, dtype=wp.int32),
                    ranges_warp if ranges_warp is not None else wp.from_torch(ranges.reshape(-1), dtype=wp.int32),
                    int(tile_ids_i32.shape[0]),
                ],
                device=str(radii.device),
            )
        # The sort buffers are reused by later forwards. Keep only the exact
        # immutable list needed by render/backward; kernel reads are range-clamped.
        point_list = point_list.clone()

        return BinningState(
            grid_x=grid_x,
            grid_y=grid_y,
            point_list=point_list,
            ranges=ranges[:tile_count],
            num_rendered=num_rendered,
        )


build_binning_state = _build_binning_state

__all__ = ["build_binning_state", "_build_binning_state"]
