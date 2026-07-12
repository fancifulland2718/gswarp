from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import (
    BLOCK_X,
    BLOCK_Y,
    TILE_COVERAGE_ACCUTILE_SWEEP,
    TILE_COVERAGE_AUTO,
    TILE_COVERAGE_CONIC_MASK_MAX_TILES,
    TILE_COVERAGE_CONIC_RECT,
    TILE_COVERAGE_SNUGBOX,
)
from .math_kernels import (
    _accutile_band_interval_wp,
    _compute_tile_rect_compat_snugbox_cov2d_wp,
    _conic_rect_intersects_tile_wp,
    _count_covered_tiles_wp,
    _resolve_tile_coverage_mode_wp,
)


if wp is not None:

    @wp.kernel
    def _recount_covered_tiles_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        tiles_touched_out: wp.array(dtype=wp.int32),
        coverage_masks_out: wp.array(dtype=wp.int64),
    ):
        idx = wp.tid()
        radius = radii[idx]
        if radius <= 0:
            tiles_touched_out[idx] = 0
            coverage_masks_out[idx] = wp.int64(0)
            return
        point = points_xy_image[idx]
        co = conic_opacity[idx]
        cov = cov2d_inv[idx]
        rect = _compute_tile_rect_compat_snugbox_cov2d_wp(
            point[0],
            point[1],
            cov[0],
            cov[2],
            co[3],
            radius,
            grid_x,
            grid_y,
        )
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        mode = _resolve_tile_coverage_mode_wp(coverage_mode, rect)
        mask = wp.int64(0)
        if mode == TILE_COVERAGE_CONIC_RECT and area <= TILE_COVERAGE_CONIC_MASK_MAX_TILES:
            if area <= 1:
                tiles_touched_out[idx] = area
                if area == 1:
                    mask = wp.int64(1)
            else:
                threshold = 2.0 * wp.log(wp.max(255.0 * co[3], 1.0))
                count = wp.int32(0)
                bit_index = wp.int32(0)
                for tile_y in range(rect[1], rect[3]):
                    for tile_x in range(rect[0], rect[2]):
                        if _conic_rect_intersects_tile_wp(
                            point[0],
                            point[1],
                            co[0],
                            co[1],
                            co[2],
                            threshold,
                            tile_x,
                            tile_y,
                        ):
                            mask = mask | (wp.int64(1) << wp.int64(bit_index))
                            count = count + 1
                        bit_index = bit_index + 1
                tiles_touched_out[idx] = count
            coverage_masks_out[idx] = mask
            return

        effective_mode = coverage_mode
        if mode == TILE_COVERAGE_CONIC_RECT:
            effective_mode = wp.int32(TILE_COVERAGE_ACCUTILE_SWEEP)
        tiles_touched_out[idx] = _count_covered_tiles_wp(
            point[0], point[1], cov[0], cov[2], co[0], co[1], co[2], co[3],
            radius, grid_x, grid_y, effective_mode,
        )
        coverage_masks_out[idx] = mask

    @wp.func
    def _emit_tile_ids_for_coverage_wp(
        point: wp.vec2,
        radius: wp.int32,
        conic_opacity: wp.vec4,
        cov2d: wp.vec3,
        grid_x: wp.int32,
        grid_y: wp.int32,
        requested_mode: wp.int32,
        coverage_mask: wp.int64,
        point_id: wp.int32,
        offset: wp.int32,
        tile_ids_out: wp.array(dtype=wp.int32),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        rect = _compute_tile_rect_compat_snugbox_cov2d_wp(
            point[0],
            point[1],
            cov2d[0],
            cov2d[2],
            conic_opacity[3],
            radius,
            grid_x,
            grid_y,
        )
        span_x = rect[2] - rect[0]
        span_y = rect[3] - rect[1]
        area = span_x * span_y
        mode = _resolve_tile_coverage_mode_wp(requested_mode, rect)
        threshold = 2.0 * wp.log(wp.max(255.0 * conic_opacity[3], 1.0))
        off = offset

        if area <= 1 or mode == TILE_COVERAGE_SNUGBOX:
            for tile_y in range(rect[1], rect[3]):
                for tile_x in range(rect[0], rect[2]):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = point_id
                    off = off + 1
            return

        if mode == TILE_COVERAGE_CONIC_RECT:
            if area <= TILE_COVERAGE_CONIC_MASK_MAX_TILES:
                bit_index = wp.int32(0)
                for tile_y in range(rect[1], rect[3]):
                    for tile_x in range(rect[0], rect[2]):
                        bit = wp.int64(1) << wp.int64(bit_index)
                        if (coverage_mask & bit) != wp.int64(0):
                            tile_ids_out[off] = tile_y * grid_x + tile_x
                            point_list_out[off] = point_id
                            off = off + 1
                        bit_index = bit_index + 1
                return

        if span_y <= span_x:
            for tile_y in range(rect[1], rect[3]):
                interval = _accutile_band_interval_wp(
                    point[0],
                    point[1],
                    conic_opacity[0],
                    conic_opacity[1],
                    conic_opacity[2],
                    threshold,
                    tile_y,
                    BLOCK_X,
                    BLOCK_Y,
                    rect[0],
                    rect[2],
                )
                for tile_x in range(interval[0], interval[1]):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = point_id
                    off = off + 1
        else:
            for tile_x in range(rect[0], rect[2]):
                interval = _accutile_band_interval_wp(
                    point[1],
                    point[0],
                    conic_opacity[2],
                    conic_opacity[1],
                    conic_opacity[0],
                    threshold,
                    tile_x,
                    BLOCK_Y,
                    BLOCK_X,
                    rect[1],
                    rect[3],
                )
                for tile_y in range(interval[0], interval[1]):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = point_id
                    off = off + 1

    @wp.func
    def _emit_packed_keys_for_coverage_wp(
        point: wp.vec2,
        radius: wp.int32,
        conic_opacity: wp.vec4,
        cov2d: wp.vec3,
        depth_key: wp.int64,
        grid_x: wp.int32,
        grid_y: wp.int32,
        requested_mode: wp.int32,
        coverage_mask: wp.int64,
        point_id: wp.int32,
        offset: wp.int32,
        packed_keys_out: wp.array(dtype=wp.int64),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        rect = _compute_tile_rect_compat_snugbox_cov2d_wp(
            point[0],
            point[1],
            cov2d[0],
            cov2d[2],
            conic_opacity[3],
            radius,
            grid_x,
            grid_y,
        )
        span_x = rect[2] - rect[0]
        span_y = rect[3] - rect[1]
        area = span_x * span_y
        mode = _resolve_tile_coverage_mode_wp(requested_mode, rect)
        threshold = 2.0 * wp.log(wp.max(255.0 * conic_opacity[3], 1.0))
        off = offset

        if area <= 1 or mode == TILE_COVERAGE_SNUGBOX:
            for tile_y in range(rect[1], rect[3]):
                for tile_x in range(rect[0], rect[2]):
                    tile_id = tile_y * grid_x + tile_x
                    packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                    point_list_out[off] = point_id
                    off = off + 1
            return

        if mode == TILE_COVERAGE_CONIC_RECT:
            if area <= TILE_COVERAGE_CONIC_MASK_MAX_TILES:
                bit_index = wp.int32(0)
                for tile_y in range(rect[1], rect[3]):
                    for tile_x in range(rect[0], rect[2]):
                        bit = wp.int64(1) << wp.int64(bit_index)
                        if (coverage_mask & bit) != wp.int64(0):
                            tile_id = tile_y * grid_x + tile_x
                            packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                            point_list_out[off] = point_id
                            off = off + 1
                        bit_index = bit_index + 1
                return

        if span_y <= span_x:
            for tile_y in range(rect[1], rect[3]):
                interval = _accutile_band_interval_wp(
                    point[0],
                    point[1],
                    conic_opacity[0],
                    conic_opacity[1],
                    conic_opacity[2],
                    threshold,
                    tile_y,
                    BLOCK_X,
                    BLOCK_Y,
                    rect[0],
                    rect[2],
                )
                for tile_x in range(interval[0], interval[1]):
                    tile_id = tile_y * grid_x + tile_x
                    packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                    point_list_out[off] = point_id
                    off = off + 1
        else:
            for tile_x in range(rect[0], rect[2]):
                interval = _accutile_band_interval_wp(
                    point[1],
                    point[0],
                    conic_opacity[2],
                    conic_opacity[1],
                    conic_opacity[0],
                    threshold,
                    tile_x,
                    BLOCK_Y,
                    BLOCK_X,
                    rect[1],
                    rect[3],
                )
                for tile_y in range(interval[0], interval[1]):
                    tile_id = tile_y * grid_x + tile_x
                    packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                    point_list_out[off] = point_id
                    off = off + 1

    @wp.kernel
    def _gather_i32_by_index_warp_kernel(
        src: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        out[idx] = src[indices[idx]]

    @wp.kernel
    def _duplicate_with_keys_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        point_offsets: wp.array(dtype=wp.int32),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        coverage_masks: wp.array(dtype=wp.int64),
        tile_ids_out: wp.array(dtype=wp.int32),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        radius = radii[idx]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        coverage_mask = wp.int64(0)
        if coverage_mode == TILE_COVERAGE_CONIC_RECT or coverage_mode == TILE_COVERAGE_AUTO:
            coverage_mask = coverage_masks[idx]
        _emit_tile_ids_for_coverage_wp(
            points_xy_image[idx],
            radius,
            conic_opacity[idx],
            cov2d_inv[idx],
            grid_x,
            grid_y,
            coverage_mode,
            coverage_mask,
            idx,
            off,
            tile_ids_out,
            point_list_out,
        )

    @wp.kernel
    def _duplicate_with_packed_keys_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        point_offsets: wp.array(dtype=wp.int32),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        depths: wp.array(dtype=wp.float32),
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        coverage_masks: wp.array(dtype=wp.int64),
        packed_keys_out: wp.array(dtype=wp.int64),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        radius = radii[idx]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        depth_bits = wp.cast(depths[idx], wp.int32)
        depth_key = wp.int64(depth_bits) & wp.int64(4294967295)
        coverage_mask = wp.int64(0)
        if coverage_mode == TILE_COVERAGE_CONIC_RECT or coverage_mode == TILE_COVERAGE_AUTO:
            coverage_mask = coverage_masks[idx]
        _emit_packed_keys_for_coverage_wp(
            points_xy_image[idx],
            radius,
            conic_opacity[idx],
            cov2d_inv[idx],
            depth_key,
            grid_x,
            grid_y,
            coverage_mode,
            coverage_mask,
            idx,
            off,
            packed_keys_out,
            point_list_out,
        )

    @wp.kernel
    def _duplicate_with_keys_from_order_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        point_ids: wp.array(dtype=wp.int32),
        point_offsets: wp.array(dtype=wp.int32),
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        coverage_masks: wp.array(dtype=wp.int64),
        tile_ids_out: wp.array(dtype=wp.int32),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        point_id = point_ids[idx]
        radius = radii[point_id]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        coverage_mask = wp.int64(0)
        if coverage_mode == TILE_COVERAGE_CONIC_RECT or coverage_mode == TILE_COVERAGE_AUTO:
            coverage_mask = coverage_masks[point_id]
        _emit_tile_ids_for_coverage_wp(
            points_xy_image[point_id],
            radius,
            conic_opacity[point_id],
            cov2d_inv[point_id],
            grid_x,
            grid_y,
            coverage_mode,
            coverage_mask,
            point_id,
            off,
            tile_ids_out,
            point_list_out,
        )

    @wp.kernel
    def _pack_binning_keys_warp_kernel(
        tile_ids: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        depths: wp.array(dtype=wp.float32),
        packed_keys_out: wp.array(dtype=wp.int64),
    ):
        idx = wp.tid()
        point_idx = point_list[idx]
        depth_bits = wp.cast(depths[point_idx], wp.int32)
        depth_key = wp.int64(depth_bits) & wp.int64(4294967295)
        tile_key = wp.int64(tile_ids[idx]) << wp.int64(32)
        packed_keys_out[idx] = tile_key | depth_key

    @wp.kernel
    def _identify_tile_ranges_warp_kernel(
        tile_ids: wp.array(dtype=wp.int32),
        range_flat: wp.array(dtype=wp.int32),
        length: wp.int32,
    ):
        idx = wp.tid()
        if idx >= length:
            return

        curr_tile = tile_ids[idx]
        if idx == 0:
            range_flat[curr_tile * 2] = 0
        else:
            prev_tile = tile_ids[idx - 1]
            if curr_tile != prev_tile:
                range_flat[prev_tile * 2 + 1] = idx
                range_flat[curr_tile * 2] = idx

        if idx == length - 1:
            range_flat[curr_tile * 2 + 1] = length

    @wp.kernel
    def _identify_tile_ranges_from_packed_keys_warp_kernel(
        packed_keys: wp.array(dtype=wp.int64),
        range_flat: wp.array(dtype=wp.int32),
        length: wp.int32,
    ):
        idx = wp.tid()
        if idx >= length:
            return

        curr_tile = wp.int32(packed_keys[idx] >> wp.int64(32))
        if idx == 0:
            range_flat[curr_tile * 2] = 0
        else:
            prev_tile = wp.int32(packed_keys[idx - 1] >> wp.int64(32))
            if curr_tile != prev_tile:
                range_flat[prev_tile * 2 + 1] = idx
                range_flat[curr_tile * 2] = idx

        if idx == length - 1:
            range_flat[curr_tile * 2 + 1] = length
else:
    pass

__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
