from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import *
from .math_kernels import *


if wp is not None:

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

        point = points_xy_image[idx]
        co = conic_opacity[idx]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]

        cov = cov2d_inv[idx]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = idx
                    off = off + 1

        # Sentinel-fill remaining slots (AccuTile row range + alpha filter
        # may write fewer entries than preprocess SnugBox AABB allocated)
        end_off = point_offsets[idx]
        sentinel_tile = grid_x * grid_y
        while off < end_off:
            tile_ids_out[off] = sentinel_tile
            point_list_out[off] = 0
            off = off + 1

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

        point = points_xy_image[idx]
        co = conic_opacity[idx]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]
        cov = cov2d_inv[idx]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        depth_bits = wp.cast(depths[idx], wp.int32)
        depth_key = wp.int64(depth_bits) & wp.int64(4294967295)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_id = tile_y * grid_x + tile_x
                    packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                    point_list_out[off] = idx
                    off = off + 1
        # Sentinel-fill remaining slots
        end_off = point_offsets[idx]
        sentinel_key = wp.int64(grid_x * grid_y) << wp.int64(32)
        while off < end_off:
            packed_keys_out[off] = sentinel_key
            point_list_out[off] = 0
            off = off + 1

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

        point = points_xy_image[point_id]
        co = conic_opacity[point_id]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]
        cov = cov2d_inv[point_id]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = point_id
                    off = off + 1
        end_off = point_offsets[idx]
        sentinel_tile = grid_x * grid_y
        while off < end_off:
            tile_ids_out[off] = sentinel_tile
            point_list_out[off] = 0
            off = off + 1

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
