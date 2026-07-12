from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import (
    BLOCK_X,
    BLOCK_Y,
    COVERAGE_Q_EPSILON,
    DET_EPSILON,
    PREPROCESS_CULL_FOV_SCALE,
    PREPROCESS_CULL_SIGMA,
    SNUGBOX_PIXEL_PADDING,
    TILE_COVERAGE_ACCUTILE_SWEEP,
    TILE_COVERAGE_AUTO,
    TILE_COVERAGE_AUTO_CONIC_RECT_MAX_TILES,
    TILE_COVERAGE_CONIC_RECT,
    TILE_COVERAGE_SNUGBOX,
)


if wp is not None:

    @wp.func
    def _cov2d_from_scale_rotation_gram_wp(
        scale: wp.vec3,
        rotation: wp.vec4,
        scale_modifier: wp.float32,
        t00: wp.float32,
        t10: wp.float32,
        t20: wp.float32,
        t01: wp.float32,
        t11: wp.float32,
        t21: wp.float32,
    ):
        s = scale * scale_modifier
        r = rotation[0]
        xq = rotation[1]
        yq = rotation[2]
        zq = rotation[3]

        r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
        r01 = 2.0 * (xq * yq + r * zq)
        r02 = 2.0 * (xq * zq - r * yq)
        r10 = 2.0 * (xq * yq - r * zq)
        r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
        r12 = 2.0 * (yq * zq + r * xq)
        r20 = 2.0 * (xq * zq + r * yq)
        r21 = 2.0 * (yq * zq - r * xq)
        r22 = 1.0 - 2.0 * (xq * xq + yq * yq)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        # A = M * T  (3x2)
        a00 = m00 * t00 + m01 * t10 + m02 * t20
        a10 = m10 * t00 + m11 * t10 + m12 * t20
        a20 = m20 * t00 + m21 * t10 + m22 * t20
        a01 = m00 * t01 + m01 * t11 + m02 * t21
        a11 = m10 * t01 + m11 * t11 + m12 * t21
        a21 = m20 * t01 + m21 * t11 + m22 * t21

        # cov2d = A^T * A + 0.3*I  (2x2 symmetric, guaranteed PSD)
        return wp.vec3(
            a00 * a00 + a10 * a10 + a20 * a20 + 0.3,
            a00 * a01 + a10 * a11 + a20 * a21,
            a01 * a01 + a11 * a11 + a21 * a21 + 0.3,
        )

    @wp.func
    def _compute_power(con_o: wp.vec4, d_x: float, d_y: float):
        return -0.5 * (con_o[0] * d_x * d_x + con_o[2] * d_y * d_y) - con_o[1] * d_x * d_y

    @wp.func
    def _compute_alpha(con_o: wp.vec4, power: float):
        return wp.min(float(0.99), con_o[3] * wp.exp(power))

    @wp.func
    def _ndc_to_pix_wp(value: float, size: int):
        return ((value + 1.0) * float(size) - 1.0) * 0.5

    @wp.func
    def _dnormvdv_wp(vector: wp.vec3, grad_vector: wp.vec3):
        sum2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
        invsum32 = 1.0 / wp.sqrt(wp.max(sum2 * sum2 * sum2, 1.0e-20))
        return wp.vec3(
            ((sum2 - vector[0] * vector[0]) * grad_vector[0] - vector[1] * vector[0] * grad_vector[1] - vector[2] * vector[0] * grad_vector[2]) * invsum32,
            (-vector[0] * vector[1] * grad_vector[0] + (sum2 - vector[1] * vector[1]) * grad_vector[1] - vector[2] * vector[1] * grad_vector[2]) * invsum32,
            (-vector[0] * vector[2] * grad_vector[0] - vector[1] * vector[2] * grad_vector[1] + (sum2 - vector[2] * vector[2]) * grad_vector[2]) * invsum32,
        )

    @wp.func
    def _conic_denom2inv_wp(denom: wp.float32):
        return 1.0 / (denom * denom + 1.0e-7)

    @wp.func
    def _conservative_cull_radius_from_cov3d_wp(
        v00: wp.float32,
        v01: wp.float32,
        v02: wp.float32,
        v11: wp.float32,
        v12: wp.float32,
        v22: wp.float32,
    ):
        row0 = wp.abs(v00) + wp.abs(v01) + wp.abs(v02)
        row1 = wp.abs(v01) + wp.abs(v11) + wp.abs(v12)
        row2 = wp.abs(v02) + wp.abs(v12) + wp.abs(v22)
        return wp.max(row0, wp.max(row1, row2))

    @wp.func
    def _preprocess_radius_upper_wp(
        cov3d_lambda_upper: wp.float32,
        p_view_z: wp.float32,
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
    ):
        limx = PREPROCESS_CULL_FOV_SCALE * tanfovx
        limy = PREPROCESS_CULL_FOV_SCALE * tanfovy
        focal_x = float(image_width) / (2.0 * tanfovx)
        focal_y = float(image_height) / (2.0 * tanfovy)
        row0_norm = (focal_x / p_view_z) * wp.sqrt(1.0 + limx * limx)
        row1_norm = (focal_y / p_view_z) * wp.sqrt(1.0 + limy * limy)
        j_frob_sq = row0_norm * row0_norm + row1_norm * row1_norm
        cov2d_lambda_upper = cov3d_lambda_upper * j_frob_sq + 0.3
        return wp.int32(wp.ceil(PREPROCESS_CULL_SIGMA * wp.sqrt(wp.max(cov2d_lambda_upper, 0.0))))

    @wp.func
    def _compute_tile_rect_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        radius: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        rect_min_x = wp.min(grid_x, wp.max(0, wp.int32((point_x - float(radius)) / float(BLOCK_X))))
        rect_min_y = wp.min(grid_y, wp.max(0, wp.int32((point_y - float(radius)) / float(BLOCK_Y))))
        rect_max_x = wp.min(grid_x, wp.max(0, wp.int32((point_x + float(radius) + float(BLOCK_X - 1)) / float(BLOCK_X))))
        rect_max_y = wp.min(grid_y, wp.max(0, wp.int32((point_y + float(radius) + float(BLOCK_Y - 1)) / float(BLOCK_Y))))
        return wp.vec4i(rect_min_x, rect_min_y, rect_max_x, rect_max_y)

    @wp.func
    def _compute_tile_rect_compat_snugbox_cov2d_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        cov2d_aa: wp.float32,
        cov2d_cc: wp.float32,
        opacity: wp.float32,
        cuda_radius: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        """Padded Speedy-Splat SnugBox clipped to CUDA's geometric rect.

        This keeps Speedy-Splat's 1/255 alpha cutoff and adds one pixel before
        tile conversion to cover boundary discretization. Clipping to CUDA's
        3-sigma rect avoids contributions the reference rasterizer never bins.
        """
        cuda_rect = _compute_tile_rect_wp(
            point_x, point_y, cuda_radius, grid_x, grid_y
        )
        t = 2.0 * wp.log(wp.max(255.0 * opacity, 1.0))
        if t <= 0.0:
            return wp.vec4i(0, 0, 0, 0)
        radius_x = wp.int32(
            wp.ceil(
                wp.sqrt(wp.max(t * cov2d_aa, 0.0))
                + SNUGBOX_PIXEL_PADDING
            )
        )
        radius_y = wp.int32(
            wp.ceil(
                wp.sqrt(wp.max(t * cov2d_cc, 0.0))
                + SNUGBOX_PIXEL_PADDING
            )
        )
        rect_min_x = wp.min(grid_x, wp.max(0, wp.int32((point_x - float(radius_x)) / float(BLOCK_X))))
        rect_min_y = wp.min(grid_y, wp.max(0, wp.int32((point_y - float(radius_y)) / float(BLOCK_Y))))
        rect_max_x = wp.min(grid_x, wp.max(0, wp.int32((point_x + float(radius_x) + float(BLOCK_X - 1)) / float(BLOCK_X))))
        rect_max_y = wp.min(grid_y, wp.max(0, wp.int32((point_y + float(radius_y) + float(BLOCK_Y - 1)) / float(BLOCK_Y))))
        return wp.vec4i(
            wp.max(cuda_rect[0], rect_min_x),
            wp.max(cuda_rect[1], rect_min_y),
            wp.min(cuda_rect[2], rect_max_x),
            wp.min(cuda_rect[3], rect_max_y),
        )

    @wp.func
    def _resolve_tile_coverage_mode_wp(
        requested_mode: wp.int32,
        rect: wp.vec4i,
    ):
        if requested_mode != TILE_COVERAGE_AUTO:
            return requested_mode
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area <= TILE_COVERAGE_AUTO_CONIC_RECT_MAX_TILES:
            return wp.int32(TILE_COVERAGE_CONIC_RECT)
        return wp.int32(TILE_COVERAGE_ACCUTILE_SWEEP)

    @wp.func
    def _conic_q_wp(
        x: wp.float32,
        y: wp.float32,
        conic_a: wp.float32,
        conic_b: wp.float32,
        conic_c: wp.float32,
    ):
        return conic_a * x * x + 2.0 * conic_b * x * y + conic_c * y * y

    @wp.func
    def _conic_rect_intersects_tile_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        conic_a: wp.float32,
        conic_b: wp.float32,
        conic_c: wp.float32,
        threshold: wp.float32,
        tile_x: wp.int32,
        tile_y: wp.int32,
    ):
        det = conic_a * conic_c - conic_b * conic_b
        if conic_a <= 0.0 or conic_c <= 0.0 or det <= DET_EPSILON:
            return True

        x0 = float(tile_x * BLOCK_X) - point_x
        x1 = float(tile_x * BLOCK_X + BLOCK_X - 1) - point_x
        y0 = float(tile_y * BLOCK_Y) - point_y
        y1 = float(tile_y * BLOCK_Y + BLOCK_Y - 1) - point_y
        if x0 <= 0.0 and x1 >= 0.0 and y0 <= 0.0 and y1 >= 0.0:
            return True

        q_min = wp.float32(3.402823466e38)

        y = wp.clamp(-conic_b * x0 / conic_c, y0, y1)
        q_min = wp.min(q_min, _conic_q_wp(x0, y, conic_a, conic_b, conic_c))
        y = wp.clamp(-conic_b * x1 / conic_c, y0, y1)
        q_min = wp.min(q_min, _conic_q_wp(x1, y, conic_a, conic_b, conic_c))

        x = wp.clamp(-conic_b * y0 / conic_a, x0, x1)
        q_min = wp.min(q_min, _conic_q_wp(x, y0, conic_a, conic_b, conic_c))
        x = wp.clamp(-conic_b * y1 / conic_a, x0, x1)
        q_min = wp.min(q_min, _conic_q_wp(x, y1, conic_a, conic_b, conic_c))

        return q_min <= threshold + COVERAGE_Q_EPSILON

    @wp.func
    def _accutile_band_interval_wp(
        point_u: wp.float32,
        point_v: wp.float32,
        conic_uu: wp.float32,
        conic_uv: wp.float32,
        conic_vv: wp.float32,
        threshold: wp.float32,
        tile_v: wp.int32,
        block_u: wp.int32,
        block_v: wp.int32,
        rect_u_min: wp.int32,
        rect_u_max: wp.int32,
    ):
        det = conic_uu * conic_vv - conic_uv * conic_uv
        if conic_uu <= 0.0 or conic_vv <= 0.0 or det <= DET_EPSILON:
            return wp.vec2i(rect_u_min, rect_u_max)

        threshold_safe = threshold + COVERAGE_Q_EPSILON
        v0 = float(tile_v * block_v) - point_v
        v1 = float(tile_v * block_v + block_v - 1) - point_v
        u_min = wp.float32(3.402823466e38)
        u_max = wp.float32(-3.402823466e38)
        found = False

        disc = conic_uu * threshold_safe - det * v0 * v0
        if disc >= -COVERAGE_Q_EPSILON:
            half = wp.sqrt(wp.max(disc, 0.0)) / conic_uu
            center = -conic_uv * v0 / conic_uu
            u_min = wp.min(u_min, center - half)
            u_max = wp.max(u_max, center + half)
            found = True

        disc = conic_uu * threshold_safe - det * v1 * v1
        if disc >= -COVERAGE_Q_EPSILON:
            half = wp.sqrt(wp.max(disc, 0.0)) / conic_uu
            center = -conic_uv * v1 / conic_uu
            u_min = wp.min(u_min, center - half)
            u_max = wp.max(u_max, center + half)
            found = True

        extent_u = wp.sqrt(wp.max(threshold_safe * conic_vv / det, 0.0))
        v_at_min_u = conic_uv * extent_u / conic_vv
        if v_at_min_u >= v0 - COVERAGE_Q_EPSILON and v_at_min_u <= v1 + COVERAGE_Q_EPSILON:
            u_min = wp.min(u_min, -extent_u)
            u_max = wp.max(u_max, -extent_u)
            found = True
        v_at_max_u = -conic_uv * extent_u / conic_vv
        if v_at_max_u >= v0 - COVERAGE_Q_EPSILON and v_at_max_u <= v1 + COVERAGE_Q_EPSILON:
            u_min = wp.min(u_min, extent_u)
            u_max = wp.max(u_max, extent_u)
            found = True

        if not found:
            return wp.vec2i(rect_u_min, rect_u_min)

        tile_u_min = wp.int32(wp.floor((point_u + u_min) / float(block_u)))
        tile_u_max = wp.int32(wp.floor((point_u + u_max) / float(block_u))) + 1
        return wp.vec2i(
            wp.max(rect_u_min, wp.min(rect_u_max, tile_u_min)),
            wp.max(rect_u_min, wp.min(rect_u_max, tile_u_max)),
        )

    @wp.func
    def _count_covered_tiles_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        cov2d_aa: wp.float32,
        cov2d_cc: wp.float32,
        conic_a: wp.float32,
        conic_b: wp.float32,
        conic_c: wp.float32,
        opacity: wp.float32,
        cuda_radius: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        requested_mode: wp.int32,
    ):
        rect = _compute_tile_rect_compat_snugbox_cov2d_wp(
            point_x,
            point_y,
            cov2d_aa,
            cov2d_cc,
            opacity,
            cuda_radius,
            grid_x,
            grid_y,
        )
        span_x = rect[2] - rect[0]
        span_y = rect[3] - rect[1]
        area = span_x * span_y
        if area <= 1:
            return area

        mode = _resolve_tile_coverage_mode_wp(requested_mode, rect)
        if mode == TILE_COVERAGE_SNUGBOX:
            return area

        threshold = 2.0 * wp.log(wp.max(255.0 * opacity, 1.0))
        count = wp.int32(0)
        if mode == TILE_COVERAGE_CONIC_RECT:
            for tile_y in range(rect[1], rect[3]):
                for tile_x in range(rect[0], rect[2]):
                    if _conic_rect_intersects_tile_wp(
                        point_x,
                        point_y,
                        conic_a,
                        conic_b,
                        conic_c,
                        threshold,
                        tile_x,
                        tile_y,
                    ):
                        count = count + 1
            return count

        if span_y <= span_x:
            for tile_y in range(rect[1], rect[3]):
                interval = _accutile_band_interval_wp(
                    point_x,
                    point_y,
                    conic_a,
                    conic_b,
                    conic_c,
                    threshold,
                    tile_y,
                    BLOCK_X,
                    BLOCK_Y,
                    rect[0],
                    rect[2],
                )
                count = count + wp.max(0, interval[1] - interval[0])
        else:
            for tile_x in range(rect[0], rect[2]):
                interval = _accutile_band_interval_wp(
                    point_y,
                    point_x,
                    conic_c,
                    conic_b,
                    conic_a,
                    threshold,
                    tile_x,
                    BLOCK_Y,
                    BLOCK_X,
                    rect[1],
                    rect[3],
                )
                count = count + wp.max(0, interval[1] - interval[0])
        return count

    @wp.func
    def _preprocess_rect_visible_wp(
        proj_value: wp.vec3,
        radius_upper: wp.int32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        rect = _compute_tile_rect_wp(point_x, point_y, radius_upper, grid_x, grid_y)
        rect_area = ( rect[2] - rect[0] ) * ( rect[3] - rect[1] )
        return rect_area != 0
else:
    pass

__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
