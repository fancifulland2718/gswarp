from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import *


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
    def _compute_tile_rect_snugbox_cov2d_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        cov2d_aa: wp.float32,
        cov2d_cc: wp.float32,
        opacity: wp.float32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        """Opacity-aware SnugBox directly from cov2d diagonal (b-insensitive).

        Mathematically equivalent to the conic version:
          r_x = ceil(sqrt(t * con_c / det_conic)) = ceil(sqrt(t * cov2d_aa))
        but avoids the det_conic = con_a*con_c - con_b^2 denominator that
        amplifies floating-point differences in the off-diagonal term b.
        """
        t = 2.0 * wp.log(wp.max(255.0 * opacity, 1.0))
        if t <= 0.0:
            return wp.vec4i(0, 0, 0, 0)
        radius_x = wp.int32(wp.ceil(wp.sqrt(wp.max(t * cov2d_aa, 0.0))))
        radius_y = wp.int32(wp.ceil(wp.sqrt(wp.max(t * cov2d_cc, 0.0))))
        rect_min_x = wp.min(grid_x, wp.max(0, wp.int32((point_x - float(radius_x)) / float(BLOCK_X))))
        rect_min_y = wp.min(grid_y, wp.max(0, wp.int32((point_y - float(radius_y)) / float(BLOCK_Y))))
        rect_max_x = wp.min(grid_x, wp.max(0, wp.int32((point_x + float(radius_x) + float(BLOCK_X - 1)) / float(BLOCK_X))))
        rect_max_y = wp.min(grid_y, wp.max(0, wp.int32((point_y + float(radius_y) + float(BLOCK_Y - 1)) / float(BLOCK_Y))))
        return wp.vec4i(rect_min_x, rect_min_y, rect_max_x, rect_max_y)

    @wp.func
    def _accutile_row_x_range_wp(
        point_x: wp.float32,
        con_a: wp.float32,
        con_b: wp.float32,
        det_conic: wp.float32,
        t: wp.float32,
        dy: wp.float32,
        grid_x: wp.int32,
    ):
        inner = con_a * t - det_conic * dy * dy
        if inner <= 0.0:
            return wp.vec2i(0, 0)
        sqrt_inner = wp.sqrt(inner)
        dx_center = -con_b * dy / con_a
        dx_half = sqrt_inner / con_a
        x_min = point_x + dx_center - dx_half
        x_max = point_x + dx_center + dx_half
        tile_x_min = wp.min(grid_x, wp.max(0, wp.int32(x_min / float(BLOCK_X))))
        tile_x_max = wp.min(grid_x, wp.max(0, wp.int32((x_max + float(BLOCK_X - 1)) / float(BLOCK_X))))
        return wp.vec2i(tile_x_min, tile_x_max)

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
