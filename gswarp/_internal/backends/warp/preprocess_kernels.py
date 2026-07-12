from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import (
    DET_EPSILON,
    VISIBILITY_NEAR_PLANE,
    sh_c0,
    sh_c1,
    sh_c2_0,
    sh_c2_1,
    sh_c2_2,
    sh_c2_3,
    sh_c2_4,
    sh_c3_0,
    sh_c3_1,
    sh_c3_2,
    sh_c3_3,
    sh_c3_4,
    sh_c3_5,
    sh_c3_6,
)
from .math_kernels import (
    _count_covered_tiles_wp,
    _compute_tile_rect_wp,
    _conservative_cull_radius_from_cov3d_wp,
    _cov2d_from_scale_rotation_gram_wp,
    _ndc_to_pix_wp,
    _preprocess_radius_upper_wp,
    _preprocess_rect_visible_wp,
)


if wp is not None:

    @wp.kernel
    def _cov3d_from_scale_rotation_warp_kernel(
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        out_cov3d_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        s = scales[tid] * scale_modifier
        q = rotations[tid]
        r = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y + r * z)
        r02 = 2.0 * (x * z - r * y)
        r10 = 2.0 * (x * y - r * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z + r * x)
        r20 = 2.0 * (x * z + r * y)
        r21 = 2.0 * (y * z - r * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        base = tid * 6
        out_cov3d_flat[base + 0] = m00 * m00 + m10 * m10 + m20 * m20
        out_cov3d_flat[base + 1] = m00 * m01 + m10 * m11 + m20 * m21
        out_cov3d_flat[base + 2] = m00 * m02 + m10 * m12 + m20 * m22
        out_cov3d_flat[base + 3] = m01 * m01 + m11 * m11 + m21 * m21
        out_cov3d_flat[base + 4] = m01 * m02 + m11 * m12 + m21 * m22
        out_cov3d_flat[base + 5] = m02 * m02 + m12 * m12 + m22 * m22

    @wp.kernel
    def _fused_project_cov3d_cov2d_preprocess_sr_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        opacities: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        # outputs
        out_cov3d_flat: wp.array(dtype=wp.float32),
        visible_mask_out: wp.array(dtype=wp.int32),
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        cov_base = idx * 6

        # zero all outputs (covers invisible / invalid cases)
        out_cov3d_flat[cov_base + 0] = float(0.0)
        out_cov3d_flat[cov_base + 1] = float(0.0)
        out_cov3d_flat[cov_base + 2] = float(0.0)
        out_cov3d_flat[cov_base + 3] = float(0.0)
        out_cov3d_flat[cov_base + 4] = float(0.0)
        out_cov3d_flat[cov_base + 5] = float(0.0)
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)

        p = means3d[idx]

        # ---- projection & visibility (from _project_visible_points) ----
        p_view_z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]
        if p_view_z <= 0.2:
            visible_mask_out[idx] = int(0)
            return
        visible_mask_out[idx] = int(1)

        # full homogeneous projection
        hom_x = proj_flat[0] * p[0] + proj_flat[4] * p[1] + proj_flat[8] * p[2] + proj_flat[12]
        hom_y = proj_flat[1] * p[0] + proj_flat[5] * p[1] + proj_flat[9] * p[2] + proj_flat[13]
        hom_w = proj_flat[3] * p[0] + proj_flat[7] * p[1] + proj_flat[11] * p[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        p_proj_x = hom_x * inv_w
        p_proj_y = hom_y * inv_w

        # ---- rotation matrix M = diag(s)*R  (shared by cov3d & cov2d) ----
        s = scales[idx] * scale_modifier
        q = rotations[idx]
        rr = q[0]
        xq = q[1]
        yq = q[2]
        zq = q[3]

        r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
        r01 = 2.0 * (xq * yq + rr * zq)
        r02 = 2.0 * (xq * zq - rr * yq)
        r10 = 2.0 * (xq * yq - rr * zq)
        r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
        r12 = 2.0 * (yq * zq + rr * xq)
        r20 = 2.0 * (xq * zq + rr * yq)
        r21 = 2.0 * (yq * zq - rr * xq)
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

        # ---- cov3d = M^T M  (6 upper-triangle values, for backward) ----
        out_cov3d_flat[cov_base + 0] = m00 * m00 + m10 * m10 + m20 * m20
        out_cov3d_flat[cov_base + 1] = m00 * m01 + m10 * m11 + m20 * m21
        out_cov3d_flat[cov_base + 2] = m00 * m02 + m10 * m12 + m20 * m22
        out_cov3d_flat[cov_base + 3] = m01 * m01 + m11 * m11 + m21 * m21
        out_cov3d_flat[cov_base + 4] = m01 * m02 + m11 * m12 + m21 * m22
        out_cov3d_flat[cov_base + 5] = m02 * m02 + m12 * m12 + m22 * m22

        # ---- cov2d via Gram factorisation (reusing M from above) ----
        vx = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        vy = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        vz = p_view_z

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = vx / vz
        tytz = vy / vz
        vx = wp.clamp(txtz, -limx, limx) * vz
        vy = wp.clamp(tytz, -limy, limy) * vz

        a_j = focal_x / vz
        b_j = focal_y / vz
        c_j = -(focal_x * vx) / (vz * vz)
        d_j = -(focal_y * vy) / (vz * vz)

        t00 = view_flat[0] * a_j + view_flat[2] * c_j
        t10 = view_flat[4] * a_j + view_flat[6] * c_j
        t20 = view_flat[8] * a_j + view_flat[10] * c_j
        t01 = view_flat[1] * b_j + view_flat[2] * d_j
        t11 = view_flat[5] * b_j + view_flat[6] * d_j
        t21 = view_flat[9] * b_j + view_flat[10] * d_j

        # A = M * T  (3x2)
        a00 = m00 * t00 + m01 * t10 + m02 * t20
        a10 = m10 * t00 + m11 * t10 + m12 * t20
        a20 = m20 * t00 + m21 * t10 + m22 * t20
        a01 = m00 * t01 + m01 * t11 + m02 * t21
        a11 = m10 * t01 + m11 * t11 + m12 * t21
        a21 = m20 * t01 + m21 * t11 + m22 * t21

        # cov2d = A^T A + 0.3*I  (2x2 symmetric PSD)
        cov_a = a00 * a00 + a10 * a10 + a20 * a20 + 0.3
        cov_b = a00 * a01 + a10 * a11 + a20 * a21
        cov_c = a01 * a01 + a11 * a11 + a21 * a21 + 0.3

        det = cov_a * cov_c - cov_b * cov_b
        if wp.abs(det) <= DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_c * det_inv, -cov_b * det_inv, cov_a * det_inv)

        mid = 0.5 * (cov_a + cov_c)
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        point_x = _ndc_to_pix_wp(p_proj_x, image_width)
        point_y = _ndc_to_pix_wp(p_proj_y, image_height)

        cuda_rect = _compute_tile_rect_wp(
            point_x, point_y, radius, grid_x, grid_y
        )
        cuda_rect_area = (
            (cuda_rect[2] - cuda_rect[0]) * (cuda_rect[3] - cuda_rect[1])
        )
        if cuda_rect_area == 0:
            return

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = wp.vec3(cov_a, cov_b, cov_c)
        points_xy_image[idx] = point_image
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])
        stored_cov = conic_2d_inv[idx]
        stored_conic_opacity = conic_opacity[idx]
        tiles_touched[idx] = _count_covered_tiles_wp(
            point_x,
            point_y,
            stored_cov[0],
            stored_cov[2],
            stored_conic_opacity[0],
            stored_conic_opacity[1],
            stored_conic_opacity[2],
            stored_conic_opacity[3],
            radius,
            grid_x,
            grid_y,
            coverage_mode,
        )

    @wp.kernel
    def _forward_rgb_from_sh_v3_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_v3: wp.array(dtype=wp.vec3),
        degree: wp.int32,
        coeff_count: wp.int32,
        rgb_v3: wp.array(dtype=wp.vec3),
        clamped_flat: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        sh_base = tid * coeff_count

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = dir_orig[0] * dir_orig[0] + dir_orig[1] * dir_orig[1] + dir_orig[2] * dir_orig[2]
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        color = sh_c0 * shs_v3[sh_base]

        if degree > 0 and coeff_count > 3:
            color = color + (-sh_c1 * y) * shs_v3[sh_base + 1]
            color = color + (sh_c1 * z) * shs_v3[sh_base + 2]
            color = color + (-sh_c1 * x) * shs_v3[sh_base + 3]

            if degree > 1 and coeff_count > 8:
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z

                color = color + (sh_c2_0 * xy) * shs_v3[sh_base + 4]
                color = color + (sh_c2_1 * yz) * shs_v3[sh_base + 5]
                color = color + (sh_c2_2 * (2.0 * zz - xx - yy)) * shs_v3[sh_base + 6]
                color = color + (sh_c2_3 * xz) * shs_v3[sh_base + 7]
                color = color + (sh_c2_4 * (xx - yy)) * shs_v3[sh_base + 8]

                if degree > 2 and coeff_count > 15:
                    color = color + (sh_c3_0 * y * (3.0 * xx - yy)) * shs_v3[sh_base + 9]
                    color = color + (sh_c3_1 * xy * z) * shs_v3[sh_base + 10]
                    color = color + (sh_c3_2 * y * (4.0 * zz - xx - yy)) * shs_v3[sh_base + 11]
                    color = color + (sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * shs_v3[sh_base + 12]
                    color = color + (sh_c3_4 * x * (4.0 * zz - xx - yy)) * shs_v3[sh_base + 13]
                    color = color + (sh_c3_5 * z * (xx - yy)) * shs_v3[sh_base + 14]
                    color = color + (sh_c3_6 * x * (xx - 3.0 * yy)) * shs_v3[sh_base + 15]

        color = color + wp.vec3(0.5, 0.5, 0.5)
        c0 = color[0]
        c1 = color[1]
        c2 = color[2]
        clamp0 = int(0)
        clamp1 = int(0)
        clamp2 = int(0)
        if c0 < 0.0:
            c0 = 0.0
            clamp0 = int(1)
        if c1 < 0.0:
            c1 = 0.0
            clamp1 = int(1)
        if c2 < 0.0:
            c2 = 0.0
            clamp2 = int(1)

        rgb_v3[tid] = wp.vec3(c0, c1, c2)
        clamped_flat[tid * 3 + 0] = clamp0
        clamped_flat[tid * 3 + 1] = clamp1
        clamped_flat[tid * 3 + 2] = clamp2

    @wp.kernel
    def _cov2d_preprocess_masked_pack_warp_kernel(
        visible_mask: wp.array(dtype=wp.int32),
        means3d: wp.array(dtype=wp.vec3),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        p_proj: wp.array(dtype=wp.vec3),
        p_view_z: wp.array(dtype=wp.float32),
        opacities: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if visible_mask[idx] == 0:
            return

        p = means3d[idx]
        base = idx * 6

        x = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        y = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = x / z
        tytz = y / z
        x = wp.clamp(txtz, -limx, limx) * z
        y = wp.clamp(tytz, -limy, limy) * z

        a = focal_x / z
        b = focal_y / z
        c = -(focal_x * x) / (z * z)
        d = -(focal_y * y) / (z * z)

        t00 = view_flat[0] * a + view_flat[2] * c
        t10 = view_flat[4] * a + view_flat[6] * c
        t20 = view_flat[8] * a + view_flat[10] * c
        t01 = view_flat[1] * b + view_flat[2] * d
        t11 = view_flat[5] * b + view_flat[6] * d
        t21 = view_flat[9] * b + view_flat[10] * d

        v00 = cov3d_flat[base + 0]
        v01 = cov3d_flat[base + 1]
        v02 = cov3d_flat[base + 2]
        v11 = cov3d_flat[base + 3]
        v12 = cov3d_flat[base + 4]
        v22 = cov3d_flat[base + 5]

        vt0x = v00 * t00 + v01 * t10 + v02 * t20
        vt0y = v01 * t00 + v11 * t10 + v12 * t20
        vt0z = v02 * t00 + v12 * t10 + v22 * t20
        vt1x = v00 * t01 + v01 * t11 + v02 * t21
        vt1y = v01 * t01 + v11 * t11 + v12 * t21
        vt1z = v02 * t01 + v12 * t11 + v22 * t21

        cov_value = wp.vec3(
            t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3,
            t00 * vt1x + t10 * vt1y + t20 * vt1z,
            t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3,
        )
        det = cov_value[0] * cov_value[2] - cov_value[1] * cov_value[1]
        if wp.abs(det) < DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_value[2] * det_inv, -cov_value[1] * det_inv, cov_value[0] * det_inv)

        mid = 0.5 * (cov_value[0] + cov_value[2])
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        proj_value = p_proj[idx]
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        cuda_rect = _compute_tile_rect_wp(
            point_x, point_y, radius, grid_x, grid_y
        )
        cuda_rect_area = (
            (cuda_rect[2] - cuda_rect[0]) * (cuda_rect[3] - cuda_rect[1])
        )
        if cuda_rect_area == 0:
            return

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z[idx]
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = cov_value
        points_xy_image[idx] = point_image
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])
        stored_cov = conic_2d_inv[idx]
        stored_conic_opacity = conic_opacity[idx]
        tiles_touched[idx] = _count_covered_tiles_wp(
            point_x,
            point_y,
            stored_cov[0],
            stored_cov[2],
            stored_conic_opacity[0],
            stored_conic_opacity[1],
            stored_conic_opacity[2],
            stored_conic_opacity[3],
            radius,
            grid_x,
            grid_y,
            coverage_mode,
        )

    @wp.kernel
    def _cov2d_preprocess_masked_pack_scale_rotation_warp_kernel(
        visible_mask: wp.array(dtype=wp.int32),
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        view_flat: wp.array(dtype=wp.float32),
        p_proj: wp.array(dtype=wp.vec3),
        p_view_z: wp.array(dtype=wp.float32),
        opacities: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        coverage_mode: wp.int32,
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if visible_mask[idx] == 0:
            return

        p = means3d[idx]

        x = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        y = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]
        inv_z = 1.0 / (z + 1.0e-7)
        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        x = wp.clamp(x * inv_z, -limx, limx)
        y = wp.clamp(y * inv_z, -limy, limy)

        a = focal_x * inv_z
        b = focal_y * inv_z

        t00 = a * ( view_flat[0] - view_flat[2] * x )
        t10 = a * ( view_flat[4] - view_flat[6] * x )
        t20 = a * ( view_flat[8] - view_flat[10] * x )
        t01 = b * ( view_flat[1] - view_flat[2] * y )
        t11 = b * ( view_flat[5] - view_flat[6] * y )
        t21 = b * ( view_flat[9] - view_flat[10] * y )

        cov_value = _cov2d_from_scale_rotation_gram_wp(
            scales[idx],
            rotations[idx],
            scale_modifier,
            t00,
            t10,
            t20,
            t01,
            t11,
            t21,
        )
        det = cov_value[0] * cov_value[2] - cov_value[1] * cov_value[1]
        if wp.abs(det) < DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_value[2] * det_inv, -cov_value[1] * det_inv, cov_value[0] * det_inv)

        mid = 0.5 * (cov_value[0] + cov_value[2])
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        proj_value = p_proj[idx]
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        cuda_rect = _compute_tile_rect_wp(
            point_x, point_y, radius, grid_x, grid_y
        )
        cuda_rect_area = (
            (cuda_rect[2] - cuda_rect[0]) * (cuda_rect[3] - cuda_rect[1])
        )
        if cuda_rect_area == 0:
            return

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z[idx]
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = cov_value
        points_xy_image[idx] = point_image
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])
        stored_cov = conic_2d_inv[idx]
        stored_conic_opacity = conic_opacity[idx]
        tiles_touched[idx] = _count_covered_tiles_wp(
            point_x,
            point_y,
            stored_cov[0],
            stored_cov[2],
            stored_conic_opacity[0],
            stored_conic_opacity[1],
            stored_conic_opacity[2],
            stored_conic_opacity[3],
            radius,
            grid_x,
            grid_y,
            coverage_mode,
        )

    @wp.kernel
    def _project_visible_points_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        visible_mask[tid] = int(p_view_z > 0.2)

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        p_proj_out[tid] = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)

    @wp.kernel
    def _project_preprocess_visible_points_cov_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]
        base = tid * 6

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        if p_view_z <= VISIBILITY_NEAR_PLANE:
            visible_mask[tid] = int(0)
            return

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        proj_value = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)

        cov3d_lambda_upper = _conservative_cull_radius_from_cov3d_wp(
            cov3d_flat[base + 0],
            cov3d_flat[base + 1],
            cov3d_flat[base + 2],
            cov3d_flat[base + 3],
            cov3d_flat[base + 4],
            cov3d_flat[base + 5],
        )
        radius_upper = _preprocess_radius_upper_wp(cov3d_lambda_upper, p_view_z, tanfovx, tanfovy, image_width, image_height)
        if not _preprocess_rect_visible_wp(proj_value, radius_upper, image_width, image_height, grid_x, grid_y):
            visible_mask[tid] = int(0)
            return

        visible_mask[tid] = int(1)
        p_proj_out[tid] = proj_value

    @wp.kernel
    def _project_preprocess_visible_points_scale_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        scale_modifier: wp.float32,
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        if p_view_z <= VISIBILITY_NEAR_PLANE:
            visible_mask[tid] = int(0)
            return

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        proj_value = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)

        scale = scales[tid]
        scaled_max = scale_modifier * wp.max(wp.abs(scale[0]), wp.max(wp.abs(scale[1]), wp.abs(scale[2])))
        cov3d_lambda_upper = scaled_max * scaled_max
        radius_upper = _preprocess_radius_upper_wp(cov3d_lambda_upper, p_view_z, tanfovx, tanfovy, image_width, image_height)
        if not _preprocess_rect_visible_wp(proj_value, radius_upper, image_width, image_height, grid_x, grid_y):
            visible_mask[tid] = int(0)
            return

        visible_mask[tid] = int(1)
        p_proj_out[tid] = proj_value
else:
    pass

__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
