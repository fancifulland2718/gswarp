from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import (
    BLOCK_X,
    BLOCK_Y,
    NUM_CHANNELS,
    ONE_MINUS_ALPHA_MIN,
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
from .math_kernels import _compute_alpha, _compute_power, _conic_denom2inv_wp, _dnormvdv_wp


if wp is not None:
    _BLOCK_PIXELS_C = wp.constant(BLOCK_X * BLOCK_Y)

    @wp.kernel
    def _fused_backward_accumulate_warp_kernel(
        grad_projected_means: wp.array(dtype=wp.vec3),
        grad_cov_means: wp.array(dtype=wp.vec3),
        grad_sh_means: wp.array(dtype=wp.vec3),
        has_sh: wp.int32,
        render_grad_points: wp.array(dtype=wp.vec2),
        grad_means3D: wp.array(dtype=wp.vec3),
        grad_means2D: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        acc = grad_projected_means[i] + grad_cov_means[i]
        if has_sh != 0:
            acc = acc + grad_sh_means[i]
        grad_means3D[i] = acc
        rp = render_grad_points[i]
        grad_means2D[i] = wp.vec3(rp[0], rp[1], 0.0)

    @wp.kernel
    def _fused_backward_preprocess_accumulate_warp_kernel(
        # --- inputs (projected means part) ---
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        proj_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        grad_mean2d: wp.array(dtype=wp.vec2),
        grad_proj_2d: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        # --- inputs (cov2d + cov3d fused part) ---
        cov3d_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        cov2d_filter_variance: wp.float32,
        grad_conic_opacity: wp.array(dtype=wp.vec4),
        grad_conic_2d_flat: wp.array(dtype=wp.float32),
        grad_conic_2d_inv_flat: wp.array(dtype=wp.float32),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        # --- inputs (accumulate part) ---
        grad_sh_means: wp.array(dtype=wp.vec3),
        has_sh: wp.int32,
        render_grad_points: wp.array(dtype=wp.vec2),
        # --- outputs ---
        grad_means3D: wp.array(dtype=wp.vec3),
        grad_means2D: wp.array(dtype=wp.vec3),
        grad_scales_out: wp.array(dtype=wp.vec3),
        grad_rotations_out: wp.array(dtype=wp.vec4),
    ):
        tid = wp.tid()
        mean = means3d[tid]
        rad = radii[tid]

        # ======== Part A: backward_projected_means ========
        hom_x = proj_flat[0] * mean[0] + proj_flat[4] * mean[1] + proj_flat[8] * mean[2] + proj_flat[12]
        hom_y = proj_flat[1] * mean[0] + proj_flat[5] * mean[1] + proj_flat[9] * mean[2] + proj_flat[13]
        hom_w = proj_flat[3] * mean[0] + proj_flat[7] * mean[1] + proj_flat[11] * mean[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)

        grad_xy = grad_mean2d[tid]
        inv_w2 = inv_w * inv_w
        if rad > 0:
            grad_xy = grad_xy + grad_proj_2d[tid]
        gx = grad_xy[0]
        gy = grad_xy[1]
        gh = hom_x * gx + hom_y * gy
        pm_x = inv_w * (proj_flat[0] * gx + proj_flat[1] * gy) - proj_flat[3] * inv_w2 * gh
        pm_y = inv_w * (proj_flat[4] * gx + proj_flat[5] * gy) - proj_flat[7] * inv_w2 * gh
        pm_z = inv_w * (proj_flat[8] * gx + proj_flat[9] * gy) - proj_flat[11] * inv_w2 * gh

        # z should be defined in Part B, but for fused kernel, it can be here
        z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
        depth_grad = grad_depths[tid]
        pm_x = pm_x + (view_flat[2] - view_flat[3] * z) * depth_grad
        pm_y = pm_y + (view_flat[6] - view_flat[7] * z) * depth_grad
        pm_z = pm_z + (view_flat[10] - view_flat[11] * z) * depth_grad
        grad_projected = wp.vec3(pm_x, pm_y, pm_z)

        # ======== Part B: backward_cov2d_cov3d_fused ========
        grad_cov_means = wp.vec3(0.0, 0.0, 0.0)
        if rad <= 0:
            grad_scales_out[tid] = wp.vec3(0.0, 0.0, 0.0)
            grad_rotations_out[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        else:
            base = tid * 6
            _gc_v4 = grad_conic_opacity[tid]
            grad_conic2_base = tid * 3

            x = view_flat[0] * mean[0] + view_flat[4] * mean[1] + view_flat[8] * mean[2] + view_flat[12]
            y = view_flat[1] * mean[0] + view_flat[5] * mean[1] + view_flat[9] * mean[2] + view_flat[13]
            z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
            tz_inv = 1.0 / (z + 1.0e-7)
            limx = 1.3 * tanfovx
            limy = 1.3 * tanfovy
            txtz = x * tz_inv
            tytz = y * tz_inv
            x_grad_mul = float(1.0)
            y_grad_mul = float(1.0)
            if txtz < -limx or txtz > limx:
                x_grad_mul = float(0.0)
            if tytz < -limy or tytz > limy:
                y_grad_mul = float(0.0)
            tx = wp.clamp(txtz, -limx, limx)
            ty = wp.clamp(tytz, -limy, limy)
            j00 = focal_x * tz_inv
            j02 = - j00 * tx
            j11 = focal_y * tz_inv
            j12 = - j11 * ty

            w00 = view_flat[0]
            w01 = view_flat[1]
            w02 = view_flat[2]
            w10 = view_flat[4]
            w11 = view_flat[5]
            w12 = view_flat[6]
            w20 = view_flat[8]
            w21 = view_flat[9]
            w22 = view_flat[10]

            t00 = w00 * j00 + w02 * j02
            t10 = w10 * j00 + w12 * j02
            t20 = w20 * j00 + w22 * j02
            t01 = w01 * j11 + w02 * j12
            t11 = w11 * j11 + w12 * j12
            t21 = w21 * j11 + w22 * j12

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

            aa = t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3 + cov2d_filter_variance
            bb = t00 * vt1x + t10 * vt1y + t20 * vt1z
            cc = t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3 + cov2d_filter_variance
            denom = aa * cc - bb * bb
            denom2inv = _conic_denom2inv_wp(denom)

            total_conic0 = _gc_v4[0] + grad_conic_2d_flat[grad_conic2_base + 0]
            total_conic1 = _gc_v4[1] + grad_conic_2d_flat[grad_conic2_base + 1]
            total_conic2 = _gc_v4[2] + grad_conic_2d_flat[grad_conic2_base + 2]

            dL_da = denom2inv * (-cc * cc * total_conic0 + 2.0 * bb * cc * total_conic1 - bb * bb * total_conic2)
            dL_dc = denom2inv * (-aa * aa * total_conic2 + 2.0 * aa * bb * total_conic1 - bb * bb  * total_conic0)
            dL_db = denom2inv * 2.0 * (bb * cc * total_conic0 - (aa * cc + bb * bb) * total_conic1 + aa * bb * total_conic2)
            dL_da = dL_da + grad_conic_2d_inv_flat[grad_conic2_base + 0]
            dL_db = dL_db + grad_conic_2d_inv_flat[grad_conic2_base + 1]
            dL_dc = dL_dc + grad_conic_2d_inv_flat[grad_conic2_base + 2]

            gc0 = t00 * t00 * dL_da + t00 * t01 * dL_db + t01 * t01 * dL_dc
            gc3 = t10 * t10 * dL_da + t10 * t11 * dL_db + t11 * t11 * dL_dc
            gc5 = t20 * t20 * dL_da + t20 * t21 * dL_db + t21 * t21 * dL_dc
            gc1 = 2.0 * t00 * t10 * dL_da + (t00 * t11 + t01 * t10) * dL_db + 2.0 * t01 * t11 * dL_dc
            gc2 = 2.0 * t00 * t20 * dL_da + (t00 * t21 + t01 * t20) * dL_db + 2.0 * t01 * t21 * dL_dc
            gc4 = 2.0 * t10 * t20 * dL_da + (t10 * t21 + t11 * t20) * dL_db + 2.0 * t11 * t21 * dL_dc

            dL_dT00 = 2.0 * vt0x * dL_da + vt1x * dL_db
            dL_dT01 = 2.0 * vt0y * dL_da + vt1y * dL_db
            dL_dT02 = 2.0 * vt0z * dL_da + vt1z * dL_db
            dL_dT10 = vt0x * dL_db + 2.0 * vt1x * dL_dc
            dL_dT11 = vt0y * dL_db + 2.0 * vt1y * dL_dc
            dL_dT12 = vt0z * dL_db + 2.0 * vt1z * dL_dc

            dL_dJ00 = w00 * dL_dT00 + w10 * dL_dT01 + w20 * dL_dT02
            dL_dJ02 = w02 * dL_dT00 + w12 * dL_dT01 + w22 * dL_dT02
            dL_dJ11 = w01 * dL_dT10 + w11 * dL_dT11 + w21 * dL_dT12
            dL_dJ12 = w02 * dL_dT10 + w12 * dL_dT11 + w22 * dL_dT12

            dL_dtx = x_grad_mul * - j00 * tz_inv * dL_dJ02
            dL_dty = y_grad_mul * - j11 * tz_inv * dL_dJ12
            dL_dtz = - ( j00 * dL_dJ00 + j11 * dL_dJ11+ 2.0 *  j02 * dL_dJ02 + 2.0 * j12 *dL_dJ12 ) * tz_inv
            grad_cov_means = wp.vec3(
                w00 * dL_dtx + w01 * dL_dty + w02 * dL_dtz,
                w10 * dL_dtx + w11 * dL_dty + w12 * dL_dtz,
                w20 * dL_dtx + w21 * dL_dty + w22 * dL_dtz,
            )

            # --- cov3d backward (uses local gc0..gc5) ---
            ss = scales[tid] * scale_modifier
            qq = rotations[tid]
            rr = qq[0]
            xq = qq[1]
            yq = qq[2]
            zq = qq[3]

            r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
            r01 = 2.0 * (xq * yq + rr * zq)
            r02 = 2.0 * (xq * zq - rr * yq)
            r10 = 2.0 * (xq * yq - rr * zq)
            r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
            r12 = 2.0 * (yq * zq + rr * xq)
            r20 = 2.0 * (xq * zq + rr * yq)
            r21 = 2.0 * (yq * zq - rr * xq)
            r22 = 1.0 - 2.0 * (xq * xq + yq * yq)

            mm00 = ss[0] * r00
            mm01 = ss[0] * r01
            mm02 = ss[0] * r02
            mm10 = ss[1] * r10
            mm11 = ss[1] * r11
            mm12 = ss[1] * r12
            mm20 = ss[2] * r20
            mm21 = ss[2] * r21
            mm22 = ss[2] * r22

            sigma00 = gc0
            sigma01 = 0.5 * gc1
            sigma02 = 0.5 * gc2
            sigma11 = gc3
            sigma12 = 0.5 * gc4
            sigma22 = gc5

            dM00 = 2.0 * (mm00 * sigma00 + mm01 * sigma01 + mm02 * sigma02)
            dM01 = 2.0 * (mm00 * sigma01 + mm01 * sigma11 + mm02 * sigma12)
            dM02 = 2.0 * (mm00 * sigma02 + mm01 * sigma12 + mm02 * sigma22)
            dM10 = 2.0 * (mm10 * sigma00 + mm11 * sigma01 + mm12 * sigma02)
            dM11 = 2.0 * (mm10 * sigma01 + mm11 * sigma11 + mm12 * sigma12)
            dM12 = 2.0 * (mm10 * sigma02 + mm11 * sigma12 + mm12 * sigma22)
            dM20 = 2.0 * (mm20 * sigma00 + mm21 * sigma01 + mm22 * sigma02)
            dM21 = 2.0 * (mm20 * sigma01 + mm21 * sigma11 + mm22 * sigma12)
            dM22 = 2.0 * (mm20 * sigma02 + mm21 * sigma12 + mm22 * sigma22)

            grad_scales_out[tid] = wp.vec3(
                scale_modifier * (dM00 * r00 + dM01 * r01 + dM02 * r02),
                scale_modifier * (dM10 * r10 + dM11 * r11 + dM12 * r12),
                scale_modifier * (dM20 * r20 + dM21 * r21 + dM22 * r22),
            )

            dR00 = dM00 * ss[0]
            dR01 = dM01 * ss[0]
            dR02 = dM02 * ss[0]
            dR10 = dM10 * ss[1]
            dR11 = dM11 * ss[1]
            dR12 = dM12 * ss[1]
            dR20 = dM20 * ss[2]
            dR21 = dM21 * ss[2]
            dR22 = dM22 * ss[2]

            grad_rotations_out[tid] = wp.vec4(
                2.0 * zq * (dR01 - dR10) + 2.0 * yq * (dR20 - dR02) + 2.0 * xq * (dR12 - dR21),
                2.0 * yq * (dR10 + dR01) + 2.0 * zq * (dR20 + dR02) + 2.0 * rr * (dR12 - dR21) - 4.0 * xq * (dR22 + dR11),
                2.0 * xq * (dR10 + dR01) + 2.0 * rr * (dR20 - dR02) + 2.0 * zq * (dR12 + dR21) - 4.0 * yq * (dR22 + dR00),
                2.0 * rr * (dR01 - dR10) + 2.0 * xq * (dR20 + dR02) + 2.0 * yq * (dR12 + dR21) - 4.0 * zq * (dR11 + dR00),
            )

        # ======== Part C: accumulate ========
        acc = grad_projected + grad_cov_means
        if has_sh != 0:
            acc = acc + grad_sh_means[tid]
        grad_means3D[tid] = acc
        rp = render_grad_points[tid]
        grad_means2D[tid] = wp.vec3(rp[0], rp[1], 0.0)

    @wp.kernel
    def _backward_rgb_from_sh_v3_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_v3: wp.array(dtype=wp.vec3),
        degree: wp.int32,
        coeff_count: wp.int32,
        point_count: wp.int32,
        grad_color_v3: wp.array(dtype=wp.vec3),
        clamped_flat: wp.array(dtype=wp.int32),
        grad_means: wp.array(dtype=wp.vec3),
        grad_sh_v3: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = wp.dot(dir_orig, dir_orig)
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        # M2: inline clamp masking (avoids _masked_grad allocation)
        _gc = grad_color_v3[tid]
        grad_rgb = wp.vec3(
            _gc[0] * (1.0 - wp.float32(clamped_flat[tid * 3])),
            _gc[1] * (1.0 - wp.float32(clamped_flat[tid * 3 + 1])),
            _gc[2] * (1.0 - wp.float32(clamped_flat[tid * 3 + 2])),
        )

        acc_dx = wp.vec3(0.0, 0.0, 0.0)
        acc_dy = wp.vec3(0.0, 0.0, 0.0)
        acc_dz = wp.vec3(0.0, 0.0, 0.0)

        # SoA stride: shs_v3 and grad_sh_v3 are (K, P, 3) layout
        # shs_v3[k * _P + tid] gives coalesced access (adjacent threads 鈫?adjacent vec3s)
        _P = point_count

        if coeff_count > 0:
            grad_sh_v3[tid] = sh_c0 * grad_rgb

        if degree > 0 and coeff_count > 3:
            grad_sh_v3[_P + tid] = (-sh_c1 * y) * grad_rgb
            grad_sh_v3[2 * _P + tid] = (sh_c1 * z) * grad_rgb
            grad_sh_v3[3 * _P + tid] = (-sh_c1 * x) * grad_rgb

            acc_dx = (-sh_c1) * shs_v3[3 * _P + tid]
            acc_dy = (-sh_c1) * shs_v3[_P + tid]
            acc_dz = sh_c1 * shs_v3[2 * _P + tid]

        if degree > 1 and coeff_count > 8:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z

            grad_sh_v3[4 * _P + tid] = (sh_c2_0 * xy) * grad_rgb
            grad_sh_v3[5 * _P + tid] = (sh_c2_1 * yz) * grad_rgb
            grad_sh_v3[6 * _P + tid] = (sh_c2_2 * (2.0 * zz - xx - yy)) * grad_rgb
            grad_sh_v3[7 * _P + tid] = (sh_c2_3 * xz) * grad_rgb
            grad_sh_v3[8 * _P + tid] = (sh_c2_4 * (xx - yy)) * grad_rgb

            s4 = shs_v3[4 * _P + tid]
            s5 = shs_v3[5 * _P + tid]
            s6 = shs_v3[6 * _P + tid]
            s7 = shs_v3[7 * _P + tid]
            s8 = shs_v3[8 * _P + tid]

            acc_dx = acc_dx + (sh_c2_0 * y) * s4 + (sh_c2_2 * (-2.0 * x)) * s6 + (sh_c2_3 * z) * s7 + (sh_c2_4 * (2.0 * x)) * s8
            acc_dy = acc_dy + (sh_c2_0 * x) * s4 + (sh_c2_1 * z) * s5 + (sh_c2_2 * (-2.0 * y)) * s6 + (sh_c2_4 * (-2.0 * y)) * s8
            acc_dz = acc_dz + (sh_c2_1 * y) * s5 + (sh_c2_2 * (4.0 * z)) * s6 + (sh_c2_3 * x) * s7

            if degree > 2 and coeff_count > 15:
                grad_sh_v3[9 * _P + tid] = (sh_c3_0 * y * (3.0 * xx - yy)) * grad_rgb
                grad_sh_v3[10 * _P + tid] = (sh_c3_1 * xy * z) * grad_rgb
                grad_sh_v3[11 * _P + tid] = (sh_c3_2 * y * (4.0 * zz - xx - yy)) * grad_rgb
                grad_sh_v3[12 * _P + tid] = (sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * grad_rgb
                grad_sh_v3[13 * _P + tid] = (sh_c3_4 * x * (4.0 * zz - xx - yy)) * grad_rgb
                grad_sh_v3[14 * _P + tid] = (sh_c3_5 * z * (xx - yy)) * grad_rgb
                grad_sh_v3[15 * _P + tid] = (sh_c3_6 * x * (xx - 3.0 * yy)) * grad_rgb

                s9 = shs_v3[9 * _P + tid]
                s10 = shs_v3[10 * _P + tid]
                s11 = shs_v3[11 * _P + tid]
                s12 = shs_v3[12 * _P + tid]
                s13 = shs_v3[13 * _P + tid]
                s14 = shs_v3[14 * _P + tid]
                s15 = shs_v3[15 * _P + tid]

                acc_dx = acc_dx + (sh_c3_0 * (6.0 * xy)) * s9 + (sh_c3_1 * yz) * s10 + (sh_c3_2 * (-2.0 * xy)) * s11 + (sh_c3_3 * (-6.0 * xz)) * s12 + (sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy)) * s13 + (sh_c3_5 * (2.0 * xz)) * s14 + (sh_c3_6 * (3.0 * (xx - yy))) * s15
                acc_dy = acc_dy + (sh_c3_0 * (3.0 * (xx - yy))) * s9 + (sh_c3_1 * xz) * s10 + (sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx)) * s11 + (sh_c3_3 * (-6.0 * yz)) * s12 + (sh_c3_4 * (-2.0 * xy)) * s13 + (sh_c3_5 * (-2.0 * yz)) * s14 + (sh_c3_6 * (-6.0 * xy)) * s15
                acc_dz = acc_dz + (sh_c3_1 * xy) * s10 + (sh_c3_2 * (8.0 * yz)) * s11 + (sh_c3_3 * (3.0 * (2.0 * zz - xx - yy))) * s12 + (sh_c3_4 * (8.0 * xz)) * s13 + (sh_c3_5 * (xx - yy)) * s14

        dL_ddir = wp.vec3(
            wp.dot(acc_dx, grad_rgb),
            wp.dot(acc_dy, grad_rgb),
            wp.dot(acc_dz, grad_rgb),
        )
        grad_means[tid] = _dnormvdv_wp(dir_orig, dL_ddir)

    @wp.kernel
    def _backward_rgb_from_sh_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_flat: wp.array(dtype=wp.float32),
        degree: wp.int32,
        coeff_count: wp.int32,
        clamped_flat: wp.array(dtype=wp.int32),
        grad_color_flat: wp.array(dtype=wp.float32),
        grad_means: wp.array(dtype=wp.vec3),
        grad_sh_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        sh_base = tid * coeff_count * NUM_CHANNELS
        color_base = tid * NUM_CHANNELS

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = dir_orig[0] * dir_orig[0] + dir_orig[1] * dir_orig[1] + dir_orig[2] * dir_orig[2]
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        grad_rgb0 = grad_color_flat[color_base + 0] * float(1 - clamped_flat[color_base + 0])
        grad_rgb1 = grad_color_flat[color_base + 1] * float(1 - clamped_flat[color_base + 1])
        grad_rgb2 = grad_color_flat[color_base + 2] * float(1 - clamped_flat[color_base + 2])

        dRGBdx0 = float(0.0)
        dRGBdx1 = float(0.0)
        dRGBdx2 = float(0.0)
        dRGBdy0 = float(0.0)
        dRGBdy1 = float(0.0)
        dRGBdy2 = float(0.0)
        dRGBdz0 = float(0.0)
        dRGBdz1 = float(0.0)
        dRGBdz2 = float(0.0)

        if coeff_count > 0:
            grad_sh_flat[sh_base + 0] = sh_c0 * grad_rgb0
            grad_sh_flat[sh_base + 1] = sh_c0 * grad_rgb1
            grad_sh_flat[sh_base + 2] = sh_c0 * grad_rgb2

        if degree > 0 and coeff_count > 3:
            grad_sh_flat[sh_base + 3] = -sh_c1 * y * grad_rgb0
            grad_sh_flat[sh_base + 4] = -sh_c1 * y * grad_rgb1
            grad_sh_flat[sh_base + 5] = -sh_c1 * y * grad_rgb2
            grad_sh_flat[sh_base + 6] = sh_c1 * z * grad_rgb0
            grad_sh_flat[sh_base + 7] = sh_c1 * z * grad_rgb1
            grad_sh_flat[sh_base + 8] = sh_c1 * z * grad_rgb2
            grad_sh_flat[sh_base + 9] = -sh_c1 * x * grad_rgb0
            grad_sh_flat[sh_base + 10] = -sh_c1 * x * grad_rgb1
            grad_sh_flat[sh_base + 11] = -sh_c1 * x * grad_rgb2

            dRGBdx0 = -sh_c1 * shs_flat[sh_base + 9]
            dRGBdx1 = -sh_c1 * shs_flat[sh_base + 10]
            dRGBdx2 = -sh_c1 * shs_flat[sh_base + 11]
            dRGBdy0 = -sh_c1 * shs_flat[sh_base + 3]
            dRGBdy1 = -sh_c1 * shs_flat[sh_base + 4]
            dRGBdy2 = -sh_c1 * shs_flat[sh_base + 5]
            dRGBdz0 = sh_c1 * shs_flat[sh_base + 6]
            dRGBdz1 = sh_c1 * shs_flat[sh_base + 7]
            dRGBdz2 = sh_c1 * shs_flat[sh_base + 8]

            if degree > 1 and coeff_count > 8:
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z

                grad_sh_flat[sh_base + 12] = sh_c2_0 * xy * grad_rgb0
                grad_sh_flat[sh_base + 13] = sh_c2_0 * xy * grad_rgb1
                grad_sh_flat[sh_base + 14] = sh_c2_0 * xy * grad_rgb2
                grad_sh_flat[sh_base + 15] = sh_c2_1 * yz * grad_rgb0
                grad_sh_flat[sh_base + 16] = sh_c2_1 * yz * grad_rgb1
                grad_sh_flat[sh_base + 17] = sh_c2_1 * yz * grad_rgb2
                grad_sh_flat[sh_base + 18] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb0
                grad_sh_flat[sh_base + 19] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb1
                grad_sh_flat[sh_base + 20] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb2
                grad_sh_flat[sh_base + 21] = sh_c2_3 * xz * grad_rgb0
                grad_sh_flat[sh_base + 22] = sh_c2_3 * xz * grad_rgb1
                grad_sh_flat[sh_base + 23] = sh_c2_3 * xz * grad_rgb2
                grad_sh_flat[sh_base + 24] = sh_c2_4 * (xx - yy) * grad_rgb0
                grad_sh_flat[sh_base + 25] = sh_c2_4 * (xx - yy) * grad_rgb1
                grad_sh_flat[sh_base + 26] = sh_c2_4 * (xx - yy) * grad_rgb2

                dRGBdx0 = dRGBdx0 + sh_c2_0 * y * shs_flat[sh_base + 12] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 18] + sh_c2_3 * z * shs_flat[sh_base + 21] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 24]
                dRGBdx1 = dRGBdx1 + sh_c2_0 * y * shs_flat[sh_base + 13] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 19] + sh_c2_3 * z * shs_flat[sh_base + 22] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 25]
                dRGBdx2 = dRGBdx2 + sh_c2_0 * y * shs_flat[sh_base + 14] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 20] + sh_c2_3 * z * shs_flat[sh_base + 23] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 26]
                dRGBdy0 = dRGBdy0 + sh_c2_0 * x * shs_flat[sh_base + 12] + sh_c2_1 * z * shs_flat[sh_base + 15] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 18] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 24]
                dRGBdy1 = dRGBdy1 + sh_c2_0 * x * shs_flat[sh_base + 13] + sh_c2_1 * z * shs_flat[sh_base + 16] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 19] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 25]
                dRGBdy2 = dRGBdy2 + sh_c2_0 * x * shs_flat[sh_base + 14] + sh_c2_1 * z * shs_flat[sh_base + 17] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 20] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 26]
                dRGBdz0 = dRGBdz0 + sh_c2_1 * y * shs_flat[sh_base + 15] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 18] + sh_c2_3 * x * shs_flat[sh_base + 21]
                dRGBdz1 = dRGBdz1 + sh_c2_1 * y * shs_flat[sh_base + 16] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 19] + sh_c2_3 * x * shs_flat[sh_base + 22]
                dRGBdz2 = dRGBdz2 + sh_c2_1 * y * shs_flat[sh_base + 17] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 20] + sh_c2_3 * x * shs_flat[sh_base + 23]

                if degree > 2 and coeff_count > 15:
                    grad_sh_flat[sh_base + 27] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 28] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 29] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 30] = sh_c3_1 * xy * z * grad_rgb0
                    grad_sh_flat[sh_base + 31] = sh_c3_1 * xy * z * grad_rgb1
                    grad_sh_flat[sh_base + 32] = sh_c3_1 * xy * z * grad_rgb2
                    grad_sh_flat[sh_base + 33] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 34] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 35] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 36] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb0
                    grad_sh_flat[sh_base + 37] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb1
                    grad_sh_flat[sh_base + 38] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb2
                    grad_sh_flat[sh_base + 39] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 40] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 41] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 42] = sh_c3_5 * z * (xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 43] = sh_c3_5 * z * (xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 44] = sh_c3_5 * z * (xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 45] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb0
                    grad_sh_flat[sh_base + 46] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb1
                    grad_sh_flat[sh_base + 47] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb2

                    dRGBdx0 = dRGBdx0 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 27] + sh_c3_1 * yz * shs_flat[sh_base + 30] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 33] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 36] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 39] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 42] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 45]
                    dRGBdx1 = dRGBdx1 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 28] + sh_c3_1 * yz * shs_flat[sh_base + 31] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 34] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 37] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 40] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 43] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 46]
                    dRGBdx2 = dRGBdx2 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 29] + sh_c3_1 * yz * shs_flat[sh_base + 32] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 35] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 38] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 41] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 44] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 47]
                    dRGBdy0 = dRGBdy0 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 27] + sh_c3_1 * xz * shs_flat[sh_base + 30] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 33] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 36] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 39] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 42] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 45]
                    dRGBdy1 = dRGBdy1 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 28] + sh_c3_1 * xz * shs_flat[sh_base + 31] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 34] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 37] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 40] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 43] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 46]
                    dRGBdy2 = dRGBdy2 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 29] + sh_c3_1 * xz * shs_flat[sh_base + 32] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 35] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 38] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 41] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 44] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 47]
                    dRGBdz0 = dRGBdz0 + sh_c3_1 * xy * shs_flat[sh_base + 30] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 33] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 36] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 39] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 42]
                    dRGBdz1 = dRGBdz1 + sh_c3_1 * xy * shs_flat[sh_base + 31] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 34] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 37] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 40] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 43]
                    dRGBdz2 = dRGBdz2 + sh_c3_1 * xy * shs_flat[sh_base + 32] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 35] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 38] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 41] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 44]

        dL_ddir = wp.vec3(
            dRGBdx0 * grad_rgb0 + dRGBdx1 * grad_rgb1 + dRGBdx2 * grad_rgb2,
            dRGBdy0 * grad_rgb0 + dRGBdy1 * grad_rgb1 + dRGBdy2 * grad_rgb2,
            dRGBdz0 * grad_rgb0 + dRGBdz1 * grad_rgb1 + dRGBdz2 * grad_rgb2,
        )
        grad_means[tid] = _dnormvdv_wp(dir_orig, dL_ddir)

    @wp.kernel
    def _backward_render_tiles_warp32_kernel(
        ranges_flat: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        points_xy_image: wp.array(dtype=wp.vec2),
        features_flat: wp.array(dtype=wp.float32),
        depths: wp.array(dtype=wp.float32),
        conic_opacity: wp.array(dtype=wp.vec4),
        background: wp.array(dtype=wp.float32),
        out_alpha_flat: wp.array(dtype=wp.float32),
        n_contrib: wp.array(dtype=wp.int32),
        grad_color_flat: wp.array(dtype=wp.float32),
        grad_depth_flat: wp.array(dtype=wp.float32),
        grad_alpha_flat: wp.array(dtype=wp.float32),
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        num_tiles: wp.int32,
        compute_depth: wp.int32,
        grad_points_xy: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        grad_conic_opacity: wp.array(dtype=wp.vec4),
        grad_feature: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        block_id = tid // _BLOCK_PIXELS_C
        local_id = tid % _BLOCK_PIXELS_C

        if block_id >= num_tiles:
            return

        tile_x = block_id % grid_x
        tile_y = block_id // grid_x
        pix_x = tile_x * BLOCK_X + (local_id % BLOCK_X)
        pix_y = tile_y * BLOCK_Y + (local_id // BLOCK_X)
        total_pixels = image_width * image_height

        inside = int(0)
        if pix_x < image_width and pix_y < image_height:
            inside = 1
        pixel_id = pix_y * image_width + pix_x

        start = ranges_flat[block_id * 2]
        end = ranges_flat[block_id * 2 + 1]
        n_gs = end - start

        last_contributor = int(0)
        if inside != 0 and n_gs > 0:
            last_contributor = n_contrib[pixel_id]

        # Resolve the warp loop bound before loading gradients and recurrence
        # state so groups with no contributors exit after one reduction.
        t_lc = wp.tile(last_contributor)
        t_max_lc = wp.tile_reduce(wp.max, t_lc)
        warp_max_lc = wp.tile_extract(t_max_lc, 0)
        if warp_max_lc == 0:
            return

        T_final = float(0.0)
        T = float(0.0)
        ddelx_dx = 0.5 * float(image_width)
        ddely_dy = 0.5 * float(image_height)
        pixf_x = float(pix_x)
        pixf_y = float(pix_y)
        dL_dpixel = wp.vec3(0.0, 0.0, 0.0)
        dL_dpixel_depth = float(0.0)
        dL_dalpha = float(0.0)
        bg_dot = float(0.0)

        if inside != 0 and n_gs > 0:
            T_final = 1.0 - out_alpha_flat[pixel_id]
            T = T_final
            dL_dpixel = wp.vec3(
                grad_color_flat[pixel_id],
                grad_color_flat[total_pixels + pixel_id],
                grad_color_flat[2 * total_pixels + pixel_id],
            )
            if compute_depth != 0:
                dL_dpixel_depth = grad_depth_flat[pixel_id]
            dL_dalpha = grad_alpha_flat[pixel_id]
            bg = wp.vec3(background[0], background[1], background[2])
            bg_dot = wp.dot(bg, dL_dpixel)

        accum_rec = wp.vec3(0.0, 0.0, 0.0)
        accum_depth_rec = float(0.0)
        accum_alpha_rec = float(0.0)
        last_alpha = float(0.0)
        last_color = wp.vec3(0.0, 0.0, 0.0)
        last_depth = float(0.0)

        for step in range(warp_max_lc):
            pos = warp_max_lc - step - 1
            coll_id = point_list[start + pos]

            my_grad_feat = wp.vec3(0.0, 0.0, 0.0)
            my_grad_depth = float(0.0)
            my_grad_xy = wp.vec2(0.0, 0.0)
            my_grad_co = wp.vec4(0.0, 0.0, 0.0, 0.0)
            has_contribution = int(0)

            if inside != 0 and pos < last_contributor:
                xy = points_xy_image[coll_id]
                con_o = conic_opacity[coll_id]
                d_x = xy[0] - pixf_x
                d_y = xy[1] - pixf_y
                power = _compute_power(con_o, d_x, d_y)

                if power <= 0.0:
                    G = wp.exp(power)
                    alpha = _compute_alpha(con_o, power)
                    if alpha >= (1.0 / 255.0):
                        has_contribution = 1
                        one_minus_alpha = wp.max(1.0 - alpha, ONE_MINUS_ALPHA_MIN)
                        T = T / one_minus_alpha
                        dchannel_dcolor = alpha * T

                        feature_base = coll_id * NUM_CHANNELS
                        color = wp.vec3(
                            features_flat[feature_base + 0],
                            features_flat[feature_base + 1],
                            features_flat[feature_base + 2],
                        )
                        accum_rec = last_alpha * last_color + (1.0 - last_alpha) * accum_rec
                        last_color = color
                        dL_dopa = wp.dot(color - accum_rec, dL_dpixel)
                        my_grad_feat = dchannel_dcolor * dL_dpixel

                        if compute_depth != 0:
                            depth_value = depths[coll_id]
                            accum_depth_rec = last_alpha * last_depth + (1.0 - last_alpha) * accum_depth_rec
                            last_depth = depth_value
                            dL_dopa = dL_dopa + (depth_value - accum_depth_rec) * dL_dpixel_depth
                            my_grad_depth = dchannel_dcolor * dL_dpixel_depth

                        accum_alpha_rec = last_alpha + (1.0 - last_alpha) * accum_alpha_rec
                        dL_dopa = dL_dopa + (1.0 - accum_alpha_rec) * dL_dalpha
                        dL_dopa = dL_dopa * T
                        last_alpha = alpha
                        dL_dopa = dL_dopa + (-T_final / one_minus_alpha) * bg_dot

                        dL_dG = con_o[3] * dL_dopa
                        gdx = G * d_x
                        gdy = G * d_y
                        dG_ddelx = -gdx * con_o[0] - gdy * con_o[1]
                        dG_ddely = -gdy * con_o[2] - gdx * con_o[1]

                        my_grad_xy = wp.vec2(dL_dG * dG_ddelx * ddelx_dx, dL_dG * dG_ddely * ddely_dy)
                        my_grad_co = wp.vec4(-0.5 * gdx * d_x * dL_dG, -0.5 * gdx * d_y * dL_dG, -0.5 * gdy * d_y * dL_dG, G * dL_dopa)

            t_has_contribution = wp.tile(has_contribution)
            s_has_contribution = wp.tile_sum(t_has_contribution)
            if wp.tile_extract(s_has_contribution, 0) > 0:
                # The vote is warp-uniform. Contributor steps with no valid
                # footprint skip three vector reductions and zero atomics.
                t_feat = wp.tile(my_grad_feat, preserve_type=True)
                s_feat = wp.tile_reduce(wp.add, t_feat)
                wp.tile_atomic_add(grad_feature, s_feat, offset=coll_id)

                if compute_depth != 0:
                    t_depth = wp.tile(my_grad_depth)
                    s_depth = wp.tile_sum(t_depth)
                    wp.tile_atomic_add(grad_depths, s_depth, offset=coll_id)

                t_xy = wp.tile(my_grad_xy, preserve_type=True)
                s_xy = wp.tile_reduce(wp.add, t_xy)
                wp.tile_atomic_add(grad_points_xy, s_xy, offset=coll_id)

                t_co = wp.tile(my_grad_co, preserve_type=True)
                s_co = wp.tile_reduce(wp.add, t_co)
                wp.tile_atomic_add(grad_conic_opacity, s_co, offset=coll_id)

    @wp.kernel
    def _backward_cov2d_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        cov2d_filter_variance: wp.float32,
        grad_conic_flat: wp.array(dtype=wp.float32),
        grad_conic_2d_flat: wp.array(dtype=wp.float32),
        grad_conic_2d_inv_flat: wp.array(dtype=wp.float32),
        grad_means_out: wp.array(dtype=wp.vec3),
        grad_cov_out_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        if radii[tid] <= 0:
            return

        mean = means3d[tid]
        base = tid * 6
        grad_conic_base = tid * 3
        grad_conic2_base = tid * 3

        x = view_flat[0] * mean[0] + view_flat[4] * mean[1] + view_flat[8] * mean[2] + view_flat[12]
        y = view_flat[1] * mean[0] + view_flat[5] * mean[1] + view_flat[9] * mean[2] + view_flat[13]
        z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = x / z
        tytz = y / z
        tx = wp.clamp(txtz, -limx, limx) * z
        ty = wp.clamp(tytz, -limy, limy) * z
        x_grad_mul = float(1.0)
        y_grad_mul = float(1.0)
        if txtz < -limx or txtz > limx:
            x_grad_mul = float(0.0)
        if tytz < -limy or tytz > limy:
            y_grad_mul = float(0.0)

        j00 = focal_x / z
        j02 = -(focal_x * tx) / (z * z)
        j11 = focal_y / z
        j12 = -(focal_y * ty) / (z * z)

        w00 = view_flat[0]
        w01 = view_flat[1]
        w02 = view_flat[2]
        w10 = view_flat[4]
        w11 = view_flat[5]
        w12 = view_flat[6]
        w20 = view_flat[8]
        w21 = view_flat[9]
        w22 = view_flat[10]

        t00 = w00 * j00 + w02 * j02
        t10 = w10 * j00 + w12 * j02
        t20 = w20 * j00 + w22 * j02
        t01 = w01 * j11 + w02 * j12
        t11 = w11 * j11 + w12 * j12
        t21 = w21 * j11 + w22 * j12

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

        a = t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3 + cov2d_filter_variance
        b = t00 * vt1x + t10 * vt1y + t20 * vt1z
        c = t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3 + cov2d_filter_variance
        denom = a * c - b * b
        denom2inv = _conic_denom2inv_wp(denom)

        total_conic0 = grad_conic_flat[grad_conic_base + 0] + grad_conic_2d_flat[grad_conic2_base + 0]
        total_conic1 = grad_conic_flat[grad_conic_base + 1] + grad_conic_2d_flat[grad_conic2_base + 1]
        total_conic2 = grad_conic_flat[grad_conic_base + 2] + grad_conic_2d_flat[grad_conic2_base + 2]

        dL_da = denom2inv * (-c * c * total_conic0 + 2.0 * b * c * total_conic1 + (denom - a * c) * total_conic2)
        dL_dc = denom2inv * (-a * a * total_conic2 + 2.0 * a * b * total_conic1 + (denom - a * c) * total_conic0)
        dL_db = denom2inv * 2.0 * (b * c * total_conic0 - (denom + 2.0 * b * b) * total_conic1 + a * b * total_conic2)
        dL_da = dL_da + grad_conic_2d_inv_flat[grad_conic2_base + 0]
        dL_db = dL_db + grad_conic_2d_inv_flat[grad_conic2_base + 1]
        dL_dc = dL_dc + grad_conic_2d_inv_flat[grad_conic2_base + 2]

        grad_cov_out_flat[base + 0] = t00 * t00 * dL_da + t00 * t01 * dL_db + t01 * t01 * dL_dc
        grad_cov_out_flat[base + 3] = t10 * t10 * dL_da + t10 * t11 * dL_db + t11 * t11 * dL_dc
        grad_cov_out_flat[base + 5] = t20 * t20 * dL_da + t20 * t21 * dL_db + t21 * t21 * dL_dc
        grad_cov_out_flat[base + 1] = 2.0 * t00 * t10 * dL_da + (t00 * t11 + t01 * t10) * dL_db + 2.0 * t01 * t11 * dL_dc
        grad_cov_out_flat[base + 2] = 2.0 * t00 * t20 * dL_da + (t00 * t21 + t01 * t20) * dL_db + 2.0 * t01 * t21 * dL_dc
        grad_cov_out_flat[base + 4] = 2.0 * t10 * t20 * dL_da + (t10 * t21 + t11 * t20) * dL_db + 2.0 * t11 * t21 * dL_dc

        dL_dT00 = 2.0 * (t00 * v00 + t10 * v01 + t20 * v02) * dL_da + (t01 * v00 + t11 * v01 + t21 * v02) * dL_db
        dL_dT01 = 2.0 * (t00 * v01 + t10 * v11 + t20 * v12) * dL_da + (t01 * v01 + t11 * v11 + t21 * v12) * dL_db
        dL_dT02 = 2.0 * (t00 * v02 + t10 * v12 + t20 * v22) * dL_da + (t01 * v02 + t11 * v12 + t21 * v22) * dL_db
        dL_dT10 = 2.0 * (t01 * v00 + t11 * v01 + t21 * v02) * dL_dc + (t00 * v00 + t10 * v01 + t20 * v02) * dL_db
        dL_dT11 = 2.0 * (t01 * v01 + t11 * v11 + t21 * v12) * dL_dc + (t00 * v01 + t10 * v11 + t20 * v12) * dL_db
        dL_dT12 = 2.0 * (t01 * v02 + t11 * v12 + t21 * v22) * dL_dc + (t00 * v02 + t10 * v12 + t20 * v22) * dL_db

        dL_dJ00 = w00 * dL_dT00 + w10 * dL_dT01 + w20 * dL_dT02
        dL_dJ02 = w02 * dL_dT00 + w12 * dL_dT01 + w22 * dL_dT02
        dL_dJ11 = w01 * dL_dT10 + w11 * dL_dT11 + w21 * dL_dT12
        dL_dJ12 = w02 * dL_dT10 + w12 * dL_dT11 + w22 * dL_dT12

        tz_inv = 1.0 / z
        tz2 = tz_inv * tz_inv
        tz3 = tz2 * tz_inv
        dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02
        dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12
        dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2.0 * focal_x * tx) * tz3 * dL_dJ02 + (2.0 * focal_y * ty) * tz3 * dL_dJ12

        grad_means_out[tid] = wp.vec3(
            view_flat[0] * dL_dtx + view_flat[1] * dL_dty + view_flat[2] * dL_dtz,
            view_flat[4] * dL_dtx + view_flat[5] * dL_dty + view_flat[6] * dL_dtz,
            view_flat[8] * dL_dtx + view_flat[9] * dL_dty + view_flat[10] * dL_dtz,
        )

    @wp.kernel
    def _backward_projected_means_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        proj_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        grad_mean2d: wp.array(dtype=wp.vec2),
        grad_proj_2d: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        out_grad_means: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        mean = means3d[tid]

        hom_x = proj_flat[0] * mean[0] + proj_flat[4] * mean[1] + proj_flat[8] * mean[2] + proj_flat[12]
        hom_y = proj_flat[1] * mean[0] + proj_flat[5] * mean[1] + proj_flat[9] * mean[2] + proj_flat[13]
        hom_w = proj_flat[3] * mean[0] + proj_flat[7] * mean[1] + proj_flat[11] * mean[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        mul1 = hom_x * inv_w * inv_w
        mul2 = hom_y * inv_w * inv_w

        grad_xy = grad_mean2d[tid]
        if radii[tid] > 0:
            grad_xy = grad_xy + grad_proj_2d[tid]
        out_x = (proj_flat[0] * inv_w - proj_flat[3] * mul1) * grad_xy[0] + (proj_flat[1] * inv_w - proj_flat[3] * mul2) * grad_xy[1]
        out_y = (proj_flat[4] * inv_w - proj_flat[7] * mul1) * grad_xy[0] + (proj_flat[5] * inv_w - proj_flat[7] * mul2) * grad_xy[1]
        out_z = (proj_flat[8] * inv_w - proj_flat[11] * mul1) * grad_xy[0] + (proj_flat[9] * inv_w - proj_flat[11] * mul2) * grad_xy[1]

        mul3 = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
        depth_grad = grad_depths[tid]
        out_x = out_x + (view_flat[2] - view_flat[3] * mul3) * depth_grad
        out_y = out_y + (view_flat[6] - view_flat[7] * mul3) * depth_grad
        out_z = out_z + (view_flat[10] - view_flat[11] * mul3) * depth_grad
        out_grad_means[tid] = wp.vec3(out_x, out_y, out_z)

    @wp.kernel
    def _backward_cov3d_from_scale_rotation_warp_kernel(
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        grad_cov3d_flat: wp.array(dtype=wp.float32),
        grad_scales: wp.array(dtype=wp.vec3),
        grad_rotations: wp.array(dtype=wp.vec4),
    ):
        tid = wp.tid()
        s = scales[tid] * scale_modifier
        q = rotations[tid]
        grad_base = tid * 6

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

        sigma00 = grad_cov3d_flat[grad_base + 0]
        sigma01 = 0.5 * grad_cov3d_flat[grad_base + 1]
        sigma02 = 0.5 * grad_cov3d_flat[grad_base + 2]
        sigma11 = grad_cov3d_flat[grad_base + 3]
        sigma12 = 0.5 * grad_cov3d_flat[grad_base + 4]
        sigma22 = grad_cov3d_flat[grad_base + 5]

        dM00 = 2.0 * (m00 * sigma00 + m01 * sigma01 + m02 * sigma02)
        dM01 = 2.0 * (m00 * sigma01 + m01 * sigma11 + m02 * sigma12)
        dM02 = 2.0 * (m00 * sigma02 + m01 * sigma12 + m02 * sigma22)
        dM10 = 2.0 * (m10 * sigma00 + m11 * sigma01 + m12 * sigma02)
        dM11 = 2.0 * (m10 * sigma01 + m11 * sigma11 + m12 * sigma12)
        dM12 = 2.0 * (m10 * sigma02 + m11 * sigma12 + m12 * sigma22)
        dM20 = 2.0 * (m20 * sigma00 + m21 * sigma01 + m22 * sigma02)
        dM21 = 2.0 * (m20 * sigma01 + m21 * sigma11 + m22 * sigma12)
        dM22 = 2.0 * (m20 * sigma02 + m21 * sigma12 + m22 * sigma22)

        grad_scales[tid] = wp.vec3(
            scale_modifier * (dM00 * r00 + dM01 * r01 + dM02 * r02),
            scale_modifier * (dM10 * r10 + dM11 * r11 + dM12 * r12),
            scale_modifier * (dM20 * r20 + dM21 * r21 + dM22 * r22),
        )

        dR00 = dM00 * s[0]
        dR01 = dM01 * s[0]
        dR02 = dM02 * s[0]
        dR10 = dM10 * s[1]
        dR11 = dM11 * s[1]
        dR12 = dM12 * s[1]
        dR20 = dM20 * s[2]
        dR21 = dM21 * s[2]
        dR22 = dM22 * s[2]

        grad_rotations[tid] = wp.vec4(
            2.0 * z * (dR01 - dR10) + 2.0 * y * (dR20 - dR02) + 2.0 * x * (dR12 - dR21),
            2.0 * y * (dR10 + dR01) + 2.0 * z * (dR20 + dR02) + 2.0 * r * (dR12 - dR21) - 4.0 * x * (dR22 + dR11),
            2.0 * x * (dR10 + dR01) + 2.0 * r * (dR20 - dR02) + 2.0 * z * (dR12 + dR21) - 4.0 * y * (dR22 + dR00),
            2.0 * r * (dR01 - dR10) + 2.0 * x * (dR20 + dR02) + 2.0 * y * (dR12 + dR21) - 4.0 * z * (dR11 + dR00),
        )
else:
    pass

__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
