from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .constants import BLOCK_X, BLOCK_Y, NUM_CHANNELS

if wp is not None:
    _BLOCK_PIXELS_C = wp.constant(BLOCK_X * BLOCK_Y)

    @wp.kernel
    def _fused_flow_grad_prep_warp_kernel(
        grad_proj_2D: wp.array(dtype=wp.vec2),
        render_grad_points: wp.array(dtype=wp.vec2),
        grad_conic_2D: wp.array(dtype=wp.vec3),
        render_grad_conic_opacity: wp.array(dtype=wp.vec4),
        grad_proj_2d_out: wp.array(dtype=wp.vec2),
        grad_conic_2d_out: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        gp = grad_proj_2D[i]
        rp = render_grad_points[i]
        grad_proj_2d_out[i] = wp.vec2(gp[0] + rp[0], gp[1] + rp[1])
        gc = grad_conic_2D[i]
        rc = render_grad_conic_opacity[i]
        grad_conic_2d_out[i] = wp.vec3(gc[0] + rc[0], gc[1] + rc[1], gc[2] + rc[2])

    @wp.kernel
    def _render_tiles_tiled256_warp_kernel(
        ranges_flat: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        points_xy_image: wp.array(dtype=wp.vec2),
        features_flat: wp.array(dtype=wp.float32),
        depths: wp.array(dtype=wp.float32),
        conic_opacity: wp.array(dtype=wp.vec4),
        background: wp.array(dtype=wp.float32),
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        num_tiles: wp.int32,
        compute_depth: wp.int32,
        write_aux: wp.int32,
        top_k: wp.int32,
        out_color_flat: wp.array(dtype=wp.float32),
        out_depth_flat: wp.array(dtype=wp.float32),
        out_alpha_flat: wp.array(dtype=wp.float32),
        n_contrib: wp.array(dtype=wp.int32),
        gs_per_pixel_flat: wp.array(dtype=wp.int32),
        weight_per_gs_pixel_flat: wp.array(dtype=wp.float32),
        x_mu_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        tile_id = tid // _BLOCK_PIXELS_C
        local_id = tid % _BLOCK_PIXELS_C

        if tile_id >= num_tiles:
            return

        # Map local_id 鈫?pixel within the 16脳16 tile
        tile_x = tile_id % grid_x
        tile_y = tile_id // grid_x
        pix_x = tile_x * BLOCK_X + (local_id % BLOCK_X)
        pix_y = tile_y * BLOCK_Y + (local_id // BLOCK_X)
        total_pixels = image_width * image_height

        inside = int(0)
        if pix_x < image_width and pix_y < image_height:
            inside = 1
        pixel_id = pix_y * image_width + pix_x

        start = ranges_flat[tile_id * 2]
        end = ranges_flat[tile_id * 2 + 1]
        n_gs = end - start

        T = float(1.0)
        contributor = int(0)
        last_contributor = int(0)
        n_success = int(0)
        color0 = float(0.0)
        color1 = float(0.0)
        color2 = float(0.0)
        weight = float(0.0)
        depth_acc = float(0.0)
        pixf_x = float(pix_x)
        pixf_y = float(pix_y)
        done = int(0)
        if inside == 0:
            done = 1

        rounds = (n_gs + 255) // 256
        for i in range(rounds):
            # ---- Block-level early-exit vote ----
            t_done = wp.tile(done)
            total_done = wp.tile_sum(t_done)
            all_done = wp.tile_extract(total_done, 0)
            if all_done == _BLOCK_PIXELS_C:
                break

            # ---- Cooperative load: each thread loads 1 Gaussian ----
            progress = i * 256 + local_id
            # 2C: simplified safe_idx 鈥?end > start guaranteed by n_gs > 0
            safe_idx = wp.min(start + progress, end - 1)
            my_id = point_list[safe_idx]

            my_xy = points_xy_image[my_id]
            my_co = conic_opacity[my_id]
            my_feature = wp.vec3(0.0, 0.0, 0.0)
            my_depth = float(0.0)
            if progress < n_gs:
                feat_base = my_id * NUM_CHANNELS
                my_feature = wp.vec3(
                    features_flat[feat_base + 0],
                    features_flat[feat_base + 1],
                    features_flat[feat_base + 2],
                )
                if compute_depth != 0:
                    my_depth = depths[my_id]

            # Broadcast each Gaussian payload once to all pixels in the tile.
            t_id = wp.tile(my_id)
            t_xy = wp.tile(my_xy, preserve_type=True)
            t_co = wp.tile(my_co, preserve_type=True)
            t_feature = wp.tile(my_feature, preserve_type=True)
            t_depth = wp.tile(my_depth)

            # ---- Inner loop: all 256 pixels process the batch ----
            batch_count = wp.min(256, n_gs - i * 256)
            for j in range(batch_count):
                contributor = contributor + 1

                if done == 0:
                    coll_id = wp.tile_extract(t_id, j)
                    # 2A: extract vec2/vec4 directly
                    xy_j = wp.tile_extract(t_xy, j)
                    co_j = wp.tile_extract(t_co, j)

                    d_x = xy_j[0] - pixf_x
                    d_y = xy_j[1] - pixf_y
                    power = -0.5 * (co_j[0] * d_x * d_x + co_j[2] * d_y * d_y) - co_j[1] * d_x * d_y

                    if power <= 0.0:
                        alpha = wp.min(float(0.99), co_j[3] * wp.exp(power))
                        if alpha >= (1.0 / 255.0):
                            test_T = T * (1.0 - alpha)
                            if test_T < 0.0001:
                                done = 1
                            else:
                                contribution = alpha * T
                                feature_j = wp.tile_extract(t_feature, j)
                                color0 = color0 + feature_j[0] * contribution
                                color1 = color1 + feature_j[1] * contribution
                                color2 = color2 + feature_j[2] * contribution
                                weight = weight + contribution
                                if compute_depth != 0:
                                    depth_j = wp.tile_extract(t_depth, j)
                                    depth_acc = depth_acc + depth_j * contribution
                                # Flow aux: record up to top_k successful contributors.
                                if write_aux != 0 and n_success < top_k:
                                    aux_idx = n_success * total_pixels + pixel_id
                                    gs_per_pixel_flat[aux_idx] = coll_id
                                    weight_per_gs_pixel_flat[aux_idx] = contribution
                                    x_mu_flat[aux_idx] = d_x
                                    x_mu_flat[top_k * total_pixels + aux_idx] = d_y
                                n_success = n_success + 1
                                T = test_T
                                last_contributor = contributor

        if inside != 0:
            n_contrib[pixel_id] = last_contributor
            out_color_flat[pixel_id] = color0 + T * background[0]
            out_color_flat[total_pixels + pixel_id] = color1 + T * background[1]
            out_color_flat[2 * total_pixels + pixel_id] = color2 + T * background[2]
            out_alpha_flat[pixel_id] = weight
            out_depth_flat[pixel_id] = depth_acc
else:
    pass

__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
