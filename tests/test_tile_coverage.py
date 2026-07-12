"""CPU oracles and static contracts for exact tile coverage."""

from __future__ import annotations

import inspect
import math
import random
import unittest

from gswarp._internal.coverage import (
    FOOTPRINT_APPROXIMATE_SCREEN_CONIC,
    FOOTPRINT_AXIS_ALIGNED,
    FOOTPRINT_CUSTOM,
    FOOTPRINT_EXACT_SCREEN_CONIC,
    resolve_tile_coverage_mode,
    tile_coverage_mode_id,
)
from gswarp._internal.backends.warp import binning_kernels, preprocess_kernels
from gswarp._internal.backends.warp.binning_ops import _depth_sort_payload_layout


BLOCK = 16
Q_EPSILON = 1.0e-4


def _tile_rect(point_x, point_y, radius_x, radius_y, grid_x, grid_y):
    return (
        min(grid_x, max(0, int((point_x - radius_x) / BLOCK))),
        min(grid_y, max(0, int((point_y - radius_y) / BLOCK))),
        min(grid_x, max(0, int((point_x + radius_x + BLOCK - 1) / BLOCK))),
        min(grid_y, max(0, int((point_y + radius_y + BLOCK - 1) / BLOCK))),
    )


def _candidate_rect(point, conic, opacity, width, height):
    cuda_rect, cov_xx, cov_yy = _cuda_rect_and_covariance(point, conic, width, height)
    threshold = 2.0 * math.log(max(255.0 * opacity, 1.0))
    grid_x = (width + BLOCK - 1) // BLOCK
    grid_y = (height + BLOCK - 1) // BLOCK
    if threshold <= 0.0:
        return (0, 0, 0, 0)
    radius_x = math.ceil(math.sqrt(max(threshold * cov_xx, 0.0)) + 1.0)
    radius_y = math.ceil(math.sqrt(max(threshold * cov_yy, 0.0)) + 1.0)
    snug = _tile_rect(point[0], point[1], radius_x, radius_y, grid_x, grid_y)
    return (
        max(cuda_rect[0], snug[0]),
        max(cuda_rect[1], snug[1]),
        min(cuda_rect[2], snug[2]),
        min(cuda_rect[3], snug[3]),
    )


def _cuda_rect_and_covariance(point, conic, width, height):
    a, b, c = conic
    det = a * c - b * b
    cov_xx = c / det
    cov_xy = -b / det
    cov_yy = a / det
    trace = cov_xx + cov_yy
    root = math.sqrt(max(0.0, 0.25 * trace * trace - (cov_xx * cov_yy - cov_xy * cov_xy)))
    lambda_max = 0.5 * trace + root
    cuda_radius = math.ceil(3.0 * math.sqrt(lambda_max))
    grid_x = (width + BLOCK - 1) // BLOCK
    grid_y = (height + BLOCK - 1) // BLOCK
    cuda_rect = _tile_rect(point[0], point[1], cuda_radius, cuda_radius, grid_x, grid_y)
    return cuda_rect, cov_xx, cov_yy


def _q(x, y, conic):
    a, b, c = conic
    return a * x * x + 2.0 * b * x * y + c * y * y


def _conic_rect_intersects(point, conic, threshold, tile_x, tile_y):
    a, b, c = conic
    x0 = tile_x * BLOCK - point[0]
    x1 = tile_x * BLOCK + BLOCK - 1 - point[0]
    y0 = tile_y * BLOCK - point[1]
    y1 = tile_y * BLOCK + BLOCK - 1 - point[1]
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        return True
    candidates = []
    for x in (x0, x1):
        y = min(y1, max(y0, -b * x / c))
        candidates.append(_q(x, y, conic))
    for y in (y0, y1):
        x = min(x1, max(x0, -b * y / a))
        candidates.append(_q(x, y, conic))
    return min(candidates) <= threshold + Q_EPSILON


def _band_interval(point_u, point_v, conic_uu, conic_uv, conic_vv, threshold, tile_v, lo, hi):
    det = conic_uu * conic_vv - conic_uv * conic_uv
    threshold += Q_EPSILON
    v0 = tile_v * BLOCK - point_v
    v1 = tile_v * BLOCK + BLOCK - 1 - point_v
    values = []
    for v in (v0, v1):
        disc = conic_uu * threshold - det * v * v
        if disc >= -Q_EPSILON:
            half = math.sqrt(max(disc, 0.0)) / conic_uu
            center = -conic_uv * v / conic_uu
            values.extend((center - half, center + half))
    extent = math.sqrt(max(threshold * conic_vv / det, 0.0))
    for u in (-extent, extent):
        v = -conic_uv * u / conic_vv
        if v0 - Q_EPSILON <= v <= v1 + Q_EPSILON:
            values.append(u)
    if not values:
        return lo, lo
    tile_lo = math.floor((point_u + min(values)) / BLOCK)
    tile_hi = math.floor((point_u + max(values)) / BLOCK) + 1
    return max(lo, min(hi, tile_lo)), max(lo, min(hi, tile_hi))


def _coverage_tiles(point, conic, opacity, width, height, mode):
    rect = _candidate_rect(point, conic, opacity, width, height)
    span_x = rect[2] - rect[0]
    span_y = rect[3] - rect[1]
    area = span_x * span_y
    if area <= 1 or mode == "snugbox":
        return {(x, y) for y in range(rect[1], rect[3]) for x in range(rect[0], rect[2])}
    threshold = 2.0 * math.log(max(255.0 * opacity, 1.0))
    if mode == "conic_rect":
        return {
            (x, y)
            for y in range(rect[1], rect[3])
            for x in range(rect[0], rect[2])
            if _conic_rect_intersects(point, conic, threshold, x, y)
        }
    result = set()
    a, b, c = conic
    if span_y <= span_x:
        for y in range(rect[1], rect[3]):
            lo, hi = _band_interval(point[0], point[1], a, b, c, threshold, y, rect[0], rect[2])
            result.update((x, y) for x in range(lo, hi))
    else:
        for x in range(rect[0], rect[2]):
            lo, hi = _band_interval(point[1], point[0], c, b, a, threshold, x, rect[1], rect[3])
            result.update((x, y) for y in range(lo, hi))
    return result


def _accepted_pixel_tiles(point, conic, opacity, width, height):
    threshold = 2.0 * math.log(max(255.0 * opacity, 1.0))
    cuda_rect, _, _ = _cuda_rect_and_covariance(point, conic, width, height)
    return {
        (x // BLOCK, y // BLOCK)
        for y in range(height)
        for x in range(width)
        if _q(x - point[0], y - point[1], conic) <= threshold
        and cuda_rect[0] <= x // BLOCK < cuda_rect[2]
        and cuda_rect[1] <= y // BLOCK < cuda_rect[3]
    }


class TileCoverageTests(unittest.TestCase):
    def test_depth_sort_payload_layout_is_bounded_and_has_explicit_fallback(self):
        self.assertEqual(
            _depth_sort_payload_layout(298_819, 2_500),
            (19, (1 << 19) - 1),
        )
        self.assertIsNone(_depth_sort_payload_layout(300_000, 32_400))
        self.assertEqual(_depth_sort_payload_layout(1, 1), (0, 0))
        self.assertIsNone(_depth_sort_payload_layout(0, 1))

    def test_policy_rejects_exact_culling_for_non_exact_footprints(self):
        self.assertEqual(resolve_tile_coverage_mode("auto", FOOTPRINT_CUSTOM), "snugbox")
        self.assertEqual(
            resolve_tile_coverage_mode("auto", FOOTPRINT_AXIS_ALIGNED),
            "snugbox",
        )
        self.assertEqual(
            resolve_tile_coverage_mode("snugbox", FOOTPRINT_APPROXIMATE_SCREEN_CONIC),
            "snugbox",
        )
        with self.assertRaisesRegex(ValueError, "exact_screen_conic"):
            resolve_tile_coverage_mode("conic_rect", FOOTPRINT_CUSTOM)
        self.assertEqual(
            resolve_tile_coverage_mode("auto", FOOTPRINT_EXACT_SCREEN_CONIC),
            "accutile_sweep",
        )
        self.assertEqual(tile_coverage_mode_id("accutile_sweep"), 1)

    def test_kernel_call_routes_all_accept_coverage_mode(self):
        kernels = (
            preprocess_kernels._fused_project_cov3d_cov2d_preprocess_sr_warp_kernel,
            preprocess_kernels._cov2d_preprocess_masked_pack_warp_kernel,
            preprocess_kernels._cov2d_preprocess_masked_pack_scale_rotation_warp_kernel,
            binning_kernels._duplicate_with_keys_warp_kernel,
            binning_kernels._duplicate_with_packed_keys_warp_kernel,
            binning_kernels._duplicate_with_keys_from_order_warp_kernel,
        )
        for kernel in kernels:
            self.assertIn("coverage_mode", inspect.signature(kernel.func).parameters)

    def test_exact_modes_cover_every_accepted_pixel_tile(self):
        rng = random.Random(20260713)
        cases = [
            ((31.4, 16.9), (2.220537, -2.480092, 3.196566), 0.12),
            ((0.25, 0.75), (0.08, 0.075, 0.09), 0.95),
            ((79.2, 47.8), (1.5, -0.02, 0.04), 0.5),
        ]
        for _ in range(250):
            angle = rng.uniform(-math.pi, math.pi)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            eigen_a = 10.0 ** rng.uniform(-1.3, 0.7)
            eigen_b = 10.0 ** rng.uniform(-1.3, 0.7)
            a = eigen_a * cos_a * cos_a + eigen_b * sin_a * sin_a
            b = (eigen_a - eigen_b) * cos_a * sin_a
            c = eigen_a * sin_a * sin_a + eigen_b * cos_a * cos_a
            cases.append(
                (
                    (rng.uniform(-4.0, 100.0), rng.uniform(-4.0, 68.0)),
                    (a, b, c),
                    rng.uniform(0.01, 0.99),
                )
            )

        for point, conic, opacity in cases:
            expected = _accepted_pixel_tiles(point, conic, opacity, 96, 64)
            snugbox = _coverage_tiles(point, conic, opacity, 96, 64, "snugbox")
            self.assertTrue(expected <= snugbox)
            for mode in ("accutile_sweep", "conic_rect"):
                actual = _coverage_tiles(point, conic, opacity, 96, 64, mode)
                self.assertTrue(
                    expected <= actual,
                    msg=f"{mode} omitted {expected - actual} for {(point, conic, opacity)}",
                )
                self.assertTrue(actual <= snugbox)


if __name__ == "__main__":
    unittest.main()
