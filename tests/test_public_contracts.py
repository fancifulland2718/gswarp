"""CPU-safe public contract tests for gswarp."""

from __future__ import annotations

import unittest

import torch

from gswarp._internal.api.validation import normalize_gaussian_inputs
from gswarp._internal.backends.warp.packing import _pack_forward_aux_buffers, _unpack_forward_aux_buffers
from gswarp._internal.backends.warp.state import BinningState, PreprocessOutputs
from gswarp.knn import distCUDA2


class GaussianInputNormalizationTests(unittest.TestCase):
    def test_requires_exactly_one_color_representation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            normalize_gaussian_inputs(
                dc=None,
                shs=None,
                colors_precomp=None,
                scales=torch.empty(0),
                rotations=torch.empty(0),
                cov3D_precomp=torch.empty((1, 6)),
            )

    def test_requires_complete_scale_rotation_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "scales/rotations"):
            normalize_gaussian_inputs(
                dc=None,
                shs=torch.empty((1, 1, 3)),
                colors_precomp=None,
                scales=torch.empty((1, 3)),
                rotations=None,
                cov3D_precomp=None,
            )


class KNNContractTests(unittest.TestCase):
    def test_empty_and_singleton_inputs_preserve_defined_results(self) -> None:
        empty = distCUDA2(torch.empty((0, 3), dtype=torch.float32))
        singleton = distCUDA2(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))

        self.assertEqual(empty.shape, (0,))
        torch.testing.assert_close(singleton, torch.zeros(1, dtype=torch.float32))

    def test_two_and_three_points_are_rejected(self) -> None:
        for point_count in (2, 3):
            points = torch.zeros((point_count, 3), dtype=torch.float32)
            with self.subTest(point_count=point_count):
                with self.assertRaisesRegex(ValueError, "at least 4 points"):
                    distCUDA2(points)


class PackedStateContractTests(unittest.TestCase):
    def test_binning_buffer_stores_exact_point_list_without_padding(self) -> None:
        point_count = 2
        preprocess = PreprocessOutputs(
            visible=torch.ones(point_count, dtype=torch.bool),
            depths=torch.tensor([1.0, 2.0]),
            radii=torch.ones(point_count, dtype=torch.int32),
            proj_2d=torch.zeros((point_count, 2)),
            conic_2d=torch.zeros((point_count, 3)),
            conic_2d_inv=torch.zeros((point_count, 3)),
            points_xy_image=torch.zeros((point_count, 2)),
            tiles_touched=torch.ones(point_count, dtype=torch.int32),
            rgb=torch.zeros((point_count, 3)),
            clamped=torch.zeros((point_count, 3), dtype=torch.int32),
            conic_opacity=torch.zeros((point_count, 4)),
            cov3d_all=torch.zeros((point_count, 6)),
        )
        binning = BinningState(
            grid_x=1,
            grid_y=1,
            point_list=torch.tensor([1, 0, 1], dtype=torch.int32),
            ranges=torch.tensor([[0, 3]], dtype=torch.int32),
            num_rendered=3,
        )

        geom_buffer, binning_buffer, img_buffer = _pack_forward_aux_buffers(
            preprocess, binning, torch.zeros((4, 4), dtype=torch.int32)
        )
        unpacked = _unpack_forward_aux_buffers(
            geom_buffer, binning_buffer, img_buffer, binning.num_rendered, 4, 4
        )

        self.assertEqual(binning_buffer.numel(), (binning.num_rendered + 2) * 4)
        self.assertIsNotNone(unpacked)
        _, unpacked_binning, _ = unpacked
        torch.testing.assert_close(unpacked_binning.point_list, binning.point_list)
        torch.testing.assert_close(unpacked_binning.ranges, binning.ranges)


if __name__ == "__main__":
    unittest.main()
