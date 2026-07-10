"""CPU-safe public contract tests for gswarp."""

from __future__ import annotations

import unittest

import torch

from gswarp._internal.api.validation import normalize_gaussian_inputs
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


if __name__ == "__main__":
    unittest.main()
