"""CUDA contract for the 2DGS screen-space gradient proxy."""

from __future__ import annotations

import unittest

import torch

from gswarp.methods.twodgs import (
    TwoDGaussianRasterizationSettings,
    rasterize_surfels,
)


DEVICE = torch.device("cuda:0")


def _settings() -> TwoDGaussianRasterizationSettings:
    return TwoDGaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.tensor((0.05, 0.15, 0.25), dtype=torch.float32, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        projmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        prefiltered=False,
        auto_tune=False,
        auto_tune_verbose=False,
    )


def _inputs(offset: tuple[float, float]) -> dict[str, torch.Tensor]:
    means2d = torch.zeros((2, 3), dtype=torch.float32, device=DEVICE)
    means2d[:, 0] = offset[0]
    means2d[:, 1] = offset[1]
    means2d.requires_grad_(True)
    rotations = torch.zeros((2, 4), dtype=torch.float32, device=DEVICE)
    rotations[:, 0] = 1.0
    return {
        "means3D": torch.tensor(
            ((-0.2, -0.1, 2.0), (0.15, 0.1, 2.2)),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        ),
        "means2D": means2d,
        "sh": torch.empty((0,), dtype=torch.float32, device=DEVICE),
        "colors_precomp": torch.tensor(
            ((0.8, 0.1, 0.2), (0.2, 0.7, 0.1)),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        ),
        "opacities": torch.full((2, 1), 0.55, dtype=torch.float32, device=DEVICE, requires_grad=True),
        "scales": torch.full((2, 2), 0.08, dtype=torch.float32, device=DEVICE, requires_grad=True),
        "rotations": rotations.requires_grad_(True),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TwoDGSMeans2DProxyCudaTests(unittest.TestCase):
    def test_means2d_is_forward_invariant_but_receives_screen_gradient(self) -> None:
        reference = rasterize_surfels(**_inputs((0.0, 0.0)), raster_settings=_settings())
        proxy_inputs = _inputs((3.0, -2.0))
        result = rasterize_surfels(**proxy_inputs, raster_settings=_settings())

        torch.testing.assert_close(result.color, reference.color, rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.alpha, reference.alpha, rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.depth, reference.depth, rtol=0.0, atol=0.0)
        loss = (
            result.color.square().mean()
            + result.alpha.mean()
            + 0.01 * result.depth.mean()
            + 0.01 * result.proj_2D.sum()
        )
        loss.backward()
        self.assertIsNotNone(proxy_inputs["means2D"].grad)
        self.assertTrue(torch.isfinite(proxy_inputs["means2D"].grad).all())
        self.assertGreater(float(proxy_inputs["means2D"].grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
