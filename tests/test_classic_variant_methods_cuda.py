"""Focused CUDA contracts for classic 3DGS method providers."""

from __future__ import annotations

import unittest

import torch

from gswarp.methods.generated_3dgs import rasterize_gaussians as rasterize_generated
from gswarp.methods.mip_3dgs import (
    MipGaussianRasterizationSettings,
    rasterize_gaussians as rasterize_mip,
)
from gswarp.methods.twodgs import (
    TwoDGaussianRasterizationSettings,
    rasterize_surfels,
)
from gswarp.rasterizer import GaussianRasterizationSettings, rasterize_gaussians


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def _settings(*, mip_filter: float | None = None):
    common = dict(
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
        binning_sort_mode="warp_depth_stable_tile",
    )
    if mip_filter is None:
        return GaussianRasterizationSettings(**common)
    return MipGaussianRasterizationSettings(**common, filter_variance=mip_filter)


def _twod_settings() -> TwoDGaussianRasterizationSettings:
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
        binning_sort_mode="warp_depth_stable_tile",
        filter_radius=1.0,
        near_plane=0.2,
        far_plane=100.0,
    )


def _inputs(count: int = 4):
    means3d = torch.tensor(
        ((-0.20, -0.10, 2.0), (0.15, -0.05, 2.2), (-0.05, 0.20, 2.4), (0.22, 0.18, 2.7)),
        dtype=torch.float32,
        device=DEVICE,
    )[:count].clone().requires_grad_(True)
    means2d = torch.zeros((count, 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
    colors = torch.tensor(
        ((0.8, 0.1, 0.2), (0.2, 0.7, 0.1), (0.1, 0.3, 0.9), (0.8, 0.7, 0.2)),
        dtype=torch.float32,
        device=DEVICE,
    )[:count].clone().requires_grad_(True)
    opacity = torch.full((count, 1), 0.55, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scales = torch.full((count, 3), 0.08, dtype=torch.float32, device=DEVICE, requires_grad=True)
    rotations = torch.zeros((count, 4), dtype=torch.float32, device=DEVICE)
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)
    empty = torch.empty((0,), dtype=torch.float32, device=DEVICE)
    return dict(
        means3D=means3d,
        means2D=means2d,
        sh=empty,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=empty,
    )


def _twod_inputs(count: int = 2):
    inputs = _inputs(count)
    return {
        "means3D": inputs["means3D"],
        "means2D": inputs["means2D"],
        "sh": inputs["sh"],
        "colors_precomp": inputs["colors_precomp"],
        "opacities": inputs["opacities"],
        "scales": inputs["scales"][:, :2].detach().clone().requires_grad_(True),
        "rotations": inputs["rotations"],
    }


def _loss(outputs):
    color, _radii, meta = outputs
    return color.square().mean() + meta.alpha.mean() + 0.01 * meta.depth.mean()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required")
class ClassicMethodCudaTests(unittest.TestCase):
    def test_generated_adapter_preserves_upstream_gradient(self) -> None:
        inputs = _inputs()
        displacement = torch.nn.Parameter(torch.full_like(inputs["means3D"], 0.01))
        outputs = rasterize_generated(
            **(inputs | {"means3D": inputs["means3D"] + displacement}),
            raster_settings=_settings(),
        )
        _loss(outputs).backward()
        self.assertIsNotNone(displacement.grad)
        self.assertTrue(torch.isfinite(displacement.grad).all())
        self.assertGreater(float(displacement.grad.abs().sum()), 0.0)

    def test_generated_adapter_matches_baseline_outputs_and_gradients(self) -> None:
        baseline_inputs = _inputs()
        baseline = rasterize_gaussians(**baseline_inputs, raster_settings=_settings())
        _loss(baseline).backward()
        generated_inputs = _inputs()
        generated = rasterize_generated(**generated_inputs, raster_settings=_settings())
        _loss(generated).backward()
        torch.testing.assert_close(generated[0], baseline[0], rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(generated[2].alpha, baseline[2].alpha, rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(generated[2].depth, baseline[2].depth, rtol=2.0e-5, atol=2.0e-5)
        for name in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations"):
            torch.testing.assert_close(generated_inputs[name].grad, baseline_inputs[name].grad, rtol=3.0e-4, atol=3.0e-5)

    def test_mip_zero_filter_matches_baseline_and_filtered_gradients_are_finite(self) -> None:
        baseline_inputs = _inputs()
        baseline = rasterize_gaussians(**baseline_inputs, raster_settings=_settings())
        baseline_loss = _loss(baseline)
        baseline_loss.backward()

        mip_inputs = _inputs()
        mip_zero = rasterize_mip(**mip_inputs, raster_settings=_settings(mip_filter=0.0))
        torch.testing.assert_close(mip_zero[0], baseline[0], rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(mip_zero[2].alpha, baseline[2].alpha, rtol=2.0e-5, atol=2.0e-5)
        _loss(mip_zero).backward()
        torch.testing.assert_close(mip_inputs["means3D"].grad, baseline_inputs["means3D"].grad, rtol=3.0e-4, atol=3.0e-5)

        filtered_inputs = _inputs()
        filtered = rasterize_mip(**filtered_inputs, raster_settings=_settings(mip_filter=0.25))
        _loss(filtered).backward()
        for name in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations"):
            self.assertTrue(torch.isfinite(filtered_inputs[name].grad).all(), name)

    def test_twodgs_raysplat_outputs_and_gradients_are_finite(self) -> None:
        inputs = _twod_inputs()
        result = rasterize_surfels(**inputs, raster_settings=_twod_settings())
        self.assertEqual(tuple(result.color.shape), (3, 16, 16))
        self.assertEqual(tuple(result.normal.shape), (3, 16, 16))
        self.assertEqual(tuple(result.distortion.shape), (1, 16, 16))
        self.assertEqual(tuple(result.median_depth.shape), (1, 16, 16))
        self.assertTrue(torch.isfinite(result.color).all())
        self.assertTrue(torch.isfinite(result.alpha).all())

        loss = (
            result.color.square().mean()
            + result.alpha.mean()
            + 0.01 * result.depth.mean()
            + 0.001 * result.normal.square().mean()
            + 0.001 * result.distortion.mean()
        )
        loss.backward()
        for name in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations"):
            self.assertIsNotNone(inputs[name].grad, name)
            self.assertTrue(torch.isfinite(inputs[name].grad).all(), name)


if __name__ == "__main__":
    unittest.main()
