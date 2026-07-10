"""CUDA regression tests for rasterizer public contracts and output gradients."""

from __future__ import annotations

import unittest

import torch

from gswarp.rasterizer import (
    GaussianRasterizationSettings as StandardSettings,
    clear_warp_caches as clear_standard_caches,
    rasterize_gaussians as rasterize_standard,
)
from gswarp.rasterizer_flow import (
    GaussianRasterizationSettings as FlowSettings,
    clear_warp_caches as clear_flow_caches,
    rasterize_gaussians as rasterize_flow,
)


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def _empty() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=DEVICE)


def _settings(background: torch.Tensor, *, flow: bool = False):
    common = dict(
        image_height=16,
        image_width=16,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        projmatrix=torch.eye(4, dtype=torch.float32, device=DEVICE),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=DEVICE),
        prefiltered=False,
        auto_tune=False,
        auto_tune_verbose=False,
    )
    return FlowSettings(**common, enable_flow_grad=False, compute_flow_aux=False) if flow else StandardSettings(**common)


def _inputs(means3d: torch.Tensor, *, scales: torch.Tensor | None = None, covariances: torch.Tensor | None = None):
    count = means3d.shape[0]
    if scales is None:
        scales = torch.full((count, 3), 0.5, dtype=torch.float32, device=DEVICE)
    if covariances is None:
        covariances = _empty()
    return dict(
        means3D=means3d,
        means2D=torch.zeros((count, 3), dtype=torch.float32, device=DEVICE, requires_grad=True),
        sh=_empty(),
        colors_precomp=torch.full((count, 3), 0.2, dtype=torch.float32, device=DEVICE),
        opacities=torch.full((count, 1), 0.8, dtype=torch.float32, device=DEVICE),
        scales=scales,
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE).expand(count, -1).clone()
        if covariances.numel() == 0
        else _empty(),
        cov3Ds_precomp=covariances,
    )


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Warp rasterizer tests")
class RasterizerCUDAContractTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_standard_caches()
        clear_flow_caches()

    def test_empty_inputs_return_the_configured_background_for_both_backends(self) -> None:
        background = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float32, device=DEVICE)
        inputs = _inputs(torch.empty((0, 3), dtype=torch.float32, device=DEVICE))
        expected = background[:, None, None].expand(3, 16, 16)

        color, radii, _ = rasterize_standard(**inputs, raster_settings=_settings(background))
        flow_outputs = rasterize_flow(**inputs, raster_settings=_settings(background, flow=True))

        torch.testing.assert_close(color, expected)
        torch.testing.assert_close(flow_outputs[0], expected)
        self.assertEqual(radii.numel(), 0)
        self.assertEqual(flow_outputs[1].numel(), 0)

    def test_all_culled_inputs_return_the_configured_background_for_both_backends(self) -> None:
        background = torch.tensor([0.2, 0.6, 0.3], dtype=torch.float32, device=DEVICE)
        inputs = _inputs(torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=DEVICE))
        expected = background[:, None, None].expand(3, 16, 16)

        color, radii, _ = rasterize_standard(**inputs, raster_settings=_settings(background))
        flow_outputs = rasterize_flow(**inputs, raster_settings=_settings(background, flow=True))

        torch.testing.assert_close(color, expected)
        torch.testing.assert_close(flow_outputs[0], expected)
        torch.testing.assert_close(radii, torch.zeros_like(radii))
        torch.testing.assert_close(flow_outputs[1], torch.zeros_like(flow_outputs[1]))

    def test_conic_inverse_gradient_matches_finite_difference_in_fused_path(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        means3d = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE)

        def loss_for(scale_value: float, requires_grad: bool):
            scales = torch.full((1, 3), scale_value, dtype=torch.float32, device=DEVICE, requires_grad=requires_grad)
            color, radii, meta = rasterize_standard(
                **_inputs(means3d, scales=scales), raster_settings=_settings(background)
            )
            del color, radii
            return meta.conic_2D_inv.sum(), scales

        loss, scales = loss_for(0.5, True)
        loss.backward()
        analytic = scales.grad.sum().item()
        epsilon = 1.0e-3
        numerical = (loss_for(0.5 + epsilon, False)[0].item() - loss_for(0.5 - epsilon, False)[0].item()) / (2.0 * epsilon)

        self.assertNotEqual(analytic, 0.0)
        self.assertAlmostEqual(analytic, numerical, delta=max(1.0e-2, abs(numerical) * 5.0e-2))

    def test_conic_inverse_gradient_reaches_precomputed_covariance_path(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        covariances = torch.tensor([[0.25, 0.0, 0.0, 0.25, 0.0, 0.25]], dtype=torch.float32, device=DEVICE, requires_grad=True)
        _, _, meta = rasterize_standard(
            **_inputs(
                torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE),
                scales=_empty(),
                covariances=covariances,
            ),
            raster_settings=_settings(background),
        )
        meta.conic_2D_inv.sum().backward()

        self.assertTrue(torch.isfinite(covariances.grad).all())
        self.assertGreater(covariances.grad.abs().sum().item(), 0.0)

    def test_flow_conic_inverse_gradient_reaches_scale_rotation_path(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        scales = torch.full((1, 3), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True)
        outputs = rasterize_flow(
            **_inputs(
                torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE),
                scales=scales,
            ),
            raster_settings=_settings(background, flow=True),
        )
        outputs[6].sum().backward()

        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertGreater(scales.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
