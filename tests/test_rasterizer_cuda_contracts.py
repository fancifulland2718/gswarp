"""CUDA regression tests for rasterizer public contracts and output gradients."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest
from unittest.mock import patch

import torch

import gswarp._stream as stream_interop
from gswarp._rasterizer import (
    rasterize_gaussians as rasterize_standard_raw,
    rasterize_gaussians_backward as rasterize_standard_backward_raw,
)
from gswarp.rasterizer import (
    GaussianRasterizer as StandardRasterizer,
    GaussianRasterizationSettings as StandardSettings,
    clear_warp_caches as clear_standard_caches,
    get_warp_cache_report as get_standard_cache_report,
    get_tile_coverage_mode,
    rasterize_gaussians as rasterize_standard,
    set_tile_coverage_mode,
)
from gswarp.rasterizer_flow import (
    GaussianRasterizationSettings as FlowSettings,
    clear_warp_caches as clear_flow_caches,
    rasterize_gaussians as rasterize_flow,
)
from gswarp._internal.backends.warp.memory import (
    _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS,
    _C4_LAUNCH_CACHE_RENDER_BWD,
    _C4_LAUNCH_CACHE_SH_V3,
)


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def _empty() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=DEVICE)


def _settings(
    background: torch.Tensor, *, flow: bool = False, image_size: int = 16
):
    common = dict(
        image_height=image_size,
        image_width=image_size,
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

    def test_subthreshold_opacity_preserves_cuda_geometric_visibility(self) -> None:
        background = torch.tensor(
            [0.2, 0.6, 0.3], dtype=torch.float32, device=DEVICE
        )
        inputs = _inputs(
            torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE)
        )
        inputs["opacities"] = torch.full(
            (1, 1), 1.0e-4, dtype=torch.float32, device=DEVICE
        )

        color, radii, _ = rasterize_standard(
            **inputs, raster_settings=_settings(background)
        )

        expected = background[:, None, None].expand(3, 16, 16)
        torch.testing.assert_close(color, expected)
        self.assertGreater(radii.item(), 0)

    def test_rotated_conic_matches_per_pixel_compositing_oracle(self) -> None:
        image_size = 64
        background = torch.tensor(
            [0.1, 0.2, 0.3], dtype=torch.float32, device=DEVICE
        )
        opacity = 0.2
        color_value = torch.tensor(
            [[0.7, 0.4, 0.2]], dtype=torch.float32, device=DEVICE
        )
        covariances = torch.tensor(
            [[0.04, 0.038, 0.0, 0.04, 0.0, 0.02]],
            dtype=torch.float32,
            device=DEVICE,
        )
        inputs = _inputs(
            torch.tensor(
                [[-0.5, -0.5, 2.0]], dtype=torch.float32, device=DEVICE
            ),
            scales=_empty(),
            covariances=covariances,
        )
        inputs["colors_precomp"] = color_value
        inputs["opacities"] = torch.full(
            (1, 1), opacity, dtype=torch.float32, device=DEVICE
        )

        rendered, _, meta = rasterize_standard(
            **inputs,
            raster_settings=_settings(background, image_size=image_size),
        )

        point = meta.proj_2D[0]
        conic = meta.conic_2D[0]
        pixels = torch.arange(
            image_size, dtype=torch.float32, device=DEVICE
        )
        pixel_y, pixel_x = torch.meshgrid(pixels, pixels, indexing="ij")
        dx = point[0] - pixel_x
        dy = point[1] - pixel_y
        power = (
            -0.5 * (conic[0] * dx.square() + conic[2] * dy.square())
            - conic[1] * dx * dy
        )
        alpha = torch.minimum(
            torch.tensor(0.99, device=DEVICE), opacity * torch.exp(power)
        )
        alpha = torch.where(alpha >= (1.0 / 255.0), alpha, 0.0)
        expected = (
            color_value[0, :, None, None] * alpha[None]
            + background[:, None, None] * (1.0 - alpha[None])
        )

        torch.testing.assert_close(
            rendered, expected, atol=2.0e-5, rtol=2.0e-4
        )

    def test_tile_coverage_modes_match_snugbox_for_large_rotated_conic(self) -> None:
        image_size = 128
        background = torch.tensor(
            [0.1, 0.2, 0.3], dtype=torch.float32, device=DEVICE
        )
        covariances = torch.tensor(
            [[0.20, 0.18, 0.0, 0.20, 0.0, 0.02]],
            dtype=torch.float32,
            device=DEVICE,
        )
        inputs = _inputs(
            torch.tensor([[-0.15, 0.1, 2.0]], dtype=torch.float32, device=DEVICE),
            scales=_empty(),
            covariances=covariances,
        )
        inputs["opacities"] = torch.full(
            (1, 1), 0.35, dtype=torch.float32, device=DEVICE
        )

        original_mode = get_tile_coverage_mode()
        outputs = {}
        try:
            for mode in ("snugbox", "accutile_sweep", "conic_rect", "auto"):
                set_tile_coverage_mode(mode)
                outputs[mode] = rasterize_standard(
                    **inputs,
                    raster_settings=_settings(background, image_size=image_size),
                )
                torch.cuda.synchronize(DEVICE)
        finally:
            set_tile_coverage_mode(original_mode)

        reference = outputs["snugbox"]
        for mode in ("accutile_sweep", "conic_rect", "auto"):
            torch.testing.assert_close(outputs[mode][0], reference[0], atol=2.0e-5, rtol=2.0e-4)
            torch.testing.assert_close(outputs[mode][1], reference[1])
            for actual, expected in zip(outputs[mode][2], reference[2]):
                torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-4)

    def test_mark_visible_returns_owned_bool_tensor(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        rasterizer = StandardRasterizer(_settings(background))
        first = rasterizer.markVisible(
            torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]], device=DEVICE)
        )
        torch.cuda.synchronize(DEVICE)
        expected = first.clone()
        second = rasterizer.markVisible(
            torch.tensor([[0.0, 0.0, -2.0], [0.2, 0.0, -2.0]], device=DEVICE)
        )
        torch.cuda.synchronize(DEVICE)

        self.assertEqual(first.dtype, torch.bool)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, torch.zeros_like(second))

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

    def test_pending_forward_owns_its_binning_state(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        first_points = torch.tensor(
            [[-0.25, 0.0, 2.0], [0.25, 0.0, 2.0]], dtype=torch.float32, device=DEVICE
        )

        reference_scales = torch.full(
            (2, 3), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        reference_color, _, _ = rasterize_standard(
            **_inputs(first_points, scales=reference_scales),
            raster_settings=_settings(background),
        )
        reference_color.sum().backward()
        reference_grad = reference_scales.grad.detach().clone()

        clear_standard_caches()
        pending_scales = torch.full(
            (2, 3), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        pending_color, _, _ = rasterize_standard(
            **_inputs(first_points, scales=pending_scales),
            raster_settings=_settings(background),
        )
        rasterize_standard(
            **_inputs(
                torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE),
            ),
            raster_settings=_settings(background),
        )
        pending_color.sum().backward()

        torch.testing.assert_close(pending_scales.grad, reference_grad)

    def test_retain_graph_repeats_standard_backward(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        scales = torch.full(
            (8, 3),
            0.5,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        inputs = _inputs(
            torch.tensor(
                [[-0.2, 0.0, 2.0], [0.2, 0.0, 2.0]] * 4,
                dtype=torch.float32,
                device=DEVICE,
            ),
            scales=scales,
        )
        inputs["colors_precomp"] = _empty()
        inputs["sh"] = torch.randn(
            (8, 16, 3),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        color, _, _ = rasterize_standard(
            **inputs,
            raster_settings=_settings(background),
        )
        loss = color.square().mean()

        loss.backward(retain_graph=True)
        first_scale_grad = scales.grad.clone()
        first_sh_grad = inputs["sh"].grad.clone()
        scales.grad = None
        inputs["sh"].grad = None

        loss.backward()

        torch.testing.assert_close(scales.grad, first_scale_grad)
        torch.testing.assert_close(inputs["sh"].grad, first_sh_grad)

    def test_standard_backward_does_not_attach_warp_autograd(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        scales = torch.full(
            (1, 3),
            0.5,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        inputs = _inputs(
            torch.tensor(
                [[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE
            ),
            scales=scales,
        )
        inputs["colors_precomp"] = _empty()
        inputs["sh"] = torch.randn(
            (1, 16, 3),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        color, _, _ = rasterize_standard(
            **inputs,
            raster_settings=_settings(background),
        )
        original = stream_interop.wp.from_torch

        with patch.object(
            stream_interop.wp, "from_torch", wraps=original
        ) as wrapped:
            color.sum().backward()

        self.assertGreater(len(wrapped.call_args_list), 0)
        self.assertLessEqual(len(wrapped.call_args_list), 23)
        for call in wrapped.call_args_list:
            self.assertIs(call.kwargs["requires_grad"], False)
            self.assertIs(call.kwargs["return_ctype"], True)

        for cache in (
            _C4_LAUNCH_CACHE_RENDER_BWD,
            _C4_LAUNCH_CACHE_SH_V3,
            _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS,
        ):
            for command in cache.values():
                for value in command.params:
                    self.assertFalse(hasattr(value, "_ref"))

    def test_raw_forward_backward_compatibility_round_trip(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        settings = _settings(background)
        means3d = torch.tensor(
            [[0.0, 0.0, 2.0]], dtype=torch.float32, device=DEVICE
        )
        inputs = _inputs(means3d)
        forward = rasterize_standard_raw(
            background,
            means3d,
            inputs["colors_precomp"],
            inputs["opacities"],
            inputs["scales"],
            inputs["rotations"],
            settings.scale_modifier,
            inputs["cov3Ds_precomp"],
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            inputs["sh"],
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
        )
        (
            num_rendered,
            color,
            depth,
            alpha,
            radii,
            geom_buffer,
            binning_buffer,
            img_buffer,
            proj_2d,
            conic_2d,
            conic_2d_inv,
        ) = forward
        grads = rasterize_standard_backward_raw(
            background,
            means3d,
            radii,
            inputs["colors_precomp"],
            inputs["opacities"],
            inputs["scales"],
            inputs["rotations"],
            settings.scale_modifier,
            inputs["cov3Ds_precomp"],
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            torch.ones_like(color),
            torch.zeros_like(depth),
            torch.zeros_like(alpha),
            torch.zeros_like(proj_2d),
            torch.zeros_like(conic_2d),
            torch.zeros_like(conic_2d_inv),
            inputs["sh"],
            settings.sh_degree,
            settings.campos,
            geom_buffer,
            num_rendered,
            binning_buffer,
            img_buffer,
            alpha,
        )

        self.assertEqual(len(grads), 8)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
        self.assertGreater(grads[6].abs().sum().item(), 0.0)

    def test_two_stream_forward_backward_matches_isolated_references(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        settings = _settings(background)
        points_a = torch.tensor(
            [[-0.25, 0.0, 2.0], [0.25, 0.0, 2.0]], device=DEVICE
        )
        points_b = torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE)

        def isolated(points):
            scales = torch.full(
                (points.shape[0], 3),
                0.5,
                dtype=torch.float32,
                device=DEVICE,
                requires_grad=True,
            )
            color, _, _ = rasterize_standard(
                **_inputs(points, scales=scales), raster_settings=settings
            )
            color.square().mean().backward()
            torch.cuda.synchronize(DEVICE)
            return color.detach().clone(), scales.grad.detach().clone()

        expected_a = isolated(points_a)
        expected_b = isolated(points_b)
        clear_standard_caches()

        scales_a = torch.full(
            (2, 3), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        scales_b = torch.full(
            (1, 3), 0.5, dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        stream_a = torch.cuda.Stream(device=DEVICE)
        stream_b = torch.cuda.Stream(device=DEVICE)
        with torch.cuda.stream(stream_a):
            color_a, _, _ = rasterize_standard(
                **_inputs(points_a, scales=scales_a), raster_settings=settings
            )
            loss_a = color_a.square().mean()
        with torch.cuda.stream(stream_b):
            color_b, _, _ = rasterize_standard(
                **_inputs(points_b, scales=scales_b), raster_settings=settings
            )
            loss_b = color_b.square().mean()
        with torch.cuda.stream(stream_a):
            loss_a.backward()
        with torch.cuda.stream(stream_b):
            loss_b.backward()
        stream_a.synchronize()
        stream_b.synchronize()

        torch.testing.assert_close(color_a, expected_a[0])
        torch.testing.assert_close(scales_a.grad, expected_a[1])
        torch.testing.assert_close(color_b, expected_b[0])
        torch.testing.assert_close(scales_b.grad, expected_b[1])
        report = get_standard_cache_report()
        self.assertEqual(report["workspace_entries"], 2)

    def test_two_threads_submit_on_independent_streams(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        settings = _settings(background)
        points = (
            torch.tensor(
                [[-0.2, 0.0, 2.0], [0.2, 0.0, 2.0]], device=DEVICE
            ),
            torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE),
        )
        inputs = tuple(_inputs(value) for value in points)
        with torch.no_grad():
            expected = tuple(
                rasterize_standard(**value, raster_settings=settings)[0].clone()
                for value in inputs
            )
        torch.cuda.synchronize(DEVICE)
        clear_standard_caches()

        streams = (
            torch.cuda.Stream(device=DEVICE),
            torch.cuda.Stream(device=DEVICE),
        )
        barrier = Barrier(2)

        def submit(index):
            with torch.cuda.stream(streams[index]), torch.no_grad():
                barrier.wait()
                color, _, _ = rasterize_standard(
                    **inputs[index], raster_settings=settings
                )
            streams[index].synchronize()
            return color

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(submit, index) for index in range(2)]
            actual = tuple(future.result() for future in futures)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_workspace_stream_slots_are_bounded_and_evict_safely(self) -> None:
        from gswarp._internal.backends.warp import memory

        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        rasterizer = StandardRasterizer(_settings(background))
        points = torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE)
        streams = [
            torch.cuda.Stream(device=DEVICE)
            for _ in range(memory._MAX_WORKSPACE_CACHE_STREAMS + 1)
        ]
        before = memory.get_warp_cache_report()["workspace_evictions"]
        outputs = []
        for stream in streams:
            with torch.cuda.stream(stream):
                outputs.append(rasterizer.markVisible(points))
        for stream in streams:
            stream.synchronize()

        report = memory.get_warp_cache_report()
        self.assertEqual(
            report["workspace_entries"], memory._MAX_WORKSPACE_CACHE_STREAMS
        )
        self.assertGreaterEqual(report["workspace_evictions"], before + 1)
        for output in outputs:
            torch.testing.assert_close(output, torch.ones_like(output))

    def test_cache_clear_waits_for_inflight_workspace_stream(self) -> None:
        background = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        rasterizer = StandardRasterizer(_settings(background))
        points = torch.zeros((4096, 3), dtype=torch.float32, device=DEVICE)
        points[:, 2] = 2.0
        stream = torch.cuda.Stream(device=DEVICE)
        with torch.cuda.stream(stream):
            visible = rasterizer.markVisible(points)

        clear_standard_caches()

        torch.testing.assert_close(visible, torch.ones_like(visible))


if __name__ == "__main__":
    unittest.main()
