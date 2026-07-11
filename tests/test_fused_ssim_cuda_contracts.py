"""CUDA regressions for SSIM workspace ownership and stream handling."""

from __future__ import annotations

import gc
import unittest

import torch

from gswarp.fused_ssim import (
    _get_fused_ssim_cache_report,
    clear_fused_ssim_caches,
    fused_ssim,
)


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def _single_gradient(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    image = image.detach().clone().requires_grad_(True)
    fused_ssim(image, target, train=True).backward()
    return image.grad.detach().clone()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for fused SSIM tests")
class FusedSSIMCUDAContractTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_fused_ssim_caches(DEVICE)

    def tearDown(self) -> None:
        gc.collect()
        clear_fused_ssim_caches(DEVICE)

    def test_multiple_live_forwards_keep_independent_backward_workspaces(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(7)
        image1 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        target1 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        image2 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        target2 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)

        expected1 = _single_gradient(image1, target1)
        expected2 = _single_gradient(image2, target2)

        actual1 = image1.detach().clone().requires_grad_(True)
        actual2 = image2.detach().clone().requires_grad_(True)
        loss1 = fused_ssim(actual1, target1, train=True)
        loss2 = fused_ssim(actual2, target2, train=True)
        loss1.backward()
        loss2.backward()
        torch.cuda.synchronize(DEVICE)

        torch.testing.assert_close(actual1.grad, expected1, rtol=1.0e-4, atol=1.0e-5)
        torch.testing.assert_close(actual2.grad, expected2, rtol=1.0e-4, atol=1.0e-5)

    def test_non_default_pytorch_stream_is_supported(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(11)
        image = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        target = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        expected = _single_gradient(image, target)
        stream = torch.cuda.Stream(device=DEVICE)

        with torch.cuda.stream(stream):
            actual = image.detach().clone().requires_grad_(True)
            fused_ssim(actual, target, train=True).backward()
        stream.synchronize()

        torch.testing.assert_close(actual.grad, expected, rtol=1.0e-4, atol=1.0e-5)

    def test_two_stream_plans_match_isolated_gradients(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(12)
        image1 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        target1 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        image2 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        target2 = torch.rand((1, 3, 16, 16), generator=generator, device=DEVICE)
        expected1 = _single_gradient(image1, target1)
        expected2 = _single_gradient(image2, target2)
        stream1 = torch.cuda.Stream(device=DEVICE)
        stream2 = torch.cuda.Stream(device=DEVICE)

        with torch.cuda.stream(stream1):
            actual1 = image1.detach().clone().requires_grad_(True)
            fused_ssim(actual1, target1, train=True).backward()
        with torch.cuda.stream(stream2):
            actual2 = image2.detach().clone().requires_grad_(True)
            fused_ssim(actual2, target2, train=True).backward()
        stream1.synchronize()
        stream2.synchronize()

        torch.testing.assert_close(
            actual1.grad, expected1, rtol=1.0e-4, atol=1.0e-5
        )
        torch.testing.assert_close(
            actual2.grad, expected2, rtol=1.0e-4, atol=1.0e-5
        )

    def test_retain_graph_keeps_workspace_lease_live(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(13)
        image = torch.rand(
            (1, 3, 16, 16), generator=generator, device=DEVICE
        ).requires_grad_(True)
        target = torch.rand(
            (1, 3, 16, 16), generator=generator, device=DEVICE
        )
        loss = fused_ssim(image, target, train=True)

        loss.backward(retain_graph=True)
        expected = image.grad.detach().clone()
        image.grad = None
        loss.backward()
        torch.cuda.synchronize(DEVICE)

        torch.testing.assert_close(
            image.grad, expected, rtol=1.0e-4, atol=1.0e-5
        )

    def test_reused_plan_does_not_overwrite_returned_gradient(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(17)

        def run_once() -> torch.Tensor:
            image = torch.rand(
                (1, 3, 16, 16), generator=generator, device=DEVICE
            ).requires_grad_(True)
            target = torch.rand(
                (1, 3, 16, 16), generator=generator, device=DEVICE
            )
            fused_ssim(image, target, train=True).backward()
            return image.grad

        first = run_once()
        first_snapshot = first.detach().clone()
        gc.collect()
        before = _get_fused_ssim_cache_report()
        second = run_once()
        torch.cuda.synchronize(DEVICE)
        after = _get_fused_ssim_cache_report()

        torch.testing.assert_close(first, first_snapshot, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(before["free_training_plans"], 1)
        self.assertGreater(after["training_plan_reuses"], 0)
        self.assertIsNot(first, second)

    def test_training_plan_pool_is_bounded_and_clearable(self) -> None:
        generator = torch.Generator(device=DEVICE).manual_seed(19)
        for size in range(16, 22):
            image = torch.rand(
                (1, 3, size, size), generator=generator, device=DEVICE
            ).requires_grad_(True)
            target = torch.rand(
                (1, 3, size, size), generator=generator, device=DEVICE
            )
            fused_ssim(image, target, train=True).backward()
            del image, target
            gc.collect()

        report = _get_fused_ssim_cache_report()
        self.assertLessEqual(
            report["free_training_plans"],
            report["free_plan_limit_per_device"],
        )
        clear_fused_ssim_caches(DEVICE)
        self.assertEqual(
            _get_fused_ssim_cache_report()["free_training_plans"], 0
        )


if __name__ == "__main__":
    unittest.main()
