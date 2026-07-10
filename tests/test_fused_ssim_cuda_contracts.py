"""CUDA regressions for SSIM workspace ownership and stream handling."""

from __future__ import annotations

import unittest

import torch

from gswarp.fused_ssim import fused_ssim


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def _single_gradient(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    image = image.detach().clone().requires_grad_(True)
    fused_ssim(image, target, train=True).backward()
    return image.grad.detach().clone()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for fused SSIM tests")
class FusedSSIMCUDAContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
