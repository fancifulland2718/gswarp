"""Small CUDA correctness and latency microbenchmark for classic GS methods.

The benchmark does not create files unless ``--output`` is passed. Its default
workload is intentionally small: it is a local integration gate, not a quality
or training benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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


DEVICE = torch.device("cuda:0")


def _three_d_settings(height: int, width: int, *, filter_variance: float | None = None):
    common = dict(
        image_height=height,
        image_width=width,
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
    if filter_variance is None:
        return GaussianRasterizationSettings(**common)
    return MipGaussianRasterizationSettings(**common, filter_variance=filter_variance)


def _twod_settings(height: int, width: int) -> TwoDGaussianRasterizationSettings:
    return TwoDGaussianRasterizationSettings(
        image_height=height,
        image_width=width,
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


def _inputs(count: int, *, surfels: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(20260715 + count + int(surfels))
    means3d = torch.empty((count, 3), dtype=torch.float32, device=DEVICE)
    means3d[:, :2].uniform_(-0.65, 0.65, generator=generator)
    means3d[:, 2].uniform_(2.0, 3.0, generator=generator)
    colors = torch.rand((count, 3), dtype=torch.float32, device=DEVICE, generator=generator)
    opacities = torch.empty((count, 1), dtype=torch.float32, device=DEVICE)
    opacities.uniform_(0.1, 0.9, generator=generator)
    scales = torch.empty((count, 2 if surfels else 3), dtype=torch.float32, device=DEVICE)
    scales.uniform_(0.025, 0.06, generator=generator)
    rotations = torch.zeros((count, 4), dtype=torch.float32, device=DEVICE)
    rotations[:, 0] = 1.0
    tensors = {
        "means3D": means3d,
        "means2D": torch.zeros((count, 3), dtype=torch.float32, device=DEVICE),
        "sh": torch.empty((0,), dtype=torch.float32, device=DEVICE),
        "colors_precomp": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }
    for tensor in tensors.values():
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
    if not surfels:
        tensors["cov3Ds_precomp"] = torch.empty((0,), dtype=torch.float32, device=DEVICE)
    return tensors


def _three_d_loss(outputs) -> torch.Tensor:
    color, _radii, meta = outputs
    return color.square().mean() + meta.alpha.mean() + 0.01 * meta.depth.mean()


def _twod_loss(outputs) -> torch.Tensor:
    return (
        outputs.color.square().mean()
        + outputs.alpha.mean()
        + 0.01 * outputs.depth.mean()
        + 0.001 * outputs.normal.square().mean()
        + 0.001 * outputs.distortion.mean()
    )


def _clear_grads(inputs: dict[str, torch.Tensor]) -> None:
    for tensor in inputs.values():
        if tensor.is_floating_point():
            tensor.grad = None


def _time_case(forward, loss_fn, *, warmups: int, repeats: int) -> dict[str, float | bool]:
    for _ in range(warmups):
        inputs, outputs = forward()
        _clear_grads(inputs)
        loss_fn(outputs).backward()
    torch.cuda.synchronize(DEVICE)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        inputs, outputs = forward()
        _clear_grads(inputs)
        loss = loss_fn(outputs)
        loss.backward()
    end.record()
    end.synchronize()
    return {
        "forward_backward_ms": start.elapsed_time(end) / repeats,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(DEVICE) / float(1024**2),
        "finite_loss": bool(torch.isfinite(loss)),
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _three_d_pair(count: int, height: int, width: int):
    baseline_inputs = _inputs(count)
    baseline = rasterize_gaussians(**baseline_inputs, raster_settings=_three_d_settings(height, width))
    generated_inputs = _inputs(count)
    generated = rasterize_generated(**generated_inputs, raster_settings=_three_d_settings(height, width))
    mip_inputs = _inputs(count)
    mip_zero = rasterize_mip(**mip_inputs, raster_settings=_three_d_settings(height, width, filter_variance=0.0))
    return baseline, generated, mip_zero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("classic_methods_microbench requires CUDA")
    if min(args.count, args.height, args.width, args.repeats) <= 0 or args.warmups < 0:
        raise ValueError("count, dimensions, and repeats must be positive; warmups must be non-negative")

    baseline, generated, mip_zero = _three_d_pair(args.count, args.height, args.width)
    filtered_inputs = _inputs(args.count)
    filtered = rasterize_mip(
        **filtered_inputs,
        raster_settings=_three_d_settings(args.height, args.width, filter_variance=0.25),
    )
    filtered_loss = _three_d_loss(filtered)
    filtered_loss.backward()

    surfel_count = min(args.count, 32)
    surfel_inputs = _inputs(surfel_count, surfels=True)
    surfel = rasterize_surfels(
        **surfel_inputs,
        raster_settings=_twod_settings(args.height, args.width),
    )
    surfel_loss = _twod_loss(surfel)
    surfel_loss.backward()

    records = {
        "generated_3dgs": _time_case(
            lambda: (
                (inputs := _inputs(args.count)),
                rasterize_generated(**inputs, raster_settings=_three_d_settings(args.height, args.width)),
            ),
            _three_d_loss,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "mip_3dgs_filter_0": _time_case(
            lambda: (
                (inputs := _inputs(args.count)),
                rasterize_mip(**inputs, raster_settings=_three_d_settings(args.height, args.width, filter_variance=0.0)),
            ),
            _three_d_loss,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "mip_3dgs_filter_025": _time_case(
            lambda: (
                (inputs := _inputs(args.count)),
                rasterize_mip(**inputs, raster_settings=_three_d_settings(args.height, args.width, filter_variance=0.25)),
            ),
            _three_d_loss,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "twodgs": _time_case(
            lambda: (
                (inputs := _inputs(surfel_count, surfels=True)),
                rasterize_surfels(**inputs, raster_settings=_twod_settings(args.height, args.width)),
            ),
            _twod_loss,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
    }
    report = {
        "config": vars(args) | {"output": None},
        "device": torch.cuda.get_device_name(DEVICE),
        "equivalence": {
            "generated_color_max_abs": _max_abs(generated[0], baseline[0]),
            "generated_alpha_max_abs": _max_abs(generated[2].alpha, baseline[2].alpha),
            "mip_zero_color_max_abs": _max_abs(mip_zero[0], baseline[0]),
            "mip_zero_alpha_max_abs": _max_abs(mip_zero[2].alpha, baseline[2].alpha),
        },
        "finite": {
            "mip_filtered_loss": bool(torch.isfinite(filtered_loss)),
            "mip_filtered_means3d_grad": bool(torch.isfinite(filtered_inputs["means3D"].grad).all()),
            "twod_loss": bool(torch.isfinite(surfel_loss)),
            "twod_means2d_grad": bool(torch.isfinite(surfel_inputs["means2D"].grad).all()),
        },
        "records": records,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
