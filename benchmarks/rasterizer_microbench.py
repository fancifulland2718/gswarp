"""Deterministic correctness and latency microbenchmark for the public rasterizer.

The script intentionally does not write artifacts by default.  Pass ``--output``
to store a JSON baseline outside the repository, then use ``--baseline`` on a
later revision to reject numerical drift before a full densification run.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import warp as wp

from gswarp.rasterizer import (
    GaussianRasterizationSettings,
    clear_warp_caches,
    rasterize_gaussians,
)


DEFAULT_COUNTS = (4_096, 16_384, 65_536, 262_144)
DEVICE = torch.device("cuda:0")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _signature(tensor: torch.Tensor) -> dict[str, float | int]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    return {
        "numel": value.numel(),
        "finite": int(finite.sum().item()),
        "sum": float(value.sum().item()),
        "abs_sum": float(value.abs().sum().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
    }


def _make_inputs(count: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed + count)
    means3d = torch.empty((count, 3), dtype=torch.float32, device=DEVICE)
    means3d[:, :2].uniform_(-0.65, 0.65, generator=generator)
    means3d[:, 2].uniform_(2.0, 3.0, generator=generator)
    means3d.requires_grad_(True)

    means2d = torch.zeros((count, 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
    colors = torch.rand((count, 3), dtype=torch.float32, device=DEVICE, generator=generator).requires_grad_(True)
    opacities = torch.empty((count, 1), dtype=torch.float32, device=DEVICE)
    opacities.uniform_(0.1, 0.9, generator=generator)
    opacities.requires_grad_(True)
    scales = torch.empty((count, 3), dtype=torch.float32, device=DEVICE)
    scales.uniform_(0.025, 0.06, generator=generator)
    scales.requires_grad_(True)
    rotations = torch.zeros((count, 4), dtype=torch.float32, device=DEVICE)
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)

    return {
        "means3D": means3d,
        "means2D": means2d,
        "sh": torch.empty((0,), dtype=torch.float32, device=DEVICE),
        "colors_precomp": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "cov3Ds_precomp": torch.empty((0,), dtype=torch.float32, device=DEVICE),
    }


def _settings(height: int, width: int) -> GaussianRasterizationSettings:
    return GaussianRasterizationSettings(
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


def _clear_grads(inputs: dict[str, torch.Tensor]) -> None:
    for name in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations"):
        inputs[name].grad = None


def _run_step(
    inputs: dict[str, torch.Tensor], settings: GaussianRasterizationSettings, *, collect: bool
) -> dict[str, Any] | None:
    _clear_grads(inputs)
    color, radii, meta = rasterize_gaussians(**inputs, raster_settings=settings)
    loss = (
        color.square().mean()
        + meta.alpha.mean()
        + 0.01 * meta.depth.mean()
        + 1.0e-4 * (meta.proj_2D.square().mean() + meta.conic_2D.sum() + meta.conic_2D_inv.sum())
    )
    loss.backward()
    if not collect:
        return None

    return {
        "loss": _signature(loss.reshape(1)),
        "visible_gaussians": int((radii > 0).sum().item()),
        "outputs": {
            "color": _signature(color),
            "depth": _signature(meta.depth),
            "alpha": _signature(meta.alpha),
            "proj_2D": _signature(meta.proj_2D),
            "conic_2D": _signature(meta.conic_2D),
            "conic_2D_inv": _signature(meta.conic_2D_inv),
        },
        "gradients": {
            name: _signature(inputs[name].grad)
            for name in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations")
        },
    }


def _difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    if left.get("visible_gaussians") != right.get("visible_gaussians"):
        return {"max_abs": math.inf, "max_rel": math.inf}
    largest_abs = 0.0
    largest_rel = 0.0
    for section in ("loss", "outputs", "gradients"):
        left_items = {"loss": left[section]} if section == "loss" else left[section]
        right_items = {"loss": right[section]} if section == "loss" else right[section]
        for name, left_value in left_items.items():
            right_value = right_items[name]
            if left_value["numel"] != right_value["numel"]:
                return {"max_abs": math.inf, "max_rel": math.inf}
            if left_value["finite"] != left_value["numel"] or right_value["finite"] != right_value["numel"]:
                return {"max_abs": math.inf, "max_rel": math.inf}
            for field in ("sum", "abs_sum", "max_abs"):
                delta = abs(float(left_value[field]) - float(right_value[field]))
                largest_abs = max(largest_abs, delta)
                largest_rel = max(largest_rel, delta / max(abs(float(right_value[field])), 1.0))
    return {"max_abs": largest_abs, "max_rel": largest_rel}


def _profile_case(count: int, args: argparse.Namespace) -> dict[str, Any]:
    clear_warp_caches()
    settings = _settings(args.height, args.width)
    inputs = _make_inputs(count, args.seed)
    for _ in range(args.warmups):
        _run_step(inputs, settings, collect=False)
    torch.cuda.synchronize(DEVICE)

    first = _run_step(inputs, settings, collect=True)
    second = _run_step(inputs, settings, collect=True)
    torch.cuda.synchronize(DEVICE)
    repeatability = _difference(first, second)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        _run_step(inputs, settings, collect=False)
    end.record()
    end.synchronize()

    return {
        "gaussian_count": count,
        "signature": first,
        "repeatability": repeatability,
        "timing_ms": {"forward_backward": start.elapsed_time(end) / args.repeats},
        "peak_allocated_mb": torch.cuda.max_memory_allocated(DEVICE) / float(1024**2),
    }


def _verify_baseline(records: list[dict[str, Any]], baseline_path: Path, args: argparse.Namespace) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {record["gaussian_count"]: record for record in baseline["records"]}
    for record in records:
        previous = expected.get(record["gaussian_count"])
        if previous is None:
            raise RuntimeError(f"baseline has no record for {record['gaussian_count']} Gaussians")
        drift = _difference(record["signature"], previous["signature"])
        if drift["max_abs"] > args.atol and drift["max_rel"] > args.rtol:
            raise RuntimeError(
                f"numerical signature drift at N={record['gaussian_count']}: "
                f"abs={drift['max_abs']:.3e}, rel={drift['max_rel']:.3e}"
            )


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(DEVICE)
    return {
        "git_sha": _git_sha(),
        "module_file": str(Path(sys.modules["gswarp"].__file__).resolve()),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "device_capability": f"{properties.major}.{properties.minor}",
    }


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(",") if item.strip())
    if not counts or any(item <= 0 for item in counts):
        raise argparse.ArgumentTypeError("counts must be a comma-separated list of positive integers")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=_parse_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("rasterizer_microbench requires a CUDA-enabled PyTorch runtime")
    if args.height <= 0 or args.width <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise ValueError("image dimensions and repeats must be positive; warmups must be non-negative")

    records = [_profile_case(count, args) for count in args.counts]
    if args.baseline is not None:
        _verify_baseline(records, args.baseline, args)
    report = {"environment": _environment(), "config": vars(args) | {"baseline": str(args.baseline) if args.baseline else None, "output": None}, "records": records}
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
