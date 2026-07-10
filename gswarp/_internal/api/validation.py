"""Public input validation helpers."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


_INT32_MAX = 2**31 - 1


class GaussianInputs(NamedTuple):
    shs: torch.Tensor
    colors_precomp: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    cov3D_precomp: torch.Tensor


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def _require_float32_cuda(name: str, value: torch.Tensor, device: torch.device) -> None:
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {value.dtype}")


def _require_shape(name: str, value: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        expected = tuple("N" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


def _debug_check_finite(name: str, value: torch.Tensor) -> None:
    if value.numel() != 0 and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf values")


def validate_rasterizer_inputs(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
) -> None:
    """Validate public rasterizer inputs before any Warp state is touched."""
    means3D = _require_tensor("means3D", means3D)
    means2D = _require_tensor("means2D", means2D)
    opacities = _require_tensor("opacities", opacities)
    if not means3D.is_cuda:
        raise ValueError("means3D must be a CUDA tensor")
    device = means3D.device
    _require_float32_cuda("means3D", means3D, device)
    _require_shape("means3D", means3D, (None, 3))

    point_count = means3D.shape[0]
    if point_count > _INT32_MAX:
        raise ValueError("the Gaussian count must fit in a signed 32-bit index")
    _require_float32_cuda("means2D", means2D, device)
    _require_shape("means2D", means2D, (point_count, 3))
    _require_float32_cuda("opacities", opacities, device)
    if opacities.numel() != point_count:
        raise ValueError(
            f"opacities must contain one value per Gaussian ({point_count}), got {opacities.numel()}"
        )

    try:
        image_height = raster_settings.image_height
        image_width = raster_settings.image_width
        tanfovx = raster_settings.tanfovx
        tanfovy = raster_settings.tanfovy
        scale_modifier = raster_settings.scale_modifier
        sh_degree = raster_settings.sh_degree
        background = _require_tensor("raster_settings.bg", raster_settings.bg)
        viewmatrix = _require_tensor("raster_settings.viewmatrix", raster_settings.viewmatrix)
        projmatrix = _require_tensor("raster_settings.projmatrix", raster_settings.projmatrix)
        campos = _require_tensor("raster_settings.campos", raster_settings.campos)
    except AttributeError as exc:
        raise ValueError("raster_settings is missing a required field") from exc

    if not isinstance(image_height, int) or not isinstance(image_width, int) or image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive integers")
    if image_height * image_width > _INT32_MAX:
        raise ValueError("image_height * image_width must fit in a signed 32-bit index")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (tanfovx, tanfovy, scale_modifier)):
        raise ValueError("tanfovx, tanfovy, and scale_modifier must be finite and positive")
    if not isinstance(sh_degree, int) or sh_degree < 0:
        raise ValueError("sh_degree must be a non-negative integer")

    for name, value, shape in (
        ("raster_settings.bg", background, (3,)),
        ("raster_settings.viewmatrix", viewmatrix, (4, 4)),
        ("raster_settings.projmatrix", projmatrix, (4, 4)),
        ("raster_settings.campos", campos, (3,)),
    ):
        _require_float32_cuda(name, value, device)
        _require_shape(name, value, shape)

    optional = {
        "sh": sh,
        "colors_precomp": colors_precomp,
        "scales": scales,
        "rotations": rotations,
        "cov3Ds_precomp": cov3Ds_precomp,
    }
    for name, value in optional.items():
        value = _require_tensor(name, value)
        if value.numel() != 0:
            _require_float32_cuda(name, value, device)

    has_sh = sh.numel() != 0
    has_colors = colors_precomp.numel() != 0
    has_scales = scales.numel() != 0
    has_rotations = rotations.numel() != 0
    has_covariances = cov3Ds_precomp.numel() != 0
    if point_count != 0 and has_sh == has_colors:
        raise ValueError("provide exactly one of sh or colors_precomp")
    if point_count != 0 and has_scales != has_rotations:
        raise ValueError("provide exactly one of scales/rotations or cov3Ds_precomp")
    if point_count != 0 and (has_scales or has_rotations) == has_covariances:
        raise ValueError("provide exactly one of scales/rotations or cov3Ds_precomp")

    if has_colors:
        _require_shape("colors_precomp", colors_precomp, (point_count, 3))
    if has_scales:
        _require_shape("scales", scales, (point_count, 3))
    if has_rotations:
        _require_shape("rotations", rotations, (point_count, 4))
    if has_covariances:
        _require_shape("cov3Ds_precomp", cov3Ds_precomp, (point_count, 6))
    if has_sh:
        if sh.numel() % (point_count * 3) != 0:
            raise ValueError("sh must contain a whole number of RGB coefficients per Gaussian")
        if sh.numel() // (point_count * 3) < (sh_degree + 1) ** 2:
            raise ValueError("sh does not contain enough coefficients for sh_degree")

    if getattr(raster_settings, "debug", False):
        for name, value in (
            ("means3D", means3D),
            ("means2D", means2D),
            ("opacities", opacities),
            ("raster_settings.bg", background),
            ("raster_settings.viewmatrix", viewmatrix),
            ("raster_settings.projmatrix", projmatrix),
            ("raster_settings.campos", campos),
            *optional.items(),
        ):
            _debug_check_finite(name, value)


def normalize_gaussian_inputs(
    *,
    dc,
    shs,
    colors_precomp,
    scales,
    rotations,
    cov3D_precomp,
) -> GaussianInputs:
    """Normalize public rasterizer optional inputs without touching backend state."""
    if dc is not None and shs is not None:
        shs = torch.cat([dc, shs], dim=1)
    elif dc is not None and shs is None:
        shs = dc

    if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
        raise ValueError("provide exactly one of shs or colors_precomp")

    if ((scales is None or rotations is None) and cov3D_precomp is None) or (
        (scales is not None or rotations is not None) and cov3D_precomp is not None
    ):
        raise ValueError("provide exactly one of scales/rotations or cov3D_precomp")

    if shs is None:
        shs = torch.Tensor([])
    if colors_precomp is None:
        colors_precomp = torch.Tensor([])
    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    return GaussianInputs(
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )


__all__ = ["GaussianInputs", "normalize_gaussian_inputs", "validate_rasterizer_inputs"]
