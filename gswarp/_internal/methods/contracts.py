"""Typed method-input contracts shared by rasterization providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class MethodInputs(Protocol):
    """Normalized per-call inputs required by the common staged executor."""

    background: torch.Tensor
    means3d: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    scale_modifier: float
    cov3d_precomp: torch.Tensor
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    tan_fovx: float
    tan_fovy: float
    image_height: int
    image_width: int
    sh: torch.Tensor
    degree: int
    campos: torch.Tensor
    prefiltered: bool


@dataclass(frozen=True, slots=True)
class RasterPipelineInputs:
    """Normalized CUDA-GS-compatible arguments for one typed forward call."""

    background: torch.Tensor
    means3d: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    scale_modifier: float
    cov3d_precomp: torch.Tensor
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    tan_fovx: float
    tan_fovy: float
    image_height: int
    image_width: int
    sh: torch.Tensor
    degree: int
    campos: torch.Tensor
    prefiltered: bool

    @classmethod
    def from_compatibility_args(cls, args: tuple[Any, ...]) -> "RasterPipelineInputs":
        if len(args) != 18:
            raise TypeError(f"rasterizer forward expects 18 arguments, received {len(args)}")
        return cls(*args)


@dataclass(frozen=True, slots=True)
class MipRasterPipelineInputs(RasterPipelineInputs):
    """Baseline-compatible inputs plus the method-owned 2D filter variance."""

    filter_variance: float

    @classmethod
    def from_mip_args(cls, args: tuple[Any, ...]) -> "MipRasterPipelineInputs":
        if len(args) != 19:
            raise TypeError(f"mip rasterizer forward expects 19 arguments, received {len(args)}")
        return cls(*args)


@dataclass(frozen=True, slots=True)
class TwoDPipelineInputs:
    """Method-owned inputs for a perspective-correct 2D Gaussian surfel call."""

    background: torch.Tensor
    means3d: torch.Tensor
    means2d: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    tan_fovx: float
    tan_fovy: float
    image_height: int
    image_width: int
    sh: torch.Tensor
    degree: int
    campos: torch.Tensor
    prefiltered: bool
    filter_radius: float
    near_plane: float
    far_plane: float

    @classmethod
    def from_twod_args(cls, args: tuple[Any, ...]) -> "TwoDPipelineInputs":
        if len(args) != 21:
            raise TypeError(f"2DGS rasterizer forward expects 21 arguments, received {len(args)}")
        return cls(*args)


MethodInputAdapter = Callable[[tuple[Any, ...]], MethodInputs]


def compatibility_3dgs_adapter(args: tuple[Any, ...]) -> RasterPipelineInputs:
    """Adapt the frozen CUDA-compatible argument tuple used by baseline 3DGS."""

    return RasterPipelineInputs.from_compatibility_args(args)


def generated_3dgs_adapter(args: tuple[Any, ...]) -> RasterPipelineInputs:
    """Adapt materialized dynamic/anchor Gaussians without detaching tensors."""

    return compatibility_3dgs_adapter(args)


def mip_3dgs_adapter(args: tuple[Any, ...]) -> MipRasterPipelineInputs:
    """Adapt the Mip method's compatibility tuple and explicit filter control."""

    return MipRasterPipelineInputs.from_mip_args(args)


def twodgs_adapter(args: tuple[Any, ...]) -> TwoDPipelineInputs:
    """Adapt 2D surfel tensors to the method-owned typed input schema."""

    return TwoDPipelineInputs.from_twod_args(args)


_INPUT_ADAPTERS: dict[str, MethodInputAdapter] = {
    "compatibility_3dgs": compatibility_3dgs_adapter,
    "generated_3dgs": generated_3dgs_adapter,
    "mip_3dgs": mip_3dgs_adapter,
    "twodgs": twodgs_adapter,
}


def resolve_input_adapter(name: str | None) -> MethodInputAdapter:
    """Resolve one registered adapter before the call enters the hot pipeline."""

    key = "compatibility_3dgs" if name is None else name
    try:
        return _INPUT_ADAPTERS[key]
    except KeyError as exc:
        raise RuntimeError(f"Unknown gswarp method input adapter: {key!r}") from exc


__all__ = [
    "MethodInputAdapter",
    "MethodInputs",
    "MipRasterPipelineInputs",
    "RasterPipelineInputs",
    "TwoDPipelineInputs",
    "compatibility_3dgs_adapter",
    "generated_3dgs_adapter",
    "mip_3dgs_adapter",
    "resolve_input_adapter",
    "twodgs_adapter",
]
