from __future__ import annotations

from typing import Any

import torch

from dataclasses import dataclass


@dataclass(slots=True)
class RenderBackwardInterop:
    """Forward-created Warp views reused by the standard render backward."""

    ranges: Any
    point_list: Any
    points_xy: Any
    features: Any
    depths: Any
    conic_opacity: Any
    background: Any
    out_alpha: Any
    n_contrib: Any


@dataclass(slots=True)
class PreprocessBackwardInterop:
    """Forward-created Warp views reused by preprocess and SH backward."""

    means3d: Any | None = None
    scales: Any | None = None
    rotations: Any | None = None
    viewmatrix: Any | None = None
    projmatrix: Any | None = None
    radii: Any | None = None
    cov3d: Any | None = None
    campos: Any | None = None
    clamped: Any | None = None


@dataclass(slots=True)
class StandardBackwardInterop:
    """Owned non-autograd views retained by one standard forward graph."""

    render: RenderBackwardInterop | None = None
    preprocess: PreprocessBackwardInterop | None = None


@dataclass(slots=True)
class PreprocessOutputs:
    visible: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    proj_2d: torch.Tensor
    conic_2d: torch.Tensor
    conic_2d_inv: torch.Tensor
    points_xy_image: torch.Tensor
    tiles_touched: torch.Tensor
    rgb: torch.Tensor
    clamped: torch.Tensor
    conic_opacity: torch.Tensor
    cov3d_all: torch.Tensor
    backward_interop: PreprocessBackwardInterop | None = None
    cov2d_filter_variance: float = 0.0


@dataclass(slots=True)
class BinningState:
    grid_x: int
    grid_y: int
    point_list: torch.Tensor
    ranges: torch.Tensor
    num_rendered: int


@dataclass(slots=True)
class ForwardState:
    """Typed tensors retained by one normal frontend forward/backward pair."""

    preprocess: PreprocessOutputs
    binning: BinningState
    n_contrib: torch.Tensor
    backward_interop: StandardBackwardInterop | None = None


@dataclass(slots=True)
class RenderStageResult:
    """Normalized render output consumed by the shared method pipeline."""

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    n_contrib: torch.Tensor
    backward_interop: RenderBackwardInterop | None = None
    aux: tuple[torch.Tensor, ...] = ()


@dataclass(slots=True)
class ForwardResult:
    """Normal frontend result that avoids raw compatibility-buffer packing."""

    num_rendered: int
    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    radii: torch.Tensor
    proj_2d: torch.Tensor
    conic_2d: torch.Tensor
    conic_2d_inv: torch.Tensor
    state: ForwardState | None
    aux: tuple[torch.Tensor, ...] = ()


__all__ = [
    "RenderBackwardInterop",
    "PreprocessBackwardInterop",
    "StandardBackwardInterop",
    "PreprocessOutputs",
    "BinningState",
    "ForwardState",
    "RenderStageResult",
    "ForwardResult",
]
