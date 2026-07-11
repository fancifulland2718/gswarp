from __future__ import annotations

from typing import Any

import torch
import warp as wp

from dataclasses import dataclass

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


__all__ = ["PreprocessOutputs", "BinningState", "ForwardState", "ForwardResult"]
