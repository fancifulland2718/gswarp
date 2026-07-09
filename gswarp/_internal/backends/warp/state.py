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
    point_offsets: torch.Tensor
    point_list: torch.Tensor
    point_list_keys: torch.Tensor
    ranges: torch.Tensor
    num_rendered: int


__all__ = ["PreprocessOutputs", "BinningState"]
