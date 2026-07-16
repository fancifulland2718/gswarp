"""Mip-Splatting stage bindings on the stable Warp backend."""

from __future__ import annotations

import math

import torch
import warp as wp

from gswarp._internal.coverage import (
    BASELINE_3DGS_COVERAGE,
    resolve_tile_coverage_mode_id,
)

from . import backend_3dgs as _base
from . import runtime as _runtime
from .binning_kernels import _recount_covered_tiles_warp_kernel
from .constants import BLOCK_X, BLOCK_Y


BACKEND_CAPABILITIES = _base.BACKEND_CAPABILITIES | frozenset({"mip_3dgs"})
ForwardState = _base.ForwardState
_build_binning_state = _base._build_binning_state
empty_forward_stage = _base.empty_forward_stage
feature_stage = _base.feature_stage
render_stage = _base.render_stage
build_state_stage = _base.build_state_stage
mark_visible = _base.mark_visible
rasterize_gaussians_backward_typed = _base.rasterize_gaussians_backward_typed


def _recount_filtered_tiles(preprocess_outputs, image_height: int, image_width: int) -> None:
    point_count = preprocess_outputs.radii.shape[0]
    if point_count == 0:
        return
    grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
    grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    coverage_mode = resolve_tile_coverage_mode_id(
        _runtime.get_active_tile_coverage_mode(), BASELINE_3DGS_COVERAGE
    )
    coverage_masks = torch.empty((point_count,), dtype=torch.int64, device=preprocess_outputs.radii.device)
    wp.launch(
        kernel=_recount_covered_tiles_warp_kernel,
        dim=point_count,
        inputs=[
            wp.from_torch(preprocess_outputs.points_xy_image.contiguous(), dtype=wp.vec2),
            wp.from_torch(preprocess_outputs.radii.contiguous(), dtype=wp.int32),
            wp.from_torch(preprocess_outputs.conic_opacity.contiguous(), dtype=wp.vec4),
            wp.from_torch(preprocess_outputs.conic_2d_inv.contiguous(), dtype=wp.vec3),
            int(grid_x),
            int(grid_y),
            int(coverage_mode),
        ],
        outputs=[
            wp.from_torch(preprocess_outputs.tiles_touched, dtype=wp.int32),
            wp.from_torch(coverage_masks, dtype=wp.int64),
        ],
        device=str(preprocess_outputs.radii.device),
    )


def preprocess_stage(inputs):
    """Apply the method-owned screen-space Mip filter after base projection."""

    filter_variance = float(inputs.filter_variance)
    if not math.isfinite(filter_variance) or filter_variance < 0.0:
        raise ValueError("filter_variance must be finite and non-negative")

    outputs = _base.preprocess_stage(inputs)
    outputs.cov2d_filter_variance = filter_variance
    if filter_variance == 0.0 or outputs.radii.numel() == 0:
        return outputs

    active = outputs.radii > 0
    cov = outputs.conic_2d_inv
    filtered = torch.stack(
        (cov[:, 0] + filter_variance, cov[:, 1], cov[:, 2] + filter_variance), dim=1
    )
    determinant = filtered[:, 0] * filtered[:, 2] - filtered[:, 1].square()
    valid = active & (determinant > 1.0e-12)
    inv_det = torch.where(valid, determinant.reciprocal(), torch.zeros_like(determinant))
    conic = torch.stack(
        (filtered[:, 2] * inv_det, -filtered[:, 1] * inv_det, filtered[:, 0] * inv_det), dim=1
    )
    outputs.conic_2d_inv = torch.where(valid[:, None], filtered, torch.zeros_like(filtered))
    outputs.conic_2d = torch.where(valid[:, None], conic, torch.zeros_like(conic))
    outputs.conic_opacity = torch.cat(
        (outputs.conic_2d, outputs.conic_opacity[:, 3:4]), dim=1
    )

    trace = filtered[:, 0] + filtered[:, 2]
    discriminant = torch.clamp(
        0.25 * (filtered[:, 0] - filtered[:, 2]).square() + filtered[:, 1].square(),
        min=0.0,
    )
    radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(0.5 * trace + torch.sqrt(discriminant), min=0.1)))
    outputs.radii = torch.where(valid, radius.to(torch.int32), torch.zeros_like(outputs.radii))
    _recount_filtered_tiles(outputs, inputs.image_height, inputs.image_width)
    return outputs




def __getattr__(name):
    return getattr(_base, name)
__all__ = [
    "BACKEND_CAPABILITIES",
    "ForwardState",
    "_build_binning_state",
    "build_state_stage",
    "empty_forward_stage",
    "feature_stage",
    "mark_visible",
    "preprocess_stage",
    "rasterize_gaussians_backward_typed",
    "render_stage",
]
