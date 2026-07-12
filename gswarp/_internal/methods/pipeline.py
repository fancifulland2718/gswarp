"""Shared typed forward pipeline for registered Gaussian methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gswarp._internal.backends.warp.state import ForwardResult, RenderStageResult


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


def execute_typed_forward(plan, args: tuple[Any, ...]) -> ForwardResult:
    """Execute one method plan without bypassing its replaceable stages."""

    inputs = RasterPipelineInputs.from_compatibility_args(args)
    if inputs.means3d.ndim != 2 or inputs.means3d.shape[1] != 3:
        raise ValueError("means3D must have dimensions (num_points, 3)")
    if inputs.means3d.shape[0] == 0:
        return plan.stages.empty_forward(inputs)

    preprocess_outputs = plan.stages.preprocess(inputs)
    binning_state = plan.stages.binning(
        preprocess_outputs, inputs.image_height, inputs.image_width
    )
    render_result = plan.stages.render(
        inputs,
        preprocess_outputs,
        binning_state,
        plan.stages.features(inputs, preprocess_outputs),
    )
    if not isinstance(render_result, RenderStageResult):
        raise TypeError("method render stage must return RenderStageResult")

    state = plan.stages.build_state(preprocess_outputs, binning_state, render_result)
    return ForwardResult(
        num_rendered=binning_state.num_rendered,
        color=render_result.color,
        depth=render_result.depth,
        alpha=render_result.alpha,
        radii=preprocess_outputs.radii,
        proj_2d=preprocess_outputs.proj_2d,
        conic_2d=preprocess_outputs.conic_2d,
        conic_2d_inv=preprocess_outputs.conic_2d_inv,
        state=state,
        aux=render_result.aux,
    )


__all__ = ["RasterPipelineInputs", "execute_typed_forward"]
