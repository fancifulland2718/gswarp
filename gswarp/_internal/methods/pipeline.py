"""Shared typed forward pipeline for registered Gaussian methods."""

from __future__ import annotations

from typing import Any

from gswarp._internal.backends.warp.state import ForwardResult, RenderStageResult
from gswarp._internal.methods.contracts import RasterPipelineInputs


def execute_typed_forward(plan, args: tuple[Any, ...]) -> ForwardResult:
    """Execute one method plan without bypassing its replaceable stages."""

    inputs = plan.input_adapter(args)
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
