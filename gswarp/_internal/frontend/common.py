"""Shared frontend helpers."""

from __future__ import annotations

from gswarp._internal.backends.select import resolve_backend
from gswarp._internal.api.runtime_context import resolve_execution_options, run_with_runtime_overrides


def plan_for(method):
    return resolve_backend(method)


def backend_for(method):
    """Compatibility helper for public runtime-control accessors."""
    return plan_for(method).backend


def run_typed_forward(plan, raster_settings, args):
    options = resolve_execution_options(plan.backend, raster_settings, flow=plan.flow)
    result = run_with_runtime_overrides(
        plan.backend,
        raster_settings,
        lambda: plan.stages.forward(*args),
        flow=plan.flow,
        options=options,
    )
    return result, options


def run_typed_backward(plan, raster_settings, args, options, forward_state):
    return run_with_runtime_overrides(
        plan.backend,
        raster_settings,
        lambda: plan.stages.backward(*args, forward_state=forward_state),
        flow=plan.flow,
        options=options,
    )


def adapt_outputs(plan, outputs, meta_type=None):
    return plan.output_adapter(outputs, meta_type)


def mark_visible(plan, positions, raster_settings):
    return plan.stages.mark_visible(
        positions,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
    )


__all__ = [
    "adapt_outputs",
    "backend_for",
    "mark_visible",
    "plan_for",
    "run_typed_backward",
    "run_typed_forward",
]
