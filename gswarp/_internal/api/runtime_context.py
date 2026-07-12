"""Call-scoped runtime option helpers."""

from __future__ import annotations

from contextlib import contextmanager

from gswarp._stream import execution_context
from gswarp._internal.backends.warp.runtime import ExecutionOptions, execution_options
from gswarp._internal.coverage import CoverageContract, resolve_tile_coverage_mode


def resolve_execution_options(
    backend,
    raster_settings,
    *,
    flow: bool = False,
    coverage_contract: CoverageContract,
) -> ExecutionOptions:
    """Merge public defaults and per-call settings into an immutable snapshot."""
    backward_mode = getattr(raster_settings, "backward_mode", None)
    binning_sort_mode = getattr(raster_settings, "binning_sort_mode", None)
    tile_coverage_mode = resolve_tile_coverage_mode(
        backend.get_tile_coverage_mode(), coverage_contract
    )
    compute_flow_aux = getattr(raster_settings, "compute_flow_aux", None) if flow else None
    return ExecutionOptions(
        backward_mode=backend.get_backward_mode() if backward_mode is None else backward_mode,
        binning_sort_mode=backend.get_binning_sort_mode() if binning_sort_mode is None else binning_sort_mode,
        tile_coverage_mode=tile_coverage_mode,
        compute_depth=backend.get_compute_depth(),
        auto_tune=bool(getattr(raster_settings, "auto_tune", True)),
        auto_tune_verbose=bool(getattr(raster_settings, "auto_tune_verbose", True)),
        compute_flow_aux=(backend.get_compute_flow_aux() if compute_flow_aux is None else bool(compute_flow_aux))
        if flow
        else None,
    )


@contextmanager
def runtime_overrides(
    backend,
    raster_settings,
    *,
    flow: bool = False,
    coverage_contract: CoverageContract | None = None,
    options: ExecutionOptions | None = None,
):
    """Bind one immutable option snapshot and the matching CUDA stream."""
    if options is None:
        if coverage_contract is None:
            raise ValueError("coverage_contract is required when options are not pre-resolved")
        options = resolve_execution_options(
            backend,
            raster_settings,
            flow=flow,
            coverage_contract=coverage_contract,
        )
    with execution_context(raster_settings.bg.device):
        with execution_options(options):
            yield options


def run_with_runtime_overrides(
    backend,
    raster_settings,
    fn,
    *,
    flow: bool = False,
    coverage_contract: CoverageContract | None = None,
    options: ExecutionOptions | None = None,
):
    with runtime_overrides(
        backend,
        raster_settings,
        flow=flow,
        coverage_contract=coverage_contract,
        options=options,
    ):
        return fn()


__all__ = [
    "ExecutionOptions",
    "resolve_execution_options",
    "runtime_overrides",
    "run_with_runtime_overrides",
]
