"""Runtime override context helpers."""

from __future__ import annotations

from contextlib import contextmanager

from gswarp._stream import ensure_aligned


@contextmanager
def runtime_overrides(backend, raster_settings, *, flow: bool = False):
    """Temporarily apply per-settings backend runtime overrides."""
    ensure_aligned()

    previous_backward_mode = backend.get_backward_mode()
    previous_binning_sort_mode = backend.get_binning_sort_mode()
    previous_auto_tuning = None
    previous_compute_flow_aux = backend.get_compute_flow_aux() if flow else None

    backward_mode = getattr(raster_settings, "backward_mode", None)
    binning_sort_mode = getattr(raster_settings, "binning_sort_mode", None)
    auto_tune = getattr(raster_settings, "auto_tune", True)
    auto_tune_verbose = getattr(raster_settings, "auto_tune_verbose", True)
    compute_flow_aux = getattr(raster_settings, "compute_flow_aux", None) if flow else None

    try:
        previous_auto_tuning = backend.get_runtime_auto_tuning_config()
        backend.set_runtime_auto_tuning(enabled=auto_tune, verbose=auto_tune_verbose)

        if backward_mode is not None and backward_mode != previous_backward_mode:
            backend.set_backward_mode(backward_mode)

        if binning_sort_mode is not None and binning_sort_mode != previous_binning_sort_mode:
            backend.set_binning_sort_mode(binning_sort_mode)

        if flow and compute_flow_aux is not None and bool(compute_flow_aux) != previous_compute_flow_aux:
            backend.set_compute_flow_aux(bool(compute_flow_aux))

        yield
    finally:
        if previous_auto_tuning is not None:
            current_auto_tuning = backend.get_runtime_auto_tuning_config()
            if current_auto_tuning != previous_auto_tuning:
                backend.set_runtime_auto_tuning(
                    enabled=previous_auto_tuning["enabled"],
                    verbose=previous_auto_tuning["verbose"],
                )
        if flow and backend.get_compute_flow_aux() != previous_compute_flow_aux:
            backend.set_compute_flow_aux(previous_compute_flow_aux)
        if backend.get_binning_sort_mode() != previous_binning_sort_mode:
            backend.set_binning_sort_mode(previous_binning_sort_mode)
        if backend.get_backward_mode() != previous_backward_mode:
            backend.set_backward_mode(previous_backward_mode)


def run_with_runtime_overrides(backend, raster_settings, fn, *, flow: bool = False):
    with runtime_overrides(backend, raster_settings, flow=flow):
        return fn()


__all__ = ["runtime_overrides", "run_with_runtime_overrides"]
