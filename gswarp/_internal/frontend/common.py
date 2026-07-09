"""Shared frontend helpers."""

from __future__ import annotations

from gswarp._internal.backends.select import resolve_backend


def backend_for(method):
    return resolve_backend(method)


def mark_visible(backend, positions, raster_settings):
    return backend.mark_visible(
        positions,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
    )


__all__ = ["backend_for", "mark_visible"]
