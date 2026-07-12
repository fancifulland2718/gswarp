"""Tests for call-scoped runtime option ownership."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gswarp._internal.api.runtime_context import resolve_execution_options, runtime_overrides
from gswarp._internal.backends.warp import memory, runtime
from gswarp._internal.coverage import FOOTPRINT_CUSTOM


class _BackendDefaults:
    def get_backward_mode(self):
        return "manual"

    def get_binning_sort_mode(self):
        return "warp_depth_stable_tile"

    def get_tile_coverage_mode(self):
        return "auto"

    def get_compute_depth(self):
        return True

    def get_compute_flow_aux(self):
        return True


def _settings(**overrides):
    values = {
        "bg": torch.zeros(3),
        "backward_mode": None,
        "binning_sort_mode": None,
        "auto_tune": True,
        "auto_tune_verbose": True,
        "compute_flow_aux": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RuntimeOptionsTests(unittest.TestCase):
    def test_settings_override_defaults_without_mutating_them(self) -> None:
        backend = _BackendDefaults()
        settings = _settings(binning_sort_mode="torch", auto_tune=False, auto_tune_verbose=False)

        options = resolve_execution_options(backend, settings)

        self.assertEqual(options.backward_mode, "manual")
        self.assertEqual(options.binning_sort_mode, "torch")
        self.assertEqual(options.tile_coverage_mode, "accutile_sweep")
        self.assertTrue(options.compute_depth)
        self.assertFalse(options.auto_tune)
        self.assertFalse(options.auto_tune_verbose)
        self.assertEqual(backend.get_binning_sort_mode(), "warp_depth_stable_tile")

    def test_nested_calls_restore_the_outer_snapshot(self) -> None:
        backend = _BackendDefaults()
        outer_settings = _settings(binning_sort_mode="torch", auto_tune=False)
        inner_settings = _settings(binning_sort_mode="warp_radix", auto_tune=True)

        with runtime_overrides(backend, outer_settings):
            self.assertEqual(runtime.get_active_binning_sort_mode(), "torch")
            self.assertEqual(runtime.get_active_tile_coverage_mode(), "accutile_sweep")
            self.assertEqual(runtime.get_active_auto_tuning_config(), (False, True))
            with runtime_overrides(backend, inner_settings):
                self.assertEqual(runtime.get_active_binning_sort_mode(), "warp_radix")
                self.assertEqual(runtime.get_active_auto_tuning_config(), (True, True))
            self.assertEqual(runtime.get_active_binning_sort_mode(), "torch")
            self.assertEqual(runtime.get_active_auto_tuning_config(), (False, True))

        self.assertEqual(runtime.get_active_binning_sort_mode(), backend.get_binning_sort_mode())

    def test_coverage_mode_is_method_safe_and_call_scoped(self) -> None:
        backend = _BackendDefaults()
        settings = _settings()

        options = resolve_execution_options(
            backend,
            settings,
            footprint_capability=FOOTPRINT_CUSTOM,
        )
        self.assertEqual(options.tile_coverage_mode, "snugbox")
        with runtime_overrides(backend, settings, options=options):
            self.assertEqual(runtime.get_active_tile_coverage_mode(), "snugbox")
        self.assertEqual(runtime.get_active_tile_coverage_mode(), runtime.get_tile_coverage_mode())

    def test_flow_option_is_scoped_with_the_same_snapshot(self) -> None:
        backend = _BackendDefaults()
        settings = _settings(compute_flow_aux=False)

        with runtime_overrides(backend, settings, flow=True):
            self.assertFalse(runtime.get_active_compute_flow_aux(True))
        self.assertTrue(runtime.get_active_compute_flow_aux(True))

    def test_tile_coverage_default_setter_validates_and_restores(self) -> None:
        original = runtime.get_tile_coverage_mode()
        try:
            runtime.set_tile_coverage_mode("conic_rect")
            self.assertEqual(runtime.get_tile_coverage_mode(), "conic_rect")
            with self.assertRaisesRegex(ValueError, "accutile_sweep"):
                runtime.set_tile_coverage_mode("not-a-mode")
        finally:
            runtime.set_tile_coverage_mode(original)

    def test_cache_report_and_bounds_are_cpu_safe(self) -> None:
        cache = memory._C4_LAUNCH_CACHE_SH
        cache.clear()
        memory._WORKSPACE_SLOT_CACHE.clear()
        for index in range(memory._MAX_LAUNCH_CACHE_ENTRIES + 1):
            cache[("cpu", index)] = object()

        memory._get_workspace_slot("cpu").sequence = torch.empty(4, dtype=torch.float32)
        report = memory.get_warp_cache_report()

        self.assertEqual(len(cache), memory._MAX_LAUNCH_CACHE_ENTRIES)
        self.assertEqual(report["workspace_entries"], 1)
        self.assertEqual(report["by_cache"]["_C4_LAUNCH_CACHE_SH"]["entries"], memory._MAX_LAUNCH_CACHE_ENTRIES)
        self.assertEqual(report["by_cache"]["_SEQUENCE_BUFFER_CACHE"]["tensor_bytes"], 16)

        memory.clear_common_warp_caches("cpu")
        report = memory.get_warp_cache_report()
        self.assertEqual(report["by_cache"]["_SEQUENCE_BUFFER_CACHE"]["entries"], 0)
        self.assertEqual(report["by_cache"]["_C4_LAUNCH_CACHE_SH"]["entries"], 0)

    def test_warp_allocation_device_does_not_refresh_tuning_report(self) -> None:
        with (
            patch.object(runtime, "_WARP_INITIALIZED", True),
            patch.object(
                runtime,
                "get_runtime_tuning_report",
                side_effect=AssertionError("allocation device lookup must not refresh runtime memory"),
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=2),
        ):
            self.assertEqual(memory._get_runtime_warp_device("cuda"), "cuda:2")
            self.assertEqual(memory._get_runtime_warp_device("cuda:1"), "cuda:1")


if __name__ == "__main__":
    unittest.main()
