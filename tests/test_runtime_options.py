"""Tests for call-scoped runtime option ownership."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from gswarp._internal.api.runtime_context import resolve_execution_options, runtime_overrides
from gswarp._internal.backends.warp import runtime


class _BackendDefaults:
    def get_backward_mode(self):
        return "manual"

    def get_binning_sort_mode(self):
        return "warp_depth_stable_tile"

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
            self.assertEqual(runtime.get_active_auto_tuning_config(), (False, True))
            with runtime_overrides(backend, inner_settings):
                self.assertEqual(runtime.get_active_binning_sort_mode(), "warp_radix")
                self.assertEqual(runtime.get_active_auto_tuning_config(), (True, True))
            self.assertEqual(runtime.get_active_binning_sort_mode(), "torch")
            self.assertEqual(runtime.get_active_auto_tuning_config(), (False, True))

        self.assertEqual(runtime.get_active_binning_sort_mode(), backend.get_binning_sort_mode())

    def test_flow_option_is_scoped_with_the_same_snapshot(self) -> None:
        backend = _BackendDefaults()
        settings = _settings(compute_flow_aux=False)

        with runtime_overrides(backend, settings, flow=True):
            self.assertFalse(runtime.get_active_compute_flow_aux(True))
        self.assertTrue(runtime.get_active_compute_flow_aux(True))


if __name__ == "__main__":
    unittest.main()
