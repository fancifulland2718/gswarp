"""CPU-safe executable method-plan contracts."""

from __future__ import annotations

from dataclasses import replace
import os
import unittest

from gswarp._internal.backends.select import (
    build_method_plan,
    clear_method_plan_cache,
    resolve_backend,
)
from gswarp._internal.methods.registry import get_method
from gswarp._internal.methods.spec import MethodSpec
from gswarp.methods.baseline_3dgs import METHOD as BASELINE_3DGS
from gswarp.methods.flow_aux import METHOD as FLOW_AUX


class MethodPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_mode = os.environ.get("GSWARP_WARP_BACKEND")
        os.environ["GSWARP_WARP_BACKEND"] = "stable"
        clear_method_plan_cache()

    def tearDown(self) -> None:
        if self._backend_mode is None:
            os.environ.pop("GSWARP_WARP_BACKEND", None)
        else:
            os.environ["GSWARP_WARP_BACKEND"] = self._backend_mode
        clear_method_plan_cache()

    def test_registered_methods_resolve_once_to_immutable_plans(self) -> None:
        baseline_first = resolve_backend(BASELINE_3DGS)
        baseline_second = resolve_backend("baseline_3dgs")
        flow = resolve_backend(FLOW_AUX)

        self.assertIs(baseline_first, baseline_second)
        self.assertEqual(baseline_first.spec, BASELINE_3DGS)
        self.assertFalse(baseline_first.flow)
        self.assertTrue(flow.flow)
        self.assertIs(baseline_first.stages.forward, baseline_first.backend.rasterize_gaussians_typed)
        self.assertIs(flow.stages.forward, flow.backend.rasterize_gaussians_typed)

    def test_registry_rejects_unregistered_spec_with_same_name(self) -> None:
        altered = replace(BASELINE_3DGS, filtering="mip")
        with self.assertRaisesRegex(ValueError, "not registered"):
            get_method(altered)

    def test_synthetic_plan_replaces_one_stage_and_reuses_the_rest(self) -> None:
        baseline = resolve_backend(BASELINE_3DGS)

        def synthetic_preprocess(*args, **kwargs):
            return baseline.stages.preprocess(*args, **kwargs)

        synthetic_spec = replace(BASELINE_3DGS, name="test_synthetic_baseline")
        synthetic = build_method_plan(
            synthetic_spec,
            baseline.backend,
            stage_overrides={"preprocess": synthetic_preprocess},
        )

        self.assertIs(synthetic.stages.preprocess, synthetic_preprocess)
        self.assertIs(synthetic.stages.binning, baseline.stages.binning)
        self.assertIs(synthetic.stages.render, baseline.stages.render)
        self.assertIs(synthetic.stages.forward, baseline.stages.forward)
        self.assertIs(synthetic.stages.backward, baseline.stages.backward)

    def test_advanced_requirement_is_not_satisfied_by_stable_backend(self) -> None:
        baseline = resolve_backend(BASELINE_3DGS)
        advanced_only = MethodSpec(
            name="test_advanced_only",
            backend_family="warp_3dgs",
            output_mode="standard_meta",
            requires_advanced_warp=True,
        )

        with self.assertRaisesRegex(RuntimeError, "advanced_warp"):
            build_method_plan(advanced_only, baseline.backend)

    def test_auto_falls_back_and_forced_advanced_fails_clearly(self) -> None:
        os.environ["GSWARP_WARP_BACKEND"] = "auto"
        clear_method_plan_cache()
        self.assertIn("stable_warp", resolve_backend(BASELINE_3DGS).capabilities)

        os.environ["GSWARP_WARP_BACKEND"] = "advanced"
        clear_method_plan_cache()
        with self.assertRaisesRegex(RuntimeError, "Advanced Warp backend unavailable"):
            resolve_backend(BASELINE_3DGS)


if __name__ == "__main__":
    unittest.main()
