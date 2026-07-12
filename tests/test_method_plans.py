"""CPU-safe executable method-plan contracts."""

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest

import torch

from gswarp._internal.backends.select import (
    build_method_plan,
    clear_method_plan_cache,
    resolve_backend,
)
from gswarp._internal.backends.warp.state import RenderStageResult
from gswarp._internal.backends.warp.advanced import validate_advanced_backend_module
from gswarp._internal.backends.warp.capabilities import (
    WarpFeatureSet,
    detect_warp_features,
    parse_warp_version,
)
from gswarp._internal.methods.registry import get_method
from gswarp._internal.methods.pipeline import execute_typed_forward
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
        self.assertIs(baseline_first.stages.preprocess, baseline_first.backend.preprocess_stage)
        self.assertIs(flow.stages.render, flow.backend.render_stage)

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
        self.assertIs(synthetic.stages.features, baseline.stages.features)
        self.assertIs(synthetic.stages.binning, baseline.stages.binning)
        self.assertIs(synthetic.stages.render, baseline.stages.render)
        self.assertIs(synthetic.stages.build_state, baseline.stages.build_state)
        self.assertIs(synthetic.stages.backward, baseline.stages.backward)

    def test_pipeline_executes_the_resolved_replaceable_stages(self) -> None:
        events: list[str] = []
        point_count = 2
        preprocess_outputs = SimpleNamespace(
            radii=torch.ones(point_count, dtype=torch.int32),
            proj_2d=torch.zeros((point_count, 2)),
            conic_2d=torch.zeros((point_count, 3)),
            conic_2d_inv=torch.zeros((point_count, 3)),
            rgb=torch.full((point_count, 3), 0.25),
        )
        binning_state = SimpleNamespace(num_rendered=3)
        state_marker = object()

        def empty_forward(_inputs):
            raise AssertionError("non-empty test must not use the empty stage")

        def preprocess(inputs):
            events.append("preprocess")
            self.assertEqual(inputs.means3d.shape, (point_count, 3))
            return preprocess_outputs

        def binning(outputs, height, width):
            events.append("binning")
            self.assertIs(outputs, preprocess_outputs)
            self.assertEqual((height, width), (4, 5))
            return binning_state

        def render(inputs, outputs, state, features):
            events.append("render")
            self.assertIs(outputs, preprocess_outputs)
            self.assertIs(state, binning_state)
            torch.testing.assert_close(features, torch.full((point_count, 3), 0.75))
            return RenderStageResult(
                color=torch.zeros((3, 4, 5)),
                depth=torch.zeros((4, 5)),
                alpha=torch.zeros((4, 5)),
                n_contrib=torch.zeros((4, 5), dtype=torch.int32),
            )

        def build_state(outputs, state, render_result):
            events.append("build_state")
            self.assertIs(outputs, preprocess_outputs)
            self.assertIs(state, binning_state)
            self.assertEqual(render_result.color.shape, (3, 4, 5))
            return state_marker

        backend = SimpleNamespace(
            BACKEND_CAPABILITIES=frozenset({"stable_warp"}),
            ForwardState=object,
            empty_forward_stage=empty_forward,
            preprocess_stage=lambda _inputs: None,
            feature_stage=lambda inputs, outputs: inputs.colors,
            _build_binning_state=binning,
            render_stage=render,
            build_state_stage=build_state,
            rasterize_gaussians_backward_typed=lambda *args, **kwargs: None,
            mark_visible=lambda *args, **kwargs: None,
        )
        plan = build_method_plan(
            replace(BASELINE_3DGS, name="test_executable_pipeline"),
            backend,
            stage_overrides={"preprocess": preprocess},
        )
        empty = torch.empty(0)
        result = execute_typed_forward(
            plan,
            (
                torch.zeros(3),
                torch.zeros((point_count, 3)),
                torch.full((point_count, 3), 0.75),
                torch.ones((point_count, 1)),
                torch.ones((point_count, 3)),
                torch.zeros((point_count, 4)),
                1.0,
                empty,
                torch.eye(4),
                torch.eye(4),
                1.0,
                1.0,
                4,
                5,
                empty,
                0,
                torch.zeros(3),
                False,
            ),
        )

        self.assertEqual(events, ["preprocess", "binning", "render", "build_state"])
        self.assertIs(result.state, state_marker)
        self.assertEqual(result.num_rendered, 3)

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

    def test_warp_feature_detection_is_version_and_symbol_gated(self) -> None:
        fake_warp = SimpleNamespace(
            __version__="1.12.0+cuda",
            launch_tiled=object(),
            tile_sum=object(),
            grad=object(),
            div_approx=object(),
            inverse_approx=object(),
            Texture2D=object(),
            capture_if=object(),
            is_conditional_graph_supported=object(),
            compile_aot_module=object(),
            load_aot_module=object(),
        )
        features = detect_warp_features(fake_warp)

        self.assertEqual(parse_warp_version("1.11.2.dev4"), (1, 11, 2))
        self.assertEqual(features.version, (1, 12, 0))
        self.assertEqual(
            features.capabilities,
            frozenset(
                {
                    "stable_warp",
                    "tile_axis_reduce",
                    "inline_function_grad",
                    "approx_math",
                    "texture_sampling",
                    "conditional_graph",
                    "aot_module",
                }
            ),
        )

    def test_advanced_backend_enforces_version_and_capabilities(self) -> None:
        backend = object()
        module = SimpleNamespace(
            BACKEND_AVAILABLE=True,
            BACKEND=backend,
            MIN_WARP_VERSION=(1, 12, 0),
            REQUIRED_WARP_CAPABILITIES=frozenset({"approx_math"}),
        )

        resolved, reason = validate_advanced_backend_module(
            module, WarpFeatureSet((1, 11, 0), frozenset({"stable_warp"}))
        )
        self.assertIsNone(resolved)
        self.assertIn("requires warp-lang >= 1.12.0", reason)

        resolved, reason = validate_advanced_backend_module(
            module, WarpFeatureSet((1, 12, 0), frozenset({"stable_warp"}))
        )
        self.assertIsNone(resolved)
        self.assertIn("approx_math", reason)

        resolved, reason = validate_advanced_backend_module(
            module,
            WarpFeatureSet((1, 12, 0), frozenset({"stable_warp", "approx_math"})),
        )
        self.assertIs(resolved, backend)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
