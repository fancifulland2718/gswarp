"""Registration and contract checks for classic 3DGS variant providers."""

from __future__ import annotations

import os
import unittest

from gswarp._internal.backends.select import clear_method_plan_cache, resolve_backend
from gswarp._internal.methods.contracts import (
    compatibility_3dgs_adapter,
    generated_3dgs_adapter,
    mip_3dgs_adapter,
    resolve_input_adapter,
    twodgs_adapter,
)
from gswarp._internal.methods.registry import METHODS


class ClassicVariantMethodContractTests(unittest.TestCase):
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

    def test_variant_plans_bind_their_declared_input_adapters(self) -> None:
        expected = {
            "baseline_3dgs": compatibility_3dgs_adapter,
            "flow_aux": compatibility_3dgs_adapter,
            "generated_3dgs": generated_3dgs_adapter,
            "mip_3dgs": mip_3dgs_adapter,
            "twodgs": twodgs_adapter,
        }
        for name, adapter in expected.items():
            with self.subTest(name=name):
                plan = resolve_backend(name)
                self.assertIs(plan.input_adapter, adapter)
                self.assertIs(plan.backend, resolve_backend(name).backend)

    def test_native_variants_keep_separate_backend_and_output_contracts(self) -> None:
        baseline = resolve_backend("baseline_3dgs")
        mip = resolve_backend("mip_3dgs")
        twod = resolve_backend("twodgs")

        self.assertIsNot(mip.backend, baseline.backend)
        self.assertIsNot(twod.backend, baseline.backend)
        self.assertEqual(mip.spec.filtering, "mip")
        self.assertEqual(twod.spec.output_mode, "twodgs")
        self.assertEqual(twod.spec.primitive, "gaussian_2d")

    def test_adapters_reject_incompatible_argument_layouts_before_execution(self) -> None:
        with self.assertRaisesRegex(TypeError, "18 arguments"):
            compatibility_3dgs_adapter(())
        with self.assertRaisesRegex(TypeError, "19 arguments"):
            mip_3dgs_adapter(())
        with self.assertRaisesRegex(TypeError, "21 arguments"):
            twodgs_adapter(())
        with self.assertRaisesRegex(RuntimeError, "Unknown gswarp method input adapter"):
            resolve_input_adapter("not_a_method")

    def test_registry_has_only_explicit_classic_variant_entries(self) -> None:
        self.assertEqual(
            set(METHODS),
            {"baseline_3dgs", "flow_aux", "generated_3dgs", "mip_3dgs", "twodgs"},
        )


if __name__ == "__main__":
    unittest.main()
