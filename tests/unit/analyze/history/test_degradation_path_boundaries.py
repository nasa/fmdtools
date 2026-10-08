#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for degradation attribute path boundaries.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import SingleFaultScenario


def combined_history(selected, other):
    return History(
        {
            "nominal." + selected: np.zeros(3),
            "nominal." + other: np.zeros(3),
            "nominal.time": np.array([0.0, 1.0, 2.0]),
            "case1." + selected: np.zeros(3),
            "case1." + other: np.array([0.0, 1.0, 2.0]),
            "case1.time": np.array([0.0, 1.0, 2.0]),
        }
    )


class NamedState(State):
    x: float = 0.0
    xy: float = 0.0


class OffsetMode(Mode):
    fault_offset = Fault(prob=0.1, disturbances=(("s.xy", 100.0),))


class NamedFunction(Function):
    container_s = NamedState
    container_m = OffsetMode

    def dynamic_behavior(self):
        self.s.x += 1.0
        self.s.xy += 2.0

    def classify(self, **kwargs):
        return {"x": self.s.x, "xy": self.s.xy}


class TestDegradationPathBoundaries(unittest.TestCase):
    """Match selected state paths without including similarly named siblings."""

    def test_substring_collisions_do_not_mark_the_selected_attribute_degraded(self):
        cases = (
            ("pump", "pump.s.x", "pump2.s.x"),
            ("pump", "plant.pump.s.x", "plant.prepump.s.x"),
            ("x", "plant.s.x", "plant.s.xy"),
            ("s.x", "plant.s.x", "plant.s.xyz"),
            ("branch.pump", "site.branch.pump.s.x", "site.branch.pump2.s.x"),
            ("mode", "m.mode", "m.mode_index"),
        )
        for selector, selected, other in cases:
            for nested in (False, True):
                with self.subTest(selector=selector, other=other, nested=nested):
                    history = combined_history(selected, other)
                    if nested:
                        history = history.nest()
                    before = history.copy()
                    actual = history.get_degraded_hist(selector)
                    np.testing.assert_array_equal(
                        actual[selector], np.zeros(3, dtype=bool)
                    )
                    np.testing.assert_array_equal(
                        actual["total"], np.zeros(3, dtype=int)
                    )
                    np.testing.assert_array_equal(actual.time, [0.0, 1.0, 2.0])
                    self.assertEqual(history, before)

    def test_parent_and_full_paths_still_include_all_matching_descendants(self):
        history = combined_history("site.pump.s.a", "site.pump.s.b")
        history["case1.site.pump.s.a"] = np.array([1.0, 0.0, 0.0])
        for selector in ("pump", "site.pump", "pump.s", "s"):
            with self.subTest(selector=selector):
                actual = history.get_degraded_hist(selector)
                np.testing.assert_array_equal(actual[selector], [True, True, True])
        actual = history.get_degraded_hist("site.pump.s.a", "site.pump.s.b")
        np.testing.assert_array_equal(actual["site.pump.s.a"], [True, False, False])
        np.testing.assert_array_equal(actual["site.pump.s.b"], [False, True, True])
        np.testing.assert_array_equal(actual["total"], [1, 1, 1])

    def test_operator_output_controls_and_default_selection_keep_existing_behavior(
        self,
    ):
        nominal = History({"s.x": np.array([1.0, 2.0, 3.0]), "time": np.arange(3)})
        faulty = History({"s.x": np.array([1.0, 4.0, 3.0]), "time": np.arange(3)})
        actual = faulty.get_degraded_hist(nomhist=nominal)
        np.testing.assert_array_equal(actual["s.x"], [False, True, False])
        for withtime in (False, True):
            for withtotal in (False, True):
                with self.subTest(withtime=withtime, withtotal=withtotal):
                    result = faulty.get_degraded_hist(
                        "s.x",
                        nomhist=nominal,
                        operator=np.all,
                        withtime=withtime,
                        withtotal=withtotal,
                    )
                    expected_keys = {"s.x"}
                    if withtime:
                        expected_keys.add("time")
                    if withtotal:
                        expected_keys.add("total")
                    self.assertEqual(set(result), expected_keys)
                    np.testing.assert_array_equal(result["s.x"], [False, True, False])

    def test_real_fault_simulation_does_not_attribute_xy_degradation_to_x(self):
        model = NamedFunction(sp={"end_time": 4.0})
        _, nominal = Simulation(mdl=model)()
        scenario = SingleFaultScenario.from_fault(
            (model.name, "offset"), 2.0, mdl=model
        )
        _, faulty = Simulation(mdl=model, scen=scenario)()
        np.testing.assert_array_equal(nominal["s.x"], faulty["s.x"])
        self.assertTrue(np.any(nominal["s.xy"] != faulty["s.xy"]))
        history = History({"nominal": nominal, "offset_case": faulty}).flatten()
        before = history.copy()
        degraded = history.get_degraded_hist("s.x", "s.xy")
        np.testing.assert_array_equal(degraded["s.x"], np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(
            degraded["s.xy"], [False, False, True, True, True]
        )
        np.testing.assert_array_equal(degraded["total"], [0, 0, 1, 1, 1])
        self.assertEqual(history, before)


if __name__ == "__main__":
    unittest.main()
