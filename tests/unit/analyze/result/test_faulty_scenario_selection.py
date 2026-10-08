#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for selecting a scenario named faulty.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “Fault Model Design tools - fmdtools version 2” software is licensed
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
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import SingleFaultScenario


class DerivedResult(Result):
    """Check that the selector retains the calling result class."""


class TestFaultyScenarioSelection(unittest.TestCase):
    def test_reserved_scenario_name_works_for_flat_and_nested_results(self):
        for cls in (Result, History, DerivedResult):
            for nested in (False, True):
                with self.subTest(cls=cls.__name__, nested=nested):
                    values = np.array([0.0, 4.0, 7.0])
                    data = cls(
                        {
                            "nominal.s.x": np.arange(3.0),
                            "faulty.s.x": values,
                            "nominal.time": np.arange(3.0),
                            "faulty.time": np.arange(3.0),
                        }
                    )
                    if nested:
                        data = data.nest()
                    before = data.copy()
                    expected = cls({"s.x": values, "time": np.arange(3.0)})
                    for actual in (data.get_faulty(), data.faulty, data.get("faulty")):
                        self.assertIs(type(actual), cls)
                        self.assertEqual(actual, expected)
                        self.assertIs(actual["s.x"], values)
                    np.testing.assert_array_equal(data.get("faulty.s.x"), values)
                    self.assertEqual(data, before)

    def test_multiple_fault_scenarios_keep_their_prefixes_and_order(self):
        for cls in (Result, History):
            for nested in (False, True):
                with self.subTest(cls=cls.__name__, nested=nested):
                    data = cls(
                        {
                            "faulty.cost": 4.0,
                            "nominal.cost": 0.0,
                            "second.cost": 8.0,
                            "faulty.time": np.arange(2),
                            "second.time": np.arange(2),
                            "nominal.time": np.arange(2),
                        }
                    )
                    if nested:
                        data = data.nest()
                    expected = [
                        "faulty.cost",
                        "faulty.time",
                        "second.cost",
                        "second.time",
                    ]
                    for actual in (data.get_faulty(), data.get("faulty")):
                        self.assertEqual(list(actual), expected)
                        self.assertEqual(actual["faulty.cost"], 4.0)
                        self.assertEqual(actual["second.cost"], 8.0)
                    self.assertEqual(data.flatten()["nominal.cost"], 0.0)

    def test_nested_sample_prefixes_and_ordinary_fault_names_keep_existing_layout(self):
        for name in ("faulty", "case1"):
            for nested in (False, True):
                with self.subTest(name=name, nested=nested):
                    flat = Result(
                        {"sample.nominal.cost": 0.0, "sample." + name + ".cost": 5.0}
                    )
                    data = flat.nest() if nested else flat
                    self.assertEqual(data.get_faulty(), Result({"cost": 5.0}))
                    two = Result(
                        {
                            "left.nominal.cost": 0.0,
                            "left." + name + ".cost": 5.0,
                            "right.nominal.cost": 1.0,
                            "right." + name + ".cost": 9.0,
                        }
                    )
                    actual = (two.nest() if nested else two).get_faulty()
                    self.assertEqual(
                        list(actual),
                        ["left." + name + ".cost", "right." + name + ".cost"],
                    )
        self.assertEqual(Result({"nominal.cost": 0.0}).get_faulty(), Result())
        with self.assertRaises(KeyError):
            Result({"case1.cost": 5.0}).get_faulty()

    def test_actual_fault_results_support_selection_and_degradation_analysis(self):
        model = ExampleFunction(sp={"end_time": 4.0})
        nominal_result, nominal_history = Simulation(mdl=model)()
        scenario = SingleFaultScenario.from_fault((model.name, "low"), 2.0, mdl=model)
        fault_result, fault_history = Simulation(mdl=model, scen=scenario)()
        for cls, nominal, fault in (
            (Result, nominal_result, fault_result),
            (History, nominal_history, fault_history),
        ):
            with self.subTest(cls=cls.__name__):
                combined = cls({"nominal": nominal, "faulty": fault}).flatten()
                before = combined.copy()
                self.assertEqual(combined.get_faulty(), fault.flatten())
                self.assertEqual(combined, before)
        history = History(
            {"nominal": nominal_history, "faulty": fault_history}
        ).flatten()
        control = History(
            {"nominal": nominal_history, "case1": fault_history}
        ).flatten()
        self.assertEqual(
            history.get_degraded_hist("s.x"), control.get_degraded_hist("s.x")
        )
        np.testing.assert_array_equal(
            history.get_degraded_hist("s.x")["s.x"], [False, False, True, True, True]
        )


if __name__ == "__main__":
    unittest.main()
