#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault summaries and model calculations.

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

import copy
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode
from fmdtools.define.container.state import ExampleState
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import Scenario, Sequence


class RootSystem(Function):
    arch_fa = ExFxnArch
    container_s = ExampleState
    container_m = Mode

    def dynamic_behavior(self):
        self.s.assign(self.fa.flows["exf"].s)


class TestFaultSummaryPaths(unittest.TestCase):
    def test_subfault_flags_are_found_at_root_and_nested_object_paths(self):
        for prefix in ("unit", "fxns.unit", "plant.fxns.unit"):
            original = History(
                {
                    prefix + ".m.sub_faults": np.array([False, True, False]),
                    "time": np.arange(3),
                }
            )
            for nested in (False, True):
                for attribute in dict.fromkeys(("unit", prefix)):
                    with self.subTest(
                        prefix=prefix, nested=nested, attribute=attribute
                    ):
                        history = original.nest() if nested else original
                        traces = history.get_faults_hist(attribute)
                        self.assertEqual(set(traces[attribute]), {"sub_faults"})
                        np.testing.assert_array_equal(
                            traces[attribute]["sub_faults"], [False, True, False]
                        )
                        result = history.get_faulty_hist(attribute)
                        np.testing.assert_array_equal(
                            result[attribute], [False, True, False]
                        )
                        np.testing.assert_array_equal(result["total"], [0, 1, 0])
                        np.testing.assert_array_equal(result.time, [0, 1, 2])

    def test_fault_like_container_names_do_not_count_as_faults(self):
        suffixes = (
            "m.faults_cached.low",
            "m.faults_label",
            "m.sub_faults_extra",
            "m.sub_faults.extra",
            "s.faults.low",
        )
        for prefix in ("unit", "fxns.unit"):
            for suffix in suffixes:
                for nested in (False, True):
                    with self.subTest(prefix=prefix, suffix=suffix, nested=nested):
                        history = History(
                            {
                                prefix + "." + suffix: np.array([0, 1, 1]),
                                "time": np.arange(3),
                            }
                        )
                        if nested:
                            history = history.nest()
                        self.assertEqual(
                            dict(history.get_faults_hist("unit")["unit"]), {}
                        )
                        np.testing.assert_array_equal(
                            history.get_faulty_hist("unit")["total"], [0, 0, 0]
                        )

    def test_fault_names_aggregate_with_subfaults_without_selecting_nearby_objects(
        self,
    ):
        history = History(
            {
                "unit.m.faults.low": np.array([False, True, False, False]),
                "unit.m.faults.high": np.array([False, False, True, False]),
                "unit.m.sub_faults": np.array([False, False, False, True]),
                "unit2.m.faults.low": np.array([False, False, False, True]),
                "myunit.m.faults.low": np.ones(4, dtype=bool),
                "unit.m.faults_cached.low": np.ones(4, dtype=bool),
                "time": np.arange(4),
            }
        )
        before = copy.deepcopy(history)
        self.assertEqual(
            set(history.get_faults_hist("unit")["unit"]), {"low", "high", "sub_faults"}
        )
        result = history.get_faulty_hist("unit", "unit2")
        np.testing.assert_array_equal(result.unit, [False, True, True, True])
        np.testing.assert_array_equal(result.unit2, [False, False, False, True])
        np.testing.assert_array_equal(result["total"], [0, 1, 1, 2])
        conjunction = history.get_faulty_hist(
            "unit", operator=np.all, withtime=False, withtotal=False
        )
        np.testing.assert_array_equal(conjunction.unit, [False] * 4)
        self.assertEqual(history, before)

    def test_legacy_direct_fault_traces_keep_their_existing_behavior(self):
        for prefix in ("unit", "fxns.unit"):
            history = History(
                {
                    prefix + ".m.faults": np.array([False, True, False]),
                    "time": np.arange(3),
                }
            )
            self.assertEqual(set(history.get_faults_hist("unit")["unit"]), {"faults"})
            np.testing.assert_array_equal(
                history.get_faulty_hist("unit").unit, [False, True, False]
            )

    def test_real_hierarchical_faults_are_visible_at_the_root_object(self):
        for onset in (1.0, 2.0):
            with self.subTest(onset=onset):
                model = RootSystem("plant", sp={"end_time": 3.0})
                scenario = Scenario(
                    name="child_fault",
                    times=(onset,),
                    sequence=Sequence(
                        faultseq={onset: {"plant.fa.fxns.ex_fxn": ["low"]}}
                    ),
                )
                _, history = Simulation(mdl=model, scen=scenario)()
                expected = history.time >= onset
                np.testing.assert_array_equal(history["m.sub_faults"], expected)
                named = History(
                    {
                        "plant." + key: value
                        for key, value in history.flatten().items()
                        if key != "time"
                    },
                    time=history.time,
                )
                for representation in (named, named.nest()):
                    summary = representation.get_faulty_hist("plant")
                    np.testing.assert_array_equal(summary.plant, expected)
                    np.testing.assert_array_equal(
                        summary["total"], expected.astype(int)
                    )
                self.assertFalse(model.m.sub_faults)


if __name__ == "__main__":
    unittest.main()
