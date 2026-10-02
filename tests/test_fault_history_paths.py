#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for complete fault-path components in histories.

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

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import Scenario, Sequence


class FaultNamedState(State):
    x: np.float64 = 0.0
    y: np.float64 = 0.0
    defaults: np.float64 = 5.0
    faults_seen: np.int64 = 2
    sub_faults: np.int64 = 3


class FaultNamedFunction(ExampleFunction):
    container_s = FaultNamedState

    def dynamic_behavior(self):
        super().dynamic_behavior()
        self.s.defaults += 1.0


class TestFaultHistoryPaths(unittest.TestCase):
    def assert_positions(self, history, positions):
        before = copy.deepcopy(history)
        for metric in ("earliest", "latest", "times", "total"):
            actual = history.get_fault_time(metric)
            if metric == "total":
                self.assertEqual(actual, len(positions))
            elif not positions:
                self.assertTrue(np.isnan(actual))
            elif metric == "times":
                np.testing.assert_array_equal(actual, positions)
            else:
                self.assertEqual(actual, positions[0 if metric == "earliest" else -1])
        self.assertEqual(history, before)

    def test_similar_attribute_and_parent_names_are_logged_as_ordinary_values(self):
        for name in ("defaults", "faults_seen", "sub_faults", "myfaults"):
            for prefix in ("", "faults_controller."):
                with self.subTest(name=name, prefix=prefix):
                    state = SimpleNamespace(**{name: 7.5})
                    obj = SimpleNamespace(s=state)
                    if prefix:
                        obj = SimpleNamespace(faults_controller=obj)
                    key = prefix + "s." + name
                    history = History({key: np.zeros(2)})
                    history.log(obj, 0)
                    setattr(state, name, 9.5)
                    history.log(obj, 1)
                    np.testing.assert_array_equal(history[key], [7.5, 9.5])
                    self.assert_positions(history, [])

        state = SimpleNamespace(faults=SimpleNamespace(low=7.5))
        obj = SimpleNamespace(s=state)
        history = History({"s.faults.low": np.zeros(2)})
        history.log(obj, 0)
        state.faults.low = 9.5
        history.log(obj, 1)
        np.testing.assert_array_equal(history["s.faults.low"], [7.5, 9.5])
        self.assert_positions(history, [])

    def test_unrelated_trace_names_do_not_create_faults_flat_or_nested(self):
        history = History(
            {
                "plant.s.defaults": [8, 9, 10],
                "plant.s.faults_seen": [5, 6, 7],
                "plant.s.sub_faults": [3, 4, 5],
                "faults_controller.s.x": [9, 9, 9],
                "plant.s.faults_label": ["healthy", "healthy", "healthy"],
                "plant.s.faults.low": [1, 2, 3],
                "time": [10.0, 20.0, 30.0],
            }
        )
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.assert_positions(history.nest() if nested else history, [])

    def test_actual_fault_membership_and_mode_summary_traces_remain_supported(self):
        for key in (
            "m.faults.low",
            "fxns.unit.m.faults.low",
            "sub_faults",
            "m.sub_faults",
            "fxns.unit.m.sub_faults",
        ):
            for nested in (False, True):
                with self.subTest(key=key, nested=nested):
                    history = History(
                        {
                            key: [False, True, True, False],
                            "s.defaults": [9, 9, 9, 9],
                            "time": [0.0, 2.0, 4.0, 6.0],
                        }
                    )
                    self.assert_positions(history.nest() if nested else history, [1, 2])
        history = History(
            {
                "m.faults.low": [False, True, True, False],
                "m.faults.high": [False, False, True, True],
                "m.sub_faults": [False, False, True, False],
            }
        )
        self.assert_positions(history, [1, 2, 3])

    def test_fault_membership_logging_and_sub_fault_flags_are_unchanged(self):
        mode = SimpleNamespace(faults={"low"}, sub_faults=True)
        obj = SimpleNamespace(m=mode)
        history = History(
            {
                "m.faults.low": np.zeros(2, bool),
                "m.faults.high": np.zeros(2, bool),
                "m.sub_faults": np.zeros(2, bool),
            }
        )
        history.log(obj, 0)
        mode.faults = {"high"}
        mode.sub_faults = False
        history.log(obj, 1)
        np.testing.assert_array_equal(history["m.faults.low"], [True, False])
        np.testing.assert_array_equal(history["m.faults.high"], [False, True])
        np.testing.assert_array_equal(history["m.sub_faults"], [True, False])
        self.assert_positions(history, [0, 1])

    def test_real_simulations_keep_state_names_and_true_fault_onsets_separate(self):
        for fault_time in (None, 1.0, 3.0):
            with self.subTest(fault_time=fault_time):
                model = FaultNamedFunction("faults_controller", sp={"end_time": 3.0})
                sequence = (
                    Sequence()
                    if fault_time is None
                    else Sequence(faultseq={fault_time: {"faults_controller": ["low"]}})
                )
                scenario = Scenario(
                    sequence=sequence,
                    name="nominal" if fault_time is None else "fault_case",
                )
                _, history = Simulation(mdl=model, scen=scenario)()
                np.testing.assert_array_equal(
                    history["s.defaults"], [5.0, 6.0, 7.0, 8.0]
                )
                np.testing.assert_array_equal(history["s.faults_seen"], [2] * 4)
                np.testing.assert_array_equal(history["s.sub_faults"], [3] * 4)
                positions = (
                    [] if fault_time is None else list(range(int(fault_time), 4))
                )
                self.assert_positions(history, positions)
                self.assertEqual(model.s.defaults, 5.0)
                self.assertFalse(model.m.any_faults())


if __name__ == "__main__":
    unittest.main()
