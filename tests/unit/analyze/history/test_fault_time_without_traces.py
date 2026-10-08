#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault-time queries with no recorded fault traces.

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
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import Scenario, Sequence


class TestFaultTimeWithoutTraces(unittest.TestCase):
    def assert_fault_positions(self, history, expected):
        before = history.copy().flatten()
        expected = np.asarray(expected, dtype=int)
        for metric in ("earliest", "latest", "times", "total"):
            with self.subTest(metric=metric):
                actual = history.get_fault_time(metric)
                if metric == "total":
                    self.assertEqual(actual, len(expected))
                elif not expected.size:
                    self.assertTrue(np.isnan(actual))
                elif metric == "times":
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(np.issubdtype(actual.dtype, np.integer))
                else:
                    self.assertEqual(
                        actual, expected[0 if metric == "earliest" else -1]
                    )
        flat = history.flatten()
        self.assertEqual(set(flat), set(before))
        for key in flat:
            np.testing.assert_array_equal(flat[key], before[key])

    def test_empty_history_has_no_recorded_fault_positions(self):
        self.assert_fault_positions(History(), [])

    def test_time_only_histories_have_no_recorded_fault_positions(self):
        for times in ([], [0.0], [10.0, 20.0, 30.0]):
            for as_array in (False, True):
                with self.subTest(times=times, as_array=as_array):
                    self.assert_fault_positions(
                        History(time=np.asarray(times) if as_array else times), []
                    )

    def test_histories_containing_only_states_are_supported_flat_or_nested(self):
        history = History(
            {
                "plant.s.x": np.array([1.0, 2.0, 3.0]),
                "plant.s.label": np.array(["a", "b", "c"]),
                "time": np.array([5.0, 7.0, 11.0]),
            }
        )
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.assert_fault_positions(history.nest() if nested else history, [])

    def test_empty_fault_containers_are_equivalent_to_untracked_faults(self):
        history = History(
            {
                "plant": History({"m": History({"faults": History()})}),
                "time": [10.0, 20.0],
            }
        )
        self.assert_fault_positions(history, [])

    def test_recorded_false_and_empty_fault_traces_keep_existing_results(self):
        for values in ([], [False], [False, False, False]):
            with self.subTest(values=values):
                history = History({"m.faults.low": np.asarray(values, dtype=bool)})
                self.assert_fault_positions(history, [])

    def test_first_last_and_multiple_fault_indices_remain_indices_not_timestamps(self):
        for positions in ([0], [2], [0, 2], [0, 1, 2]):
            for dtype in (bool, np.int64):
                with self.subTest(positions=positions, dtype=dtype):
                    values = np.zeros(3, dtype=dtype)
                    values[positions] = 1
                    history = History(
                        {"m.faults.low": values, "time": [10.0, 20.0, 30.0]}
                    )
                    self.assert_fault_positions(history, positions)

    def test_multiple_fault_modes_count_each_faulty_timestep_once(self):
        history = History(
            {
                "first.m.faults.low": [True, True, False, False],
                "second.m.faults.high": [True, False, False, True],
                "time": [10.0, 20.0, 30.0, 40.0],
            }
        )
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.assert_fault_positions(
                    history.nest() if nested else history, [0, 1, 3]
                )

    def test_real_simulations_with_state_only_tracking_have_no_recorded_faults(self):
        model = ExampleFunction(sp={"end_time": 3.0}, track={"s": "x"})
        for sequence in (
            Sequence(),
            Sequence(faultseq={1.0: {"examplefunction": ["low"]}}),
        ):
            with self.subTest(sequence=sequence):
                _, history = Simulation(mdl=model, scen=Scenario(sequence=sequence))()
                self.assertEqual(set(history), {"s.x", "time"})
                self.assert_fault_positions(history, [])

    def test_real_simulation_with_fault_tracking_retains_fault_onset(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        for injection_time in (0.0, 1.0, 3.0):
            with self.subTest(injection_time=injection_time):
                sequence = Sequence(
                    faultseq={injection_time: {"examplefunction": ["low"]}}
                )
                _, history = Simulation(mdl=model, scen=Scenario(sequence=sequence))()
                expected = np.flatnonzero(history["m.faults.low"])
                self.assertGreater(len(expected), 0)
                self.assert_fault_positions(history, expected)


if __name__ == "__main__":
    unittest.main()
