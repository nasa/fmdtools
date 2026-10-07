#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for logging only explicitly requested simulation times.

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
import itertools
import unittest

import numpy as np

from fmdtools.define.block.base import SimParam
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation


class TestExplicitHistoryTimes(unittest.TestCase):
    def test_unrequested_steps_are_skipped_and_selected_positions_are_resolved(self):
        for wrap in (list, tuple, np.asarray):
            for requested in (
                [0.0, 2.0, 4.0],
                [1.0, 3.0],
                [],
                [4.0],
                [4.0, 0.0, 2.0],
                [0.0, 2.0, 2.0, 4.0],
            ):
                with self.subTest(wrap=wrap.__name__, requested=requested):
                    times = wrap(requested)
                    before = np.asarray(times).copy()
                    params = SimParam(end_time=4.0, track_times=("times", times))
                    for index in range(5):
                        log, position = params.get_hist_ind(index, float(index))
                        self.assertEqual(log, float(index) in requested)
                        if log:
                            self.assertEqual(position, requested.index(float(index)))
                    np.testing.assert_array_equal(times, before)

    def test_explicit_recording_vectors_are_independent_writable_arrays(self):
        for wrap in (list, tuple, np.asarray):
            source = wrap([0.0, 2.0, 4.0])
            params = SimParam(track_times=("times", source))
            first = params.get_histrange()
            second = params.get_histrange()
            self.assertIsInstance(first, np.ndarray)
            self.assertTrue(first.flags.writeable)
            first[0] = 99.0
            np.testing.assert_array_equal(second, [0.0, 2.0, 4.0])
            np.testing.assert_array_equal(source, [0.0, 2.0, 4.0])

    def test_requested_times_still_respect_local_timestep_alignment(self):
        params = SimParam(dt=1.0, track_times=("times", [0.0, 0.5, 1.0, 1.5, 2.0]))
        for index in range(5):
            log, position = params.get_hist_ind(index, index * 0.5, local_dt=0.5)
            self.assertEqual(log, index % 2 == 0)
            if log:
                self.assertEqual(position, index)
        self.assertEqual(SimParam().get_hist_ind(2, 2.0), (True, 2))
        self.assertEqual(
            SimParam(track_times=("interval", 2)).get_hist_ind(4, 4.0), (True, 2)
        )
        self.assertEqual(SimParam().get_hist_ind(2, 2.0, 0.1), (False, 1))
        self.assertEqual(SimParam().get_hist_ind(10, 2.0, 0.1), (True, 1))

    def test_sparse_simulations_match_full_history_at_requested_times(self):
        for dt, wrap in itertools.product((0.25, 1.0), (list, tuple, np.asarray)):
            for indices in ([0, 2, 4], [1, 3], [4]):
                with self.subTest(dt=dt, wrap=wrap.__name__, indices=indices):
                    times = np.asarray(indices, dtype=float) * dt
                    reference = ExampleFunction(
                        sp={"end_time": 4 * dt, "dt": dt, "use_local": False}
                    )
                    expected_result, expected_history = Simulation(mdl=reference)()
                    supplied = wrap(times)
                    before = copy.deepcopy(supplied)
                    model = ExampleFunction(
                        sp={
                            "end_time": 4 * dt,
                            "dt": dt,
                            "use_local": False,
                            "track_times": ("times", supplied),
                        }
                    )
                    result, history = Simulation(mdl=model)()
                    self.assertEqual(result, expected_result)
                    self.assertIsInstance(history.time, np.ndarray)
                    np.testing.assert_array_equal(history.time, times)
                    for name in expected_history:
                        np.testing.assert_array_equal(
                            history[name], expected_history[name][indices]
                        )
                    np.testing.assert_array_equal(supplied, before)
                    self.assertEqual(model.s.x, 0.0)
                    self.assertFalse(model.m.any_faults())

    def test_empty_requested_times_do_not_prevent_simulation(self):
        reference = ExampleFunction(sp={"end_time": 3.0})
        expected, _ = Simulation(mdl=reference)()
        for times in ([], (), np.array([])):
            with self.subTest(times=type(times).__name__):
                model = ExampleFunction(
                    sp={"end_time": 3.0, "track_times": ("times", times)}
                )
                result, history = Simulation(mdl=model)()
                self.assertEqual(result, expected)
                self.assertEqual(len(history.time), 0)
                self.assertEqual(len(history["s.x"]), 0)

    def test_readonly_array_and_copy_are_independent_of_execution(self):
        times = np.array([0.0, 2.0, 4.0])
        times.setflags(write=False)
        model = ExampleFunction(sp={"end_time": 4.0, "track_times": ("times", times)})
        duplicate = model.copy()
        first, h1 = Simulation(mdl=model)()
        second, h2 = Simulation(mdl=duplicate)()
        self.assertEqual(first, second)
        self.assertEqual(h1, h2)
        np.testing.assert_array_equal(times, [0.0, 2.0, 4.0])
        self.assertFalse(times.flags.writeable)


if __name__ == "__main__":
    unittest.main()
