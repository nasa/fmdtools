#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for nonmutating history alignment during comparison.

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


def trace(times, offset=0.0, nested=False):
    times = np.asarray(times)
    hist = History(
        {
            "time": times,
            "s.x": times.astype(float) + offset,
            "s.vector": np.column_stack((times, times)).astype(float) + offset,
        }
    )
    return hist.nest() if nested else hist


class TestHistoryAlignmentOwnership(unittest.TestCase):
    def assert_unchanged(self, history, original):
        self.assertEqual(list(history.flatten()), list(original))
        for key, value in history.flatten().items():
            self.assertIs(value, original[key])
            np.testing.assert_array_equal(value, original[key])

    def test_direct_histories_retain_original_arrays_for_all_overlap_directions(self):
        for nominal_times, faulty_times in [
            (range(2, 5), range(7)),
            (range(7), range(2, 5)),
            (range(5), range(2, 7)),
            (range(2, 7), range(5)),
            (range(4), range(4)),
            ([2], range(5)),
        ]:
            with self.subTest(nominal=list(nominal_times), faulty=list(faulty_times)):
                nominal, faulty = trace(nominal_times), trace(faulty_times, offset=1)
                before_nom, before_fault = dict(nominal), dict(faulty)
                common = np.intersect1d(nominal.time, faulty.time)
                actual = faulty.get_degraded_hist("s.x", nomhist=nominal)
                np.testing.assert_array_equal(actual.time, common)
                np.testing.assert_array_equal(
                    actual["s.x"], np.ones(len(common), dtype=bool)
                )
                np.testing.assert_array_equal(
                    actual["total"], np.ones(len(common), dtype=int)
                )
                self.assert_unchanged(nominal, before_nom)
                self.assert_unchanged(faulty, before_fault)

    def test_repeated_comparisons_do_not_lose_later_deviations(self):
        faulty = trace(range(7))
        faulty["s.x"][-1] += 1
        original = dict(faulty)
        first = faulty.get_degraded_hist("s.x", nomhist=trace(range(2, 5)))
        np.testing.assert_array_equal(first["s.x"], [False] * 3)
        second = faulty.get_degraded_hist("s.x", nomhist=trace(range(7)))
        np.testing.assert_array_equal(second.time, np.arange(7))
        np.testing.assert_array_equal(second["s.x"], [False] * 6 + [True])
        self.assert_unchanged(faulty, original)

    def test_nested_readonly_and_list_histories_keep_input_structure(self):
        for use_lists in (False, True):
            with self.subTest(use_lists=use_lists):
                nominal, faulty = trace(range(1, 4)), trace(range(5), offset=1)
                for hist in (nominal, faulty):
                    for key, value in hist.items():
                        if use_lists and key != "time":
                            hist[key] = value.tolist()
                        else:
                            value.setflags(write=False)
                nominal, faulty = nominal.nest(), faulty.nest()
                original = dict(faulty.flatten())
                original_nom = dict(nominal.flatten())
                actual = faulty.get_degraded_hist("s.x", nomhist=nominal)
                np.testing.assert_array_equal(actual.time, [1, 2, 3])
                np.testing.assert_array_equal(actual["s.x"], [True] * 3)
                self.assertIsInstance(faulty.s, History)
                self.assert_unchanged(faulty, original)
                self.assert_unchanged(nominal, original_nom)

    def test_bundled_nominal_faulty_histories_keep_their_full_traces(self):
        original = History(
            {"nominal": trace(range(1, 4)), "faulty": trace(range(5), 1)}
        ).flatten()
        for nested in (False, True):
            history = original.nest() if nested else History(original)
            before = dict(history.flatten())
            actual = history.get_degraded_hist("s.x")
            np.testing.assert_array_equal(actual.time, [1, 2, 3])
            np.testing.assert_array_equal(actual["s.x"], [True] * 3)
            self.assert_unchanged(history, before)

    def test_real_simulation_histories_remain_reusable_after_short_comparison(self):
        _, nominal = Simulation(
            mdl=ExampleFunction(p={"x": 1.0}, sp={"end_time": 5.0})
        )()
        _, faulty = Simulation(
            mdl=ExampleFunction(p={"x": 2.0}, sp={"end_time": 5.0})
        )()
        short_nominal = nominal.cut(3, start_ind=1, newcopy=True)
        original = dict(faulty)
        first = faulty.get_degraded_hist("s.x", nomhist=short_nominal)
        np.testing.assert_array_equal(first.time, [1.0, 2.0, 3.0])
        second = faulty.get_degraded_hist("s.x", nomhist=nominal)
        np.testing.assert_array_equal(second.time, np.arange(6))
        np.testing.assert_array_equal(
            second["s.x"], [False, True, True, True, True, True]
        )
        self.assert_unchanged(faulty, original)
        np.testing.assert_array_equal(faulty["s.x"], 2 * np.arange(6))


if __name__ == "__main__":
    unittest.main()
