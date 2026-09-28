#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for time-local standard deviation in history aggregation.

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


class TestHistoryStandardDeviation(unittest.TestCase):
    """Keep variation across runs separate from variation over time."""

    def make_history(self, values, time_key="time", times=None):
        if times is None:
            times = np.arange(len(values[0]), dtype=float)
        data = {f"run{i}.signal": value for i, value in enumerate(values)}
        data[time_key] = times
        return History(data)

    def test_identical_varying_histories_have_zero_spread(self):
        for count in (1, 2, 5):
            for array_type in (list, np.array):
                with self.subTest(count=count, array_type=array_type.__name__):
                    values = [array_type([0.0, 10.0, -20.0]) for _ in range(count)]
                    result = self.make_history(values).get_mean_std_errhist("signal")
                    for key in ("stat", "low", "high"):
                        np.testing.assert_array_equal(result[key], values[0])

    def test_spread_is_calculated_at_each_time_sample(self):
        history = self.make_history([[1.0, 10.0, 100.0], [1.0, 14.0, 108.0]])
        result = history.get_mean_std_errhist("signal")
        np.testing.assert_array_equal(result.stat, [1.0, 12.0, 104.0])
        # Retain the existing mean +/- half a standard deviation convention.
        np.testing.assert_array_equal(result.low, [1.0, 11.0, 102.0])
        np.testing.assert_array_equal(result.high, [1.0, 13.0, 106.0])
        np.testing.assert_array_equal(result.time, [0.0, 1.0, 2.0])

    def test_shared_time_dependent_offset_does_not_change_spread(self):
        values = np.array([[0.0, 2.0, 0.0], [4.0, 6.0, 8.0]])
        offset = np.array([100.0, -200.0, 1000.0])
        baseline = self.make_history(values).get_mean_std_errhist("signal")
        shifted = self.make_history(values + offset).get_mean_std_errhist("signal")
        np.testing.assert_allclose(shifted.stat, baseline.stat + offset)
        np.testing.assert_allclose(
            shifted.high - shifted.low, baseline.high - baseline.low
        )

    def test_custom_time_and_input_history_are_preserved(self):
        history = self.make_history(
            [[0.0, 2.0], [0.0, 6.0]], time_key="elapsed", times=[3.0, 7.5]
        )
        before = history.copy()
        result = history.get_mean_std_errhist("signal", time="elapsed")
        self.assertIs(type(result), History)
        self.assertEqual(set(result), {"elapsed", "stat", "low", "high"})
        np.testing.assert_array_equal(result.elapsed, [3.0, 7.5])
        np.testing.assert_array_equal(result.high - result.low, [0.0, 2.0])
        for key in history:
            np.testing.assert_array_equal(history[key], before[key])

    def test_multiple_components_keep_their_own_spread(self):
        values = np.array([[[0.0, 10.0], [20.0, 30.0]], [[0.0, 14.0], [28.0, 30.0]]])
        result = self.make_history(values).get_mean_std_errhist("signal")
        np.testing.assert_array_equal(result.stat, [[0.0, 12.0], [24.0, 30.0]])
        np.testing.assert_array_equal(
            result.high - result.low, [[0.0, 2.0], [4.0, 0.0]]
        )
        self.assertEqual(result.stat.shape, (2, 2))

    def test_single_time_sample_retains_existing_values(self):
        result = self.make_history([[1.0], [5.0]]).get_mean_std_errhist("signal")
        np.testing.assert_array_equal(result.stat, [3.0])
        np.testing.assert_array_equal(result.low, [2.0])
        np.testing.assert_array_equal(result.high, [4.0])


if __name__ == "__main__":
    unittest.main()
