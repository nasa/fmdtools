#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for weighted percentage metrics.

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

from fmdtools.analyze.common import calc_metric, calc_percent
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result


class TestWeightedPercentMetrics(unittest.TestCase):
    """Percent metrics must use the same weighted averaging as NumPy."""

    def test_weights_change_the_nonzero_fraction(self):
        for data, weights in (
            ([0, 1], [9, 1]),
            ([0, -3, 2], [0, 1, 3]),
            ([False, True, True], [8, 1, 1]),
            ([0, 0], [1, 2]),
            ([1, 2], [1, 2]),
        ):
            with self.subTest(data=data, weights=weights):
                expected = np.average(np.asarray(data, dtype=bool), weights=weights)
                self.assertAlmostEqual(
                    calc_percent(data, weights=weights, round_value=False), expected
                )
                self.assertAlmostEqual(
                    calc_metric(data, "percent", weights=weights, round_value=False),
                    expected,
                )

    def test_axis_and_same_shape_weights(self):
        values = np.array([[0, 1, 2], [3, 0, 1]])
        for axis, weights in (
            (0, [3, 1]),
            (1, [1, 2, 4]),
            (None, [[1, 2, 3], [6, 5, 4]]),
        ):
            with self.subTest(axis=axis):
                original = values.copy()
                expected = np.average(values != 0, axis=axis, weights=weights)
                actual = calc_metric(values, "percent", weights=weights, axis=axis)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_array_equal(values, original)

    def test_named_result_weights_are_used_after_alignment(self):
        flat = Result({"a.outcome": 0, "b.outcome": 1, "b.weight": 1, "a.weight": 9})
        for result in (flat, flat.nest()):
            with self.subTest(nested=result is not flat):
                for weights in ("weight", {"b": 1, "a": 9}, [9, 1]):
                    with self.subTest(weights=weights):
                        self.assertAlmostEqual(
                            result.get_metric("outcome", "percent", weights=weights),
                            0.1,
                        )

    def test_history_weights_preserve_time_axis(self):
        history = History(
            {
                "a.outcome": np.array([0, 1, 1]),
                "b.outcome": np.array([1, 0, 1]),
                "a.weight": 9,
                "b.weight": 1,
            }
        )
        np.testing.assert_allclose(
            history.get_metric("outcome", "percent", weights="weight", axis=0),
            [0.1, 0.9, 1.0],
        )

    def test_invalid_weights_are_not_silently_ignored(self):
        with self.assertRaises(ZeroDivisionError):
            calc_percent([0, 1], weights=[0, 0])
        with self.assertRaises((TypeError, ValueError)):
            calc_percent([0, 1], weights=[1, 2, 3])

    def test_unweighted_and_rounding_behavior_is_unchanged(self):
        self.assertEqual(calc_percent([0, 1, 1]), 0.666667)
        self.assertEqual(calc_metric([0, 1], "percent", weights=[1, 2], res=0.01), 0.67)
        self.assertAlmostEqual(
            calc_percent([0, 1], weights=[1, 2], round_value=False), 2 / 3
        )
        # Rates remain distinct from weights in percentage calculations.
        self.assertEqual(calc_percent([0, 1], rates=[9, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()
