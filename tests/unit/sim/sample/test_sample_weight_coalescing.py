#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check shared accumulation of coincident sample weights.

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

from fmdtools.sim.sample import (
    _combine_sample_weights,
    sample_times_even,
    sample_times_quad,
)


class TestSampleWeightCoalescing(unittest.TestCase):
    def test_uniform_and_explicit_weights_keep_order_mass_and_inputs(self):
        for wrap in (list, tuple, np.asarray):
            for explicit in (False, True):
                with self.subTest(container=wrap.__name__, explicit=explicit):
                    times = wrap([4, 1, 4, 2, 1])
                    weights = wrap([0.125, 0.0, 0.25, 0.125, 0.5])
                    if isinstance(times, np.ndarray):
                        times.setflags(write=False)
                        weights.setflags(write=False)
                    before = np.array(times), np.array(weights)
                    selected, combined = _combine_sample_weights(
                        times, weights if explicit else None
                    )
                    self.assertEqual(selected, [4, 1, 2])
                    self.assertEqual(
                        combined, [0.375, 0.5, 0.125] if explicit else [0.4, 0.4, 0.2]
                    )
                    self.assertEqual(sum(combined), 1.0)
                    # Combining equal times preserves any time-dependent loss.
                    self.assertAlmostEqual(
                        sum(t * t * w for t, w in zip(selected, combined)),
                        7.0 if explicit else 7.6,
                    )
                    np.testing.assert_array_equal(times, before[0])
                    np.testing.assert_array_equal(weights, before[1])

    def test_empty_zero_and_unnormalized_weights_retain_their_meaning(self):
        with np.errstate(all="raise"):
            for weights in (None, [], np.array([])):
                self.assertEqual(_combine_sample_weights([], weights), ([], []))
            self.assertEqual(
                _combine_sample_weights([2, 2, 1], [1.0, -1.0, 0.0]),
                ([2, 1], [0.0, 0.0]),
            )
            self.assertEqual(
                _combine_sample_weights([2, 1, 2], [2.0, 0.0, 3.0]),
                ([2, 1], [5.0, 0.0]),
            )
        self.assertEqual(sample_times_even([], 1), ([], []))
        self.assertEqual(sample_times_quad([], [], []), ([], []))
        # Indexing must not silently drop a sample when a weight is missing.
        with self.assertRaises(IndexError):
            sample_times_quad([0, 1], [-1, 1], [1.0])

    def test_precision_scalar_types_and_distinct_times_are_preserved(self):
        for dtype in (float, np.float32, np.float64):
            weights = [dtype(0.125), dtype(0.25), dtype(0.375)]
            selected, combined = _combine_sample_weights([2, 1, 2], weights)
            self.assertEqual(selected, [2, 1])
            self.assertEqual(combined, [0.5, 0.25])
            self.assertTrue(all(type(weight) is dtype for weight in combined))
        for weights, expected in (
            ([1e16, 1.0, -1e16], 0.0),
            ([1e16, -1e16, 1.0], 1.0),
            ([1e-200, 2e-200], 3e-200),
        ):
            with self.subTest(weights=weights):
                _, combined = _combine_sample_weights([2] * len(weights), weights)
                np.testing.assert_allclose(combined, [expected], rtol=1e-15, atol=0)
        adjacent = np.nextafter(1.0, 2.0)
        self.assertEqual(
            _combine_sample_weights([1.0, adjacent, 1.0], [0.25, 0.5, 0.25]),
            ([1.0, adjacent], [0.5, 0.5]),
        )


if __name__ == "__main__":
    unittest.main()
