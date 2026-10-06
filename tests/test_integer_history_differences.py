#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exact signed differences of integer histories.

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

from fmdtools.analyze.common import diff
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class CountState(State):
    count: np.array = np.array([0, 255], dtype=np.uint8)


class CountFunction(Function):
    container_s = CountState

    def dynamic_behavior(self):
        self.s.count = self.s.count[::-1].copy()

    def classify(self, **kwargs):
        return {"count": self.s.count.copy()}


def exact_difference(first, second):
    left, right = np.broadcast_arrays(first, second)
    return np.array(
        [int(a) - int(b) for a, b in zip(left.flat, right.flat)], dtype=object
    ).reshape(left.shape)


class TestIntegerHistoryDifferences(unittest.TestCase):
    def test_signed_unsigned_extremes_and_tolerances_match_python_integers(self):
        for dtype in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            limits = np.iinfo(dtype)
            left = np.array([limits.min, limits.max, 0, 1], dtype=dtype)
            right = left[::-1].copy()
            for shape in ((4,), (2, 2), (1, 2, 2)):
                first, second = left.reshape(shape), right.reshape(shape)
                expected = exact_difference(first, second)
                with self.subTest(dtype=dtype, shape=shape):
                    np.testing.assert_array_equal(diff(first, second, "diff"), expected)
                    np.testing.assert_array_equal(diff(first, second), first != second)
                    for threshold in (0.5, 2.0, 128.0, float(2**63)):
                        np.testing.assert_array_equal(
                            diff(first, second, threshold), np.abs(expected) > threshold
                        )
                    np.testing.assert_array_equal(first, left.reshape(shape))
                    np.testing.assert_array_equal(second, right.reshape(shape))

    def test_all_eight_bit_pairs_keep_their_exact_difference_and_threshold(self):
        for dtype in (np.int8, np.uint8):
            limits = np.iinfo(dtype)
            values = np.arange(limits.min, limits.max + 1, dtype=np.int64)
            left = values.astype(dtype)[:, None]
            right = values.astype(dtype)[None, :]
            expected = values[:, None] - values[None, :]
            with self.subTest(dtype=dtype):
                np.testing.assert_array_equal(diff(left, right, "diff"), expected)
                for threshold in (0.5, 2.0, 127.5, 254.5):
                    np.testing.assert_array_equal(
                        diff(left, right, threshold), np.abs(expected) > threshold
                    )

    def test_extreme_difference_is_not_rounded_through_float(self):
        pairs = [
            (np.array([2**53 + 1], dtype=np.int64), np.array([2**53], dtype=np.int64)),
            (np.array([-(2**63)], dtype=np.int64), np.array([0], dtype=np.int64)),
            (
                np.array([-(2**63)], dtype=np.int64),
                np.array([2**63 - 1], dtype=np.int64),
            ),
            (
                np.array([0, 2**64 - 1], dtype=np.uint64),
                np.array([2**64 - 1, 0], dtype=np.uint64),
            ),
            (np.array([-1], dtype=np.int64), np.array([2**64 - 1], dtype=np.uint64)),
        ]
        for first, second in pairs:
            with self.subTest(first=first.tolist(), second=second.tolist()):
                expected = exact_difference(first, second)
                np.testing.assert_array_equal(diff(first, second, "diff"), expected)
                np.testing.assert_array_equal(
                    diff(first, second, 0.5), np.abs(expected) > 0.5
                )

    def test_broadcasting_empty_views_and_scalar_results(self):
        first = np.arange(12, dtype=np.uint8).reshape(3, 4)[:, ::2]
        second = np.array([255, 1], dtype=np.uint8)
        first.setflags(write=False)
        second.setflags(write=False)
        before = first.copy(), second.copy()
        np.testing.assert_array_equal(
            diff(first, second, "diff"), exact_difference(first, second)
        )
        for actual, saved in zip((first, second), before):
            np.testing.assert_array_equal(actual, saved)
        for shape in ((0,), (2, 0), ()):
            left = np.zeros(shape, dtype=np.uint64)
            right = np.ones(shape, dtype=np.uint64)
            self.assertEqual(np.shape(diff(left, right, "diff")), shape)
            np.testing.assert_array_equal(
                diff(left, right, "diff"), exact_difference(left, right)
            )
        for first, second in (
            (3, 5),
            (np.uint8(0), np.uint8(255)),
            (np.int64(-(2**63)), np.int64(1)),
        ):
            self.assertEqual(diff(first, second, "diff"), int(first) - int(second))
            self.assertEqual(np.ndim(diff(first, second, "diff")), 0)

    def test_float_boolean_and_error_behavior_is_unchanged(self):
        for dtype in (np.float32, np.float64):
            left = np.array([1.0, np.nan, np.inf], dtype=dtype)
            right = np.zeros(3, dtype=dtype)
            actual = diff(left, right, "diff")
            np.testing.assert_array_equal(actual, left - right)
            self.assertEqual(actual.dtype, np.dtype(dtype))
        np.testing.assert_array_equal(
            diff([True, False], [False, False]), [True, False]
        )
        np.testing.assert_array_equal(
            diff(np.array(["a", "b"]), np.array(["b", "b"])), [True, False]
        )
        with self.assertRaisesRegex(Exception, "Unable to diff"):
            diff(np.ones(2, dtype=np.uint8), np.ones(3, dtype=np.uint8), "diff")

    def test_real_simulation_histories_do_not_hide_large_integer_deviations(self):
        _, nominal = Simulation(mdl=CountFunction(sp={"end_time": 3.0}))()
        _, faulty = Simulation(
            mdl=CountFunction(
                sp={"end_time": 3.0}, s={"count": np.array([255, 0], dtype=np.uint8)}
            )
        )()
        before = nominal.copy(), faulty.copy()
        expected = exact_difference(nominal["s.count"], faulty["s.count"])
        numeric = faulty.get_degraded_hist(
            "s.count", nomhist=nominal, difftype="diff", operator=np.sum
        )
        np.testing.assert_array_equal(numeric["s.count"], expected)
        for threshold in (2.0, 254.0, 255.0):
            actual = faulty.get_degraded_hist(
                "s.count", nomhist=nominal, difftype=threshold
            )
            np.testing.assert_array_equal(
                actual["s.count"], np.abs(expected) > threshold
            )
            np.testing.assert_array_equal(actual.time, nominal.time)
        for actual, saved in zip((nominal, faulty), before):
            self.assertEqual(actual, saved)


if __name__ == "__main__":
    unittest.main()
