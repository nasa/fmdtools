#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for integer probability bounds accepted by NumPy.

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
import math
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    Rand,
    calc_prob_for_integers,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class IntegerState(State):
    value: np.array = np.zeros((2, 2), dtype=np.int64)
    value_update = ("integers", (-2.9, 3.9, (2, 2)))


class IntegerRand(Rand):
    s: IntegerState = IntegerState()


class TotalState(State):
    total: np.float64 = 0.0


class IntegerFunction(Function):
    container_r = IntegerRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.value.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


def integer_mass(values, low, high, endpoint=False):
    if high is None:
        low, high = 0, low
    values, low, high = np.broadcast_arrays(values, low, high)
    return math.prod(
        1.0 / (int(b) - int(a) + int(endpoint))
        if int(a) <= int(x) < int(b) + int(endpoint)
        else 0.0
        for x, a, b in zip(values.flat, low.flat, high.flat)
    )


class TestIntegerBoundConversion(unittest.TestCase):
    def test_scalar_support_and_mass_follow_integer_bounds(self):
        for low, high in (
            (1.9, 4.2),
            (-2.9, 2.9),
            (-5.2, -1.8),
            (4.8, None),
            (0.9, 1.1),
        ):
            for endpoint in (False, True):
                with self.subTest(low=low, high=high, endpoint=endpoint):
                    lo, hi = (0, int(low)) if high is None else (int(low), int(high))
                    hi += int(endpoint)
                    draws = np.random.default_rng(12).integers(
                        low, high, size=40, endpoint=endpoint
                    )
                    self.assertTrue(np.all((lo <= draws) & (draws < hi)))
                    masses = []
                    for value in range(lo - 1, hi + 1):
                        expected = 1 / (hi - lo) if lo <= value < hi else 0.0
                        actual = get_prob_for_rand(
                            value, "integers", low, high, None, np.int64, endpoint
                        )
                        self.assertAlmostEqual(actual, expected, 14)
                        masses.append(actual)
                    self.assertAlmostEqual(sum(masses), 1.0, 14)

    def test_broadcast_and_sized_draws_keep_their_exact_joint_mass(self):
        cases = [([-2.9, 1.9], [3.8, 6.2]), ([[1.9], [-2.9]], [4.2, 5.9])]
        for (low, high), endpoint in itertools.product(cases, (False, True)):
            shape = np.broadcast_shapes(np.shape(low), np.shape(high))
            for size in (None, (2, *shape), (0, *shape)):
                with self.subTest(low=low, high=high, size=size, endpoint=endpoint):
                    originals = copy.deepcopy((low, high))
                    values = np.random.default_rng(17).integers(
                        low, high, size=size, endpoint=endpoint
                    )
                    expected = integer_mass(values, low, high, endpoint)
                    actual = get_prob_for_rand(
                        values, "integers", low, high, size, np.int64, endpoint
                    )
                    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
                    self.assertEqual((low, high), originals)

    def test_large_exact_integer_endpoints_and_invalid_support_remain_correct(self):
        for low, high, dtype, endpoint in (
            (2**64 - 4, 2**64, np.uint64, False),
            (2**64 - 4, 2**64 - 1, np.uint64, True),
            (-(2**63), -(2**63) + 4, np.int64, False),
        ):
            values = np.random.default_rng(8).integers(
                low, high, size=3, dtype=dtype, endpoint=endpoint
            )
            self.assertAlmostEqual(
                get_prob_for_rand(values, "integers", low, high, 3, dtype, endpoint),
                1 / 64.0,
            )
        for value in (1.5, np.nan, np.inf):
            self.assertEqual(calc_prob_for_integers(value, 0.0, 3.9), 0.0)
        for low, high in ((1.1, 1.9), (-1.9, -1.1)):
            with self.assertRaises(ValueError):
                np.random.default_rng(3).integers(low, high)
            with self.assertRaises(ValueError):
                calc_prob_for_integers(1, low, high)
        self.assertEqual(calc_prob_for_integers([], 0, 5), 1.0)

    def test_tracking_preserves_rng_copies_and_reset(self):
        rand = IntegerRand(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        for _ in range(3):
            rand.set_rand_state("value", "integers", -2.9, 3.9, (2, 2))
            np.testing.assert_array_equal(
                rand.s.value, reference.integers(-2.9, 3.9, (2, 2))
            )
            self.assertAlmostEqual(rand.probs[-1], 5.0**-4)
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        self.assertAlmostEqual(rand.return_probdens(), 5.0**-12, 18)
        clone = rand.copy()
        for item in (rand, clone):
            item.set_rand_state("value", "integers", -2.9, 3.9, (2, 2))
        np.testing.assert_array_equal(rand.s.value, clone.s.value)
        rand.reset()
        rand.set_rand_state("value", "integers", -2.9, 3.9, (2, 2))
        np.testing.assert_array_equal(
            rand.s.value, np.random.default_rng(17).integers(-2.9, 3.9, (2, 2))
        )

    def test_simulation_records_probability_and_draw_histories(self):
        model = IntegerFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        values = np.array([rng.integers(-2.9, 3.9, (2, 2)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.value"], values)
        np.testing.assert_allclose(history["r.probdens"], 5.0**-4)
        totals = np.r_[0.0, np.cumsum(values[1:].sum(axis=(1, 2)))]
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.value, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
