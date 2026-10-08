#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for accepted hypergeometric count conversions.

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
import math
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def urn_mass(values, good, bad, sample):
    values, good, bad, sample = np.broadcast_arrays(values, good, bad, sample)
    masses = []
    for value, g, b, n in zip(values.flat, good.flat, bad.flat, sample.flat):
        g, b, n = int(g), int(b), int(n)
        x = int(value)
        mass = (
            math.comb(g, x) * math.comb(b, n - x) / math.comb(g + b, n)
            if 0 <= x <= g and 0 <= n - x <= b
            else 0.0
        )
        masses.append(mass)
    return math.prod(masses)


class CountState(State):
    successes: np.array = np.zeros((2, 2), dtype=int)
    successes_update = ("hypergeometric", (5.9, 6.2, 3.9, (2, 2)))


class CountRand(Rand):
    s: CountState = CountState()


class TotalState(State):
    total: np.float64 = 0.0


class CountFunction(Function):
    container_r = CountRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.successes.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestHypergeometricCountConversion(unittest.TestCase):
    def test_each_count_is_converted_before_probability_evaluation(self):
        for args in (
            (5.9, 6, 3),
            (5, 6.9, 3),
            (5, 6, 3.9),
            (5.9, 6.2, 3.9),
            (0.9, 6.9, 2.9),
            (5.2, 6.8, 0.9),
        ):
            with self.subTest(args=args):
                draws = np.random.default_rng(8).hypergeometric(*args, size=30)
                self.assertTrue(np.all(draws <= int(args[2])))
                masses = []
                for value in range(-1, int(args[2]) + 2):
                    mass = get_prob_for_rand(value, "hypergeometric", *args)
                    self.assertAlmostEqual(mass, urn_mass(value, *args), 14)
                    masses.append(mass)
                self.assertAlmostEqual(sum(masses), 1.0, 14)

    def test_list_and_integer_array_broadcasts_and_sizes_are_preserved(self):
        cases = [
            ([5.9, 3.2], [6.1, 9.2], [3.9, 1.8]),
            ([[5.9], [3.2]], [6.1, 9.2], 2.8),
            (
                np.array([100, 120], dtype=np.int8),
                np.array([100, 110], dtype=np.int8),
                [2, 3],
            ),
        ]
        for args in cases:
            shape = np.broadcast_shapes(*(np.shape(x) for x in args))
            for size in (None, (2, *shape), (0, *shape)):
                with self.subTest(args=args, size=size):
                    before = copy.deepcopy(args)
                    draws = np.random.default_rng(17).hypergeometric(*args, size=size)
                    expected = urn_mass(draws, *args)
                    actual = get_prob_for_rand(draws, "hypergeometric", *args, size)
                    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
                    for value, original in zip(args, before):
                        np.testing.assert_array_equal(value, original)

    def test_integer_counts_and_impossible_outcomes_keep_existing_behavior(self):
        for args in ((5, 6, 3), (0, 6, 0), (0, 6, 2), (5, 0, 2), (5, 6, 11)):
            for value in range(-1, args[2] + 2):
                with self.subTest(args=args, value=value):
                    actual = get_prob_for_rand(value, "hypergeometric", *args)
                    self.assertAlmostEqual(actual, urn_mass(value, *args), 14)
        self.assertEqual(get_prob_for_rand(1.5, "hypergeometric", 5.9, 6.2, 3.9), 0.0)

    def test_tracked_draws_and_generator_state_are_unchanged(self):
        rand = CountRand(seed=19, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(19)
        masses = []
        for _ in range(3):
            values = reference.hypergeometric(5.9, 6.2, 3.9, (2, 2))
            rand.set_rand_state("successes", "hypergeometric", 5.9, 6.2, 3.9, (2, 2))
            np.testing.assert_array_equal(rand.s.successes, values)
            masses.append(urn_mass(values, 5, 6, 3))
            np.testing.assert_allclose(rand.probs[-1], masses[-1], rtol=1e-13)
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        np.testing.assert_allclose(
            rand.return_probdens(), math.prod(masses), rtol=1e-13
        )
        clone = rand.copy()
        for item in (rand, clone):
            item.set_rand_state("successes", "hypergeometric", 5.9, 6.2, 3.9, (2, 2))
        np.testing.assert_array_equal(rand.s.successes, clone.s.successes)
        rand.reset()
        rand.set_rand_state("successes", "hypergeometric", 5.9, 6.2, 3.9, (2, 2))
        np.testing.assert_array_equal(
            rand.s.successes,
            np.random.default_rng(19).hypergeometric(5.9, 6.2, 3.9, (2, 2)),
        )

    def test_simulation_records_finite_masses_and_complete_histories(self):
        model = CountFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        values = np.array(
            [reference.hypergeometric(5.9, 6.2, 3.9, (2, 2)) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.successes"], values)
        np.testing.assert_allclose(
            history["r.probdens"], [urn_mass(x, 5, 6, 3) for x in values], rtol=1e-13
        )
        totals = np.r_[0.0, np.cumsum(values[1:].sum(axis=(1, 2)))]
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.successes, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
