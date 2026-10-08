#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for binomial probability tracking with truncated trial counts.

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

import math
import unittest

import numpy as np
from scipy.stats import nbinom

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation

TRIALS = [3.7, 5.2]
SUCCESS = [0.25, 0.6]


class BinomialState(State):
    count: np.array = np.zeros((2, 2), dtype=np.int64)
    count_update = ("binomial", (TRIALS, SUCCESS, (2, 2)))


class BinomialRand(Rand):
    s: BinomialState = BinomialState()


class TotalState(State):
    total: np.int64 = 0


class BinomialFunction(Function):
    container_r = BinomialRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.count.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


def exact_mass(values, n, p):
    values, trials, chances = np.broadcast_arrays(
        values, np.asarray(n, dtype=np.int64), p
    )
    factors = []
    for value, count, chance in zip(values.flat, trials.flat, chances.flat):
        value, count = int(value), int(count)
        factors.append(
            math.comb(count, value) * chance**value * (1 - chance) ** (count - value)
            if 0 <= value <= count
            else 0.0
        )
    return math.prod(factors)


class TestBinomialTrialCounts(unittest.TestCase):
    def test_float_trial_counts_match_truncated_binomial_formula(self):
        for n in (
            0.0,
            0.9,
            -0.7,
            1.9,
            3.7,
            np.float32(5.8),
            np.float64(8.2),
            3,
            np.int64(5),
        ):
            for p in (0.0, 0.2, 0.5, 1.0):
                with self.subTest(n=n, p=p):
                    mass = get_pfunc_for_dist("binomial", n, p)
                    probabilities = [mass(k) for k in range(int(n) + 1)]
                    np.testing.assert_allclose(
                        probabilities,
                        [exact_mass(k, n, p) for k in range(int(n) + 1)],
                        rtol=1e-13,
                        atol=0,
                    )
                    self.assertAlmostEqual(sum(probabilities), 1.0)
                    self.assertEqual(mass(int(n) + 1), 0.0)
                    self.assertEqual(mass(-1), 0.0)
                    self.assertEqual(mass(0.5), 0.0)

    def test_list_tuple_and_integral_array_parameters_broadcast_with_size(self):
        parameters = [
            ([3.7, 5.2], [0.2, 0.7]),
            ((3.7, 5.2), (0.2, 0.7)),
            ([[1.9], [4.7]], [0.2, 0.5, 0.8]),
            (np.array([3, 5], dtype=np.int32), np.array([0.2, 0.7])),
        ]
        for n, p in parameters:
            shape = np.broadcast_shapes(np.shape(n), np.shape(p))
            for size in (None, (2, *shape), (0, *shape)):
                with self.subTest(shape=shape, size=size, type=type(n).__name__):
                    before = np.copy(n), np.copy(p)
                    values = np.random.default_rng(11).binomial(n, p, size)
                    actual = get_prob_for_rand(values, "binomial", n, p, size)
                    np.testing.assert_allclose(
                        actual, exact_mass(values, n, p), rtol=1e-13, atol=0
                    )
                    self.assertEqual(np.ndim(actual), 0)
                    for got, saved in zip((n, p), before):
                        np.testing.assert_array_equal(got, saved)

    def test_varargs_and_other_distributions_keep_their_existing_parameters(self):
        mass = get_pfunc_for_dist("binomial", 3.7, 0.5, 3)
        np.testing.assert_allclose(
            mass(0, 1, 2), exact_mass([0, 1, 2], 3.7, 0.5), rtol=1e-13
        )
        np.testing.assert_allclose(
            get_prob_for_rand([1, 2], "negative_binomial", 2.5, 0.4),
            np.prod(nbinom.pmf([1, 2], 2.5, 0.4)),
            rtol=1e-13,
        )
        for params in ((), (3.7,), (3.7, 0.5, 1, "extra")):
            with self.assertRaises(TypeError):
                get_pfunc_for_dist("binomial", *params)

    def test_tracked_draws_preserve_rng_state_copy_and_reset(self):
        rand = BinomialRand(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        masses = []
        for _ in range(3):
            rand.set_rand_state("count", "binomial", TRIALS, SUCCESS, (2, 2))
            expected = reference.binomial(TRIALS, SUCCESS, (2, 2))
            np.testing.assert_array_equal(rand.s.count, expected)
            probability = exact_mass(expected, TRIALS, SUCCESS)
            masses.append(probability)
            np.testing.assert_allclose(rand.probs[-1], probability, rtol=1e-13, atol=0)
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        np.testing.assert_allclose(
            rand.return_probdens(), math.prod(masses), rtol=1e-13, atol=0
        )
        clone = rand.copy()
        for item in (rand, clone):
            item.set_rand_state("count", "binomial", TRIALS, SUCCESS, (2, 2))
        np.testing.assert_array_equal(rand.s.count, clone.s.count)
        rand.reset()
        rand.set_rand_state("count", "binomial", TRIALS, SUCCESS, (2, 2))
        np.testing.assert_array_equal(
            rand.s.count, np.random.default_rng(17).binomial(TRIALS, SUCCESS, (2, 2))
        )

    def test_actual_simulation_records_normalized_probabilities_and_unchanged_counts(
        self,
    ):
        model = BinomialFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        expected = np.array(
            [rng.binomial(TRIALS, SUCCESS, (2, 2)) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.count"], expected)
        np.testing.assert_allclose(
            history["r.probdens"],
            [exact_mass(x, TRIALS, SUCCESS) for x in expected],
            rtol=1e-13,
            atol=0,
        )
        np.testing.assert_array_equal(
            history["s.total"], np.r_[0, np.cumsum(expected[1:].sum(axis=(1, 2)))]
        )
        self.assertEqual(result["tend.classify.total"], history["s.total"][-1])
        np.testing.assert_array_equal(model.r.s.count, np.zeros((2, 2), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
