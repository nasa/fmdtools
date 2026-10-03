#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for random-distribution parameter preparation and probability tracking.

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

import copy
import math
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def hypergeometric_mass(values, good, bad, draws):
    """Evaluate joint mass using exact integer binomial coefficients."""
    arrays = np.broadcast_arrays(values, good, bad, draws)
    result = 1.0
    for value, g, b, n in zip(*(a.flat for a in arrays)):
        value, g, b, n = int(value), int(g), int(b), int(n)
        if not 0 <= value <= g or not 0 <= n - value <= b:
            return 0.0
        result *= math.comb(g, value) * math.comb(b, n - value) / math.comb(g + b, n)
    return result


class HypergeometricState(State):
    counts: np.array = np.zeros(2, dtype=np.int64)
    counts_update = ("hypergeometric", ([4, 7], [6, 3], [3, 4]))


class HypergeometricRand(Rand):
    s: HypergeometricState = HypergeometricState()


class CountTotal(State):
    total: np.float64 = 0.0


class HypergeometricFunction(Function):
    container_r = HypergeometricRand
    container_s = CountTotal

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.counts)

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestHypergeometricParameters(unittest.TestCase):
    def test_arraylike_counts_and_batch_shapes_match_exact_mass(self):
        for wrap in (list, tuple, np.asarray):
            for size in (None, (2,), (3, 2), (2, 1, 2), (0, 2)):
                with self.subTest(container=wrap.__name__, size=size):
                    parameters = tuple(wrap(x) for x in ([4, 7], [6, 3], [3, 4]))
                    before = copy.deepcopy(parameters)
                    values = np.random.default_rng(19).hypergeometric(*parameters, size)
                    expected = hypergeometric_mass(values, *parameters)
                    for args in (parameters, (*parameters, size)):
                        actual = get_prob_for_rand(values, "hypergeometric", *args)
                        self.assertAlmostEqual(actual, expected, places=13)
                        self.assertEqual(np.ndim(actual), 0)
                    for value, previous in zip(parameters, before):
                        np.testing.assert_array_equal(value, previous)

    def test_population_sum_does_not_overflow_narrow_integer_counts(self):
        for dtype, counts in (
            (np.int8, [100, 110]),
            (np.uint8, [200, 240]),
            (np.int16, [20000, 25000]),
        ):
            with self.subTest(dtype=dtype):
                good = np.array(counts, dtype=dtype)
                bad = good[::-1].copy()
                values = np.array([1, 2])
                self.assertAlmostEqual(
                    get_prob_for_rand(values, "hypergeometric", good, bad, 3),
                    hypergeometric_mass(values, good, bad, 3),
                    places=12,
                )

    def test_mixed_parameters_broadcast_and_scalar_distributions_normalize(self):
        for parameters in (
            ([2, 5], 8, [2, 3]),
            (5, [3, 7], 2),
            ([[2], [4]], [3, 5, 7], 2),
        ):
            with self.subTest(parameters=parameters):
                values = np.random.default_rng(5).hypergeometric(*parameters)
                self.assertAlmostEqual(
                    get_prob_for_rand(values, "hypergeometric", *parameters),
                    hypergeometric_mass(values, *parameters),
                    places=13,
                )
        for good, bad, draws in ((4, 6, 3), (0, 5, 2), (5, 0, 2), (4, 6, 0)):
            with self.subTest(good=good, bad=bad, draws=draws):
                total = sum(
                    get_prob_for_rand(k, "hypergeometric", good, bad, draws)
                    for k in range(draws + 1)
                )
                self.assertAlmostEqual(total, 1.0, places=13)
                self.assertEqual(
                    get_prob_for_rand(-1, "hypergeometric", good, bad, draws), 0
                )
                self.assertEqual(
                    get_prob_for_rand(draws + 1, "hypergeometric", good, bad, draws), 0
                )

    def test_tracked_updates_copies_and_reset_preserve_draws_and_rng_state(self):
        rand = HypergeometricRand(seed=11, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(11)
        parameters = ([4, 7], [6, 3], [3, 4])
        masses = []
        for _ in range(3):
            expected = reference.hypergeometric(*parameters)
            rand.set_rand_state("counts", "hypergeometric", *parameters)
            np.testing.assert_array_equal(rand.s.counts, expected)
            self.assertEqual(
                rand.rng.bit_generator.state, reference.bit_generator.state
            )
            masses.append(hypergeometric_mass(expected, *parameters))
        np.testing.assert_allclose(rand.probs, masses, rtol=1e-13)
        self.assertAlmostEqual(rand.return_probdens(), np.prod(masses), places=13)
        clone = rand.copy()
        for instance in (rand, clone):
            instance.set_rand_state("counts", "hypergeometric", *parameters)
        np.testing.assert_array_equal(rand.s.counts, clone.s.counts)
        rand.reset()
        self.assertEqual(rand.probs, [])
        rand.set_rand_state("counts", "hypergeometric", *parameters)
        np.testing.assert_array_equal(
            rand.s.counts, np.random.default_rng(11).hypergeometric(*parameters)
        )

    def test_real_simulation_records_correct_masses_without_changing_samples(self):
        model = HypergeometricFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        parameters = ([4, 7], [6, 3], [3, 4])
        expected = np.array(
            [reference.hypergeometric(*parameters) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.counts"], expected)
        np.testing.assert_allclose(
            history["r.probdens"],
            [hypergeometric_mass(v, *parameters) for v in expected],
            rtol=1e-13,
        )
        totals = np.r_[0.0, np.cumsum(expected[1:].sum(axis=1))]
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.counts, np.zeros(2, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
