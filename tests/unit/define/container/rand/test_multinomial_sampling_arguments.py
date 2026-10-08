#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for multinomial sampling arguments.

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
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def analytic_mass(values, trials, probabilities):
    """Multiply factorial-formula masses without calling scipy.stats."""
    values = np.asarray(values)
    probabilities = np.asarray(probabilities)
    shape = np.broadcast_shapes(
        values.shape[:-1], np.shape(trials), probabilities.shape[:-1]
    )
    rows = np.broadcast_to(values, shape + (values.shape[-1],))
    counts = np.broadcast_to(trials, shape)
    probs = np.broadcast_to(probabilities, shape + (values.shape[-1],))
    mass = 1.0
    for index in np.ndindex(shape):
        row = rows[index]
        probability = math.factorial(int(counts[index]))
        for count, p in zip(row, probs[index]):
            probability *= float(p) ** int(count) / math.factorial(int(count))
        mass *= probability
    return mass


class CountState(State):
    counts: np.array = np.zeros((2, 3), dtype=int)
    counts_update = ("multinomial", (5, [0.2, 0.3, 0.5], 2))


class CountRand(Rand):
    s: CountState = CountState()


class ScoreState(State):
    score: np.float64 = 0.0


class CountFunction(Function):
    container_r = CountRand
    container_s = ScoreState

    def dynamic_behavior(self):
        self.s.score += (self.r.s.counts @ np.array([1, 2, 4])).sum()

    def classify(self, **kwargs):
        return {"score": self.s.score}


class TestMultinomialSamplingArguments(unittest.TestCase):
    """Check whole-vector probabilities and retain NumPy sampling behavior."""

    def test_complete_vectors_and_batch_shapes_match_factorial_formula(self):
        for trials, probabilities in (
            (5, [0.2, 0.3, 0.5]),
            (0, [0.25, 0.75]),
            (3, [1.0]),
        ):
            for size in (None, (), 1, 3, (2, 3), (1, 2, 1), 0, (2, 0)):
                with self.subTest(
                    trials=trials, categories=len(probabilities), size=size
                ):
                    draws = np.random.default_rng(11).multinomial(
                        trials, probabilities, size
                    )
                    before = draws.copy()
                    expected = analytic_mass(draws, trials, probabilities)
                    for arguments in (
                        (trials, probabilities),
                        (trials, probabilities, size),
                    ):
                        actual = get_prob_for_rand(draws, "multinomial", *arguments)
                        self.assertAlmostEqual(actual, expected, places=13)
                        self.assertEqual(np.ndim(actual), 0)
                    np.testing.assert_array_equal(draws, before)

    def test_broadcast_trials_and_category_parameters_keep_the_last_axis(self):
        trials = np.array([2, 5])
        probabilities = np.array([[0.25, 0.75], [0.6, 0.4]])
        for size in (None, (2,), (3, 2)):
            with self.subTest(size=size):
                draws = np.random.default_rng(7).multinomial(
                    trials, probabilities, size
                )
                actual = get_prob_for_rand(
                    draws, "multinomial", trials, probabilities, size
                )
                self.assertAlmostEqual(
                    actual, analytic_mass(draws, trials, probabilities), 13
                )
        np.testing.assert_array_equal(trials, [2, 5])
        np.testing.assert_array_equal(probabilities, [[0.25, 0.75], [0.6, 0.4]])

    def test_enumerated_outcomes_normalize_and_invalid_counts_keep_zero_mass(self):
        probabilities = [0.25, 0.25, 0.5]
        pfunc = get_pfunc_for_dist("multinomial", 3, probabilities, None)
        total = 0.0
        for first in range(4):
            for second in range(4 - first):
                counts = [first, second, 3 - first - second]
                expected = analytic_mass(counts, 3, probabilities)
                self.assertAlmostEqual(pfunc(*counts), expected, 13)
                total += pfunc(counts)
        self.assertAlmostEqual(total, 1.0, 13)
        for counts in ((1, 1, 0), (-1, 2, 2), (0.5, 0.5, 2)):
            with self.subTest(counts=counts):
                self.assertEqual(pfunc(counts), 0.0)
        self.assertEqual(
            get_prob_for_rand([0, 3], "multinomial", 3, [0.0, 1.0], None), 1.0
        )
        with self.assertRaises(TypeError):
            get_pfunc_for_dist("multinomial", 3, probabilities, None, "extra")

    def test_tracked_updates_copies_and_reset_preserve_draws_and_generator_state(self):
        for size in (None, (2, 3), 0):
            with self.subTest(size=size):
                rand = CountRand(seed=17, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(17)
                masses = []
                for _ in range(2):
                    expected = reference.multinomial(5, [0.2, 0.3, 0.5], size)
                    rand.set_rand_state(
                        "counts", "multinomial", 5, [0.2, 0.3, 0.5], size
                    )
                    np.testing.assert_array_equal(rand.s.counts, expected)
                    self.assertEqual(
                        rand.rng.bit_generator.state, reference.bit_generator.state
                    )
                    masses.append(analytic_mass(expected, 5, [0.2, 0.3, 0.5]))
                np.testing.assert_allclose(rand.probs, masses)
                self.assertAlmostEqual(rand.return_probdens(), np.prod(masses), 13)
                clone = rand.copy()
                rand.set_rand_state("counts", "multinomial", 5, [0.2, 0.3, 0.5], size)
                clone.set_rand_state("counts", "multinomial", 5, [0.2, 0.3, 0.5], size)
                np.testing.assert_array_equal(rand.s.counts, clone.s.counts)
                rand.reset()
                self.assertEqual(rand.probs, [])
                rand.set_rand_state("counts", "multinomial", 5, [0.2, 0.3, 0.5], size)
                expected = np.random.default_rng(17).multinomial(
                    5, [0.2, 0.3, 0.5], size
                )
                np.testing.assert_array_equal(rand.s.counts, expected)
        inactive = CountRand(seed=3, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(inactive.rng.bit_generator.state)
        inactive.set_rand_state("counts", "multinomial", 5, [0.2, 0.3, 0.5], 2)
        self.assertEqual(inactive.rng.bit_generator.state, before)
        self.assertEqual(inactive.probs, [])

    def test_real_simulation_records_counts_joint_masses_and_weighted_totals(self):
        model = CountFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        counts = np.array(
            [reference.multinomial(5, [0.2, 0.3, 0.5], 2) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.counts"], counts)
        masses = [analytic_mass(value, 5, [0.2, 0.3, 0.5]) for value in counts]
        np.testing.assert_allclose(history["r.probdens"], masses, rtol=1e-13)
        totals = np.r_[0.0, np.cumsum((counts[1:] @ np.array([1, 2, 4])).sum(axis=1))]
        np.testing.assert_array_equal(history["s.score"], totals)
        self.assertEqual(result["tend.classify.score"], totals[-1])
        np.testing.assert_array_equal(model.r.s.counts, np.zeros((2, 3), dtype=int))


if __name__ == "__main__":
    unittest.main()
