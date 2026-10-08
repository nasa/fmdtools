#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for numerical correctness.

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

import math
import unittest
from collections import Counter
from fractions import Fraction
from itertools import permutations, product

import numpy as np

from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, calc_prob_for_choice, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class ChoiceVector(State):
    values: np.array = np.zeros((2, 15), dtype=np.int64)
    values_update = ("choice", (2, (2, 15)))


class ChoiceRandomness(Rand):
    s: ChoiceVector = ChoiceVector()


class SumState(State):
    total: np.int64 = 0


class ChoiceModel(Function):
    container_r = ChoiceRandomness
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += self.r.s.values.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total, "prob": self.r.return_probdens()}


class TestChoiceProbabilityPrecision(unittest.TestCase):
    def test_uniform_joint_masses_are_not_rounded_to_zero(self):
        for size in (1, 20, 21, 30, 60, 100):
            for count in (2, 3, 7):
                with self.subTest(size=size, count=count):
                    values = np.arange(size) % count
                    expected = float(Fraction(1, count) ** size)
                    for actual in (
                        calc_prob_for_choice(values, count),
                        get_prob_for_rand(values, "choice", count, size),
                    ):
                        self.assertGreater(actual, 0.0)
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-14, atol=0.0
                        )

    def test_rare_weighted_and_repeated_labels_keep_their_mass(self):
        for rare in (1e-7, 1e-9, 1e-20, 1e-100):
            options = np.array(["rare", "rare", "common"])
            probabilities = np.array([rare / 4, rare * 3 / 4, 1 - rare])
            for size in (1, 2):
                with self.subTest(rare=rare, size=size):
                    actual = get_prob_for_rand(
                        ["rare"] * size, "choice", options, size, True, probabilities
                    )
                    self.assertGreater(actual, 0.0)
                    np.testing.assert_allclose(actual, rare**size, rtol=2e-14, atol=0.0)
            np.testing.assert_array_equal(options, ["rare", "rare", "common"])

    def test_enumerated_outcome_masses_sum_to_one(self):
        options = [0, 0, 1, 2]
        for replace in (False, True):
            for size in (1, 2, 3):
                orders = (
                    product(range(4), repeat=size)
                    if replace
                    else permutations(range(4), size)
                )
                counts = Counter(tuple(options[i] for i in order) for order in orders)
                total = sum(counts.values())
                masses = []
                for outcome, multiplicity in counts.items():
                    actual = get_prob_for_rand(
                        outcome, "choice", options, size, replace
                    )
                    np.testing.assert_allclose(
                        actual, multiplicity / total, rtol=1e-14, atol=0.0
                    )
                    masses.append(actual)
                self.assertAlmostEqual(math.fsum(masses), 1.0, places=14)
        masses = [get_prob_for_rand([i], "choice", 3) for i in range(3)]
        self.assertEqual(math.fsum(masses), 1.0)

    def test_ordered_draws_without_replacement_retain_small_masses(self):
        for count, size in ((10, 10), (12, 11), (20, 10)):
            with self.subTest(count=count, size=size):
                actual = get_prob_for_rand(
                    np.arange(size), "choice", count, size, False
                )
                expected = float(
                    Fraction(math.factorial(count - size), math.factorial(count))
                )
                self.assertGreater(actual, 0.0)
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
        self.assertEqual(get_prob_for_rand([0, 0], "choice", 2, 2, False), 0.0)
        self.assertEqual(get_prob_for_rand([], "choice", []), 1.0)
        self.assertEqual(get_prob_for_rand([9], "choice", 2), 0.0)

    def test_tracked_samples_copy_and_reset_preserve_the_seeded_sequence(self):
        rand = ChoiceRandomness(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        for step in range(3):
            rand.set_rand_state("values", "choice", 2, (2, 15))
            np.testing.assert_array_equal(rand.s.values, reference.choice(2, (2, 15)))
            self.assertEqual(rand.probs[-1], 2.0**-30)
            self.assertEqual(rand.return_probdens(), 2.0 ** (-30 * (step + 1)))
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        clone = rand.copy()
        for state in (rand, clone):
            state.update_stochastic_states()
        np.testing.assert_array_equal(rand.s.values, clone.s.values)
        self.assertEqual(rand.return_probdens(), 2.0**-30)
        rand.reset()
        rand.update_stochastic_states()
        np.testing.assert_array_equal(
            rand.s.values, np.random.default_rng(17).choice(2, (2, 15))
        )
        self.assertEqual(rand.return_probdens(), 2.0**-30)

    def test_simulation_history_and_risk_weight_keep_nonzero_probabilities(self):
        model = ChoiceModel(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(19)
        values = np.array([rng.choice(2, (2, 15)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.values"], values)
        np.testing.assert_array_equal(history["r.probdens"], 2.0**-30)
        expected = np.r_[0, np.cumsum(values[1:].sum(axis=(1, 2)))]
        np.testing.assert_array_equal(history["s.total"], expected)
        self.assertEqual(result["tend.classify.prob"], 2.0**-30)
        weighted = Result(
            {"scenario.cost": 2.0**30, "scenario.prob": result["tend.classify.prob"]}
        )
        self.assertEqual(
            weighted.get_metric(
                "cost", method="expected", rates="prob", round_value=False
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
