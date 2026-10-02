#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for probabilities of scalar and array choice draws.

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
from collections import Counter
from itertools import permutations, product
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, calc_prob_for_choice, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class ChoiceState(State):
    noise: np.array = np.zeros((2, 2), dtype=np.int64)
    noise_update = ("choice", ([0, 1, 2], (2, 2), True, [0.25, 0.25, 0.5]))


class ChoiceRand(Rand):
    s: ChoiceState = ChoiceState()


class TotalState(State):
    total: np.int64 = 0


class ChoiceFunction(Function):
    container_r = ChoiceRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.noise.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total, "prob": self.r.return_probdens()}


class TestChoiceDrawMass(unittest.TestCase):
    def test_scalar_empty_and_multidimensional_draws_use_every_value(self):
        for options in (3, np.int64(3), [0, 1, 2], (0, 1, 2), np.arange(3)):
            for shape in (None, (), 1, 3, (2, 2), 0, (2, 0)):
                with self.subTest(options_type=type(options).__name__, shape=shape):
                    values = np.random.default_rng(7).choice(options, shape)
                    expected = (1.0 / 3) ** np.size(values)
                    before = np.asarray(values).copy()
                    for actual in (
                        calc_prob_for_choice(values, options, shape),
                        get_prob_for_rand(values, "choice", options, shape),
                    ):
                        self.assertAlmostEqual(actual, expected, places=6)
                    np.testing.assert_array_equal(values, before)

    def test_weighted_string_choices_and_duplicate_values_have_correct_joint_mass(self):
        options = np.array(["a", "b", "b"])
        probabilities = np.array([0.25, 0.25, 0.5])
        for values in ("b", ["a", "b"], [["a", "b"], ["b", "b"]], []):
            with self.subTest(values=values):
                expected = np.prod(
                    [0.25 if x == "a" else 0.75 for x in np.asarray(values).ravel()]
                )
                actual = get_prob_for_rand(
                    values, "choice", options, None, True, probabilities
                )
                self.assertAlmostEqual(actual, expected, places=6)
        np.testing.assert_array_equal(options, ["a", "b", "b"])
        np.testing.assert_array_equal(probabilities, [0.25, 0.25, 0.5])
        self.assertEqual(get_prob_for_rand(["c"], "choice", options), 0.0)

    def test_uniform_draws_without_replacement_match_enumerated_index_orders(self):
        for options in ([0, 1, 2, 3], [0, 0, 1, 2]):
            for count in (0, 1, 2, 3, 4):
                with self.subTest(options=options, count=count):
                    outcomes = Counter(
                        tuple(options[i] for i in order)
                        for order in permutations(range(len(options)), count)
                    )
                    denominator = sum(outcomes.values())
                    for outcome, multiplicity in outcomes.items():
                        values = (
                            np.array(outcome).reshape(2, 2) if count == 4 else outcome
                        )
                        self.assertAlmostEqual(
                            get_prob_for_rand(
                                values, "choice", np.array(options), None, False
                            ),
                            multiplicity / denominator,
                            places=6,
                        )
        self.assertEqual(get_prob_for_rand([0, 0], "choice", [0, 1], 2, False), 0.0)
        self.assertEqual(get_prob_for_rand([9], "choice", [0, 1], 1, False), 0.0)

    def test_with_replacement_masses_sum_to_one_for_each_ordered_outcome(self):
        for options, probabilities in (
            ([0, 1], None),
            ([0, 1], [0.25, 0.75]),
            ([0, 0, 1], [0.25, 0.25, 0.5]),
        ):
            with self.subTest(options=options, probabilities=probabilities):
                masses = [
                    get_prob_for_rand(
                        outcome, "choice", options, 3, True, probabilities
                    )
                    for outcome in product(set(options), repeat=3)
                ]
                self.assertAlmostEqual(sum(masses), 1.0)

    def test_unsupported_inputs_and_empty_draws_have_explicit_results(self):
        with self.assertRaises(Exception):
            calc_prob_for_choice([0], [0, 1], replace=False, p=[0.5, 0.5])
        with self.assertRaises(ValueError):
            calc_prob_for_choice([0, 1, 2], [0, 1], replace=False)
        with self.assertRaises(ValueError):
            calc_prob_for_choice([0], [[0, 1]])
        with self.assertRaises(ValueError):
            calc_prob_for_choice([0], [0, 1], p=[1.0])
        with self.assertRaises(ValueError):
            calc_prob_for_choice([0], [])
        self.assertEqual(calc_prob_for_choice([], []), 1.0)

    def test_tracked_updates_preserve_samples_rng_state_and_reset(self):
        for replace in (True, False):
            with self.subTest(replace=replace):
                state = ChoiceRand(seed=19, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(19)
                for _ in range(3):
                    expected = reference.choice(np.arange(4), (2, 2), replace)
                    state.set_rand_state(
                        "noise", "choice", np.arange(4), (2, 2), replace
                    )
                    np.testing.assert_array_equal(state.s.noise, expected)
                    self.assertAlmostEqual(
                        state.probs[-1], 1.0 / 256 if replace else 1.0 / 24, places=6
                    )
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                clone = state.copy()
                state.update_stochastic_states()
                clone.update_stochastic_states()
                np.testing.assert_array_equal(state.s.noise, clone.s.noise)
                self.assertEqual(state.probs, clone.probs)
                state.reset()
                state.update_stochastic_states()
                np.testing.assert_array_equal(
                    state.s.noise,
                    np.random.default_rng(19).choice(
                        [0, 1, 2], (2, 2), True, [0.25, 0.25, 0.5]
                    ),
                )
        disabled = ChoiceRand(seed=19, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)

    def test_actual_simulation_records_choice_draws_and_joint_probabilities(self):
        model = ChoiceFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        samples = np.array(
            [
                reference.choice([0, 1, 2], (2, 2), True, [0.25, 0.25, 0.5])
                for _ in history.time
            ]
        )
        np.testing.assert_array_equal(history["r.s.noise"], samples)
        masses = np.prod(np.where(samples == 2, 0.5, 0.25), axis=(1, 2))
        np.testing.assert_allclose(history["r.probdens"], masses, atol=5e-7, rtol=0)
        totals = np.r_[0, np.cumsum(samples[1:].sum(axis=(1, 2)))]
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        self.assertAlmostEqual(result["tend.classify.prob"], masses[-1], places=6)


if __name__ == "__main__":
    unittest.main()
