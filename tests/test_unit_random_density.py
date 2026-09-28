#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for the joint density of unit-uniform random draws.

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
import unittest

import numpy as np
from scipy import stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    calc_prob_density_for_random,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class UnitUniformState(State):
    noise: np.array = np.zeros(3)
    noise_update = ("random", (3,))


class UnitUniformRand(Rand):
    s: UnitUniformState = UnitUniformState()


class AccumulatedState(State):
    total: np.float64 = 0.0


class UnitUniformFunction(Function):
    container_r = UnitUniformRand
    container_s = AccumulatedState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.noise)

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestUnitUniformJointDensity(unittest.TestCase):
    def test_joint_density_is_one_for_every_supported_draw_size(self):
        for shape in ((), (1,), (3,), (2, 3), (2, 1, 4), (0,), (2, 0, 3)):
            for dtype in (np.float32, np.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    values = np.random.default_rng(12).random(shape, dtype=dtype)
                    before = values.copy()
                    expected = np.prod(stats.uniform.pdf(values))
                    for density in (
                        calc_prob_density_for_random(values),
                        get_prob_for_rand(values, "random", shape),
                        get_pfunc_for_dist("random", shape)(values),
                    ):
                        self.assertIs(type(density), np.float64)
                        self.assertEqual(density, expected)
                        self.assertEqual(density, 1.0)
                    np.testing.assert_array_equal(values, before)

    def test_joint_density_does_not_depend_on_batching_or_shape(self):
        values = np.random.default_rng(6).random(12)
        separate = np.prod([get_prob_for_rand(x, "random") for x in values])
        self.assertEqual(get_pfunc_for_dist("random")(*values), separate)
        self.assertEqual(calc_prob_density_for_random(values), separate)
        for batched in (
            values,
            values.reshape(3, 4),
            values.reshape(2, 2, 3),
            values.tolist(),
            tuple(values),
            values[::-1],
        ):
            with self.subTest(shape=np.shape(batched)):
                self.assertEqual(get_prob_for_rand(batched, "random"), separate)

    def test_support_boundaries_and_nonfinite_values_have_zero_density(self):
        for invalid in (-np.finfo(float).eps, 1.0, 2.0, np.inf, -np.inf, np.nan):
            for position in (0, 2):
                with self.subTest(invalid=invalid, position=position):
                    values = np.array([0.0, 0.5, np.nextafter(1.0, 0.0)])
                    values[position] = invalid
                    self.assertEqual(get_prob_for_rand(values, "random"), 0.0)
        self.assertEqual(
            get_prob_for_rand([0.0, np.nextafter(1.0, 0.0)], "random"), 1.0
        )
        self.assertEqual(get_prob_for_rand(0.5, "random"), 1.0)

    def test_random_state_tracking_keeps_draws_and_generator_state_unchanged(self):
        for shape in (1, 3, (2, 3), 0, ()):
            with self.subTest(shape=shape):
                state = UnitUniformRand(seed=17, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(17)
                for _ in range(3):
                    state.set_rand_state("noise", "random", shape)
                    np.testing.assert_array_equal(
                        state.s.noise, reference.random(shape)
                    )
                    self.assertEqual(state.probs[-1], 1.0)
                    self.assertEqual(state.return_probdens(), 1.0)
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                    self.assertEqual(state.gen_state(), reference.bit_generator.state)
                self.assertEqual(state.probs, [1.0, 1.0, 1.0])

    def test_automatic_updates_and_copy_reset_keep_density_consistent(self):
        state = UnitUniformRand(seed=9, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(9)
        for _ in range(3):
            state.update_stochastic_states()
            np.testing.assert_array_equal(state.s.noise, reference.random(3))
            self.assertEqual(state.probs, [1.0])
        clone = state.copy()
        clone.update_stochastic_states()
        state.update_stochastic_states()
        np.testing.assert_array_equal(state.s.noise, clone.s.noise)
        self.assertIsNot(state.s.noise, clone.s.noise)
        self.assertEqual(clone.return_probdens(), 1.0)
        state.reset()
        self.assertEqual(state.probs, [])
        state.update_stochastic_states()
        np.testing.assert_array_equal(state.s.noise, np.random.default_rng(9).random(3))
        self.assertEqual(state.return_probdens(), 1.0)

    def test_scalar_and_disabled_updates_preserve_existing_behavior(self):
        scalar = ExampleRand(seed=3, run_stochastic=True, track_pdf=True)
        scalar.set_rand_state("noise", "random")
        self.assertEqual(scalar.s.noise, np.random.default_rng(3).random())
        self.assertEqual(scalar.probs, [1.0])
        disabled = UnitUniformRand(seed=3, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        np.testing.assert_array_equal(disabled.s.noise, np.zeros(3))
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)

    def test_simulation_probability_history_matches_independent_uniform_density(self):
        model = UnitUniformFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 11},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(11)
        draws = np.array([reference.random(3) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], draws)
        np.testing.assert_array_equal(history["r.probdens"], np.ones(len(history.time)))
        totals = np.concatenate(([0.0], np.cumsum(draws[1:].sum(axis=1))))
        np.testing.assert_allclose(history["s.total"], totals)
        self.assertEqual(result["tend.classify.density"], 1.0)
        self.assertAlmostEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.noise, np.zeros(3))
        self.assertEqual(model.r.probs, [])


if __name__ == "__main__":
    unittest.main()
