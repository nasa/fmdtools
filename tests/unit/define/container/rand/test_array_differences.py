#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for array-valued random-state updates.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import ExampleRand, Rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class VectorState(State):
    noise: np.array = np.zeros(3)


class VectorRand(Rand):
    s: VectorState = VectorState()


class AutoVectorState(VectorState):
    noise_update = ("uniform", (-2.0, 3.0, 3))


class AutoVectorRand(Rand):
    s: AutoVectorState = AutoVectorState()


class ListState(State):
    noise: list = [0.0, 0.0, 0.0]


class ListRand(Rand):
    s: ListState = ListState()


class SumState(State):
    total: np.float64 = 0.0


class VectorFunction(Function):
    container_r = AutoVectorRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.noise)

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestArrayRandomStates(unittest.TestCase):
    def test_array_draws_preserve_shapes_values_and_generator_sequence(self):
        for method, args in (
            ("normal", (0.0, 1.0, 3)),
            ("uniform", (-2.0, 3.0, (2, 3))),
            ("integers", (1, 10, 4)),
            ("choice", ([1.0, 4.0, 9.0], 5)),
            ("normal", (0.0, 1.0, 0)),
            ("normal", (0.0, 1.0, ())),
        ):
            with self.subTest(method=method, args=args):
                state = VectorRand(seed=23, run_stochastic=True)
                original = state.s.noise
                reference = np.random.default_rng(23)
                for _ in range(3):
                    expected = getattr(reference, method)(*args)
                    state.set_rand_state("noise", method, *args)
                    self.assertIsInstance(state.s.noise, np.ndarray)
                    np.testing.assert_array_equal(state.s.noise, expected)
                    self.assertEqual(state.s.noise.dtype, expected.dtype)
                    self.assertEqual(state.s.noise.shape, expected.shape)
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                    self.assertEqual(state.gen_state(), reference.bit_generator.state)
                np.testing.assert_array_equal(original, np.zeros(3))
                self.assertEqual(state.probs, [])

    def test_probability_tracking_uses_the_complete_uniform_draw(self):
        for shape in (1, 3, (2, 3)):
            with self.subTest(shape=shape):
                state = VectorRand(seed=41, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(41)
                for _ in range(2):
                    state.set_rand_state("noise", "uniform", -2.0, 3.0, shape)
                    np.testing.assert_array_equal(
                        state.s.noise, reference.uniform(-2.0, 3.0, shape)
                    )
                joint_density = 5.0**-state.s.noise.size
                np.testing.assert_allclose(state.probs, [joint_density] * 2)
                self.assertAlmostEqual(state.return_probdens(), joint_density**2)

    def test_automatic_updates_reset_density_per_update_and_match_direct_draws(self):
        state = AutoVectorRand(seed=19, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(19)
        for _ in range(3):
            state.update_stochastic_states()
            np.testing.assert_array_equal(
                state.s.noise, reference.uniform(-2.0, 3.0, 3)
            )
            np.testing.assert_allclose(state.probs, [5.0**-3])
        state.reset()
        self.assertEqual(state.probs, [])
        np.testing.assert_array_equal(state.s.noise, np.zeros(3))
        state.update_stochastic_states()
        np.testing.assert_array_equal(
            state.s.noise, np.random.default_rng(19).uniform(-2.0, 3.0, 3)
        )

    def test_copy_preserves_draw_sequence_without_sharing_state(self):
        state = VectorRand(seed=11, run_stochastic=True)
        state.set_rand_state("noise", "normal", 0.0, 1.0, 3)
        clone = state.copy()
        self.assertIsNot(clone.s.noise, state.s.noise)
        np.testing.assert_array_equal(clone.s.noise, state.s.noise)
        state.set_rand_state("noise", "normal", 0.0, 1.0, 3)
        clone.set_rand_state("noise", "normal", 0.0, 1.0, 3)
        np.testing.assert_array_equal(clone.s.noise, state.s.noise)
        clone.s.noise[0] = 999.0
        self.assertNotEqual(clone.s.noise[0], state.s.noise[0])

    def test_list_and_scalar_state_controls_retain_existing_behavior(self):
        state = ListRand(seed=7, run_stochastic=True)
        reference = np.random.default_rng(7)
        for _ in range(2):
            state.set_rand_state("noise", "normal", 0.0, 1.0, 3)
            self.assertIs(type(state.s.noise), list)
            self.assertEqual(state.s.noise, reference.normal(0.0, 1.0, 3).tolist())
        scalar = ExampleRand(seed=7, run_stochastic=True)
        scalar.set_rand_state("noise", "normal", 0.0, 1.0)
        self.assertEqual(scalar.s.noise, np.random.default_rng(7).normal())
        for count in (1, 3):
            with self.subTest(count=count):
                with self.assertRaisesRegex(Exception, "returned array"):
                    scalar.set_rand_state("noise", "normal", 0.0, 1.0, count)

    def test_disabled_stochastic_updates_leave_rng_and_values_unchanged(self):
        state = AutoVectorRand(seed=3, run_stochastic=False)
        before = copy.deepcopy(state.rng.bit_generator.state)
        state.set_rand_state("noise", "normal", 0.0, 1.0, 3)
        state.update_stochastic_states()
        np.testing.assert_array_equal(state.s.noise, np.zeros(3))
        self.assertEqual(state.rng.bit_generator.state, before)
        self.assertEqual(state.probs, [])

    def test_full_simulation_records_array_noise_and_its_cumulative_effect(self):
        model = VectorFunction(
            sp={"end_time": 3.0, "run_stochastic": True}, r={"seed": 13}
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(13)
        expected = np.array([reference.uniform(-2.0, 3.0, 3) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        # The initialization draw is recorded at t=0; dynamic behavior starts at t=1.
        totals = np.concatenate(([0.0], np.cumsum(np.sum(expected[1:], axis=1))))
        np.testing.assert_allclose(history["s.total"], totals)
        self.assertAlmostEqual(result["tend.classify.total"], np.sum(expected[1:]))
        np.testing.assert_array_equal(model.r.s.noise, np.zeros(3))
        self.assertEqual(model.s.total, 0.0)


if __name__ == "__main__":
    unittest.main()
