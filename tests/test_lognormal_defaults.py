#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for default lognormal probability tracking.

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
import unittest

import numpy as np
from scipy.integrate import quad

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_lognormal_pdf,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def density(values, mean=0.0, sigma=1.0):
    """Evaluate the lognormal formula without the distribution adapter."""
    values = np.asarray(values)
    return np.prod(
        np.exp(-0.5 * ((np.log(values) - mean) / sigma) ** 2)
        / (values * sigma * np.sqrt(2.0 * np.pi))
    )


class LognormalState(State):
    default: np.float64 = 1.0
    shifted: np.float64 = 1.0
    default_update = ("lognormal", ())
    shifted_update = ("lognormal", (0.5,))


class LognormalRand(Rand):
    s: LognormalState = LognormalState()


class SumState(State):
    total: np.float64 = 0.0


class LognormalFunction(Function):
    container_r = LognormalRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += self.r.s.default + self.r.s.shifted

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestLognormalDefaults(unittest.TestCase):
    def test_default_and_partial_arguments_match_explicit_density(self):
        """Defaults must apply to both scalar and array-valued parameters."""
        for args in ((), (0.5,), (-1.0,), (np.array([-0.5, 0.5]),), (0.5, 0.75)):
            mean = args[0] if args else 0.0
            sigma = args[1] if len(args) > 1 else 1.0
            for values in (1.0, np.array([0.5, 2.0])):
                with self.subTest(args=args, values=values):
                    expected = density(values, mean, sigma)
                    np.testing.assert_allclose(
                        get_prob_for_rand(values, "lognormal", *args),
                        expected,
                        rtol=1e-13,
                        atol=0.0,
                    )
                    np.testing.assert_allclose(
                        get_lognormal_pdf(*args)(values), expected, rtol=1e-13, atol=0.0
                    )

    def test_explicit_draw_shapes_and_broadcast_parameters_are_unchanged(self):
        for size in (None, (), 3, (2, 3), 0, (2, 0)):
            with self.subTest(size=size):
                values = np.random.default_rng(17).lognormal(0.5, 0.75, size)
                before = np.copy(values)
                actual = get_prob_for_rand(values, "lognormal", 0.5, 0.75, size)
                np.testing.assert_allclose(
                    actual, density(values, 0.5, 0.75), rtol=1e-13, atol=0.0
                )
                np.testing.assert_array_equal(values, before)
        mean = np.array([-0.5, 0.5, 1.0])
        sigma = np.array([[0.5], [1.0]])
        values = np.full((2, 3), 2.0)
        np.testing.assert_allclose(
            get_prob_for_rand(values, "lognormal", mean, sigma),
            density(values, mean, sigma),
            rtol=1e-13,
            atol=0.0,
        )
        np.testing.assert_allclose(
            get_lognormal_pdf(0.5)(0.5, 2.0), density([0.5, 2.0], 0.5)
        )

    def test_default_density_support_normalization_and_keywords(self):
        pdf = get_lognormal_pdf()
        self.assertAlmostEqual(quad(pdf, 0.0, np.inf)[0], 1.0, places=10)
        for value in (-1.0, 0.0, np.inf):
            with self.subTest(value=value):
                self.assertEqual(pdf(value), 0.0)
        self.assertTrue(np.isnan(pdf(np.nan)))
        np.testing.assert_allclose(
            get_lognormal_pdf(mean=0.5, size=(2,))(1.0), density(1.0, 0.5)
        )

    def test_random_updates_preserve_samples_state_copy_and_reset(self):
        for args in ((), (0.5,)):
            with self.subTest(args=args):
                state = ExampleRand(seed=31, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(31)
                expected_densities = []
                for _ in range(3):
                    value = reference.lognormal(*args)
                    state.set_rand_state("noise", "lognormal", *args)
                    self.assertEqual(state.s.noise, value)
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                    expected_densities.append(density(value, args[0] if args else 0.0))
                np.testing.assert_allclose(state.probs, expected_densities)
                np.testing.assert_allclose(
                    state.return_probdens(), np.prod(expected_densities)
                )
                clone = state.copy()
                state.set_rand_state("noise", "lognormal", *args)
                clone.set_rand_state("noise", "lognormal", *args)
                self.assertEqual(clone.s.noise, state.s.noise)
                state.reset()
                state.set_rand_state("noise", "lognormal", *args)
                self.assertEqual(
                    state.s.noise, np.random.default_rng(31).lognormal(*args)
                )
        disabled = ExampleRand(seed=31, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.set_rand_state("noise", "lognormal")
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)

    def test_real_simulation_tracks_default_and_partial_lognormal_draws(self):
        model = LognormalFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 41},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(41)
        draws = np.array(
            [(reference.lognormal(), reference.lognormal(0.5)) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.default"], draws[:, 0])
        np.testing.assert_array_equal(history["r.s.shifted"], draws[:, 1])
        expected = [density(row[0]) * density(row[1], 0.5) for row in draws]
        np.testing.assert_allclose(history["r.probdens"], expected)
        totals = np.concatenate(([0.0], np.cumsum(draws[1:].sum(axis=1))))
        np.testing.assert_allclose(history["s.total"], totals)
        np.testing.assert_allclose(result["tend.classify.density"], expected[-1])
        self.assertEqual(model.r.probs, [])
        self.assertEqual(model.s.total, 0.0)


if __name__ == "__main__":
    unittest.main()
