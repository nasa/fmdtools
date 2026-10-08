#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exponential and Rayleigh sampling arguments.

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
    get_exp_ray_pdf,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def expected_factors(values, scale, method):
    """Evaluate positive-scale densities independently of scipy.stats."""
    values, scale = np.broadcast_arrays(values, scale)
    if method == "exponential":
        density = np.exp(-values / scale) / scale
    else:
        density = values / scale**2 * np.exp(-0.5 * (values / scale) ** 2)
    return np.where(values >= 0, density, 0.0)


class PositiveDrawState(State):
    waiting: np.array = np.zeros((2, 3))
    waiting_update = ("exponential", (2.0, (2, 3)))
    amplitude: np.array = np.zeros((2, 3))
    amplitude_update = ("rayleigh", (0.75, (2, 3)))


class PositiveDrawRand(Rand):
    s: PositiveDrawState = PositiveDrawState()


class TotalState(State):
    total: np.float64 = 0.0


class PositiveDrawFunction(Function):
    container_r = PositiveDrawRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.waiting) + np.sum(self.r.s.amplitude)

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestExponentialRayleighDrawSizes(unittest.TestCase):
    def test_draw_size_does_not_shift_density_and_all_shapes_are_supported(self):
        for method in ("exponential", "rayleigh"):
            for size in (None, (), 1, 4, (2, 3), (2, 1, 3), 0, (2, 0)):
                with self.subTest(method=method, size=size):
                    values = getattr(np.random.default_rng(31), method)(2.0, size)
                    before = np.copy(values)
                    expected = np.prod(expected_factors(values, 2.0, method))
                    for args in ((2.0,), (2.0, size)):
                        for actual in (
                            get_exp_ray_pdf(method, *args)(values),
                            get_pfunc_for_dist(method, *args)(values),
                            get_prob_for_rand(values, method, *args),
                        ):
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-13, atol=0.0
                            )
                    np.testing.assert_array_equal(values, before)

    def test_default_scale_scalar_and_variadic_controls(self):
        for method in ("exponential", "rayleigh"):
            for scale in (0.25, 1.0, 2.5):
                with self.subTest(method=method, scale=scale):
                    values = np.array([0.0, 0.25, 1.0, 4.0])
                    for value in values:
                        expected = expected_factors(value, scale, method)
                        np.testing.assert_allclose(
                            get_exp_ray_pdf(method, scale)(value), expected
                        )
                    positive = values[1:]
                    np.testing.assert_allclose(
                        get_pfunc_for_dist(method, scale, 3)(*positive),
                        np.prod(expected_factors(positive, scale, method)),
                    )
            np.testing.assert_allclose(
                get_prob_for_rand(0.5, method), expected_factors(0.5, 1.0, method)
            )

    def test_scale_broadcasting_preserves_values_and_parameters(self):
        for method in ("exponential", "rayleigh"):
            with self.subTest(method=method):
                scale = np.array([[0.5], [2.0]])
                values = np.array([[0.25, 0.5, 1.0], [2.0, 3.0, 4.0]])
                original_scale, original_values = scale.copy(), values.copy()
                actual = get_prob_for_rand(values, method, scale, (2, 3))
                np.testing.assert_allclose(
                    actual,
                    np.prod(expected_factors(values, scale, method)),
                    rtol=1e-13,
                    atol=0.0,
                )
                np.testing.assert_array_equal(scale, original_scale)
                np.testing.assert_array_equal(values, original_values)

    def test_density_normalization_support_and_early_interval_mass(self):
        for method in ("exponential", "rayleigh"):
            for scale in (0.5, 2.0):
                with self.subTest(method=method, scale=scale):
                    density = get_exp_ray_pdf(method, scale, 5)
                    total, _ = quad(density, 0.0, np.inf)
                    early, _ = quad(density, 0.0, scale)
                    np.testing.assert_allclose(total, 1.0, rtol=1e-10)
                    exponent = -1.0 if method == "exponential" else -0.5
                    np.testing.assert_allclose(
                        early, 1.0 - np.exp(exponent), rtol=1e-10
                    )
                    self.assertEqual(density(-0.1), 0.0)
                    self.assertTrue(np.isnan(density(np.nan)))

    def test_tracked_draws_keep_samples_generator_state_and_optional_scalar_size(self):
        for method in ("exponential", "rayleigh"):
            for size in (None, (), 3, (2, 3), 0):
                with self.subTest(method=method, size=size):
                    state = PositiveDrawRand(
                        seed=11, run_stochastic=True, track_pdf=True
                    )
                    reference = np.random.default_rng(11)
                    probabilities = []
                    for _ in range(3):
                        expected = getattr(reference, method)(2.0, size)
                        state.set_rand_state("waiting", method, 2.0, size)
                        np.testing.assert_array_equal(state.s.waiting, expected)
                        probabilities.append(
                            np.prod(expected_factors(expected, 2.0, method))
                        )
                        self.assertEqual(
                            state.rng.bit_generator.state, reference.bit_generator.state
                        )
                    np.testing.assert_allclose(state.probs, probabilities, rtol=1e-13)
                    np.testing.assert_allclose(
                        state.return_probdens(), np.prod(probabilities), rtol=1e-13
                    )
            scalar = ExampleRand(seed=17, run_stochastic=True, track_pdf=True)
            scalar.set_rand_state("noise", method, 2.0, None)
            expected = getattr(np.random.default_rng(17), method)(2.0)
            self.assertEqual(scalar.s.noise, expected)
            np.testing.assert_allclose(
                scalar.probs, [expected_factors(expected, 2.0, method)]
            )

    def test_automatic_updates_copy_reset_and_disabled_controls(self):
        state = PositiveDrawRand(seed=29, run_stochastic=True, track_pdf=True)
        state.update_stochastic_states()
        first_waiting, first_amplitude = (
            state.s.waiting.copy(),
            state.s.amplitude.copy(),
        )
        clone = state.copy()
        state.update_stochastic_states()
        clone.update_stochastic_states()
        np.testing.assert_array_equal(state.s.waiting, clone.s.waiting)
        np.testing.assert_array_equal(state.s.amplitude, clone.s.amplitude)
        self.assertIsNot(state.s.waiting, clone.s.waiting)
        np.testing.assert_allclose(state.probs, clone.probs)
        state.reset()
        state.update_stochastic_states()
        np.testing.assert_array_equal(state.s.waiting, first_waiting)
        np.testing.assert_array_equal(state.s.amplitude, first_amplitude)
        disabled = PositiveDrawRand(seed=29, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)
        untracked = PositiveDrawRand(seed=29, run_stochastic=True, track_pdf=False)
        untracked.update_stochastic_states()
        np.testing.assert_array_equal(untracked.s.waiting, first_waiting)
        np.testing.assert_array_equal(untracked.s.amplitude, first_amplitude)
        self.assertEqual(untracked.probs, [])

    def test_actual_simulation_records_both_distributions_and_joint_density(self):
        model = PositiveDrawFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(19)
        waiting, amplitude, densities = [], [], []
        for _ in history.time:
            wait = reference.exponential(2.0, (2, 3))
            ampl = reference.rayleigh(0.75, (2, 3))
            waiting.append(wait)
            amplitude.append(ampl)
            densities.append(
                np.prod(expected_factors(wait, 2.0, "exponential"))
                * np.prod(expected_factors(ampl, 0.75, "rayleigh"))
            )
        np.testing.assert_array_equal(history["r.s.waiting"], waiting)
        np.testing.assert_array_equal(history["r.s.amplitude"], amplitude)
        np.testing.assert_allclose(
            history["r.probdens"], densities, rtol=1e-12, atol=0.0
        )
        increments = np.array(waiting).sum(axis=(1, 2)) + np.array(amplitude).sum(
            axis=(1, 2)
        )
        totals = np.concatenate(([0.0], np.cumsum(increments[1:])))
        np.testing.assert_allclose(history["s.total"], totals)
        np.testing.assert_allclose(result["tend.classify.total"], totals[-1])
        np.testing.assert_allclose(
            result["tend.classify.density"], densities[-1], rtol=1e-12, atol=0.0
        )
        np.testing.assert_array_equal(model.r.s.waiting, np.zeros((2, 3)))
        self.assertEqual(model.r.probs, [])


if __name__ == "__main__":
    unittest.main()
