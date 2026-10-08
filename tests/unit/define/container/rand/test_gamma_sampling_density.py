#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for Gamma sampling parameter conversion.

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
from scipy import integrate, special, stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class GammaState(State):
    noise: np.array = np.zeros((2, 3))
    noise_update = ("gamma", (2.5, 1.75, (2, 3)))


class GammaRand(Rand):
    s: GammaState = GammaState()


class TotalState(State):
    total: np.float64 = 0.0


class GammaFunction(Function):
    container_r = GammaRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.noise)

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestGammaSamplingDensity(unittest.TestCase):
    def test_shape_and_scale_match_the_analytic_density(self):
        for shape, scale in ((0.5, 0.25), (1.0, 2.0), (2.0, 3.0), (5.0, 0.5)):
            for value in (0.125, 1.0, 4.0):
                with self.subTest(shape=shape, scale=scale, value=value):
                    expected = value ** (shape - 1.0) * np.exp(-value / scale)
                    expected /= special.gamma(shape) * scale**shape
                    np.testing.assert_allclose(
                        get_prob_for_rand(value, "gamma", shape, scale),
                        expected,
                        rtol=1e-13,
                        atol=0.0,
                    )
                    np.testing.assert_allclose(
                        get_pfunc_for_dist("gamma", shape, scale)(value),
                        expected,
                        rtol=1e-13,
                        atol=0.0,
                    )
        density = get_pfunc_for_dist("gamma", 2.0, 3.0)
        self.assertAlmostEqual(integrate.quad(density, 0.0, np.inf)[0], 1.0)
        expected_below_scale = 1.0 - 2.0 / np.e
        self.assertAlmostEqual(
            integrate.quad(density, 0.0, 3.0)[0], expected_below_scale
        )

    def test_optional_sampling_arguments_do_not_change_density_parameters(self):
        for method in ("gamma", "standard_gamma"):
            for size in (None, (), 1, 3, (2, 3), 0, (2, 0)):
                with self.subTest(method=method, size=size):
                    args = (2.5, 1.75, size) if method == "gamma" else (2.5, size)
                    sample = getattr(np.random.default_rng(19), method)(*args)
                    before = np.copy(sample)
                    scale = 1.75 if method == "gamma" else 1.0
                    expected = np.prod(stats.gamma.pdf(sample, 2.5, scale=scale))
                    np.testing.assert_allclose(
                        get_prob_for_rand(sample, method, *args),
                        expected,
                        rtol=1e-13,
                        atol=0.0,
                    )
                    np.testing.assert_array_equal(sample, before)

    def test_parameter_broadcasting_and_variadic_values_preserve_inputs(self):
        shape = np.array([0.5, 1.0, 3.0])
        scale = np.array([[0.25], [2.0]])
        values = np.array([[0.125, 0.25, 0.5], [1.0, 2.0, 4.0]])
        before = [x.copy() for x in (shape, scale, values)]
        for method, args, expected in (
            (
                "gamma",
                (shape, scale, (2, 3)),
                np.prod(stats.gamma.pdf(values, shape, scale=scale)),
            ),
            (
                "standard_gamma",
                (shape, (2, 3)),
                np.prod(stats.gamma.pdf(values, shape)),
            ),
        ):
            with self.subTest(method=method):
                np.testing.assert_allclose(
                    get_prob_for_rand(values, method, *args),
                    expected,
                    rtol=1e-13,
                    atol=0.0,
                )
        pdf = get_pfunc_for_dist("gamma", 2.0, 3.0)
        np.testing.assert_allclose(pdf(*values.ravel()), pdf(values), rtol=1e-13)
        for actual, original in zip((shape, scale, values), before):
            np.testing.assert_array_equal(actual, original)

    def test_standard_gamma_dtype_and_output_buffer_are_generation_only(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                buffer = np.empty((2, 3), dtype=dtype)
                reference_buffer = np.empty_like(buffer)
                state = GammaRand(seed=23, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(23)
                for _ in range(2):
                    reference.standard_gamma(2.5, (2, 3), dtype, reference_buffer)
                    state.set_rand_state(
                        "noise", "standard_gamma", 2.5, (2, 3), dtype, buffer
                    )
                    np.testing.assert_array_equal(buffer, reference_buffer)
                    np.testing.assert_array_equal(state.s.noise, reference_buffer)
                    self.assertEqual(state.s.noise.dtype, dtype)
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                    np.testing.assert_allclose(
                        state.probs[-1],
                        np.prod(stats.gamma.pdf(buffer, 2.5)),
                        rtol=1e-13,
                        atol=0.0,
                    )

    def test_tracked_updates_copy_reset_and_disabled_controls_preserve_draws(self):
        state = GammaRand(seed=11, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(11)
        for _ in range(3):
            state.update_stochastic_states()
            expected = reference.gamma(2.5, 1.75, (2, 3))
            np.testing.assert_array_equal(state.s.noise, expected)
            np.testing.assert_allclose(
                state.probs, [np.prod(stats.gamma.pdf(expected, 2.5, scale=1.75))]
            )
            self.assertEqual(
                state.rng.bit_generator.state, reference.bit_generator.state
            )
        clone = state.copy()
        for instance in (state, clone):
            instance.update_stochastic_states()
        np.testing.assert_array_equal(state.s.noise, clone.s.noise)
        self.assertIsNot(state.s.noise, clone.s.noise)
        state.reset()
        state.update_stochastic_states()
        np.testing.assert_array_equal(
            state.s.noise, np.random.default_rng(11).gamma(2.5, 1.75, (2, 3))
        )
        disabled = GammaRand(seed=7, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        self.assertEqual(disabled.rng.bit_generator.state, before)
        self.assertEqual(disabled.probs, [])
        np.testing.assert_array_equal(disabled.s.noise, np.zeros((2, 3)))

    def test_scalar_defaults_support_and_untracked_sampling_are_unchanged(self):
        for method in ("gamma", "standard_gamma"):
            with self.subTest(method=method):
                self.assertEqual(get_prob_for_rand(0.0, method, 1.0), 1.0)
                self.assertEqual(get_prob_for_rand(-1.0, method, 2.0), 0.0)
                self.assertTrue(np.isnan(get_prob_for_rand(np.nan, method, 2.0)))
                state = ExampleRand(seed=13, run_stochastic=True, track_pdf=True)
                state.set_rand_state("noise", method, 2.5)
                self.assertEqual(
                    state.s.noise, getattr(np.random.default_rng(13), method)(2.5)
                )
                np.testing.assert_allclose(
                    state.probs, [stats.gamma.pdf(state.s.noise, 2.5)]
                )
        untracked = GammaRand(seed=7, run_stochastic=True, track_pdf=False)
        untracked.update_stochastic_states()
        np.testing.assert_array_equal(
            untracked.s.noise, np.random.default_rng(7).gamma(2.5, 1.75, (2, 3))
        )
        self.assertEqual(untracked.probs, [])

    def test_actual_simulation_records_matrix_draws_density_and_cumulative_state(self):
        model = GammaFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        expected = np.array([reference.gamma(2.5, 1.75, (2, 3)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        densities = np.prod(stats.gamma.pdf(expected, 2.5, scale=1.75), axis=(1, 2))
        np.testing.assert_allclose(
            history["r.probdens"], densities, rtol=1e-13, atol=0.0
        )
        totals = np.concatenate(([0.0], np.cumsum(expected[1:].sum(axis=(1, 2)))))
        np.testing.assert_allclose(history["s.total"], totals)
        self.assertAlmostEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_allclose(
            result["tend.classify.density"], densities[-1], rtol=1e-13, atol=0.0
        )
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 3)))
        self.assertEqual(model.r.probs, [])


if __name__ == "__main__":
    unittest.main()
