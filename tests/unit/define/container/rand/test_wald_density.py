#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for NumPy Wald density parameter mapping.

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
from scipy import stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def density(values, mean, scale):
    """NumPy's inverse Gaussian formula, independent of scipy.stats."""
    values, mean, scale = np.broadcast_arrays(values, mean, scale)
    return np.prod(
        np.sqrt(scale / (2.0 * np.pi * values**3))
        * np.exp(-scale * (values - mean) ** 2 / (2.0 * mean**2 * values))
    )


class WaldState(State):
    noise: np.array = np.zeros((2, 2))
    noise_update = ("wald", (2.0, 3.0, (2, 2)))


class WaldRand(Rand):
    s: WaldState = WaldState()


class SumState(State):
    total: np.float64 = 0.0


class WaldFunction(Function):
    container_r = WaldRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += self.r.s.noise.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestWaldDensity(unittest.TestCase):
    def test_mean_and_scale_match_the_analytic_density(self):
        for mean, scale in ((1.0, 1.0), (0.5, 2.0), (2.0, 3.0), (3.0, 0.75)):
            for value in (0.25, 1.0, 2.0, 5.0):
                with self.subTest(mean=mean, scale=scale, value=value):
                    expected = density(value, mean, scale)
                    np.testing.assert_allclose(
                        get_prob_for_rand(value, "wald", mean, scale),
                        expected,
                        rtol=1e-13,
                        atol=0.0,
                    )

    def test_density_normalizes_and_has_the_requested_mean(self):
        for mean, scale in ((1.0, 1.0), (0.5, 2.0), (2.0, 3.0)):
            with self.subTest(mean=mean, scale=scale):
                pdf = get_pfunc_for_dist("wald", mean, scale)
                self.assertAlmostEqual(quad(pdf, 0.0, np.inf)[0], 1.0, places=9)
                self.assertAlmostEqual(
                    quad(lambda x: x * pdf(x), 0.0, np.inf)[0], mean, places=9
                )
                below_mean = quad(pdf, 0.0, mean)[0]
                self.assertGreater(below_mean, 0.0)
                self.assertLess(below_mean, 1.0)

    def test_sizes_broadcast_parameters_and_variadic_calls_preserve_inputs(self):
        for size in (None, (), 3, (2, 3), (2, 1, 3), 0, (2, 0)):
            with self.subTest(size=size):
                values = np.random.default_rng(7).wald(2.0, 3.0, size)
                before = np.copy(values)
                np.testing.assert_allclose(
                    get_prob_for_rand(values, "wald", 2.0, 3.0, size),
                    density(values, 2.0, 3.0),
                    rtol=1e-13,
                    atol=0.0,
                )
                np.testing.assert_array_equal(values, before)
        mean = np.array([[0.5], [2.0]])
        scale = np.array([1.0, 2.0, 3.0])
        values = np.random.default_rng(7).wald(mean, scale)
        expected = density(values, mean, scale)
        np.testing.assert_allclose(
            get_prob_for_rand(values, "wald", mean, scale, (2, 3)), expected
        )
        np.testing.assert_array_equal(mean, [[0.5], [2.0]])
        np.testing.assert_array_equal(scale, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            get_pfunc_for_dist("wald", 2.0, 3.0)(1.0, 2.0),
            density([1.0, 2.0], 2.0, 3.0),
        )

    def test_support_and_unit_parameter_control(self):
        pdf = get_pfunc_for_dist("wald", 1.0, 1.0)
        for value in (-1.0, 0.0, 0.5, 1.0, 2.0):
            with self.subTest(value=value):
                np.testing.assert_allclose(
                    pdf(value), stats.wald.pdf(value), rtol=1e-13, atol=0.0
                )
        self.assertTrue(np.isnan(pdf(np.nan)))
        self.assertEqual(get_prob_for_rand([], "wald", 1.0, 1.0, 0), 1.0)

    def test_tracked_draws_preserve_generator_state_copy_reset_and_controls(self):
        for size in (None, (2, 2)):
            with self.subTest(size=size):
                cls = ExampleRand if size is None else WaldRand
                state = cls(seed=17, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(17)
                for _ in range(3):
                    expected = reference.wald(2.0, 3.0, size)
                    state.set_rand_state("noise", "wald", 2.0, 3.0, size)
                    np.testing.assert_array_equal(state.s.noise, expected)
                    np.testing.assert_allclose(
                        state.probs[-1], density(expected, 2.0, 3.0)
                    )
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                clone = state.copy()
                state.set_rand_state("noise", "wald", 2.0, 3.0, size)
                clone.set_rand_state("noise", "wald", 2.0, 3.0, size)
                np.testing.assert_array_equal(state.s.noise, clone.s.noise)
                state.reset()
                state.set_rand_state("noise", "wald", 2.0, 3.0, size)
                np.testing.assert_array_equal(
                    state.s.noise, np.random.default_rng(17).wald(2.0, 3.0, size)
                )
        disabled = WaldRand(seed=17, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)
        untracked = WaldRand(seed=17, run_stochastic=True, track_pdf=False)
        untracked.update_stochastic_states()
        np.testing.assert_array_equal(
            untracked.s.noise, np.random.default_rng(17).wald(2.0, 3.0, (2, 2))
        )
        self.assertEqual(untracked.probs, [])

    def test_real_matrix_simulation_records_density_and_unchanged_samples(self):
        model = WaldFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(19)
        expected = np.array([reference.wald(2.0, 3.0, (2, 2)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        masses = [density(row, 2.0, 3.0) for row in expected]
        np.testing.assert_allclose(history["r.probdens"], masses)
        totals = np.concatenate(([0.0], np.cumsum(expected[1:].sum(axis=(1, 2)))))
        np.testing.assert_allclose(history["s.total"], totals)
        np.testing.assert_allclose(result["tend.classify.density"], masses[-1])
        self.assertEqual(model.r.probs, [])
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
