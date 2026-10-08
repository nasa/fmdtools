#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for shape-distribution sampling arguments.

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
from scipy import stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


DISTRIBUTIONS = (
    ("beta", (2.0, 3.0), "beta"),
    ("f", (3.0, 5.0), "f"),
    ("chisquare", (3.0,), "chi2"),
    ("noncentral_chisquare", (3.0, 1.0), "ncx2"),
    ("noncentral_f", (3.0, 5.0, 1.0), "ncf"),
    ("power", (2.0,), "powerlaw"),
    ("weibull", (2.0,), "weibull_min"),
)


class ShapeState(State):
    noise: np.array = np.zeros((2, 2))


class ShapeRand(Rand):
    s: ShapeState = ShapeState()


class BatchState(State):
    beta: np.array = np.zeros((2, 2))
    f: np.array = np.zeros((2, 2))
    chisquare: np.array = np.zeros((2, 2))
    noncentral_chisquare: np.array = np.zeros((2, 2))
    noncentral_f: np.array = np.zeros((2, 2))
    power: np.array = np.zeros((2, 2))
    weibull: np.array = np.zeros((2, 2))
    beta_update = ("beta", (2.0, 3.0, (2, 2)))
    f_update = ("f", (3.0, 5.0, (2, 2)))
    chisquare_update = ("chisquare", (3.0, (2, 2)))
    noncentral_chisquare_update = ("noncentral_chisquare", (3.0, 1.0, (2, 2)))
    noncentral_f_update = ("noncentral_f", (3.0, 5.0, 1.0, (2, 2)))
    power_update = ("power", (2.0, (2, 2)))
    weibull_update = ("weibull", (2.0, (2, 2)))


class BatchRand(Rand):
    s: BatchState = BatchState()


class TotalState(State):
    total: float = 0.0


class BatchFunction(Function):
    container_r = BatchRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += sum(
            np.sum(getattr(self.r.s, name)) for name in self.r.s.__fields__
        )

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestShapeDistributionSizes(unittest.TestCase):
    """Keep generator-only arguments separate from distribution parameters."""

    def test_scalar_empty_and_matrix_sizes_do_not_shift_the_density(self):
        for name, params, scipy_name in DISTRIBUTIONS:
            for size in (None, (), 1, 3, (2, 3), (2, 1, 3), 0, (2, 0)):
                with self.subTest(name=name, size=size):
                    values = getattr(np.random.default_rng(7), name)(*params, size)
                    before = np.copy(values)
                    expected = np.prod(getattr(stats, scipy_name).pdf(values, *params))
                    for arguments in (params, (*params, size)):
                        actual = get_prob_for_rand(values, name, *arguments)
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-13, atol=0.0
                        )
                    np.testing.assert_array_equal(values, before)

    def test_known_formulas_remain_valid_with_an_optional_size(self):
        cases = (
            ("beta", (2.0, 3.0), 0.5, 1.5),
            ("f", (2.0, 2.0), 1.0, 0.25),
            ("chisquare", (2.0,), 1.0, np.exp(-0.5) / 2),
            ("noncentral_chisquare", (2.0, 0.0), 1.0, np.exp(-0.5) / 2),
            ("power", (2.0,), 0.5, 1.0),
            ("weibull", (2.0,), 1.0, 2 * np.exp(-1.0)),
        )
        for name, params, value, density in cases:
            with self.subTest(name=name):
                actual = get_prob_for_rand([value] * 3, name, *params, 3)
                np.testing.assert_allclose(actual, density**3, rtol=1e-13, atol=0.0)

    def test_broadcast_parameters_and_variadic_values_are_unchanged(self):
        for name, parameters, scipy_name in DISTRIBUTIONS:
            with self.subTest(name=name):
                params = (
                    np.array([[parameters[0]], [parameters[0] + 0.5]]),
                    *parameters[1:],
                )
                values = getattr(np.random.default_rng(5), name)(*params, (2, 3))
                before = params[0].copy(), values.copy()
                expected = np.prod(getattr(stats, scipy_name).pdf(values, *params))
                actual = get_prob_for_rand(values, name, *params, (2, 3))
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
                pdf = get_pfunc_for_dist(name, *parameters, 3)
                draws = getattr(np.random.default_rng(5), name)(*parameters, 3)
                np.testing.assert_allclose(
                    pdf(*draws), pdf(draws), rtol=1e-13, atol=0.0
                )
                np.testing.assert_array_equal(params[0], before[0])
                np.testing.assert_array_equal(values, before[1])

    def test_tracked_updates_preserve_draws_generator_state_and_reset(self):
        for name, params, scipy_name in DISTRIBUTIONS:
            for size in (None, (2, 2)):
                with self.subTest(name=name, size=size):
                    state = ShapeRand(seed=23, run_stochastic=True, track_pdf=True)
                    reference = np.random.default_rng(23)
                    expected_probs = []
                    for _ in range(3):
                        expected = getattr(reference, name)(*params, size)
                        state.set_rand_state("noise", name, *params, size)
                        np.testing.assert_array_equal(state.s.noise, expected)
                        self.assertEqual(
                            state.rng.bit_generator.state, reference.bit_generator.state
                        )
                        expected_probs.append(
                            np.prod(getattr(stats, scipy_name).pdf(expected, *params))
                        )
                    np.testing.assert_allclose(
                        state.probs, expected_probs, rtol=1e-13, atol=0.0
                    )
                    clone = state.copy()
                    for item in (state, clone):
                        item.set_rand_state("noise", name, *params, size)
                    np.testing.assert_array_equal(state.s.noise, clone.s.noise)
                    state.reset()
                    state.set_rand_state("noise", name, *params, size)
                    np.testing.assert_array_equal(
                        state.s.noise,
                        getattr(np.random.default_rng(23), name)(*params, size),
                    )

    def test_invalid_argument_counts_and_disabled_controls(self):
        for name, params, _ in DISTRIBUTIONS:
            with self.subTest(name=name):
                for invalid in (params[:-1], (*params, None, 7)):
                    with self.assertRaises(TypeError):
                        get_pfunc_for_dist(name, *invalid)
                disabled = ShapeRand(seed=3, run_stochastic=False, track_pdf=True)
                before = copy.deepcopy(disabled.rng.bit_generator.state)
                disabled.set_rand_state("noise", name, *params, (2, 2))
                self.assertEqual(disabled.rng.bit_generator.state, before)
                self.assertEqual(disabled.probs, [])
                untracked = ShapeRand(seed=3, run_stochastic=True, track_pdf=False)
                untracked.set_rand_state("noise", name, *params, (2, 2))
                np.testing.assert_array_equal(
                    untracked.s.noise,
                    getattr(np.random.default_rng(3), name)(*params, (2, 2)),
                )
                self.assertEqual(untracked.probs, [])

    def test_real_simulation_preserves_samples_and_tracks_all_seven_families(self):
        model = BatchFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(19)
        total_increments, densities = [], []
        for index, _ in enumerate(history.time):
            total, density = 0.0, 1.0
            for name, params, scipy_name in DISTRIBUTIONS:
                values = getattr(reference, name)(*params, (2, 2))
                np.testing.assert_array_equal(history["r.s." + name][index], values)
                total += np.sum(values)
                density *= np.prod(getattr(stats, scipy_name).pdf(values, *params))
            total_increments.append(total)
            densities.append(density)
        expected_totals = np.concatenate(([0.0], np.cumsum(total_increments[1:])))
        np.testing.assert_allclose(history["s.total"], expected_totals)
        np.testing.assert_allclose(
            history["r.probdens"], densities, rtol=1e-12, atol=0.0
        )
        self.assertAlmostEqual(result["tend.classify.total"], expected_totals[-1])
        np.testing.assert_allclose(
            result["tend.classify.density"], densities[-1], rtol=1e-12, atol=0.0
        )
        self.assertEqual(model.s.total, 0.0)
        self.assertEqual(model.r.probs, [])


if __name__ == "__main__":
    unittest.main()
