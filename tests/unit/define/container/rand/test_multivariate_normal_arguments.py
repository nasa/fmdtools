#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for multivariate normal generation arguments.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


MEAN = np.array([1.0, -2.0])
COV = np.array([[2.0, 0.5], [0.5, 1.0]])


def analytic_density(values, mean, cov):
    """Independent full-rank Gaussian formula using determinant and solve."""
    centered = np.asarray(values).reshape(-1, len(mean)) - mean
    exponent = np.sum(centered * np.linalg.solve(cov, centered.T).T, axis=-1)
    normalizer = np.sqrt((2 * np.pi) ** len(mean) * np.linalg.det(cov))
    return np.prod(np.exp(-0.5 * exponent) / normalizer)


class GaussianState(State):
    noise: np.array = np.zeros((2, 2))
    noise_update = ("multivariate_normal", (MEAN, COV, 2, "raise", 1e-8))


class GaussianRand(Rand):
    s: GaussianState = GaussianState()


class SumState(State):
    total: float = 0.0


class GaussianFunction(Function):
    container_r = GaussianRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += self.r.s.noise.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestMultivariateNormalArguments(unittest.TestCase):
    def test_generation_options_do_not_change_full_rank_densities(self):
        for mean, cov in ((np.array([1.0]), np.array([[4.0]])), (MEAN, COV)):
            for size in (None, (), 1, 3, (2, 3), 0, (2, 0)):
                for policy in ("warn", "raise", "ignore"):
                    with self.subTest(dimension=len(mean), size=size, policy=policy):
                        values = np.random.default_rng(19).multivariate_normal(
                            mean, cov, size, policy, 1e-7
                        )
                        before = values.copy()
                        expected = analytic_density(values, mean, cov)
                        for arguments in (
                            (mean, cov),
                            (mean, cov, size),
                            (mean, cov, size, policy),
                            (mean, cov, size, policy, 1e-7),
                        ):
                            actual = get_prob_for_rand(
                                values, "multivariate_normal", *arguments
                            )
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-13, atol=0
                            )
                            self.assertEqual(np.ndim(actual), 0)
                        np.testing.assert_array_equal(values, before)

    def test_singular_covariance_uses_density_on_its_support(self):
        mean = np.array([1.0, -2.0])
        cov = np.diag([4.0, 0.0])
        for size in (None, (), 1, 3, (2, 3), 0):
            with self.subTest(size=size):
                values = np.random.default_rng(23).multivariate_normal(mean, cov, size)
                expected = np.prod(
                    np.exp(-((values[..., 0] - mean[0]) ** 2) / 8) / np.sqrt(8 * np.pi)
                )
                for args in (
                    (mean, cov),
                    (mean, cov, size),
                    (mean, cov, size, "raise"),
                ):
                    actual = get_prob_for_rand(values, "multivariate_normal", *args)
                    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
        pfunc = get_pfunc_for_dist("multivariate_normal", mean, cov, None)
        self.assertEqual(pfunc([1.0, -1.0]), 0.0)
        np.testing.assert_array_equal(cov, np.diag([4.0, 0.0]))

    def test_variadic_views_and_one_dimensional_controls_remain_usable(self):
        values = np.array([[1.0, -2.0], [2.0, -1.0], [0.0, -3.0]])
        pfunc = get_pfunc_for_dist("multivariate_normal", MEAN, COV, (3,), "raise")
        for view in (values, values[::-1], np.asfortranarray(values)):
            with self.subTest(strides=view.strides):
                view.setflags(write=False)
                np.testing.assert_allclose(
                    pfunc(view), analytic_density(view, MEAN, COV)
                )
                np.testing.assert_allclose(
                    pfunc(*view), analytic_density(view, MEAN, COV)
                )
        np.testing.assert_allclose(
            get_prob_for_rand(0.0, "multivariate_normal", 0.0, 1.0),
            1 / np.sqrt(2 * np.pi),
        )
        with self.assertRaises(ValueError):
            get_prob_for_rand(
                [0.0, 0.0], "multivariate_normal", [0.0, 0.0], [[1.0, 0.0], [0.0, -1.0]]
            )

    def test_tracked_updates_preserve_samples_state_copy_and_reset(self):
        for size in (None, (2, 3), 0):
            with self.subTest(size=size):
                rand = GaussianRand(seed=29, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(29)
                masses = []
                for _ in range(2):
                    expected = reference.multivariate_normal(
                        MEAN, COV, size, "raise", 1e-8
                    )
                    rand.set_rand_state(
                        "noise", "multivariate_normal", MEAN, COV, size, "raise", 1e-8
                    )
                    np.testing.assert_array_equal(rand.s.noise, expected)
                    masses.append(analytic_density(expected, MEAN, COV))
                    self.assertEqual(
                        rand.rng.bit_generator.state, reference.bit_generator.state
                    )
                np.testing.assert_allclose(rand.probs, masses)
                clone = rand.copy()
                for state in (rand, clone):
                    state.set_rand_state(
                        "noise", "multivariate_normal", MEAN, COV, size, "raise"
                    )
                np.testing.assert_array_equal(rand.s.noise, clone.s.noise)
                rand.reset()
                self.assertEqual(rand.probs, [])
                rand.set_rand_state(
                    "noise", "multivariate_normal", MEAN, COV, size, "raise"
                )
                np.testing.assert_array_equal(
                    rand.s.noise,
                    np.random.default_rng(29).multivariate_normal(
                        MEAN, COV, size, "raise"
                    ),
                )
        disabled = GaussianRand(seed=3, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.set_rand_state("noise", "multivariate_normal", MEAN, COV, 2, "raise")
        self.assertEqual(disabled.rng.bit_generator.state, before)
        self.assertEqual(disabled.probs, [])

    def test_real_simulation_retains_correlated_draws_and_joint_densities(self):
        model = GaussianFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        expected = np.array(
            [
                reference.multivariate_normal(MEAN, COV, 2, "raise", 1e-8)
                for _ in history.time
            ]
        )
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        masses = [analytic_density(row, MEAN, COV) for row in expected]
        np.testing.assert_allclose(history["r.probdens"], masses, rtol=1e-13)
        totals = np.r_[0.0, np.cumsum(expected[1:].sum(axis=(1, 2)))]
        np.testing.assert_allclose(history["s.total"], totals)
        self.assertAlmostEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
