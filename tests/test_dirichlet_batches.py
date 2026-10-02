#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for complete Dirichlet vectors and batches.

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
import math
import unittest

import numpy as np
from scipy import stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def analytic_density(values, alpha):
    """Independent product of the full-vector Dirichlet density formula."""
    alpha = np.asarray(alpha)
    coefficient = math.gamma(float(alpha.sum())) / math.prod(
        math.gamma(float(a)) for a in alpha
    )
    rows = np.asarray(values).reshape(-1, len(alpha))
    return np.prod(coefficient * np.prod(rows ** (alpha - 1), axis=-1))


class ProportionState(State):
    proportions: np.array = np.full((2, 3), 1.0 / 3)
    proportions_update = ("dirichlet", ([2.0, 3.0, 4.0], 2))


class ProportionRand(Rand):
    s: ProportionState = ProportionState()


class TotalState(State):
    total: np.float64 = 0.0


class ProportionFunction(Function):
    container_r = ProportionRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.proportions[:, 0])

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestDirichletBatches(unittest.TestCase):
    def test_single_vectors_and_all_batch_shapes_match_the_analytic_density(self):
        for alpha in ([1.0], [1.0, 1.0], [0.5, 2.0, 3.0]):
            for size in (None, (), 1, 3, (2, 3), (1, 2, 1), 0, (2, 0)):
                with self.subTest(alpha=alpha, size=size):
                    values = np.random.default_rng(31).dirichlet(alpha, size=size)
                    before = values.copy()
                    expected = analytic_density(values, alpha)
                    for arguments in ((alpha,), (alpha, size)):
                        actual = get_prob_for_rand(values, "dirichlet", *arguments)
                        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)
                        self.assertEqual(np.ndim(actual), 0)
                    np.testing.assert_array_equal(values, before)

    def test_square_batches_use_the_last_axis_even_when_both_axes_sum_to_one(self):
        values = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.4, 0.4]])
        alpha = [2.0, 3.0, 4.0]
        expected = analytic_density(values, alpha)
        self.assertFalse(np.isclose(expected, analytic_density(values.T, alpha)))
        np.testing.assert_allclose(
            get_prob_for_rand(values, "dirichlet", alpha), expected
        )
        uniform = np.tile([0.1, 0.2, 0.3, 0.4], (5, 1))
        self.assertAlmostEqual(
            get_prob_for_rand(uniform, "dirichlet", [1.0] * 4, 5), 6.0**5
        )

    def test_variadic_noncontiguous_and_readonly_inputs_keep_their_values(self):
        alpha = np.array([2.0, 3.0, 4.0])
        source = np.random.default_rng(15).dirichlet(alpha, size=4)
        for values in (source[::2], np.asfortranarray(source), source[::-1]):
            with self.subTest(strides=values.strides):
                before = values.copy()
                values.setflags(write=False)
                actual = get_pfunc_for_dist("dirichlet", alpha, None)(values)
                np.testing.assert_allclose(actual, analytic_density(values, alpha))
                np.testing.assert_array_equal(values, before)
        np.testing.assert_allclose(
            get_pfunc_for_dist("dirichlet", alpha)(0.2, 0.3, 0.5),
            analytic_density([0.2, 0.3, 0.5], alpha),
        )
        np.testing.assert_array_equal(alpha, [2.0, 3.0, 4.0])

    def test_incomplete_vectors_and_invalid_simplex_points_are_rejected(self):
        for values in (
            [0.5, 0.5],
            [[0.1, 0.2, 0.1]],
            [[-0.1, 0.4, 0.7]],
            np.empty((2, 0)),
        ):
            with self.subTest(values=np.shape(values)):
                with self.assertRaises(ValueError):
                    get_prob_for_rand(values, "dirichlet", [2.0, 3.0, 4.0], None)
        for alpha in ([-1.0, 2.0], [0.0, 2.0], [[1.0, 2.0]]):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    get_prob_for_rand([0.5, 0.5], "dirichlet", alpha)
        np.testing.assert_allclose(
            get_prob_for_rand([0.0, 0.5, 0.5], "dirichlet", [1.0] * 3), 2.0
        )

    def test_tracked_draws_copies_reset_and_disabled_controls_preserve_random_state(
        self,
    ):
        alpha = [2.0, 3.0, 4.0]
        for size in (None, 1, (2, 3), 0):
            with self.subTest(size=size):
                state = ProportionRand(seed=47, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(47)
                for _ in range(2):
                    state.set_rand_state("proportions", "dirichlet", alpha, size)
                    expected = reference.dirichlet(alpha, size)
                    np.testing.assert_array_equal(state.s.proportions, expected)
                    np.testing.assert_allclose(
                        state.probs[-1], analytic_density(expected, alpha)
                    )
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                duplicate = state.copy()
                state.set_rand_state("proportions", "dirichlet", alpha, size)
                duplicate.set_rand_state("proportions", "dirichlet", alpha, size)
                np.testing.assert_array_equal(
                    state.s.proportions, duplicate.s.proportions
                )
                state.reset()
                state.set_rand_state("proportions", "dirichlet", alpha, size)
                np.testing.assert_array_equal(
                    state.s.proportions,
                    np.random.default_rng(47).dirichlet(alpha, size),
                )
        disabled = ProportionRand(seed=47, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.set_rand_state("proportions", "dirichlet", alpha, 2)
        self.assertEqual(disabled.rng.bit_generator.state, before)
        self.assertEqual(disabled.probs, [])
        untracked = ProportionRand(seed=47, run_stochastic=True, track_pdf=False)
        untracked.set_rand_state("proportions", "dirichlet", alpha, 2)
        np.testing.assert_array_equal(
            untracked.s.proportions, np.random.default_rng(47).dirichlet(alpha, 2)
        )
        self.assertEqual(untracked.probs, [])

    def test_real_simulation_records_complete_vectors_and_independent_densities(self):
        model = ProportionFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 29},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(29)
        expected = np.array([reference.dirichlet([2.0, 3.0, 4.0], 2) for _ in range(4)])
        np.testing.assert_array_equal(history["r.s.proportions"], expected)
        densities = [analytic_density(x, [2.0, 3.0, 4.0]) for x in expected]
        np.testing.assert_allclose(history["r.probdens"], densities)
        np.testing.assert_allclose(
            history["s.total"], np.r_[0.0, np.cumsum(expected[1:, :, 0].sum(axis=1))]
        )
        self.assertAlmostEqual(
            result.get("tend.classify.total"), history["s.total"][-1]
        )
        np.testing.assert_allclose(expected.sum(axis=-1), 1.0)
        for x, d in zip(expected, densities):
            np.testing.assert_allclose(
                d, np.prod(stats.dirichlet.pdf(x.T, [2.0, 3.0, 4.0]))
            )


if __name__ == "__main__":
    unittest.main()
