#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for array-like triangular distribution parameters.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_prob_for_rand, get_triangular_pdf
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def triangular_mass(values, left, mode, right):
    """Multiply the elementary piecewise triangular densities."""
    arrays = np.broadcast_arrays(
        values,
        np.asarray(left, dtype=float),
        np.asarray(mode, dtype=float),
        np.asarray(right, dtype=float),
    )
    densities = []
    for value, low, peak, high in zip(*(a.flat for a in arrays)):
        if value < low or value > high:
            density = 0.0
        elif value == peak:
            density = 2.0 / (high - low)
        elif value < peak:
            density = 2.0 * (value - low) / ((high - low) * (peak - low))
        else:
            density = 2.0 * (high - value) / ((high - low) * (high - peak))
        densities.append(density)
    return math.prod(densities)


class TriangularState(State):
    noise: np.array = np.zeros((2, 2))
    noise_update = ("triangular", ([0.0, 1.0], [1.0, 2.0], [2.0, 4.0], (2, 2)))


class TriangularRand(Rand):
    s: TriangularState = TriangularState()


class TotalState(State):
    total: np.float64 = 0.0


class TriangularFunction(Function):
    container_r = TriangularRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.noise.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestTriangularArraylikeParameters(unittest.TestCase):
    def test_broadcast_parameters_and_draw_sizes_match_piecewise_density(self):
        for wrap in (list, tuple, np.asarray):
            for size in (None, 2, (3, 2), (1, 2, 2), (0, 2)):
                for left, mode, right in (
                    ([0.0, 1.0], [1.0, 2.0], [2.0, 4.0]),
                    ([0.0, 1.0], [0.0, 4.0], [2.0, 4.0]),
                ):
                    with self.subTest(wrap=wrap.__name__, size=size, mode=mode):
                        params = tuple(wrap(p) for p in (left, mode, right))
                        before = copy.deepcopy(params)
                        draws = np.random.default_rng(13).triangular(*params, size)
                        expected = triangular_mass(draws, *params)
                        actual = get_prob_for_rand(draws, "triangular", *params, size)
                        self.assertAlmostEqual(actual, expected, places=13)
                        self.assertEqual(np.ndim(actual), 0)
                        for original, saved in zip(params, before):
                            np.testing.assert_array_equal(original, saved)

    def test_mixed_scalars_and_nested_array_parameters_broadcast(self):
        parameters = (
            (0.0, [1.0, 2.0], 4.0),
            ([0.0, 1.0], 2.0, [3.0, 4.0]),
            ([[0.0], [1.0]], [1.0, 2.0, 3.0], [[4.0], [5.0]]),
        )
        for params in parameters:
            with self.subTest(params=params):
                draws = np.random.default_rng(21).triangular(*params)
                self.assertAlmostEqual(
                    get_triangular_pdf(*params)(draws),
                    triangular_mass(draws, *params),
                    places=13,
                )

    def test_fixed_width_integer_bounds_are_promoted_before_subtraction(self):
        for dtype in (np.int8, np.int16, np.int32):
            with self.subTest(dtype=dtype):
                left = np.array([-120, -100], dtype=dtype)
                mode = np.array([0, 10], dtype=dtype)
                right = np.array([120, 100], dtype=dtype)
                self.assertAlmostEqual(
                    get_triangular_pdf(left, mode, right)([0.0, 10.0]),
                    4.0 / (240.0 * 200.0),
                    places=15,
                )

    def test_scalar_support_and_variadic_density_inputs_are_unchanged(self):
        density = get_triangular_pdf(0.0, 1.0, 2.0)
        for values in ((-1.0,), (0.0,), (1.0,), (2.0,), (3.0,), (0.5, 1.5)):
            with self.subTest(values=values):
                self.assertAlmostEqual(
                    density(*values), triangular_mass(values, 0.0, 1.0, 2.0), places=14
                )

    def test_tracked_updates_preserve_samples_generator_state_copy_and_reset(self):
        parameters = ([0.0, 1.0], [1.0, 2.0], [2.0, 4.0], (2, 2))
        rand = TriangularRand(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        for _ in range(3):
            expected = reference.triangular(*parameters)
            rand.set_rand_state("noise", "triangular", *parameters)
            np.testing.assert_array_equal(rand.s.noise, expected)
            self.assertAlmostEqual(
                rand.probs[-1], triangular_mass(expected, *parameters[:3])
            )
            self.assertEqual(
                rand.rng.bit_generator.state, reference.bit_generator.state
            )
        clone = rand.copy()
        rand.set_rand_state("noise", "triangular", *parameters)
        clone.set_rand_state("noise", "triangular", *parameters)
        np.testing.assert_array_equal(rand.s.noise, clone.s.noise)
        self.assertEqual(rand.probs, clone.probs)
        rand.reset()
        self.assertEqual(rand.probs, [])
        rand.set_rand_state("noise", "triangular", *parameters)
        np.testing.assert_array_equal(
            rand.s.noise, np.random.default_rng(17).triangular(*parameters)
        )

    def test_real_simulation_records_densities_for_array_parameter_draws(self):
        model = TriangularFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        generator = np.random.default_rng(31)
        parameters = ([0.0, 1.0], [1.0, 2.0], [2.0, 4.0], (2, 2))
        expected = np.array([generator.triangular(*parameters) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        np.testing.assert_allclose(
            history["r.probdens"],
            [triangular_mass(x, *parameters[:3]) for x in expected],
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            history["s.total"], np.r_[0.0, np.cumsum(expected[1:].sum(axis=(1, 2)))]
        )
        self.assertAlmostEqual(result["tend.classify.total"], history["s.total"][-1])
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
