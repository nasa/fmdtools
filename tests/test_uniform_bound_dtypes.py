#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for uniform bounds with fixed-width numeric types.

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

import math
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_prob_for_rand, get_uniform_pdf
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation

LOW = np.array([-120, -100], dtype=np.int8)
HIGH = np.array([120, 100], dtype=np.int8)


class UniformState(State):
    noise: np.array = np.zeros((2, 2))
    noise_update = ("uniform", (LOW, HIGH, (2, 2)))


class UniformRand(Rand):
    s: UniformState = UniformState()


class TotalState(State):
    total: np.float64 = 0.0


class UniformFunction(Function):
    container_r = UniformRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.noise.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


def uniform_density(values, low, high):
    values, low, high = np.broadcast_arrays(
        values, np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    )
    return math.prod(
        1.0 / (b - a) if a <= x <= b else 0.0
        for x, a, b in zip(values.flat, low.flat, high.flat)
    )


class TestUniformBoundDtypes(unittest.TestCase):
    def test_fixed_width_scalar_bounds_preserve_positive_interval_widths(self):
        bounds = []
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            info = np.iinfo(dtype)
            bounds.append((dtype(info.min + 1), dtype(info.max - 1)))
        bounds.append((np.float16(-60000), np.float16(60000)))
        for low, high in bounds:
            with self.subTest(dtype=type(low).__name__):
                width = float(high) - float(low)
                with np.errstate(over="raise", invalid="raise"):
                    actual = get_prob_for_rand(0.0, "uniform", low, high)
                np.testing.assert_allclose(actual, 1 / width, rtol=1e-14, atol=0)
                self.assertTrue(np.isfinite(actual) and actual > 0)
                np.testing.assert_allclose(actual * width, 1.0, rtol=1e-14)

    def test_broadcast_bounds_shapes_and_generation_sizes_preserve_density(self):
        cases = [
            (LOW, HIGH),
            (LOW.tolist(), HIGH.tolist()),
            (tuple(LOW), tuple(HIGH)),
            (
                np.array([[-120], [-100]], dtype=np.int8),
                np.array([100, 110, 120], dtype=np.int8),
            ),
        ]
        for low, high in cases:
            shape = np.broadcast_shapes(np.shape(low), np.shape(high))
            for size in (None, (3, *shape), (0, *shape)):
                with self.subTest(
                    shape=shape, size=size, bound_type=type(low).__name__
                ):
                    before = np.copy(low), np.copy(high)
                    draws = np.random.default_rng(17).uniform(low, high, size)
                    expected = uniform_density(draws, low, high)
                    actual = get_prob_for_rand(draws, "uniform", low, high, size)
                    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
                    self.assertEqual(np.ndim(actual), 0)
                    for values, saved in zip((low, high), before):
                        np.testing.assert_array_equal(values, saved)

    def test_support_readonly_inputs_defaults_and_variadic_draws(self):
        low, high = LOW.copy(), HIGH.copy()
        low.setflags(write=False)
        high.setflags(write=False)
        pdf = get_uniform_pdf(low, high)
        self.assertEqual(pdf([-121.0, 0.0]), 0.0)
        self.assertEqual(pdf([0.0, 101.0]), 0.0)
        np.testing.assert_allclose(
            pdf([0.0, 0.0]), 1 / (240.0 * 200.0), rtol=1e-14, atol=0
        )
        self.assertEqual(get_uniform_pdf()(0.25, 0.75), 1.0)
        self.assertEqual(get_uniform_pdf(5.0, 6.0)(5.5), 1.0)
        self.assertEqual(get_uniform_pdf(5.0, 6.0)(7.0), 0.0)
        np.testing.assert_array_equal(low, LOW)
        np.testing.assert_array_equal(high, HIGH)

    def test_tracked_draws_retain_rng_state_copy_and_reset(self):
        rand = UniformRand(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        masses = []
        for _ in range(3):
            rand.set_rand_state("noise", "uniform", LOW, HIGH, (2, 2))
            expected = reference.uniform(LOW, HIGH, (2, 2))
            np.testing.assert_array_equal(rand.s.noise, expected)
            mass = uniform_density(expected, LOW, HIGH)
            masses.append(mass)
            np.testing.assert_allclose(rand.probs[-1], mass, rtol=1e-14, atol=0)
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        np.testing.assert_allclose(
            rand.return_probdens(), math.prod(masses), rtol=1e-14, atol=0
        )
        clone = rand.copy()
        for item in (rand, clone):
            item.set_rand_state("noise", "uniform", LOW, HIGH, (2, 2))
        np.testing.assert_array_equal(rand.s.noise, clone.s.noise)
        self.assertEqual(rand.probs, clone.probs)
        rand.reset()
        rand.set_rand_state("noise", "uniform", LOW, HIGH, (2, 2))
        np.testing.assert_array_equal(
            rand.s.noise, np.random.default_rng(17).uniform(LOW, HIGH, (2, 2))
        )

    def test_actual_simulation_records_finite_densities_without_changing_draws(self):
        model = UniformFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        expected = np.array([rng.uniform(LOW, HIGH, (2, 2)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        np.testing.assert_allclose(
            history["r.probdens"],
            [uniform_density(x, LOW, HIGH) for x in expected],
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(
            history["s.total"], np.r_[0.0, np.cumsum(expected[1:].sum(axis=(1, 2)))]
        )
        self.assertAlmostEqual(result["tend.classify.total"], history["s.total"][-1])
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
