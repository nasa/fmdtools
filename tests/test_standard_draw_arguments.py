#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for standard normal and Cauchy sampling signatures.

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
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_prob_for_rand,
    get_pfunc_for_dist,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def density(values, name):
    values = np.asarray(values, dtype=np.float64)
    if name == "standard_normal":
        return np.prod(np.exp(-(values**2) / 2) / np.sqrt(2 * np.pi))
    return np.prod(1 / (np.pi * (1 + values**2)))


class NoiseState(State):
    gaussian: np.array = np.zeros((2, 3))
    cauchy: np.array = np.zeros((2, 3))
    gaussian_update = ("standard_normal", ((2, 3),))
    cauchy_update = ("standard_cauchy", ((2, 3),))


class NoiseRand(Rand):
    s: NoiseState = NoiseState()


class TotalState(State):
    total: np.float64 = 0.0


class NoisyFunction(Function):
    container_r = NoiseRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.r.s.gaussian.sum() + self.r.s.cauchy.sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestStandardDrawArguments(unittest.TestCase):
    """Sampling options must not shift standard-distribution densities."""

    def test_scalar_and_array_sizes_preserve_standard_densities(self):
        for name in ("standard_normal", "standard_cauchy"):
            for size in (None, (), 1, 4, (2, 3), (2, 1, 3), 0, (2, 0)):
                with self.subTest(name=name, size=size):
                    values = getattr(np.random.default_rng(14), name)(size)
                    before = np.copy(values)
                    expected = density(values, name)
                    for args in ((), (size,)):
                        np.testing.assert_allclose(
                            get_prob_for_rand(values, name, *args),
                            expected,
                            rtol=1e-13,
                            atol=0.0,
                        )
                        np.testing.assert_allclose(
                            get_pfunc_for_dist(name, *args)(values),
                            expected,
                            rtol=1e-13,
                            atol=0.0,
                        )
                    np.testing.assert_array_equal(values, before)

    def test_normal_dtype_and_output_buffer_do_not_become_parameters(self):
        for dtype in (np.float32, np.float64):
            for size in (None, (2, 3)):
                with self.subTest(dtype=dtype, size=size):
                    out = np.empty((2, 3), dtype=dtype)
                    values = np.random.default_rng(5).standard_normal(size, dtype, out)
                    before = out.copy()
                    args = (size, dtype, out)
                    np.testing.assert_allclose(
                        get_prob_for_rand(values, "standard_normal", *args),
                        density(values, "standard_normal"),
                        rtol=1e-13,
                        atol=0.0,
                    )
                    np.testing.assert_array_equal(out, before)
                    state = NoiseRand(seed=5, run_stochastic=True, track_pdf=True)
                    buffer = np.empty((2, 3), dtype=dtype)
                    state.set_rand_state(
                        "gaussian", "standard_normal", size, dtype, buffer
                    )
                    np.testing.assert_array_equal(state.s.gaussian, before)
                    self.assertEqual(state.s.gaussian.dtype, dtype)
                    np.testing.assert_allclose(
                        state.probs[-1], density(before, "standard_normal")
                    )

    def test_tracked_updates_preserve_generator_state_copy_and_reset(self):
        for name, field in (
            ("standard_normal", "gaussian"),
            ("standard_cauchy", "cauchy"),
        ):
            for size in (None, (2, 3), 0):
                with self.subTest(name=name, size=size):
                    state = NoiseRand(seed=23, run_stochastic=True, track_pdf=True)
                    reference = np.random.default_rng(23)
                    for _ in range(3):
                        expected = getattr(reference, name)(size)
                        state.set_rand_state(field, name, size)
                        np.testing.assert_array_equal(state.s[field], expected)
                        np.testing.assert_allclose(
                            state.probs[-1], density(expected, name)
                        )
                        self.assertEqual(
                            state.rng.bit_generator.state, reference.bit_generator.state
                        )
                    clone = state.copy()
                    for instance in (state, clone):
                        instance.set_rand_state(field, name, size)
                    np.testing.assert_array_equal(state.s[field], clone.s[field])
                    self.assertEqual(state.probs, clone.probs)
                    state.reset()
                    state.set_rand_state(field, name, size)
                    np.testing.assert_array_equal(
                        state.s[field], getattr(np.random.default_rng(23), name)(size)
                    )

    def test_standard_support_variadic_and_disabled_controls_are_unchanged(self):
        for name in ("standard_normal", "standard_cauchy"):
            with self.subTest(name=name):
                pdf = get_pfunc_for_dist(name)
                np.testing.assert_allclose(
                    pdf(-1.0, 0.0, 1.0), density([-1.0, 0.0, 1.0], name)
                )
                self.assertEqual(pdf(np.inf), 0.0)
                self.assertEqual(pdf(-np.inf), 0.0)
                self.assertTrue(np.isnan(pdf(np.nan)))
                disabled = ExampleRand(seed=11, run_stochastic=False, track_pdf=True)
                before = copy.deepcopy(disabled.rng.bit_generator.state)
                disabled.set_rand_state("noise", name)
                self.assertEqual(disabled.probs, [])
                self.assertEqual(disabled.rng.bit_generator.state, before)
                untracked = ExampleRand(seed=11, run_stochastic=True, track_pdf=False)
                untracked.set_rand_state("noise", name)
                self.assertEqual(
                    untracked.s.noise, getattr(np.random.default_rng(11), name)()
                )
                self.assertEqual(untracked.probs, [])

    def test_real_simulation_records_both_standard_distributions(self):
        model = NoisyFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        gaussian, cauchy, masses = [], [], []
        for _ in history.time:
            normal_values = rng.standard_normal((2, 3))
            cauchy_values = rng.standard_cauchy((2, 3))
            gaussian.append(normal_values)
            cauchy.append(cauchy_values)
            masses.append(
                density(normal_values, "standard_normal")
                * density(cauchy_values, "standard_cauchy")
            )
        np.testing.assert_array_equal(history["r.s.gaussian"], gaussian)
        np.testing.assert_array_equal(history["r.s.cauchy"], cauchy)
        np.testing.assert_allclose(history["r.probdens"], masses, rtol=1e-13)
        totals = np.r_[
            0.0,
            np.cumsum(
                np.asarray(gaussian)[1:].sum(axis=(1, 2))
                + np.asarray(cauchy)[1:].sum(axis=(1, 2))
            ),
        ]
        np.testing.assert_allclose(history["s.total"], totals)
        self.assertAlmostEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.gaussian, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
