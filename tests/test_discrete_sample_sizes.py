#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for sample sizes in discrete probability tracking.

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
from scipy import special, stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


LAWS = {
    "poisson": ("poisson", (2.0,)),
    "binomial": ("binom", (5, 0.3)),
    "negative_binomial": ("nbinom", (2.5, 0.4)),
    "geometric": ("geom", (0.3,)),
    "zipf": ("zipf", (3.0,)),
    "logseries": ("logser", (0.5,)),
}


def analytic_mass(name, value):
    if name == "poisson":
        return math.exp(-2.0) * 2.0**value / math.factorial(value)
    if name == "binomial":
        return math.comb(5, value) * 0.3**value * 0.7 ** (5 - value)
    if name == "negative_binomial":
        return (
            math.gamma(value + 2.5)
            / (math.factorial(value) * math.gamma(2.5))
            * 0.4**2.5
            * 0.6**value
        )
    if value == 0:
        return 0.0
    if name == "geometric":
        return 0.3 * 0.7 ** (value - 1)
    if name == "zipf":
        return value**-3.0 / special.zeta(3.0, 1.0)
    return -(0.5**value) / (value * math.log1p(-0.5))


class CountState(State):
    poisson: np.array = np.zeros((2, 2), dtype=np.int64)
    poisson_update = ("poisson", (2.0, (2, 2)))
    binomial: np.array = np.zeros((2, 2), dtype=np.int64)
    binomial_update = ("binomial", (5, 0.3, (2, 2)))
    negative_binomial: np.array = np.zeros((2, 2), dtype=np.int64)
    negative_binomial_update = ("negative_binomial", (2.5, 0.4, (2, 2)))
    geometric: np.array = np.zeros((2, 2), dtype=np.int64)
    geometric_update = ("geometric", (0.3, (2, 2)))
    zipf: np.array = np.zeros((2, 2), dtype=np.int64)
    zipf_update = ("zipf", (3.0, (2, 2)))
    logseries: np.array = np.zeros((2, 2), dtype=np.int64)
    logseries_update = ("logseries", (0.5, (2, 2)))


class CountRand(Rand):
    s: CountState = CountState()


class TotalState(State):
    total: np.int64 = 0


class CountFunction(Function):
    container_r = CountRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += sum(np.sum(self.r.s[name]) for name in LAWS)

    def classify(self, **kwargs):
        return {"total": self.s.total, "mass": self.r.return_probdens()}


class TestDiscreteSampleSizes(unittest.TestCase):
    def test_sizes_do_not_shift_discrete_support_or_change_independent_masses(self):
        for name, (scipy_name, parameters) in LAWS.items():
            for size in (None, (), 1, 4, (2, 3), (2, 1, 3), 0, (2, 0)):
                with self.subTest(name=name, size=size):
                    samples = getattr(np.random.default_rng(12), name)(
                        *parameters, size
                    )
                    before = np.copy(samples)
                    expected = np.prod(
                        getattr(stats, scipy_name).pmf(samples, *parameters)
                    )
                    for arguments in (parameters, (*parameters, size)):
                        actual = get_prob_for_rand(samples, name, *arguments)
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-13, atol=0.0
                        )
                    np.testing.assert_array_equal(samples, before)

    def test_analytic_masses_variadic_values_defaults_and_invalid_arguments(self):
        for name, (_, parameters) in LAWS.items():
            with self.subTest(name=name):
                mass = get_pfunc_for_dist(name, *parameters, 4)
                for value in range(4):
                    np.testing.assert_allclose(
                        mass(value), analytic_mass(name, value), rtol=1e-13, atol=0.0
                    )
                expected = np.prod([analytic_mass(name, k) for k in (1, 2, 3)])
                np.testing.assert_allclose(
                    mass(1, 2, 3), expected, rtol=1e-13, atol=0.0
                )
                self.assertEqual(mass(-1), 0.0)
                self.assertEqual(mass(0.5), 0.0)
                with self.assertRaises(TypeError):
                    get_pfunc_for_dist(name, *parameters, 4, "extra")
                if name != "poisson":
                    with self.assertRaises(TypeError):
                        get_pfunc_for_dist(name)
        np.testing.assert_allclose(
            get_prob_for_rand(1, "poisson"), math.exp(-1.0), rtol=1e-13
        )

    def test_parameter_arrays_broadcast_without_mutation(self):
        for name, (scipy_name, parameters) in LAWS.items():
            with self.subTest(name=name):
                if len(parameters) == 1:
                    params = (np.array([parameters[0], parameters[0] * 1.2]),)
                else:
                    params = (
                        np.array([[parameters[0]], [parameters[0] + 2]]),
                        np.array([0.2, 0.5, 0.8]),
                    )
                shape = (2, 2) if len(params) == 1 else (2, 3)
                original = copy.deepcopy(params)
                samples = getattr(np.random.default_rng(7), name)(*params, shape)
                expected = np.prod(getattr(stats, scipy_name).pmf(samples, *params))
                np.testing.assert_allclose(
                    get_prob_for_rand(samples, name, *params, shape),
                    expected,
                    rtol=1e-13,
                    atol=0.0,
                )
                for actual, before in zip(params, original):
                    np.testing.assert_array_equal(actual, before)

    def test_tracked_updates_keep_samples_generator_state_and_copies(self):
        for name, (scipy_name, parameters) in LAWS.items():
            for size in (None, (2, 2)):
                with self.subTest(name=name, size=size):
                    state = CountRand(seed=23, run_stochastic=True, track_pdf=True)
                    reference = np.random.default_rng(23)
                    for _ in range(3):
                        values = getattr(reference, name)(*parameters, size)
                        state.set_rand_state(name, name, *parameters, size)
                        np.testing.assert_array_equal(state.s[name], values)
                        expected = np.prod(
                            getattr(stats, scipy_name).pmf(values, *parameters)
                        )
                        np.testing.assert_allclose(
                            state.probs[-1], expected, rtol=1e-13, atol=0.0
                        )
                        self.assertEqual(
                            state.rng.bit_generator.state, reference.bit_generator.state
                        )
        state = CountRand(seed=17, run_stochastic=True, track_pdf=True)
        state.update_stochastic_states()
        clone = state.copy()
        state.update_stochastic_states()
        clone.update_stochastic_states()
        for name in LAWS:
            np.testing.assert_array_equal(state.s[name], clone.s[name])
            self.assertIsNot(state.s[name], clone.s[name])
        state.reset()
        state.update_stochastic_states()
        reference = CountRand(seed=17, run_stochastic=True, track_pdf=True)
        reference.update_stochastic_states()
        np.testing.assert_array_equal(state.probs, reference.probs)

    def test_disabled_and_scalar_default_controls_keep_generator_behavior(self):
        for stochastic, tracked in ((False, True), (True, False)):
            with self.subTest(stochastic=stochastic, tracked=tracked):
                state = CountRand(seed=19, run_stochastic=stochastic, track_pdf=tracked)
                before = copy.deepcopy(state.rng.bit_generator.state)
                state.update_stochastic_states()
                self.assertEqual(state.probs, [])
                if not stochastic:
                    self.assertEqual(state.rng.bit_generator.state, before)
        scalar = ExampleRand(seed=19, run_stochastic=True, track_pdf=True)
        scalar.set_rand_state("noise", "poisson")
        expected = np.random.default_rng(19).poisson()
        self.assertEqual(scalar.s.noise, expected)
        np.testing.assert_allclose(scalar.probs, [stats.poisson.pmf(expected, 1.0)])
        np.testing.assert_allclose(
            get_prob_for_rand(0.0, "normal", 0.0, 1.0), 1.0 / np.sqrt(2.0 * np.pi)
        )

    def test_actual_simulation_records_all_discrete_draws_and_joint_masses(self):
        model = CountFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        expected = {name: [] for name in LAWS}
        masses, increments = [], []
        for _ in history.time:
            factors, increment = [], 0
            for name, (scipy_name, parameters) in LAWS.items():
                value = getattr(reference, name)(*parameters, (2, 2))
                expected[name].append(value)
                factors.append(
                    np.prod(getattr(stats, scipy_name).pmf(value, *parameters))
                )
                increment += value.sum()
            masses.append(np.prod(factors))
            increments.append(increment)
        for name in LAWS:
            np.testing.assert_array_equal(history["r.s." + name], expected[name])
            np.testing.assert_array_equal(model.r.s[name], np.zeros((2, 2)))
        np.testing.assert_allclose(history["r.probdens"], masses, rtol=1e-13, atol=0.0)
        totals = np.concatenate(([0], np.cumsum(increments[1:])))
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_allclose(
            result["tend.classify.mass"], masses[-1], rtol=1e-13, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
