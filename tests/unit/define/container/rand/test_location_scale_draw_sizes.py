#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for location-scale random draws with optional sample sizes.

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
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


DISTRIBUTIONS = {
    "normal": "norm",
    "laplace": "laplace",
    "logistic": "logistic",
    "gumbel": "gumbel_r",
}


def analytic_density(values, method, loc=0.0, scale=1.0):
    """Independent positive-scale formulas for the four supported families."""
    z = (np.asarray(values) - loc) / scale
    if method == "normal":
        density = np.exp(-0.5 * z * z) / (scale * np.sqrt(2 * np.pi))
    elif method == "laplace":
        density = np.exp(-np.abs(z)) / (2 * scale)
    elif method == "logistic":
        density = np.exp(-z) / (scale * (1 + np.exp(-z)) ** 2)
    else:
        density = np.exp(-z - np.exp(-z)) / scale
    return np.prod(density)


class NoiseState(State):
    normal: np.array = np.zeros((2, 3))
    normal_update = ("normal", (1.5, 0.75, (2, 3)))
    laplace: np.array = np.zeros((2, 3))
    laplace_update = ("laplace", (1.5, 0.75, (2, 3)))
    logistic: np.array = np.zeros((2, 3))
    logistic_update = ("logistic", (1.5, 0.75, (2, 3)))
    gumbel: np.array = np.zeros((2, 3))
    gumbel_update = ("gumbel", (1.5, 0.75, (2, 3)))


class NoiseRand(Rand):
    s: NoiseState = NoiseState()


class TotalState(State):
    total: np.float64 = 0.0


class NoiseFunction(Function):
    container_r = NoiseRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += sum(np.sum(getattr(self.r.s, name)) for name in DISTRIBUTIONS)

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


class TestLocationScaleDrawSizes(unittest.TestCase):
    def test_optional_sizes_preserve_the_density_of_every_supplied_value(self):
        for method in DISTRIBUTIONS:
            for size in (None, (), 1, 4, (2, 3), (2, 1, 3), 0, (2, 0)):
                with self.subTest(method=method, size=size):
                    values = getattr(np.random.default_rng(3), method)(1.5, 0.75, size)
                    before = np.copy(values)
                    expected = analytic_density(values, method, 1.5, 0.75)
                    for args in ((1.5, 0.75), (1.5, 0.75, size)):
                        np.testing.assert_allclose(
                            get_prob_for_rand(values, method, *args),
                            expected,
                            rtol=1e-12,
                            atol=0.0,
                        )
                        np.testing.assert_allclose(
                            get_pfunc_for_dist(method, *args)(values),
                            expected,
                            rtol=1e-12,
                            atol=0.0,
                        )
                    np.testing.assert_array_equal(values, before)

    def test_default_location_scale_and_variadic_controls_are_unchanged(self):
        for method in DISTRIBUTIONS:
            for args in ((), (1.5,), (1.5, 0.75)):
                with self.subTest(method=method, args=args):
                    loc, scale = (
                        (args + (1.0,)) if len(args) == 1 else (args or (0.0, 1.0))
                    )
                    values = np.array([-1.0, 0.5, 3.0])
                    expected = analytic_density(values, method, loc, scale)
                    np.testing.assert_allclose(
                        get_pfunc_for_dist(method, *args)(*values),
                        expected,
                        rtol=1e-12,
                        atol=0.0,
                    )
                    np.testing.assert_allclose(
                        get_prob_for_rand(values, method, *args),
                        expected,
                        rtol=1e-12,
                        atol=0.0,
                    )

    def test_broadcast_location_and_scale_do_not_change_inputs(self):
        loc, scale = np.array([[1.0], [-2.0]]), np.array([0.5, 1.0, 2.0])
        for method in DISTRIBUTIONS:
            with self.subTest(method=method):
                values = getattr(np.random.default_rng(9), method)(loc, scale, (2, 3))
                before = [value.copy() for value in (values, loc, scale)]
                expected = analytic_density(values, method, loc, scale)
                np.testing.assert_allclose(
                    get_prob_for_rand(values, method, loc, scale, (2, 3)),
                    expected,
                    rtol=1e-12,
                    atol=0.0,
                )
                for value, original in zip((values, loc, scale), before):
                    np.testing.assert_array_equal(value, original)

    def test_tracked_scalar_and_matrix_updates_preserve_generator_state(self):
        for method in DISTRIBUTIONS:
            for size in (None, (2, 3)):
                with self.subTest(method=method, size=size):
                    cls, name = (
                        (ExampleRand, "noise") if size is None else (NoiseRand, method)
                    )
                    state = cls(seed=11, run_stochastic=True, track_pdf=True)
                    reference = np.random.default_rng(11)
                    densities = []
                    for _ in range(3):
                        expected = getattr(reference, method)(1.5, 0.75, size)
                        state.set_rand_state(name, method, 1.5, 0.75, size)
                        np.testing.assert_array_equal(getattr(state.s, name), expected)
                        densities.append(analytic_density(expected, method, 1.5, 0.75))
                        np.testing.assert_allclose(
                            state.probs, densities, rtol=1e-12, atol=0.0
                        )
                        self.assertEqual(
                            state.rng.bit_generator.state, reference.bit_generator.state
                        )
                    np.testing.assert_allclose(
                        state.return_probdens(),
                        np.prod(densities),
                        rtol=1e-12,
                        atol=0.0,
                    )

    def test_automatic_updates_copy_reset_and_disabled_tracking_controls(self):
        state = NoiseRand(seed=17, run_stochastic=True, track_pdf=True)
        state.update_stochastic_states()
        initial = {name: getattr(state.s, name).copy() for name in DISTRIBUTIONS}
        clone = state.copy()
        for instance in (state, clone):
            instance.update_stochastic_states()
        for name in DISTRIBUTIONS:
            np.testing.assert_array_equal(
                getattr(state.s, name), getattr(clone.s, name)
            )
            self.assertFalse(
                np.shares_memory(getattr(state.s, name), getattr(clone.s, name))
            )
        state.reset()
        state.update_stochastic_states()
        for name in DISTRIBUTIONS:
            np.testing.assert_array_equal(getattr(state.s, name), initial[name])
        for stochastic, tracking in ((False, True), (True, False)):
            with self.subTest(stochastic=stochastic, tracking=tracking):
                control = NoiseRand(
                    seed=17, run_stochastic=stochastic, track_pdf=tracking
                )
                before = copy.deepcopy(control.rng.bit_generator.state)
                control.update_stochastic_states()
                self.assertEqual(control.probs, [])
                if stochastic:
                    for name in DISTRIBUTIONS:
                        np.testing.assert_array_equal(
                            getattr(control.s, name), initial[name]
                        )
                else:
                    self.assertEqual(control.rng.bit_generator.state, before)

    def test_real_simulation_records_all_four_distributions_and_joint_densities(self):
        model = NoiseFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(19)
        expected = {name: [] for name in DISTRIBUTIONS}
        densities, totals = [], []
        for _ in history.time:
            draws = {
                name: getattr(reference, name)(1.5, 0.75, (2, 3))
                for name in DISTRIBUTIONS
            }
            densities.append(
                np.prod(
                    [
                        analytic_density(value, name, 1.5, 0.75)
                        for name, value in draws.items()
                    ]
                )
            )
            totals.append(sum(np.sum(value) for value in draws.values()))
            for name, value in draws.items():
                expected[name].append(value)
        for name, values in expected.items():
            np.testing.assert_array_equal(history["r.s." + name], values)
            np.testing.assert_array_equal(getattr(model.r.s, name), np.zeros((2, 3)))
        np.testing.assert_allclose(
            history["r.probdens"], densities, rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            history["s.total"], np.concatenate(([0.0], np.cumsum(totals[1:])))
        )
        np.testing.assert_allclose(
            result["tend.classify.density"], densities[-1], rtol=1e-12, atol=0.0
        )
        self.assertEqual(model.r.probs, [])

    def test_other_distribution_mappings_are_unchanged(self):
        for name, args, reference in (
            ("beta", (2.0, 3.0), stats.beta(2.0, 3.0)),
            ("gamma", (2.0, 3.0), stats.gamma(2.0, scale=3.0)),
            ("standard_normal", (), stats.norm()),
        ):
            with self.subTest(method=name):
                values = [0.2, 0.5]
                np.testing.assert_allclose(
                    get_prob_for_rand(values, name, *args),
                    np.prod(reference.pdf(values)),
                    rtol=1e-12,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
