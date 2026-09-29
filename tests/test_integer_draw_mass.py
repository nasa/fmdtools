#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for the joint mass of integer random draws.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import copy
from fractions import Fraction
import unittest

import numpy as np
from scipy import stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    calc_prob_for_integers,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class IntegerState(State):
    noise: np.array = np.zeros((2, 2), dtype=np.int32)
    noise_update = ("integers", (-2, 3, (2, 2), np.int32, True))


class IntegerRand(Rand):
    s: IntegerState = IntegerState()


class TotalState(State):
    total: np.int64 = 0


class IntegerFunction(Function):
    container_r = IntegerRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.noise)

    def classify(self, **kwargs):
        return {"total": self.s.total, "mass": self.r.return_probdens()}


class TestIntegerDrawMass(unittest.TestCase):
    """Run the regression cases with standard unittest discovery."""

    def test_joint_mass_counts_all_draws_and_accepts_numpy_sampling_arguments(self):
        for endpoint in [False, True]:
            for shape in [(), (1,), (3,), (2, 3), (2, 1, 3), (0,), (2, 0)]:
                with self.subTest(endpoint=endpoint, shape=shape):
                    values = np.random.default_rng(7).integers(
                        -2, 3, shape, np.int32, endpoint
                    )
                    expected = float(Fraction(1, 6 if endpoint else 5) ** values.size)
                    before = values.copy()
                    args = (-2, 3, shape, np.int32, endpoint)
                    np.testing.assert_allclose(
                        calc_prob_for_integers(values, *args),
                        expected,
                        rtol=1e-06,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        get_prob_for_rand(values, "integers", *args),
                        expected,
                        rtol=1e-06,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        get_pfunc_for_dist("integers", *args)(values),
                        expected,
                        rtol=1e-06,
                        atol=1e-12,
                    )
                    np.testing.assert_array_equal(values, before)

    def test_array_and_variadic_dispatch_agree_without_a_size_argument(self):
        for shape in [(3,), (2, 3)]:
            with self.subTest(shape=shape):
                values = np.arange(np.prod(shape)).reshape(shape)
                expected = 8.0**-values.size
                self.assertEqual(get_prob_for_rand(values, "integers", 8), expected)
                self.assertEqual(
                    get_pfunc_for_dist("integers", 0, 8)(*values.ravel()), expected
                )
                self.assertEqual(calc_prob_for_integers(values, 0, 8), expected)

    def test_broadcast_bounds_have_one_mass_factor_per_value(self):
        for endpoint in [False, True]:
            with self.subTest(endpoint=endpoint):
                low = np.array([-2, 0, 1])
                high = np.array([[4], [6]])
                values = np.random.default_rng(9).integers(low, high, endpoint=endpoint)
                before = low.copy(), high.copy(), values.copy()
                expected = np.prod(stats.randint.pmf(values, low, high + endpoint))
                actual = get_prob_for_rand(
                    values, "integers", low, high, None, np.int64, endpoint
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-06, atol=1e-12)
                for actual_array, original in zip((low, high, values), before):
                    np.testing.assert_array_equal(actual_array, original)

    def test_impossible_values_have_zero_probability_mass(self):
        for value in [-1, 4, 0.5, np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                self.assertEqual(get_prob_for_rand([0, value], "integers", 0, 4), 0.0)

    def test_upper_bound_is_included_only_when_requested(self):
        for endpoint, expected in [(False, 0.0), (True, 0.04)]:
            with self.subTest(endpoint=endpoint, expected=expected):
                np.testing.assert_allclose(
                    get_prob_for_rand([0, 4], "integers", 0, 4, 2, np.int64, endpoint),
                    expected,
                    rtol=1e-06,
                    atol=1e-12,
                )

    def test_integer_bounds_are_not_rounded_or_overflowed(self):
        for low, high, dtype, endpoint in [
            (-(2**63), 2**63 - 1, np.int64, True),
            (0, 2**64 - 1, np.uint64, True),
            (0, 2**64, np.uint64, False),
            (2**64 - 4, 2**64, np.uint64, False),
        ]:
            with self.subTest(low=low, high=high, dtype=dtype, endpoint=endpoint):
                values = np.random.default_rng(12).integers(
                    low, high, 3, dtype, endpoint
                )
                expected = float(Fraction(1, high - low + int(endpoint)) ** values.size)
                actual = get_prob_for_rand(
                    values, "integers", low, high, 3, dtype, endpoint
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
                upper = high if endpoint else high - 1
                np.testing.assert_allclose(
                    calc_prob_for_integers(
                        np.array([upper], dtype=dtype), low, high, None, dtype, endpoint
                    ),
                    float(Fraction(1, high - low + int(endpoint))),
                    rtol=1e-14,
                    atol=0.0,
                )
                if low > 0:
                    self.assertEqual(
                        calc_prob_for_integers(
                            np.array([low - 1], dtype=dtype), low, high
                        ),
                        0.0,
                    )

    def test_tracking_preserves_values_dtype_and_random_generator_state(self):
        for shape in [None, (), 3, (2, 2), 0]:
            with self.subTest(shape=shape):
                state = IntegerRand(seed=23, run_stochastic=True, track_pdf=True)
                reference = np.random.default_rng(23)
                for _ in range(3):
                    expected = reference.integers(8, None, shape, np.int64)
                    state.set_rand_state("noise", "integers", 8, None, shape, np.int64)
                    np.testing.assert_array_equal(state.s.noise, expected)
                    self.assertEqual(state.probs[-1], 8.0 ** (-np.size(expected)))
                    self.assertEqual(
                        state.rng.bit_generator.state, reference.bit_generator.state
                    )
                self.assertEqual(
                    state.return_probdens(), 8.0 ** (-3 * np.size(expected))
                )

    def test_auto_updates_copy_reset_and_disabled_controls(self):
        state = IntegerRand(seed=13, run_stochastic=True, track_pdf=True)
        state.update_stochastic_states()
        clone = state.copy()
        state.update_stochastic_states()
        clone.update_stochastic_states()
        np.testing.assert_array_equal(state.s.noise, clone.s.noise)
        self.assertIsNot(clone.s.noise, state.s.noise)
        np.testing.assert_allclose(state.probs, [6.0 ** (-4)], rtol=1e-06, atol=1e-12)
        state.reset()
        state.update_stochastic_states()
        np.testing.assert_array_equal(
            state.s.noise,
            np.random.default_rng(13).integers(-2, 3, (2, 2), np.int32, True),
        )
        disabled = IntegerRand(seed=13, run_stochastic=False, track_pdf=True)
        before = copy.deepcopy(disabled.rng.bit_generator.state)
        disabled.update_stochastic_states()
        np.testing.assert_array_equal(disabled.s.noise, np.zeros((2, 2)))
        self.assertEqual(disabled.probs, [])
        self.assertEqual(disabled.rng.bit_generator.state, before)
        scalar = ExampleRand(seed=13, run_stochastic=True, track_pdf=True)
        scalar.set_rand_state("noise", "integers", 4)
        self.assertEqual(scalar.probs, [0.25])

    def test_actual_matrix_noise_simulation_records_each_draw_and_joint_mass(self):
        model = IntegerFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 19},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(19)
        expected = np.array(
            [reference.integers(-2, 3, (2, 2), np.int32, True) for _ in history.time]
        )
        np.testing.assert_array_equal(history["r.s.noise"], expected)
        np.testing.assert_allclose(history["r.probdens"], 6.0**-4)
        totals = np.concatenate(([0], np.cumsum(expected[1:].sum(axis=(1, 2)))))
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_allclose(
            result["tend.classify.mass"], 6.0 ** (-4), rtol=1e-06, atol=1e-12
        )
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))
        self.assertEqual(model.r.probs, [])

    def test_empty_or_reversed_integer_intervals_are_rejected(self):
        for low, high in [(2, 2), (3, 2)]:
            with self.subTest(low=low, high=high):
                with self.assertRaisesRegex(ValueError, "interval"):
                    calc_prob_for_integers([2], low, high)

    def test_closed_single_integer_interval_has_unit_mass(self):
        self.assertEqual(
            calc_prob_for_integers([7, 7, 7], 7, 7, 3, np.int64, True), 1.0
        )
        self.assertEqual(
            calc_prob_for_integers([7, 6, 7], 7, 7, 3, np.int64, True), 0.0
        )

    def test_numpy_uint64_and_object_bounds_retain_exact_endpoints(self):
        low = np.array([2**64 - 4], dtype=np.uint64)
        high = np.array([2**64], dtype=object)
        samples = np.random.default_rng(1).integers(
            low, high, size=(2, 1), dtype=np.uint64
        )
        self.assertEqual(
            get_prob_for_rand(samples, "integers", low, high, (2, 1), np.uint64),
            1.0 / 16.0,
        )
        self.assertEqual(
            calc_prob_for_integers(np.array([2**64 - 5], dtype=np.uint64), low, high),
            0.0,
        )
        np.testing.assert_array_equal(low, [2**64 - 4])
        np.testing.assert_array_equal(high, [2**64])


if __name__ == "__main__":
    unittest.main()
