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

import numpy as np
import pytest
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


@pytest.mark.parametrize("shape", [(), (1,), (3,), (2, 3), (2, 1, 3), (0,), (2, 0)])
@pytest.mark.parametrize("endpoint", [False, True])
def test_joint_mass_counts_all_draws_and_accepts_numpy_sampling_arguments(
    shape, endpoint
):
    values = np.random.default_rng(7).integers(-2, 3, shape, np.int32, endpoint)
    expected = float(Fraction(1, 6 if endpoint else 5) ** values.size)
    before = values.copy()
    args = (-2, 3, shape, np.int32, endpoint)
    assert calc_prob_for_integers(values, *args) == pytest.approx(expected)
    assert get_prob_for_rand(values, "integers", *args) == pytest.approx(expected)
    assert get_pfunc_for_dist("integers", *args)(values) == pytest.approx(expected)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_array_and_variadic_dispatch_agree_without_a_size_argument(shape):
    values = np.arange(np.prod(shape)).reshape(shape)
    expected = 8.0**-values.size
    assert get_prob_for_rand(values, "integers", 8) == expected
    assert get_pfunc_for_dist("integers", 0, 8)(*values.ravel()) == expected
    assert calc_prob_for_integers(values, 0, 8) == expected


@pytest.mark.parametrize("endpoint", [False, True])
def test_broadcast_bounds_have_one_mass_factor_per_value(endpoint):
    low = np.array([-2, 0, 1])
    high = np.array([[4], [6]])
    values = np.random.default_rng(9).integers(low, high, endpoint=endpoint)
    before = low.copy(), high.copy(), values.copy()
    expected = np.prod(stats.randint.pmf(values, low, high + endpoint))
    actual = get_prob_for_rand(values, "integers", low, high, None, np.int64, endpoint)
    assert actual == pytest.approx(expected)
    for actual_array, original in zip((low, high, values), before):
        np.testing.assert_array_equal(actual_array, original)


@pytest.mark.parametrize("value", [-1, 4, 0.5, np.nan, np.inf, -np.inf])
def test_impossible_values_have_zero_probability_mass(value):
    assert get_prob_for_rand([0, value], "integers", 0, 4) == 0.0


@pytest.mark.parametrize("endpoint,expected", [(False, 0.0), (True, 0.04)])
def test_upper_bound_is_included_only_when_requested(endpoint, expected):
    assert get_prob_for_rand(
        [0, 4], "integers", 0, 4, 2, np.int64, endpoint
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "low,high,dtype,endpoint",
    [
        (-(2**63), 2**63 - 1, np.int64, True),
        (0, 2**64 - 1, np.uint64, True),
        (0, 2**64, np.uint64, False),
        (2**64 - 4, 2**64, np.uint64, False),
    ],
)
def test_integer_bounds_are_not_rounded_or_overflowed(low, high, dtype, endpoint):
    values = np.random.default_rng(12).integers(low, high, 3, dtype, endpoint)
    expected = float(Fraction(1, high - low + int(endpoint)) ** values.size)
    actual = get_prob_for_rand(values, "integers", low, high, 3, dtype, endpoint)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)
    upper = high if endpoint else high - 1
    assert calc_prob_for_integers(
        np.array([upper], dtype=dtype), low, high, None, dtype, endpoint
    ) == pytest.approx(
        float(Fraction(1, high - low + int(endpoint))), rel=1e-14, abs=0.0
    )
    if low > 0:
        assert (
            calc_prob_for_integers(np.array([low - 1], dtype=dtype), low, high) == 0.0
        )


@pytest.mark.parametrize("shape", [None, (), 3, (2, 2), 0])
def test_tracking_preserves_values_dtype_and_random_generator_state(shape):
    state = IntegerRand(seed=23, run_stochastic=True, track_pdf=True)
    reference = np.random.default_rng(23)
    for _ in range(3):
        expected = reference.integers(8, None, shape, np.int64)
        state.set_rand_state("noise", "integers", 8, None, shape, np.int64)
        np.testing.assert_array_equal(state.s.noise, expected)
        assert state.probs[-1] == 8.0 ** -np.size(expected)
        assert state.rng.bit_generator.state == reference.bit_generator.state
    assert state.return_probdens() == 8.0 ** (-3 * np.size(expected))


def test_auto_updates_copy_reset_and_disabled_controls():
    state = IntegerRand(seed=13, run_stochastic=True, track_pdf=True)
    state.update_stochastic_states()
    clone = state.copy()
    state.update_stochastic_states()
    clone.update_stochastic_states()
    np.testing.assert_array_equal(state.s.noise, clone.s.noise)
    assert clone.s.noise is not state.s.noise
    assert state.probs == pytest.approx([6.0**-4])
    state.reset()
    state.update_stochastic_states()
    np.testing.assert_array_equal(
        state.s.noise, np.random.default_rng(13).integers(-2, 3, (2, 2), np.int32, True)
    )
    disabled = IntegerRand(seed=13, run_stochastic=False, track_pdf=True)
    before = copy.deepcopy(disabled.rng.bit_generator.state)
    disabled.update_stochastic_states()
    np.testing.assert_array_equal(disabled.s.noise, np.zeros((2, 2)))
    assert disabled.probs == []
    assert disabled.rng.bit_generator.state == before
    scalar = ExampleRand(seed=13, run_stochastic=True, track_pdf=True)
    scalar.set_rand_state("noise", "integers", 4)
    assert scalar.probs == [0.25]


def test_actual_matrix_noise_simulation_records_each_draw_and_joint_mass():
    model = IntegerFunction(
        sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True}, r={"seed": 19}
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
    assert result["tend.classify.total"] == totals[-1]
    assert result["tend.classify.mass"] == pytest.approx(6.0**-4)
    np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 2)))
    assert model.r.probs == []


@pytest.mark.parametrize("low,high", [(2, 2), (3, 2)])
def test_empty_or_reversed_integer_intervals_are_rejected(low, high):
    with pytest.raises(ValueError, match="interval"):
        calc_prob_for_integers([2], low, high)


def test_closed_single_integer_interval_has_unit_mass():
    assert calc_prob_for_integers([7, 7, 7], 7, 7, 3, np.int64, True) == 1.0
    assert calc_prob_for_integers([7, 6, 7], 7, 7, 3, np.int64, True) == 0.0


def test_numpy_uint64_and_object_bounds_retain_exact_endpoints():
    low = np.array([2**64 - 4], dtype=np.uint64)
    high = np.array([2**64], dtype=object)
    samples = np.random.default_rng(1).integers(low, high, size=(2, 1), dtype=np.uint64)
    assert (
        get_prob_for_rand(samples, "integers", low, high, (2, 1), np.uint64)
        == 1.0 / 16.0
    )
    assert (
        calc_prob_for_integers(np.array([2**64 - 5], dtype=np.uint64), low, high) == 0.0
    )
    np.testing.assert_array_equal(low, [2**64 - 4])
    np.testing.assert_array_equal(high, [2**64])
