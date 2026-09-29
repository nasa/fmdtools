#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent Student t random-state probability tracking.

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
import math

import numpy as np
import pytest
from scipy import integrate, stats

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_pfunc_for_dist,
    get_prob_for_rand,
    get_standard_t_pdf,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class StudentState(State):
    noise: np.array = np.zeros(3)
    noise_update = ("standard_t", (5.0, 3))


class StudentRand(Rand):
    s: StudentState = StudentState()


class SumState(State):
    total: np.float64 = 0.0


class StudentFunction(Function):
    container_r = StudentRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.total += np.sum(self.r.s.noise)

    def classify(self, **kwargs):
        return {"total": self.s.total, "density": self.r.return_probdens()}


def analytic_density(value, df):
    return (
        math.exp(math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0))
        / math.sqrt(math.pi * df)
        * (1.0 + value * value / df) ** (-(df + 1.0) / 2.0)
    )


@pytest.mark.parametrize("df", [0.5, 1.0, 3.0, 20.0])
@pytest.mark.parametrize(
    "values", [0.0, [-1.0, 0.0, 2.0], np.array([[0.0, 0.5], [1.0, 2.0]])]
)
def test_density_equals_product_of_independent_analytic_factors(df, values):
    before = np.array(values, copy=True)
    expected = math.prod(analytic_density(float(value), df) for value in before.flat)
    actual = get_prob_for_rand(values, "standard_t", df)
    assert actual == pytest.approx(expected, rel=2e-13, abs=0.0)
    assert np.ndim(actual) == 0
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("size", [None, (), 1, 3, (2, 3), 0, (2, 0, 3)])
def test_optional_size_and_empty_draws_preserve_joint_density(size):
    values = np.random.default_rng(12).standard_t(4.0, size=size)
    expected = np.prod(stats.t.pdf(values, 4.0))
    assert get_prob_for_rand(values, "standard_t", 4.0, size) == pytest.approx(expected)
    assert get_standard_t_pdf(4.0, size)(values) == pytest.approx(expected)


def test_array_degrees_of_freedom_and_variadic_draws():
    df = np.array([1.0, 3.0, 8.0])
    values = np.random.default_rng(7).standard_t(df, size=(2, 3))
    expected = np.prod(stats.t.pdf(values, df))
    assert get_prob_for_rand(values, "standard_t", df, (2, 3)) == pytest.approx(
        expected
    )
    assert get_pfunc_for_dist("standard_t", df)(*values) == pytest.approx(expected)
    flat_values = np.array([-1.0, 0.0, 2.0])
    assert get_standard_t_pdf(3.0)(*flat_values) == pytest.approx(
        np.prod(stats.t.pdf(flat_values, 3.0))
    )


@pytest.mark.parametrize("df", [1.0, 5.0])
def test_scalar_density_integrates_to_one_and_matches_known_cases(df):
    total, _ = integrate.quad(get_standard_t_pdf(df), -np.inf, np.inf)
    assert total == pytest.approx(1.0, abs=1e-10)
    if df == 1.0:
        assert get_standard_t_pdf(df)(0.0, 1.0) == pytest.approx(
            1.0 / (2.0 * math.pi**2)
        )


@pytest.mark.parametrize("values", [[np.inf, 0.0], [-np.inf, 1.0], [np.nan, 0.0]])
def test_nonfinite_samples_follow_univariate_density_semantics(values):
    np.testing.assert_allclose(
        get_prob_for_rand(values, "standard_t", 3.0),
        np.prod(stats.t.pdf(values, 3.0)),
        equal_nan=True,
    )


@pytest.mark.parametrize("size", [(), 1, 3, (2, 3), 0])
def test_tracking_retains_generated_samples_and_generator_state(size):
    state = StudentRand(seed=13, run_stochastic=True, track_pdf=True)
    reference = np.random.default_rng(13)
    expected_densities = []
    for _ in range(3):
        expected = reference.standard_t(5.0, size)
        state.set_rand_state("noise", "standard_t", 5.0, size)
        np.testing.assert_array_equal(state.s.noise, expected)
        assert state.s.noise.shape == expected.shape
        expected_densities.append(np.prod(stats.t.pdf(expected, 5.0)))
        np.testing.assert_allclose(state.probs, expected_densities)
        assert state.return_probdens() == pytest.approx(np.prod(expected_densities))
        assert state.rng.bit_generator.state == reference.bit_generator.state
        assert state.gen_state() == reference.bit_generator.state


def test_automatic_updates_copy_and_reset_preserve_per_step_density():
    state = StudentRand(seed=23, run_stochastic=True, track_pdf=True)
    for _ in range(2):
        state.update_stochastic_states()
        np.testing.assert_allclose(
            state.probs, [np.prod(stats.t.pdf(state.s.noise, 5.0))]
        )
    clone = state.copy()
    state.update_stochastic_states()
    clone.update_stochastic_states()
    np.testing.assert_array_equal(state.s.noise, clone.s.noise)
    assert state.s.noise is not clone.s.noise
    assert state.probs == clone.probs
    state.reset()
    assert state.probs == []
    state.update_stochastic_states()
    np.testing.assert_array_equal(
        state.s.noise, np.random.default_rng(23).standard_t(5.0, 3)
    )


def test_scalar_and_disabled_random_state_controls():
    state = ExampleRand(seed=29, run_stochastic=True, track_pdf=True)
    state.set_rand_state("noise", "standard_t", 3.0)
    expected = np.random.default_rng(29).standard_t(3.0)
    assert state.s.noise == expected
    assert state.probs == pytest.approx([analytic_density(expected, 3.0)])
    disabled = StudentRand(seed=29, run_stochastic=False, track_pdf=True)
    before = copy.deepcopy(disabled.rng.bit_generator.state)
    disabled.update_stochastic_states()
    np.testing.assert_array_equal(disabled.s.noise, np.zeros(3))
    assert disabled.rng.bit_generator.state == before
    assert disabled.probs == []


def test_actual_stochastic_simulation_records_each_independent_draw():
    model = StudentFunction(
        sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True}, r={"seed": 17}
    )
    result, history = Simulation(mdl=model)()
    reference = np.random.default_rng(17)
    draws = np.array([reference.standard_t(5.0, 3) for _ in history.time])
    densities = np.prod(stats.t.pdf(draws, 5.0), axis=1)
    totals = np.concatenate(([0.0], np.cumsum(draws[1:].sum(axis=1))))
    np.testing.assert_array_equal(history["r.s.noise"], draws)
    np.testing.assert_allclose(history["r.probdens"], densities)
    np.testing.assert_allclose(history["s.total"], totals)
    assert result["tend.classify.density"] == pytest.approx(densities[-1])
    assert result["tend.classify.total"] == pytest.approx(totals[-1])
    np.testing.assert_array_equal(model.r.s.noise, np.zeros(3))
    assert model.r.probs == []
