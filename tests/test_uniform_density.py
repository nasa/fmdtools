#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for uniform random-state probability densities.

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

import numpy as np
import pytest
from scipy.integrate import quad

from fmdtools.define.container.rand import ExampleRand, get_prob_for_rand


@pytest.mark.parametrize(
    "low, high", [(0.0, 1.0), (5.0, 6.0), (-4.0, -1.0), (-2.0, 3.0), (0.25, 0.5)]
)
@pytest.mark.parametrize("count", [1, 3])
def test_uniform_density_matches_interval_width(low, high, count):
    """Joint density is the reciprocal volume of the sampling box."""
    values = low + (high - low) * np.linspace(0.25, 0.75, count)
    supplied = values[0] if count == 1 else values
    assert get_prob_for_rand(supplied, "uniform", low, high) == pytest.approx(
        (high - low) ** -count
    )


@pytest.mark.parametrize(
    "low, high, value",
    [
        (5.0, 6.0, 7.0),
        (5.0, 6.0, 4.0),
        (-4.0, -1.0, -0.5),
        (-2.0, 3.0, 4.0),
        (0.25, 0.5, 0.6),
    ],
)
def test_uniform_density_is_zero_outside_requested_interval(low, high, value):
    assert get_prob_for_rand(value, "uniform", low, high) == 0.0


@pytest.mark.parametrize(
    "args, value, expected",
    [
        ((), 0.5, 1.0),
        ((0.0,), 0.5, 1.0),
        ((-2.0,), 0.0, 1.0 / 3),
        ((5.0, 6.0, 3), [5.25, 5.5, 5.75], 1.0),
    ],
)
def test_uniform_defaults_and_sampling_size(args, value, expected):
    """NumPy's optional sample shape does not parameterize the density."""
    assert get_prob_for_rand(value, "uniform", *args) == pytest.approx(expected)


@pytest.mark.parametrize("low, high", [(5.0, 6.0), (-4.0, -1.0), (-2.0, 3.0)])
def test_uniform_density_integrates_to_one(low, high):
    integral, _ = quad(lambda x: get_prob_for_rand(x, "uniform", low, high), low, high)
    assert integral == pytest.approx(1.0)


@pytest.mark.parametrize("low, high", [(0.0, 1.0), (5.0, 6.0), (-4.0, -1.0)])
def test_random_state_records_correct_uniform_density(low, high):
    """Use the real random generator and PDF tracking without replacing draws."""
    random_state = ExampleRand(run_stochastic=True, track_pdf=True)
    reference = np.random.default_rng(random_state.seed)
    for _ in range(2):
        random_state.set_rand_state("noise", "uniform", low, high)
        assert random_state.s.noise == reference.uniform(low, high)
    assert random_state.probs == pytest.approx([1.0 / (high - low)] * 2)
    assert random_state.return_probdens() == pytest.approx((high - low) ** -2)
