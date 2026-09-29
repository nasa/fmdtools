#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for von Mises random-state probability densities.

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

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    ExampleRand,
    Rand,
    get_prob_for_rand,
    get_vonmises_pdf,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def expected_density(values, mu, kappa):
    """Independent normalized circular density, evaluated without scipy.stats."""
    values, mu, kappa = np.broadcast_arrays(values, mu, kappa)
    density = np.exp(kappa * (np.cos(values - mu) - 1.0)) / (2 * np.pi * i0e(kappa))
    return np.prod(density)


class AngleState(State):
    angles: np.array = np.zeros(3)
    angles_update = ("vonmises", (0.75, 2.0, 3))


class AngleRand(Rand):
    s: AngleState = AngleState()


class DirectionState(State):
    displacement: np.float64 = 0.0


class DirectionFunction(Function):
    container_r = AngleRand
    container_s = DirectionState

    def dynamic_behavior(self):
        self.s.displacement += np.cos(self.r.s.angles).sum()

    def classify(self, **kwargs):
        return {
            "displacement": self.s.displacement,
            "density": self.r.return_probdens(),
        }


class TestVonMisesDensity(unittest.TestCase):
    """Run the regression cases with standard unittest discovery."""

    def test_scalar_density_maps_mean_and_concentration(self):
        for offset in [0.0, 0.3, -1.25]:
            for mu, kappa in [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.25, 2.0),
                (-2.5, 0.1),
                (2.8, 500.0),
            ]:
                with self.subTest(offset=offset, mu=mu, kappa=kappa):
                    value = mu + offset
                    expected = expected_density(value, mu, kappa)
                    np.testing.assert_allclose(
                        get_prob_for_rand(value, "vonmises", mu, kappa),
                        expected,
                        rtol=1e-11,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        get_vonmises_pdf(mu, kappa)(value),
                        expected,
                        rtol=1e-11,
                        atol=1e-12,
                    )

    def test_joint_density_ignores_optional_draw_shape(self):
        for shape in [(), (1,), (4,), (2, 3), (0,), (2, 0)]:
            with self.subTest(shape=shape):
                values = np.full(shape, 0.4)
                expected = expected_density(values, 0.75, 2.0)
                for args in ((0.75, 2.0), (0.75, 2.0, shape)):
                    np.testing.assert_allclose(
                        get_prob_for_rand(values, "vonmises", *args),
                        expected,
                        rtol=1e-06,
                        atol=1e-12,
                    )

    def test_array_parameters_broadcast_without_changing_inputs(self):
        values = np.array([[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]])
        mu = np.array([[-1.0], [1.0]])
        kappa = np.array([0.0, 1.0, 3.0])
        originals = [x.copy() for x in (values, mu, kappa)]
        actual = get_prob_for_rand(values, "vonmises", mu, kappa, (2, 3))
        np.testing.assert_allclose(
            actual, expected_density(values, mu, kappa), rtol=1e-06, atol=1e-12
        )
        for value, before in zip((values, mu, kappa), originals):
            np.testing.assert_array_equal(value, before)

    def test_density_is_normalized_and_periodic(self):
        for mu, kappa in [(0.0, 0.0), (1.5, 2.0), (-2.0, 8.0)]:
            with self.subTest(mu=mu, kappa=kappa):
                pdf = get_vonmises_pdf(mu, kappa)
                integral, _ = quad(pdf, -np.pi, np.pi, epsabs=1e-10)
                np.testing.assert_allclose(integral, 1.0, rtol=1e-06, atol=1e-12)
                for turns in (-2, -1, 1, 2):
                    np.testing.assert_allclose(
                        pdf(0.25 + turns * 2 * np.pi), pdf(0.25), rtol=1e-06, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        get_vonmises_pdf(mu + turns * 2 * np.pi, kappa)(0.25),
                        pdf(0.25),
                        rtol=1e-06,
                        atol=1e-12,
                    )

    def test_scalar_random_state_keeps_real_draws_and_tracks_density(self):
        state = ExampleRand(seed=19, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(19)
        expected = []
        for _ in range(3):
            value = reference.vonmises(-0.5, 1.75)
            state.set_rand_state("noise", "vonmises", -0.5, 1.75)
            self.assertEqual(state.s.noise, value)
            expected.append(expected_density(value, -0.5, 1.75))
            np.testing.assert_allclose(state.probs, expected)
        np.testing.assert_allclose(
            state.return_probdens(), np.prod(expected), rtol=1e-06, atol=1e-12
        )
        self.assertEqual(state.rng.bit_generator.state, reference.bit_generator.state)

    def test_automatic_vector_updates_preserve_draws_reset_and_joint_density(self):
        state = AngleRand(seed=23, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(23)
        for _ in range(3):
            state.update_stochastic_states()
            values = reference.vonmises(0.75, 2.0, 3)
            np.testing.assert_array_equal(state.s.angles, values)
            np.testing.assert_allclose(
                state.probs, [expected_density(values, 0.75, 2.0)]
            )
            np.testing.assert_allclose(
                state.return_probdens(),
                expected_density(values, 0.75, 2.0),
                rtol=1e-06,
                atol=1e-12,
            )
        state.reset()
        state.update_stochastic_states()
        np.testing.assert_array_equal(
            state.s.angles, np.random.default_rng(23).vonmises(0.75, 2.0, 3)
        )

    def test_simulation_records_angles_and_reports_density_without_changing_source_model(
        self,
    ):
        model = DirectionFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        values = np.array([reference.vonmises(0.75, 2.0, 3) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.angles"], values)
        np.testing.assert_allclose(
            history["r.probdens"], [expected_density(row, 0.75, 2.0) for row in values]
        )
        np.testing.assert_allclose(
            result["tend.classify.density"],
            expected_density(values[-1], 0.75, 2.0),
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["tend.classify.displacement"],
            np.cos(values[1:]).sum(),
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_array_equal(model.r.s.angles, np.zeros(3))
        self.assertEqual(model.s.displacement, 0.0)

    def test_other_distribution_and_untracked_controls_are_unchanged(self):
        np.testing.assert_allclose(
            get_prob_for_rand(0.0, "normal", 0.0, 1.0),
            1 / np.sqrt(2 * np.pi),
            rtol=1e-06,
            atol=1e-12,
        )
        self.assertEqual(get_prob_for_rand([5.25, 5.75], "uniform", 5.0, 6.0), 1.0)
        state = ExampleRand(seed=29, run_stochastic=True)
        state.set_rand_state("noise", "vonmises", 0.75, 2.0)
        self.assertEqual(state.s.noise, np.random.default_rng(29).vonmises(0.75, 2.0))
        self.assertEqual(state.probs, [])


if __name__ == "__main__":
    unittest.main()
