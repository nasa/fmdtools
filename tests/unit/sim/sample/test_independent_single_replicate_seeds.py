#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent seeds with one replicate per parameter value.

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
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.define.container.rand import Rand
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class NoiseState(State):
    noise: np.float64 = 0.0
    noise_update = ("normal", (0.0, 1.0))


class NoiseRand(Rand):
    s: NoiseState = NoiseState()


class TotalState(State):
    total: np.float64 = 0.0


class ParameterNoiseFunction(Function):
    container_p = ExampleParameter
    container_r = NoiseRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.p.x + self.r.s.noise

    def classify(self, **kwargs):
        return {"total": self.s.total}


def make_sample(seed):
    domain = ParameterDomain(ExampleParameter)
    domain.add_variables("x")
    return ParameterSample(domain, seed=seed)


class TestIndependentSingleReplicateSeeds(unittest.TestCase):
    def test_independent_mode_generates_a_seed_per_parameter_value(self):
        for seed in (0, 17, None):
            for count in (2, 3):
                with self.subTest(seed=seed, count=count):
                    sample = make_sample(seed)
                    combinations = [[float(i + 1)] for i in range(count)]
                    original = copy.deepcopy(combinations)
                    expected = sample.seedsequence.generate_state(count)
                    sample.add_variable_replicates(
                        combinations, seed_comb="independent", weight=3.0, name="noise"
                    )
                    scenarios = sample.scenarios()
                    self.assertEqual([s.r["seed"] for s in scenarios], list(expected))
                    self.assertEqual(len({s.r["seed"] for s in scenarios}), count)
                    self.assertEqual(
                        [s.p["x"] for s in scenarios], [x[0] for x in combinations]
                    )
                    self.assertEqual(
                        [s.name for s in scenarios],
                        [f"rep0_noise_{i}" for i in range(count)],
                    )
                    self.assertEqual(
                        [s.inputparams for s in scenarios],
                        [{0: x[0]} for x in combinations],
                    )
                    self.assertEqual([s.prob for s in scenarios], [3.0 / count] * count)
                    self.assertEqual(combinations, original)

    def test_multiple_replicates_keep_the_existing_seed_order(self):
        for seed in (0, 17):
            for replicates in (2, 3):
                for mode in ("shared", "independent"):
                    with self.subTest(seed=seed, replicates=replicates, mode=mode):
                        sample = make_sample(seed)
                        sample.add_variable_replicates(
                            [[1.0], [2.0]], replicates=replicates, seed_comb=mode
                        )
                        count = replicates if mode == "shared" else 2 * replicates
                        expected = list(
                            np.random.SeedSequence(seed).generate_state(count)
                        )
                        if mode == "shared":
                            expected *= 2
                        scenarios = sample.scenarios()
                        self.assertEqual([s.r["seed"] for s in scenarios], expected)
                        np.testing.assert_allclose(
                            [s.prob for s in scenarios],
                            [1.0 / (2 * replicates)] * (2 * replicates),
                        )

    def test_single_scenario_and_default_shared_mode_keep_the_sample_seed(self):
        for seed in (0, 17, None):
            for combinations, mode in (
                ([[1.0]], "independent"),
                ([], "independent"),
                ([[1.0], [2.0]], "shared"),
            ):
                with self.subTest(seed=seed, combinations=combinations, mode=mode):
                    sample = make_sample(seed)
                    sample.add_variable_replicates(combinations, seed_comb=mode)
                    expected = {} if seed is None else {"seed": seed}
                    self.assertEqual(
                        [s.r for s in sample.scenarios()],
                        [expected] * max(1, len(combinations)),
                    )

    def test_range_sampling_forwards_independent_mode_with_its_default_replicates(self):
        for seed in (0, 17):
            with self.subTest(seed=seed):
                domain = ParameterDomain(ExampleParameter)
                domain.add_variable("x", var_set=(1.0, 2.0, 3.0))
                sample = ParameterSample(domain, seed=seed)
                sample.add_variable_ranges(seed_comb="independent")
                expected = np.random.SeedSequence(seed).generate_state(3)
                self.assertEqual(
                    [s.r["seed"] for s in sample.scenarios()], list(expected)
                )
                self.assertEqual(
                    {s.p["x"] for s in sample.scenarios()}, {1.0, 2.0, 3.0}
                )

    def test_generated_seeds_and_scenarios_are_reproducible(self):
        for seed in (0, 27):
            with self.subTest(seed=seed):
                first, second = make_sample(seed), make_sample(seed)
                for sample in (first, second):
                    sample.add_variable_replicates(
                        [[1.0], [3.0]], seed_comb="independent"
                    )
                self.assertEqual(
                    [s.asdict() for s in first.scenarios()],
                    [s.asdict() for s in second.scenarios()],
                )
                for a, b in zip(first.scenarios(), second.scenarios()):
                    np.testing.assert_array_equal(
                        Rand(**a.r).rng.normal(size=6), Rand(**b.r).rng.normal(size=6)
                    )

    def test_real_parameter_simulations_use_the_requested_seed_combination(self):
        for mode in ("shared", "independent"):
            with self.subTest(mode=mode):
                sample = make_sample(19)
                sample.add_variable_replicates([[1.0], [3.0]], seed_comb=mode)
                model = ParameterNoiseFunction(
                    sp={"end_time": 3.0, "run_stochastic": True}
                )
                results, histories = propagate.parameter_sample(
                    model, sample, showprogress=False
                )
                expected_seeds = (
                    np.random.SeedSequence(19).generate_state(2)
                    if mode == "independent"
                    else [19, 19]
                )
                draws = []
                for scenario, seed in zip(sample.scenarios(), expected_seeds):
                    time = histories[scenario.name + ".time"]
                    expected = np.random.default_rng(seed).normal(size=len(time))
                    actual = histories[scenario.name + ".r.s.noise"]
                    np.testing.assert_array_equal(actual, expected)
                    totals = np.concatenate(
                        ([0.0], np.cumsum(scenario.p["x"] + expected[1:]))
                    )
                    np.testing.assert_allclose(
                        histories[scenario.name + ".s.total"], totals
                    )
                    self.assertAlmostEqual(
                        results[scenario.name + ".tend.classify.total"], totals[-1]
                    )
                    draws.append(actual)
                self.assertEqual(np.array_equal(*draws), mode == "shared")
                self.assertEqual(model.r.s.noise, 0.0)
                self.assertEqual(model.s.total, 0.0)


if __name__ == "__main__":
    unittest.main()
