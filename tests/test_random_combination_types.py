#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exact discrete random parameter choices.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample, combine_random


class OptionParameter(Parameter, readonly=True):
    setting: tuple = ("primary", 2**53 + 1)
    setting_set = (("primary", 2**53 + 1), ("backup", 2**53 + 3))


class OptionState(State):
    total: np.int64 = 0


class OptionFunction(Function):
    container_p = OptionParameter
    container_s = OptionState

    def dynamic_behavior(self):
        self.s.total += self.p.setting[1] - 2**53

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestRandomCombinationTypes(unittest.TestCase):
    def test_discrete_choices_preserve_values_types_and_tuple_elements(self):
        choices = (
            {2**53 + 1, 0.5},
            {2**63 + 1, -1},
            {"auto", 3},
            {("primary", 2**53 + 1), ("backup", 2**53 + 3)},
            {np.int64(4), np.float32(0.25)},
            {False, "auto"},
        )
        for options in choices:
            for seed in (0, 1, 19):
                with self.subTest(options=options, seed=seed):
                    values = list(options)
                    rng = np.random.default_rng(seed)
                    expected = [values[rng.choice(len(values))] for _ in range(12)]
                    actual = combine_random([options], seed=seed, num_combos=12)
                    for row, value in zip(actual, expected):
                        self.assertIs(type(row[0]), type(value))
                        self.assertEqual(row[0], value)
                        self.assertIs(row[0], value)

    def test_continuous_ranges_keep_the_seeded_sequence_and_input_objects(self):
        options = {"auto", 2**53 + 1}
        ranges = [options, (-2.0, 3.0), {4, 7}, (0.0, 1.0)]
        original = options.copy()
        rng = np.random.default_rng(7)
        expected = [
            [
                list(options)[rng.choice(len(options))],
                rng.uniform(-2.0, 3.0),
                list(ranges[2])[rng.choice(2)],
                rng.uniform(0.0, 1.0),
            ]
            for _ in range(8)
        ]
        actual = combine_random(ranges, seed=7, num_combos=8)
        self.assertEqual(actual, expected)
        self.assertEqual(options, original)
        self.assertEqual(combine_random(ranges, seed=7, num_combos=0), [])
        with self.assertRaises(ValueError):
            combine_random([set()], seed=7)

    def test_homogeneous_numeric_ranges_keep_existing_values_and_random_draws(self):
        for options in ({1, 2, 3}, {0.25, 0.75}, {np.int64(3), np.int64(7)}):
            rng = np.random.default_rng(9)
            expected = [
                [rng.choice(list(options)), rng.uniform(0.0, 1.0)] for _ in range(10)
            ]
            np.testing.assert_array_equal(
                combine_random([options, (0.0, 1.0)], seed=9, num_combos=10), expected
            )

    def test_random_samples_replicates_and_subsampling_keep_typed_parameters(self):
        domain = ParameterDomain(dict)
        domain.add_variable("setting", var_set=("auto", 2**53 + 1))
        domain.add_variable("scale", var_lim=(0.1, 0.9))
        for subsample in (False, 5):
            with self.subTest(subsample=subsample):
                sample = ParameterSample(domain, seed=13)
                combinations = sample.combine_random(num_combos=8)
                values = list(domain.variables["setting"])
                rng = np.random.default_rng(13)
                expected = [
                    [values[rng.choice(len(values))], rng.uniform(0.1, 0.9)]
                    for _ in range(8)
                ]
                for actual, reference in zip(combinations, expected):
                    self.assertIs(type(actual[0]), type(reference[0]))
                    self.assertEqual(actual, reference)
                sample.add_variable_ranges(
                    combmethod="random",
                    comb_kwargs={"num_combos": 8},
                    n_samp=subsample,
                    replicates=2,
                    weight=0.6,
                )
                selected = (
                    expected
                    if not subsample
                    else [
                        expected[i]
                        for i in np.random.default_rng(13).choice(8, subsample)
                    ]
                )
                scenarios = sample.scenarios()
                self.assertEqual(len(scenarios), 2 * len(selected))
                for scenario, reference in zip(
                    scenarios, [row for row in selected for _ in range(2)]
                ):
                    self.assertEqual(scenario.p, dict(zip(domain.variables, reference)))
                    self.assertIs(type(scenario.p["setting"]), type(reference[0]))
                self.assertAlmostEqual(sum(s.prob for s in scenarios), 0.6)

    def test_real_parameter_sample_preserves_exact_identifiers_and_model_outputs(self):
        domain = ParameterDomain(OptionParameter)
        domain.add_variable("setting")
        sample = ParameterSample(domain, seed=11)
        sample.add_variable_ranges(combmethod="random", comb_kwargs={"num_combos": 8})
        result, history = propagate.parameter_sample(
            OptionFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        for scenario in sample.scenarios():
            with self.subTest(scenario=scenario.name):
                setting = scenario.p["setting"]
                self.assertIs(type(setting), tuple)
                self.assertIs(type(setting[1]), int)
                self.assertIn(setting, OptionParameter.setting_set)
                step = setting[1] - 2**53
                self.assertEqual(
                    result[scenario.name + ".tend.classify.total"], 3 * step
                )
                np.testing.assert_array_equal(
                    history[scenario.name + ".s.total"], np.arange(4) * step
                )


if __name__ == "__main__":
    unittest.main()
