#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for preserving parameter types during random subsampling.

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

import unittest

import numpy as np

from fmdtools.sim.sample import ParameterDomain, ParameterSample


def parameter_defaults(**kwargs):
    """Provide mixed-type defaults for orthogonal sampling."""
    return {"label": "slow", "gain": 0.5, "count": 1, "enabled": True, **kwargs}


class TestParameterSubsamplingTypes(unittest.TestCase):
    """Select existing parameter combinations without coercing their values."""

    def make_domain(self):
        domain = ParameterDomain(parameter_defaults)
        domain.add_variable("label", var_set=("slow", "fast"))
        domain.add_variable("gain", var_lim=(0.0, 1.0))
        domain.add_variable("count", var_set=(1, 3))
        domain.add_variable("enabled", var_set=(False, True))
        return domain

    def assert_typed_values(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            self.assertIs(type(actual[key]), type(value), msg=str(key))
            self.assertEqual(actual[key], value, msg=str(key))

    def test_subsamples_preserve_mixed_types_for_each_combination_method(self):
        for method in ("product", "orthogonal", "random"):
            for seed in (0, 9):
                for count in (1, 7):
                    with self.subTest(method=method, seed=seed, count=count):
                        sample = ParameterSample(self.make_domain(), seed=seed)
                        kwargs = {"num_combos": 4} if method == "random" else {}
                        combinations = getattr(sample, "combine_" + method)(**kwargs)
                        indices = np.random.default_rng(seed).choice(
                            len(combinations), count
                        )
                        expected = [combinations[i] for i in indices]
                        sample.add_variable_ranges(
                            combmethod=method, comb_kwargs=kwargs, n_samp=count
                        )
                        scenarios = sample.scenarios()
                        self.assertEqual(len(scenarios), count)
                        for scenario, values in zip(scenarios, expected):
                            self.assert_typed_values(
                                scenario.p,
                                dict(zip(sample.paramdomain.variables, values)),
                            )
                            self.assert_typed_values(
                                scenario.inputparams, dict(enumerate(values))
                            )
                            self.assertEqual(scenario.r, {"seed": seed})
                            self.assertAlmostEqual(scenario.prob, 1.0 / count)

    def test_large_integers_are_not_rounded_by_floating_point_columns(self):
        domain = ParameterDomain(dict)
        domain.add_variable("identifier", var_set=(2**53 + 1, 2**53 + 3))
        domain.add_variable("gain", var_set=(0.25, 0.75))
        sample = ParameterSample(domain, seed=3)
        combinations = sample.combine_product()
        indices = np.random.default_rng(3).choice(len(combinations), 8)
        sample.add_variable_ranges(n_samp=8)
        for scenario, index in zip(sample.scenarios(), indices):
            expected = combinations[index][0]
            self.assertEqual(int(scenario.p["identifier"]), int(expected))
            self.assertIs(type(scenario.p["identifier"]), type(expected))

    def test_numeric_variable_mapping_receives_numeric_inputs(self):
        domain = ParameterDomain(dict)
        domain.add_variable("label", var_set=("slow", "fast"))
        domain.add_variable(
            "gain", var_set=(0.25, 0.75), var_map=lambda value: (value + 0.5,)
        )
        sample = ParameterSample(domain, seed=7)
        sample.add_variable_ranges(n_samp=4)
        for scenario in sample.scenarios():
            self.assertEqual(scenario.p["gain"], scenario.inputparams[1] + 0.5)
            self.assertIn(scenario.p["label"], ("slow", "fast"))

    def test_replicate_probabilities_names_and_seeds_are_preserved(self):
        for seed_mode in ("shared", "independent"):
            with self.subTest(seed_mode=seed_mode):
                domain = self.make_domain()
                sample = ParameterSample(domain, seed=11)
                combinations = sample.combine_product()
                indices = np.random.default_rng(11).choice(len(combinations), 3)
                selected = [combinations[i] for i in indices]
                reference = ParameterSample(domain, seed=11)
                options = {
                    "replicates": 2,
                    "seed_comb": seed_mode,
                    "weight": 0.6,
                    "name": "selected",
                }
                reference.add_variable_replicates(selected, **options)
                sample.add_variable_ranges(n_samp=3, **options)
                for actual, expected in zip(sample.scenarios(), reference.scenarios()):
                    self.assertEqual(actual.asdict(), expected.asdict())
                    self.assert_typed_values(actual.p, expected.p)
                self.assertEqual(len(sample.scenarios()), 6)
                self.assertAlmostEqual(sum(s.prob for s in sample.scenarios()), 0.6)

    def test_homogeneous_numeric_sampling_keeps_seeded_row_selection(self):
        domain = ParameterDomain(dict)
        domain.add_variable("x", var_set=(0.0, 1.0))
        domain.add_variable("y", var_set=(2.0, 3.0))
        sample = ParameterSample(domain, seed=5)
        expected = np.random.default_rng(5).choice(sample.combine_product(), 10)
        sample.add_variable_ranges(n_samp=10)
        actual = [[s.p["x"], s.p["y"]] for s in sample.scenarios()]
        np.testing.assert_array_equal(actual, expected)

    def test_sampling_all_combinations_is_unchanged(self):
        sample = ParameterSample(self.make_domain(), seed=0)
        expected = sample.combine_product()
        sample.add_variable_ranges()
        self.assertEqual(len(sample.scenarios()), len(expected))
        for scenario, values in zip(sample.scenarios(), expected):
            self.assert_typed_values(
                scenario.p, dict(zip(sample.paramdomain.variables, values))
            )


if __name__ == "__main__":
    unittest.main()
