#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for zero-valued parameter-scenario seeds.

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

from fmdtools.define.container.rand import Rand
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestParameterSampleSeeds(unittest.TestCase):
    """Preserve zero seeds while retaining the existing fallback behavior."""

    def make_sample(self, seed=None):
        domain = ParameterDomain(dict)
        domain.add_variable("value", var_set=(0.25, 0.5))
        return ParameterSample(domain, seed=seed)

    def test_explicit_zero_overrides_the_sample_seed(self):
        for base_seed in (None, 0, 17):
            for zero in (0, np.int64(0), np.uint32(0)):
                with self.subTest(base_seed=base_seed, zero_type=type(zero).__name__):
                    sample = self.make_sample(base_seed)
                    sample.add_variable_scenario(0.5, seed=zero, weight=0.25)
                    scenario = sample.scenarios()[0]
                    self.assertEqual(scenario.r, {"seed": 0})
                    self.assertEqual(scenario.p, {"value": 0.5})
                    self.assertEqual(scenario.prob, 0.25)
                    self.assertEqual(scenario.inputparams, {0: 0.5})
                    actual = Rand(**scenario.r).rng.random(5)
                    np.testing.assert_array_equal(
                        actual, np.random.default_rng(0).random(5)
                    )

    def test_zero_sample_seed_is_used_for_unspecified_scenario_seed(self):
        for zero in (0, np.int64(0), np.uint32(0)):
            for kwargs in ({}, {"seed": False}, {"seed": None}):
                with self.subTest(zero_type=type(zero).__name__, kwargs=kwargs):
                    sample = self.make_sample(zero)
                    sample.add_variable_scenario(0.25, **kwargs)
                    self.assertEqual(sample.scenarios()[0].r, {"seed": 0})

    def test_nonzero_overrides_and_no_seed_defaults_are_unchanged(self):
        for base_seed in (None, False, 17):
            for seed in (None, False, 7):
                with self.subTest(base_seed=base_seed, seed=seed):
                    sample = self.make_sample(base_seed)
                    sample.add_variable_scenario(0.25, seed=seed)
                    expected = seed or base_seed
                    self.assertEqual(
                        sample.scenarios()[0].r, {"seed": expected} if expected else {}
                    )

    def test_single_replicates_and_ranges_preserve_zero_seed(self):
        for method in ("replicates", "ranges"):
            with self.subTest(method=method):
                sample = self.make_sample(0)
                if method == "replicates":
                    sample.add_variable_replicates([[0.25], [0.5]])
                else:
                    sample.add_variable_ranges()
                scenarios = sample.scenarios()
                self.assertEqual(len(scenarios), 2)
                self.assertEqual([s.r for s in scenarios], [{"seed": 0}, {"seed": 0}])
                self.assertEqual([s.prob for s in scenarios], [0.5, 0.5])
                self.assertEqual({s.p["value"] for s in scenarios}, {0.25, 0.5})

    def test_multiple_replicate_seed_generation_is_unchanged(self):
        for base_seed in (0, 17):
            for mode in ("shared", "independent"):
                with self.subTest(base_seed=base_seed, mode=mode):
                    sample = self.make_sample(base_seed)
                    sample.add_variable_replicates(
                        [[0.25], [0.5]], replicates=2, seed_comb=mode
                    )
                    seeds = np.random.SeedSequence(base_seed).generate_state(
                        2 if mode == "shared" else 4
                    )
                    expected = list(seeds) * 2 if mode == "shared" else list(seeds)
                    self.assertEqual(
                        [s.r["seed"] for s in sample.scenarios()], expected
                    )
                    self.assertEqual([s.prob for s in sample.scenarios()], [0.25] * 4)


if __name__ == "__main__":
    unittest.main()
