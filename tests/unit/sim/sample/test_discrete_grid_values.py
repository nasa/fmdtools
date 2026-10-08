#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exact discrete product and orthogonal sample values.

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

import itertools
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import ExampleParameter, Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


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


def option_factory(setting="auto", gain=1.0):
    return {"setting": setting, "gain": gain}


class TestDiscreteGridValues(unittest.TestCase):
    def test_grid_members_keep_exact_values_types_and_tuple_structure(self):
        cases = (
            {2**53 + 1, 0.5},
            {2**63 + 1, -1},
            {"auto", 3},
            {("primary", 2**53 + 1), ("backup", 2**53 + 3)},
            {(1,), (2, 3)},
            {()},
            {False, "auto"},
            {np.int64(4), np.float32(0.25)},
        )
        for options in cases:
            with self.subTest(options=options):
                domain = ParameterDomain(dict)
                domain.add_variable("setting", var_set=options)
                expected = list(domain.variables["setting"])
                values = domain.get_var_iters()["setting"]
                self.assertEqual(values.shape, (len(expected),))
                for actual, original in zip(values, expected):
                    self.assertIs(actual, original)
                    self.assertIs(type(actual), type(original))
                    self.assertEqual(actual, original)
                self.assertEqual(domain.variables["setting"], options)

    def test_continuous_grids_and_independent_discrete_array_storage_are_retained(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x", var_lim=(0, 5))
        domain.add_variable("y", var_set=(1.0, 2.0, 3.0))
        first = domain.get_var_iters(2)
        second = domain.get_var_iters(2)
        np.testing.assert_array_equal(first["x"], [0, 2, 4])
        self.assertTrue(np.issubdtype(first["x"].dtype, np.integer))
        np.testing.assert_array_equal(first["y"], list(domain.variables["y"]))
        first["y"][0] = 99.0
        self.assertNotEqual(second["y"][0], 99.0)
        self.assertEqual(domain.variables["y"], {1.0, 2.0, 3.0})

    def test_product_and_orthogonal_combinations_preserve_complete_option_members(self):
        domain = ParameterDomain(option_factory)
        domain.add_variable("setting", var_set={"auto", 2**53 + 1})
        domain.add_variable("gain", var_lim=(0.0, 1.0))
        sample = ParameterSample(domain, seed=7)
        settings = list(domain.variables["setting"])
        product = sample.combine_product(resolution=0.5)
        expected = list(itertools.product(settings, [0.0, 0.5, 1.0]))
        self.assertEqual(product, expected)
        for row, reference in zip(product, expected):
            self.assertIs(type(row[0]), type(reference[0]))
        orthogonal = sample.combine_orthogonal(resolution=0.5)
        self.assertEqual(
            orthogonal,
            [[x, 1.0] for x in settings] + [["auto", x] for x in [0.0, 0.5, 1.0]],
        )
        for row, setting in zip(orthogonal, settings):
            self.assertIs(row[0], setting)

    def test_sample_replicates_weights_and_subsampling_preserve_grid_members(self):
        domain = ParameterDomain(option_factory)
        domain.add_variable("setting", var_set={"auto", 2**53 + 1})
        domain.add_variable("gain", var_lim=(0.0, 1.0))
        for method in ("product", "orthogonal"):
            for count in (False, 4):
                with self.subTest(method=method, count=count):
                    sample = ParameterSample(domain, seed=11)
                    expected = getattr(sample, "combine_" + method)(resolution=0.5)
                    if count:
                        expected = [
                            expected[i]
                            for i in np.random.default_rng(11).choice(
                                len(expected), count
                            )
                        ]
                    sample.add_variable_ranges(
                        combmethod=method,
                        comb_kwargs={"resolution": 0.5},
                        n_samp=count,
                        replicates=2,
                        weight=0.6,
                    )
                    scenarios = sample.scenarios()
                    self.assertEqual(len(scenarios), 2 * len(expected))
                    for scenario, row in zip(
                        scenarios, [x for x in expected for _ in range(2)]
                    ):
                        self.assertEqual(scenario.p, dict(zip(domain.variables, row)))
                        self.assertIs(type(scenario.p["setting"]), type(row[0]))
                    self.assertAlmostEqual(sum(x.prob for x in scenarios), 0.6)

    def test_real_simulations_receive_exact_tuple_options_from_both_grid_methods(self):
        domain = ParameterDomain(OptionParameter)
        domain.add_variable("setting")
        for method in ("product", "orthogonal"):
            with self.subTest(method=method):
                sample = ParameterSample(domain, seed=11)
                sample.add_variable_ranges(combmethod=method)
                result, history = propagate.parameter_sample(
                    OptionFunction(sp={"end_time": 3.0}), sample, showprogress=False
                )
                self.assertEqual(len(sample.scenarios()), 2)
                for scenario in sample.scenarios():
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
