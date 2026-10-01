#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for independently owned ParameterSample defaults.

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

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter, Parameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    ParameterDomain,
    ParameterHistSample,
    ParameterResultSample,
    ParameterSample,
)


class TestParameterSampleDefaults(unittest.TestCase):
    """Only explicitly supplied objects may be shared between samples."""

    def preserve_inputs(self, sample):
        # Keep a baseline regression run from leaking shared-default mutations.
        original_domain = copy.deepcopy(sample.paramdomain.__dict__)
        original_sp = copy.deepcopy(sample.sp)

        def restore():
            sample.paramdomain.__dict__.clear()
            sample.paramdomain.__dict__.update(original_domain)
            sample.sp.clear()
            sample.sp.update(original_sp)

        self.addCleanup(restore)

    def test_omitted_domains_and_simulation_settings_are_independent(self):
        first, second = ParameterSample(), ParameterSample()
        self.preserve_inputs(first)
        first.paramdomain.add_constant("x", 9.0)
        first.paramdomain.add_variable("y", var_set={1.0, 2.0})
        first.sp["end_time"] = 1.0
        for sample in (second, ParameterSample()):
            with self.subTest(existing=sample is second):
                self.assertIsNot(sample.paramdomain, first.paramdomain)
                self.assertEqual(sample.paramdomain.constants, {})
                self.assertEqual(sample.paramdomain.variables, {})
                self.assertEqual(sample.paramdomain.var_maps, {})
                self.assertEqual(sample.sp, {})
                self.assertIs(sample.paramdomain.parameter_init, Parameter)
        self.assertEqual(second.scenarios(), [])

    def test_explicit_domain_and_settings_preserve_shared_identity(self):
        domain = ParameterDomain(ExampleParameter)
        settings = {}
        first = ParameterSample(domain, seed=13, sp=settings)
        second = ParameterSample(domain, seed=13, sp=settings)
        self.assertIs(first.paramdomain, domain)
        self.assertIs(second.paramdomain, domain)
        self.assertIs(first.sp, settings)
        self.assertIs(second.sp, settings)
        domain.add_variable("x")
        settings["end_time"] = 2.0
        first.add_variable_scenario(2.0, weight=0.25)
        second.add_variable_scenario(3.0, weight=0.75)
        self.assertEqual(first.scenarios()[0].p["x"], 2.0)
        self.assertEqual(second.scenarios()[0].p["x"], 3.0)
        self.assertEqual(first.scenarios()[0].sp["end_time"], 2.0)
        self.assertEqual(first.scenarios()[0].r, {"seed": 13})
        self.assertEqual(first.scenarios()[0].prob, 0.25)
        self.assertEqual(second.scenarios()[0].prob, 0.75)
        self.assertIsNot(first._scenarios, second._scenarios)
        np.testing.assert_array_equal(
            first.seedsequence.generate_state(4),
            np.random.SeedSequence(13).generate_state(4),
        )

    def test_result_and_history_sample_subclasses_get_independent_defaults(self):
        source = Result({"a.x": 1.0, "b.x": 2.0})
        history = History({"a.x": np.array([1.0, 2.0]), "b.x": np.array([3.0, 4.0])})
        samples = [
            ParameterSample(),
            ParameterResultSample(source, "x"),
            ParameterHistSample(history, "x"),
        ]
        for first in samples:
            self.preserve_inputs(first)
        for i, first in enumerate(samples):
            for second in samples[i + 1 :]:
                with self.subTest(
                    first=type(first).__name__, second=type(second).__name__
                ):
                    self.assertIsNot(first.paramdomain, second.paramdomain)
                    self.assertIsNot(first.sp, second.sp)
        samples[1].paramdomain.add_constant("x", 7.0)
        samples[1].sp["end_time"] = 1.0
        self.assertEqual(samples[2].sp, {})
        self.assertEqual(samples[0].paramdomain.constants, {})
        self.assertEqual(source, Result({"a.x": 1.0, "b.x": 2.0}))

    def test_actual_simulations_do_not_inherit_another_samples_settings(self):
        configured, control = ParameterSample(), ParameterSample()
        self.preserve_inputs(configured)
        configured.paramdomain.add_constant("x", 7.0)
        configured.sp["end_time"] = 1.0
        configured.add_variable_scenario()
        control.add_variable_scenario()
        model = ExampleFunction(sp={"end_time": 3.0})
        result, histories = propagate.parameter_sample(
            model, control, showprogress=False
        )
        reference = ParameterSample(ParameterDomain(Parameter), sp={})
        reference.add_variable_scenario()
        expected, expected_hist = propagate.parameter_sample(
            model, reference, showprogress=False
        )
        self.assertEqual(result, expected)
        self.assertEqual(histories, expected_hist)
        self.assertEqual(control.scenarios()[0].p, {})
        self.assertEqual(control.scenarios()[0].sp, {})
        actual_history = histories.nest(1)["var_0"]
        self.assertEqual(actual_history["time"][-1], 3.0)
        np.testing.assert_allclose(actual_history["s.x"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(result.get_values("xy")["var_0.tend.classify.xy"], 3.0)
        self.assertEqual(model.p.x, 1.0)


if __name__ == "__main__":
    unittest.main()
