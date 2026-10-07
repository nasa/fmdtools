#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for expectations when no nominal run is stored.

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
from types import SimpleNamespace

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestExpectationWithoutNominal(unittest.TestCase):
    def test_numeric_results_do_not_require_an_unrelated_nominal_run(self):
        for cls in (Result, History):
            for order in (("a", "b"), ("b", "a")):
                with self.subTest(cls=cls, order=order):
                    source = cls()
                    for name in order:
                        source[name + ".cost"] = 2.0 if name == "a" else 6.0
                        source[name + ".label"] = name
                    before = copy.deepcopy(dict(source))
                    result = source.get_expected(round_value=False)
                    self.assertIsInstance(result, cls)
                    self.assertEqual(dict(result), {"cost": 4.0})
                    self.assertEqual(dict(source), before)
                    self.assertEqual(source.get_expected(with_nominal=True).cost, 4.0)

    def test_rates_follow_scenario_names_with_no_nominal_entry(self):
        app = SimpleNamespace(
            scenarios=lambda: [
                SimpleNamespace(name="b", rate=3.0),
                SimpleNamespace(name="a", rate=1.0),
            ]
        )
        source = Result({"a.cost": 2.0, "b.cost": 6.0})
        for included in (False, True):
            self.assertEqual(source.get_expected(app, with_nominal=included).cost, 5.0)
        source["a_nominal_variant.cost"] = 10.0
        self.assertEqual(source.get_expected(round_value=False).cost, 6.0)

    def test_vector_histories_preserve_shape_timestamps_and_sources(self):
        first = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        second = first + 4
        source = History(
            {
                "a.value": first,
                "a.time": np.arange(3),
                "b.value": second,
                "b.time": np.arange(3),
            }
        )
        first.setflags(write=False)
        second.setflags(write=False)
        output = source.get_expected(round_value=False)
        np.testing.assert_array_equal(output.value, first + 2)
        np.testing.assert_array_equal(output.time, np.arange(3))
        self.assertFalse(np.shares_memory(output.value, first))
        self.assertFalse(first.flags.writeable)
        np.testing.assert_array_equal(second, first + 4)

    def test_nominal_differences_require_reference_and_existing_semantics_remain(self):
        with self.assertRaisesRegex(ValueError, "nominal"):
            Result({"a.cost": 2.0, "b.cost": 6.0}).get_expected(
                difference_from_nominal=True
            )
        source = Result({"nominal.cost": 10.0, "a.cost": 2.0, "b.cost": 6.0})
        self.assertEqual(source.get_expected(difference_from_nominal=True).cost, 6.0)
        self.assertEqual(source.get_expected().cost, 4.0)
        self.assertEqual(
            source.get_expected(with_nominal=True, round_value=False).cost, 6.0
        )
        self.assertEqual(Result({"only.cost": 4.0}).get_expected().cost, 4.0)
        self.assertEqual(dict(Result().get_expected()), {})
        self.assertEqual(dict(Result({"a.label": "name"}).get_expected()), {})

    def test_real_fault_sample_without_nominal_matches_retained_reference(self):
        model = ExampleFunction(sp={"end_time": 4.0})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "low")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0, 2.0, 3.0], weights=[0.1, 0.2, 0.7])
        result, history = propagate.fault_sample(
            model, sample, staged=False, include_nominal=False, showprogress=False
        )
        full_result, full_history = propagate.fault_sample(
            model, sample, staged=False, include_nominal=True, showprogress=False
        )
        self.assertFalse(any(k.startswith("nominal.") for k in result))
        for actual, reference in ((result, full_result), (history, full_history)):
            expected = reference.get_expected(sample, round_value=False)
            got = actual.get_expected(sample, round_value=False)
            self.assertEqual(set(got), set(expected))
            for key in got:
                np.testing.assert_allclose(got[key], expected[key])
        self.assertFalse(model.m.any_faults())


if __name__ == "__main__":
    unittest.main()
