#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exact nominal-scenario filtering in expectations.

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
from types import SimpleNamespace
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestExactNominalExpectation(unittest.TestCase):
    def test_similar_names_remain_in_scalar_and_history_expectations(self):
        for name in ("nominal_fault", "non_nominal", "fault_nominal", "nominal1"):
            for cls in (Result, History):
                with self.subTest(name=name, cls=cls.__name__):
                    scale = np.array([1.0, 2.0, 3.0]) if cls is History else 1.0
                    result = cls(
                        {
                            "nominal.metric": 2.0 * scale,
                            "fault.metric": 10.0 * scale,
                            name + ".metric": 100.0 * scale,
                            "nominal.label": "reference",
                        }
                    )
                    original = copy.deepcopy(dict(result))
                    for include, difference in (
                        (False, False),
                        (False, True),
                        (True, False),
                        (True, True),
                    ):
                        expected = 112.0 / 3 if include else 55.0
                        if difference:
                            expected = 2.0 - expected
                        actual = result.get_expected(
                            with_nominal=include, difference_from_nominal=difference
                        )
                        self.assertIs(type(actual), cls)
                        self.assertEqual(set(actual), {"metric"})
                        np.testing.assert_allclose(actual["metric"], expected * scale)
                    for key, value in result.items():
                        np.testing.assert_array_equal(value, original[key])

    def test_weighted_expectation_uses_rates_of_retained_similar_names(self):
        for cls in (Result, History):
            for order in (
                ("fault", "nominal", "non_nominal"),
                ("non_nominal", "nominal", "fault"),
            ):
                with self.subTest(cls=cls.__name__, order=order):
                    scale = np.array([1.0, 3.0]) if cls is History else 1.0
                    values = {"nominal": 2.0, "fault": 10.0, "non_nominal": 100.0}
                    result = cls(
                        {name + ".metric": values[name] * scale for name in order}
                    )
                    rates = [
                        SimpleNamespace(name="non_nominal", rate=3.0),
                        SimpleNamespace(name="fault", rate=1.0),
                    ]
                    sample = SimpleNamespace(scenarios=lambda: rates)
                    for include in (False, True):
                        expected = 312.0 / 5 if include else 310.0 / 4
                        for difference in (False, True):
                            reference = 2.0 - expected if difference else expected
                            actual = result.get_expected(
                                sample,
                                with_nominal=include,
                                difference_from_nominal=difference,
                            )
                            np.testing.assert_allclose(
                                actual["metric"], reference * scale
                            )

    def test_one_similarly_named_fault_does_not_leave_an_empty_average(self):
        for name in ("nominal_fault", "non_nominal", "fault_nominal"):
            for cls in (Result, History):
                with self.subTest(name=name, cls=cls.__name__):
                    value = np.array([4.0, 9.0]) if cls is History else 9.0
                    result = cls({"nominal.metric": value * 0, name + ".metric": value})
                    np.testing.assert_array_equal(
                        result.get_expected()["metric"], value
                    )

    def test_exact_reference_filter_and_explicit_inclusion_are_unchanged(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                scale = np.array([1.0, 2.0]) if cls is History else 1.0
                result = cls(
                    {"nominal.metric": 2.0 * scale, "fault.metric": 10.0 * scale}
                )
                np.testing.assert_array_equal(
                    result.get_expected()["metric"], 10.0 * scale
                )
                np.testing.assert_array_equal(
                    result.get_expected(with_nominal=True)["metric"], 6.0 * scale
                )
                np.testing.assert_array_equal(
                    result.get_expected(difference_from_nominal=True)["metric"],
                    -8.0 * scale,
                )

    def test_fault_simulations_with_nominal_in_the_function_name_can_be_averaged(self):
        model = ExampleFunction("nominal_controller", sp={"end_time": 3.0})
        domain = FaultDomain(model)
        domain.add_fault("nominal_controller", "low")
        sample = FaultSample(domain)
        sample.add_fault_times([1.0, 2.0])
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        scenarios = sample.scenarios()
        self.assertEqual(
            [scenario.name for scenario in scenarios],
            ["nominal_controller_low_t1p0", "nominal_controller_low_t2p0"],
        )
        for data, metric in ((result, "tend.classify.xy"), (history, "s.x")):
            before = copy.deepcopy(dict(data))
            for include in (False, True):
                for difference in (False, True):
                    with self.subTest(
                        cls=type(data).__name__, include=include, difference=difference
                    ):
                        numerator = sum(
                            data[scenario.name + "." + metric] * scenario.rate
                            for scenario in scenarios
                        )
                        denominator = sum(scenario.rate for scenario in scenarios)
                        nominal = data["nominal." + metric]
                        if include:
                            numerator += nominal
                            denominator += 1.0
                        expected = numerator / denominator
                        if difference:
                            expected = nominal - expected
                        actual = data.get_expected(
                            sample,
                            with_nominal=include,
                            difference_from_nominal=difference,
                        )
                        np.testing.assert_allclose(actual[metric], expected)
            for key, value in data.items():
                np.testing.assert_array_equal(value, before[key])


if __name__ == "__main__":
    unittest.main()
