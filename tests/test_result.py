#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for result aggregation and expected-value calculations.

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

from itertools import permutations
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim.sample import BaseSample, FaultDomain, FaultSample
from fmdtools.sim.scenario import Scenario


class OrderedSample(BaseSample):
    """Small sample of actual scenarios with known, unequal rates."""

    def __init__(self, order=("a", "b")):
        rates = {"a": 1.0, "b": 3.0}
        self.sampled = tuple(
            Scenario(name=name, rate=rates[name]) for name in order
        )

    def scenarios(self):
        return list(self.sampled)


def result_with_order(result_type, order):
    """Give histories three time samples and Results one scalar per scenario."""
    values = {"nominal": 2.0, "a": 10.0, "b": 100.0}
    scale = np.array([1.0, 2.0, -1.0]) if result_type is History else 1.0
    return ResultLike(result_type, values, order, scale), scale


def ResultLike(result_type, values, order, scale):
    """Construct a Result or History with a specified scenario ordering."""
    return result_type({name + ".metric": values[name] * scale for name in order})


class TestResultExpected(unittest.TestCase):
    """Test scenario-name alignment in expected-value calculations."""

    def test_expected_metric_is_independent_of_result_order(self):
        for order in permutations(("nominal", "a", "b")):
            for result_type in (Result, History):
                for with_nominal in (False, True):
                    with self.subTest(
                        order=order,
                        result_type=result_type.__name__,
                        with_nominal=with_nominal,
                    ):
                        result, scale = result_with_order(result_type, order)
                        expected = (
                            (2.0 + 10.0 + 3.0 * 100.0) / 5.0
                            if with_nominal
                            else (10.0 + 3.0 * 100.0) / 4.0
                        )
                        original_keys = list(result)
                        for difference in (False, True):
                            actual = result.get_expected(
                                OrderedSample(),
                                with_nominal=with_nominal,
                                difference_from_nominal=difference,
                            )
                            reference = 2.0 - expected if difference else expected
                            self.assertIs(type(actual), result_type)
                            np.testing.assert_allclose(
                                actual.metric,
                                reference * scale,
                                rtol=1e-14,
                                atol=1e-14,
                            )
                        self.assertEqual(list(result), original_keys)

    def test_reversed_sample_order_keeps_named_rates(self):
        for result_type in (Result, History):
            for with_nominal in (False, True):
                with self.subTest(
                    result_type=result_type.__name__,
                    with_nominal=with_nominal,
                ):
                    result, scale = result_with_order(
                        result_type, ("nominal", "a", "b")
                    )
                    expected = 62.4 if with_nominal else 77.5
                    actual = result.get_expected(
                        OrderedSample(("b", "a")),
                        with_nominal=with_nominal,
                    )
                    np.testing.assert_allclose(
                        actual.metric,
                        expected * scale,
                        rtol=1e-14,
                        atol=1e-14,
                    )

    def test_unweighted_expectation_is_unchanged(self):
        for result_type in (Result, History):
            for with_nominal in (False, True):
                with self.subTest(
                    result_type=result_type.__name__,
                    with_nominal=with_nominal,
                ):
                    result, scale = result_with_order(
                        result_type, ("b", "nominal", "a")
                    )
                    expected = 112.0 / 3.0 if with_nominal else 55.0
                    actual = result.get_expected(with_nominal=with_nominal)
                    np.testing.assert_allclose(
                        actual.metric,
                        expected * scale,
                        rtol=1e-14,
                        atol=1e-14,
                    )

    def test_real_fault_sample_rates_follow_scenario_names(self):
        model = ExFxnArch()
        domain = FaultDomain(model)
        domain.add_fault("ex_fxn", "low")
        domain.add_fault("ex_fxn2", "short")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0])
        scenarios = sample.scenarios()
        values = {
            scenario.name: 10.0 * (index + 1)
            for index, scenario in enumerate(scenarios)
        }
        result = Result(
            {
                "nominal.metric": 3.0,
                **{
                    scenario.name + ".metric": values[scenario.name]
                    for scenario in reversed(scenarios)
                },
            }
        )
        for with_nominal in (False, True):
            with self.subTest(with_nominal=with_nominal):
                numerator = sum(
                    values[scenario.name] * scenario.rate
                    for scenario in scenarios
                )
                denominator = sum(scenario.rate for scenario in scenarios)
                if with_nominal:
                    numerator += 3.0
                    denominator += 1.0
                actual = result.get_expected(
                    sample, with_nominal=with_nominal
                ).metric
                np.testing.assert_allclose(actual, numerator / denominator)


if __name__ == "__main__":
    unittest.main()
