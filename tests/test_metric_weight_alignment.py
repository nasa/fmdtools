#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for path-aligned metric rates and weights.

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
from itertools import permutations
import unittest

import numpy as np
from scipy.stats import bootstrap

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestMetricWeightAlignment(unittest.TestCase):
    """Keep each metric value paired with its own named rate or weight."""

    def test_named_rates_and_weights_are_independent_of_key_order(self):
        pairs = [("a.cost", 100.0), ("b.cost", 10.0), ("a.rate", 0.1), ("b.rate", 0.9)]
        for order in permutations(pairs):
            with self.subTest(order=[key for key, _ in order]):
                result = Result(dict(order))
                before = copy.deepcopy(result)
                self.assertAlmostEqual(
                    result.get_metric("cost", "expected", rates="rate"), 19.0
                )
                self.assertAlmostEqual(result.get_metric("cost", weights="rate"), 19.0)
                self.assertEqual(list(result), list(before))
                self.assertEqual(result, before)

    def test_array_histories_nested_paths_and_prefixes_keep_alignment(self):
        for cls in (Result, History):
            for nested in (False, True):
                for prefix in ("", "."):
                    with self.subTest(cls=cls.__name__, nested=nested, prefix=prefix):
                        result = cls(
                            {
                                "group.b.s.cost": np.array([10.0, 20.0]),
                                "group.a.s.cost": np.array([100.0, 200.0]),
                                "group.a.rate": 0.1,
                                "group.b.rate": 0.9,
                            }
                        )
                        if nested:
                            result = result.nest()
                        actual = result.get_metric(
                            "s.cost", "expected", rates="rate", prefix=prefix, axis=0
                        )
                        np.testing.assert_allclose(actual, [19.0, 38.0])
                        actual = result.get_metric(
                            "s.cost", weights="rate", prefix=prefix, axis=0
                        )
                        np.testing.assert_allclose(actual, [19.0, 38.0])

    def test_external_mappings_follow_selected_values_not_other_fields(self):
        result = Result(
            {"a.note": 5.0, "b.cost": 10.0, "a.cost": 100.0, "unused.note": 2.0}
        )
        for weights in ({"a": 0.1, "b": 0.9}, {"b": 0.9, "a": 0.1}):
            with self.subTest(order=list(weights)):
                before = copy.deepcopy(weights)
                self.assertAlmostEqual(
                    result.get_metric("cost", "expected", rates=weights), 19.0
                )
                self.assertAlmostEqual(result.get_metric("cost", weights=weights), 19.0)
                self.assertEqual(weights, before)
        actual = result.get_metric("cost", "expected", rates=[0.9, 0.1])
        self.assertAlmostEqual(actual, 19.0)
        self.assertEqual(result.get_vals("cost", rates=None), ([10.0, 100.0], None))

    def test_missing_named_rates_raise_and_unselected_rates_do_not_enter(self):
        result = Result({"a.cost": 100.0, "b.cost": 10.0, "a.rate": 0.1, "c.rate": 0.9})
        for kwargs in ({"rates": "rate"}, {"weights": "rate"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(KeyError):
                    result.get_metric("cost", **kwargs)
        result["b.rate"] = 0.9
        self.assertAlmostEqual(
            result.get_metric("cost", "expected", rates="rate"), 19.0
        )
        scalar = Result({"cost": 100.0, "rate": 0.25})
        self.assertEqual(scalar.get_vals("cost", rates="rate"), ([100.0], [0.25]))
        self.assertEqual(scalar.get_vals(rates="rate"), ([0.25],))

    def test_actual_simulation_metrics_and_bootstrap_pair_the_same_scenarios(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x")
        sample = ParameterSample(domain)
        for x in (1.0, 2.0, 4.0, 7.0, 11.0, 19.0):
            sample.add_variable_scenario(x)
        result, history = propagate.parameter_sample(
            ExampleFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        for output, key in ((result, "xy"), (history, "s.x")):
            with self.subTest(cls=type(output).__name__):
                selected = output.get_values(key)
                names = list(selected)
                rates = np.arange(1.0, len(names) + 1)
                mixed = type(output)(selected)
                for name, rate in reversed(list(zip(names, rates))):
                    mixed[name.removesuffix(key) + "rate"] = rate
                values = np.asarray(list(selected.values()))
                prepared = (values.T * rates).T
                np.testing.assert_allclose(
                    mixed.get_metric(key, "expected", rates="rate", axis=0),
                    np.sum(prepared, axis=0),
                )
                controls = {"random_state": 17, "n_resamples": 199, "batch": 31}
                algorithm = "basic" if output is history else "BCa"
                expected = bootstrap(
                    (prepared,), np.mean, axis=0, method=algorithm, **controls
                )
                actual = mixed.get_metric_ci(key, rates="rate", **controls)
                np.testing.assert_allclose(actual[0], np.mean(prepared, axis=0))
                np.testing.assert_allclose(actual[1], expected.confidence_interval.low)
                np.testing.assert_allclose(actual[2], expected.confidence_interval.high)


if __name__ == "__main__":
    unittest.main()
