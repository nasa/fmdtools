#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for rounding controls with named metrics.

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

from fmdtools.analyze.common import calc_metric
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestNamedMetricRounding(unittest.TestCase):
    def scalar_examples(self):
        for name in ("average", "expected", "rate", "percent"):
            yield name, [0.0, 1.0, 0.0], 1.0 / 3.0
        yield "sum", [1.0 / 6.0, 1.0 / 6.0], 1.0 / 3.0

    def test_disabling_rounding_keeps_full_precision_for_every_scalar_alias(self):
        for name, values, expected in self.scalar_examples():
            with self.subTest(method=name):
                before = list(values)
                actual = calc_metric(values, method=name, round_value=False)
                self.assertEqual(actual, expected)
                self.assertNotEqual(actual, 0.333333)
                self.assertEqual(values, before)

    def test_custom_resolution_and_decimal_precision_are_applied_once(self):
        for name, values, _ in self.scalar_examples():
            for kwargs, expected in (
                ({"res": 1e-10, "min_r": 12}, 0.3333333333),
                ({"res": 0.125, "min_r": 3}, 0.375),
                ({"res": 1e-10, "min_r": 4}, 0.3333),
            ):
                with self.subTest(method=name, kwargs=kwargs):
                    self.assertEqual(
                        calc_metric(values, method=name, **kwargs), expected
                    )
        for name in ("average", "expected", "sum"):
            with self.subTest(method=name, boundary=True):
                self.assertEqual(calc_metric([0.2500004], method=name, res=0.1), 0.3)

    def test_rate_and_weight_preprocessing_keeps_unrounded_results(self):
        values = [0.0, 1.0, 2.0]
        cases = (
            ("average", {"weights": [1.0, 2.0, 4.0]}, 10.0 / 7.0),
            ("expected", {"rates": [1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0]}, 8.0 / 7.0),
            ("rate", {"rates": [1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0]}, 5.0 / 7.0),
        )
        for name, kwargs, expected in cases:
            with self.subTest(method=name):
                actual = calc_metric(values, method=name, round_value=False, **kwargs)
                self.assertAlmostEqual(actual, expected, places=14)
                self.assertGreater(abs(actual - np.round(actual, 6)), 1e-8)

    def test_default_rounding_and_nonfloating_results_are_preserved(self):
        for name, values, _ in self.scalar_examples():
            with self.subTest(method=name):
                self.assertEqual(calc_metric(values, method=name), 0.333333)
        for kwargs in ({}, {"round_value": False}, {"res": 0.3, "min_r": 9}):
            with self.subTest(kwargs=kwargs):
                actual = calc_metric([0, 1, 2], method="total", **kwargs)
                self.assertIsInstance(actual, np.integer)
                self.assertEqual(actual, 2)
                actual = calc_metric([1, 2], method="sum", **kwargs)
                self.assertIsInstance(actual, np.integer)
                self.assertEqual(actual, 3)

    def test_array_outputs_keep_the_existing_unrounded_axis_behavior(self):
        values = np.array([[1.0 / 7.0, 2.0 / 7.0], [3.0 / 7.0, 4.0 / 7.0]])
        before = values.copy()
        for name, method in (("average", np.average), ("sum", np.sum)):
            for axis in (0, 1, -1):
                for kwargs in ({}, {"round_value": False}, {"res": 1.0, "min_r": 0}):
                    with self.subTest(method=name, axis=axis, kwargs=kwargs):
                        actual = calc_metric(values, method=name, axis=axis, **kwargs)
                        np.testing.assert_array_equal(actual, method(values, axis=axis))
                        self.assertIsInstance(actual, np.ndarray)
        np.testing.assert_array_equal(values, before)

    def test_result_and_history_metric_apis_preserve_requested_precision(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                values = [0.0, 1.0, 0.0]
                output = cls(
                    {
                        name + ".value": value
                        for name, value in zip(("a", "b", "c"), values)
                    }
                )
                for method in ("average", "expected", "rate", "percent"):
                    with self.subTest(method=method):
                        self.assertEqual(
                            output.get_metric(
                                "value", method=method, round_value=False
                            ),
                            1.0 / 3.0,
                        )
                        self.assertEqual(
                            output.get_metric(
                                "value", method=method, res=1e-10, min_r=12
                            ),
                            0.3333333333,
                        )

    def test_actual_parameter_simulation_metrics_match_callable_reductions(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x")
        sample = ParameterSample(domain)
        for value in (1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0):
            sample.add_variable_scenario(value)
        result, history = propagate.parameter_sample(model, sample, showprogress=False)
        for output, key in ((result, "xy"), (history, "s.x")):
            before = output.copy()
            for name, function in (("average", np.average), ("sum", np.sum)):
                with self.subTest(cls=type(output).__name__, method=name):
                    actual = output.get_metric(key, method=name, round_value=False)
                    expected = output.get_metric(
                        key, method=function, round_value=False
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        output.get_metric(key, method=name, res=1e-10, min_r=12),
                        output.get_metric(key, method=function, res=1e-10, min_r=12),
                    )
            for name in output:
                np.testing.assert_array_equal(output[name], before[name])


if __name__ == "__main__":
    unittest.main()
