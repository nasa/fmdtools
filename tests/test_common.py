#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for common analysis metric reductions.

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

from fmdtools.analyze.common import calc_expected, calc_metric, calc_rate
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestDefaultMetricRates(unittest.TestCase):
    """Uniform probabilities sum to one over the selected reduction axes."""

    def test_expected_values_and_event_rates_use_reduced_axes_only(self):
        values = np.arange(24.0, dtype=float).reshape(2, 3, 4)
        values[values % 3 == 0] = 0.0
        for axis in (None, 0, 1, -1, (0, 2), (-1, 0), (0, 1, 2), ()):
            for keepdims in (False, True):
                for name, direct in (("expected", calc_expected), ("rate", calc_rate)):
                    with self.subTest(axis=axis, keepdims=keepdims, metric=name):
                        data = values != 0 if name == "rate" else values
                        expected = np.mean(data, axis=axis, keepdims=keepdims)
                        before = values.copy()
                        for function in (
                            direct,
                            lambda data, **kw: calc_metric(data, name, **kw),
                        ):
                            actual = function(
                                values, axis=axis, keepdims=keepdims, round_value=False
                            )
                            np.testing.assert_allclose(actual, expected, rtol=1e-13)
                            self.assertEqual(np.shape(actual), np.shape(expected))
                        np.testing.assert_array_equal(values, before)

    def test_repeating_unreduced_dimensions_does_not_dilute_the_metric(self):
        column = np.array([[0.0], [2.0], [4.0]])
        for repetitions in (1, 3, 11):
            for name, expected in (("expected", 2.0), ("rate", 2.0 / 3.0)):
                with self.subTest(repetitions=repetitions, name=name):
                    values = np.repeat(column, repetitions, axis=1)
                    actual = calc_metric(values, name, axis=0, round_value=False)
                    np.testing.assert_allclose(actual, np.full(repetitions, expected))

    def test_explicit_rates_normalization_and_preprocessing_are_unchanged(self):
        values = np.array([[0.0, 2.0, 4.0], [3.0, 0.0, 6.0]])
        for name, direct in (("expected", calc_expected), ("rate", calc_rate)):
            for rates in (
                0.4,
                [2.0, 6.0],
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ):
                for normalize in (False, True):
                    with self.subTest(
                        name=name, rates=np.shape(rates), normalize=normalize
                    ):
                        r = np.asarray(rates)
                        if normalize:
                            r = r / r.sum()
                        data = values != 0 if name == "rate" else values
                        weighted = (data.T * r.T).T
                        actual = direct(
                            values,
                            axis=0,
                            rates=rates,
                            r_norm=normalize,
                            round_value=False,
                        )
                        np.testing.assert_allclose(actual, weighted.sum(axis=0))
        actual = calc_expected(values, axis=0, dtype=bool, round_value=False)
        np.testing.assert_allclose(actual, (values != 0).mean(axis=0))

    def test_scalar_flattened_and_invalid_axis_controls_are_preserved(self):
        for axis in (None, 0, -1, ()):
            with self.subTest(axis=axis):
                self.assertEqual(calc_expected(4.0, axis=axis), 4.0)
                self.assertEqual(calc_rate(4.0, axis=axis), 1.0)
        self.assertEqual(calc_expected([0.0, 1.0, 0.0]), 0.333333)
        self.assertEqual(calc_rate([0.0, 1.0, 0.0]), 0.333333)
        self.assertEqual(calc_expected([0.0, 1.0, 0.0], res=0.1), 0.3)
        for name in ("expected", "rate"):
            with self.subTest(name=name):
                for axis in (3, -4, (0, 0)):
                    with (
                        self.subTest(axis=axis),
                        self.assertRaises((IndexError, ValueError)),
                    ):
                        calc_metric(np.ones((2, 3)), name, axis=axis)

    def test_result_and_history_metrics_agree_with_uniform_sample_means(self):
        values = np.array([[0.0, 2.0, 4.0], [3.0, 0.0, 6.0]])
        for cls in (Result, History):
            for nested in (False, True):
                for name in ("expected", "rate"):
                    with self.subTest(cls=cls.__name__, nested=nested, metric=name):
                        output = cls(
                            {
                                str(i) + ".signal": row.copy()
                                for i, row in enumerate(values)
                            }
                        )
                        if nested:
                            output = output.nest()
                        before = output.copy()
                        actual = output.get_metric(
                            "signal", method=name, axis=0, round_value=False
                        )
                        expected = (
                            (values != 0).mean(0) if name == "rate" else values.mean(0)
                        )
                        np.testing.assert_allclose(actual, expected)
                        self.assertEqual(output, before)

    def test_real_parameter_simulations_do_not_dilute_per_time_expectations(self):
        for duration in (2.0, 5.0):
            with self.subTest(duration=duration):
                model = ExampleFunction(sp={"end_time": duration})
                domain = ParameterDomain(ExampleParameter)
                domain.add_variable("x")
                sample = ParameterSample(domain)
                for value in (0.0, 1.0, 3.0):
                    sample.add_variable_scenario(value)
                result, history = propagate.parameter_sample(
                    model, sample, showprogress=False
                )
                before = history.copy()
                values = np.array(list(history.get_values("s.x").values()))
                for name in ("expected", "rate"):
                    expected = (
                        (values != 0).mean(0) if name == "rate" else values.mean(0)
                    )
                    actual = history.get_metric(
                        "s.x", method=name, axis=0, round_value=False
                    )
                    np.testing.assert_allclose(actual, expected)
                final_values = np.array(list(result.get_values("xy").values()))
                self.assertAlmostEqual(
                    result.get_metric("xy", method="expected", round_value=False),
                    final_values.mean(),
                )
                self.assertEqual(history, before)


if __name__ == "__main__":
    unittest.main()
