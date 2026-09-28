#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for statistic arguments in bootstrap confidence intervals.

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

from functools import partial
import unittest

import numpy as np
from scipy.stats import bootstrap

from fmdtools.analyze.common import calc_metric_ci
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


VALUES = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 19.0])
BOOTSTRAP = {"random_state": 41, "n_resamples": 199, "batch": 37}


def adjusted_mean(values, axis=0, offset=0.0, scale=1.0):
    return offset + scale * np.mean(values, axis=axis)


class TestBootstrapStatisticArguments(unittest.TestCase):
    def assert_matches_reference(
        self, data, method, arguments, axis=0, algorithm="BCa", **options
    ):
        def statistic(sample, axis=0):
            return method(sample, axis=axis, **arguments)

        before = np.array(data, copy=True)
        expected = bootstrap(
            (data,), statistic, axis=axis, method=algorithm, **BOOTSTRAP, **options
        )
        actual = calc_metric_ci(
            data, method=method, axis=axis, **BOOTSTRAP, **options, **arguments
        )
        for value, reference in zip(
            actual,
            (
                statistic(data, axis),
                expected.confidence_interval.low,
                expected.confidence_interval.high,
            ),
        ):
            np.testing.assert_allclose(value, reference, rtol=1e-13, atol=1e-13)
            self.assertEqual(np.shape(value), np.shape(reference))
        np.testing.assert_array_equal(data, before)
        return actual

    def test_required_quantile_arguments_reach_every_resampled_statistic(self):
        for method, arguments in (
            (np.quantile, {"q": 0.75}),
            (np.percentile, {"q": 25.0}),
        ):
            for vectorized in (True, False):
                with self.subTest(method=method.__name__, vectorized=vectorized):
                    self.assert_matches_reference(
                        VALUES, method, arguments, vectorized=vectorized
                    )

    def test_standard_deviation_correction_matches_the_point_estimate(self):
        for method in (np.std, np.var):
            for correction in (0, 1, 2):
                with self.subTest(method=method.__name__, ddof=correction):
                    self.assert_matches_reference(VALUES, method, {"ddof": correction})

    def test_parameters_are_bound_on_bca_and_basic_paths_and_nonzero_axes(self):
        for axis in (0, 1, -1):
            for constant_column in (False, True):
                with self.subTest(axis=axis, constant=constant_column):
                    second = (
                        np.full(VALUES.size, 3.0)
                        if constant_column
                        else VALUES[::-1] * 2
                    )
                    data = np.stack((VALUES, second), axis=1)
                    if axis != 0:
                        data = data.T
                    self.assert_matches_reference(
                        data,
                        adjusted_mean,
                        {"offset": 4.0, "scale": 3.0},
                        axis=axis,
                        algorithm="basic" if constant_column else "BCa",
                    )

    def test_affine_mean_interval_has_the_same_shift_and_scale(self):
        unmodified = bootstrap((VALUES,), np.mean, **BOOTSTRAP)
        actual = calc_metric_ci(
            VALUES, method=adjusted_mean, offset=7.0, scale=2.0, **BOOTSTRAP
        )
        expected = (
            7.0 + 2.0 * np.mean(VALUES),
            7.0 + 2.0 * unmodified.confidence_interval.low,
            7.0 + 2.0 * unmodified.confidence_interval.high,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_partial_callable_and_bootstrap_controls_remain_supported(self):
        for function, args in (
            (partial(adjusted_mean, offset=2.0), {"scale": 3.0}),
            (np.mean, {}),
        ):
            with self.subTest(function=function):
                self.assert_matches_reference(
                    VALUES, function, args, confidence_level=0.8, alternative="less"
                )
        reference = bootstrap(
            (VALUES,),
            partial(adjusted_mean, offset=2.0),
            confidence_level=0.8,
            **BOOTSTRAP,
        )
        actual = calc_metric_ci(
            VALUES, method=adjusted_mean, offset=2.0, interval=80, **BOOTSTRAP
        )
        np.testing.assert_allclose(actual[1:], reference.confidence_interval)

    def test_preprocessing_rates_is_shared_by_point_and_interval(self):
        rates = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
        prepared = VALUES * rates / rates.sum()
        expected = bootstrap((prepared,), partial(np.std, ddof=1), **BOOTSTRAP)
        actual = calc_metric_ci(
            VALUES, method=np.std, ddof=1, rates=rates, r_norm=True, **BOOTSTRAP
        )
        np.testing.assert_allclose(
            actual, (np.std(prepared, ddof=1), *expected.confidence_interval)
        )
        np.testing.assert_array_equal(rates, [1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

    def test_constant_data_and_rejected_weights_keep_existing_policy(self):
        for method, arguments, expected in (
            (np.std, {"ddof": 1}, 0.0),
            (np.quantile, {"q": 0.75}, 5.0),
            (adjusted_mean, {"offset": 2.0}, 7.0),
        ):
            with self.subTest(method=method.__name__):
                actual = calc_metric_ci(
                    np.full(6, 5.0), method=method, return_anyway=True, **arguments
                )
                np.testing.assert_allclose(actual, [expected] * 3)
                with self.assertRaisesRegex(Exception, "All data are the same"):
                    calc_metric_ci(np.full(6, 5.0), method=method, **arguments)
        with self.assertRaisesRegex(Exception, "Weights not able"):
            calc_metric_ci(VALUES, method=np.std, ddof=1, weights=np.ones(6))

    def test_actual_parameter_simulation_results_and_histories_use_statistic_args(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x")
        sample = ParameterSample(domain)
        for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            sample.add_variable_scenario(value)
        result, history = propagate.parameter_sample(model, sample, showprogress=False)
        for output, key, method, args, algorithm in (
            (result, "xy", np.quantile, {"q": 0.75}, "BCa"),
            (history, "s.x", np.std, {"ddof": 1}, "basic"),
        ):
            with self.subTest(cls=type(output).__name__):
                values = np.array(list(output.get_values(key).values()))
                before = output.copy()
                expected = bootstrap(
                    (values,),
                    partial(method, **args),
                    axis=0,
                    method=algorithm,
                    **BOOTSTRAP,
                )
                actual = output.get_metric_ci(key, method=method, **args, **BOOTSTRAP)
                np.testing.assert_allclose(actual[0], method(values, axis=0, **args))
                np.testing.assert_allclose(actual[1], expected.confidence_interval.low)
                np.testing.assert_allclose(actual[2], expected.confidence_interval.high)
                for name in output:
                    np.testing.assert_array_equal(output[name], before[name])


if __name__ == "__main__":
    unittest.main()
