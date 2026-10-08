#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for confidence-interval metric registration in tables.

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
from scipy.stats import bootstrap

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import BaseComparison, Comparison, NestedComparison
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


def ci_reference(values, seed=43):
    values = np.asarray(values)
    if np.all(values == values.flat[0]):
        return values.mean(axis=0), values.mean(axis=0), values.mean(axis=0)
    result = bootstrap((values,), np.mean, axis=0, n_resamples=199, rng=seed)
    return (
        values.mean(axis=0),
        result.confidence_interval.low,
        result.confidence_interval.high,
    )


def make_sample(values):
    domain = ParameterDomain(dict)
    domain.add_variables("group", "value")
    sample = ParameterSample(domain)
    for group, value in values:
        sample.add_variable_scenario(group, value)
    return sample


class TestComparisonCIMetrics(unittest.TestCase):
    def test_ci_only_metrics_have_point_estimates_and_bounds(self):
        for container in (Result, History):
            with self.subTest(container=container.__name__):
                values = [1.0, 2.0, 8.0, 12.0]
                data = container(
                    {"case" + str(i) + ".risk": value for i, value in enumerate(values)}
                )
                before = copy.deepcopy(data)
                table = BaseComparison(
                    data,
                    {("all",): ["case" + str(i) for i in range(4)]},
                    metrics=[],
                    ci_metrics=["risk"],
                    stats={"risk": np.mean},
                    ci_kwargs={"n_resamples": 199, "rng": 43},
                )
                expected = ci_reference(values)
                self.assertEqual(list(table.data), ["risk", "risk_lb", "risk_ub"])
                for column, value in zip(table.data, expected):
                    np.testing.assert_allclose(table[column][("all",)], value)
                self.assertEqual(data, before)

    def test_overlapping_and_repeated_metric_names_are_evaluated_once(self):
        result = Result(
            {
                "a.risk": 1.0,
                "b.risk": 2.0,
                "c.risk": 8.0,
                "d.risk": 12.0,
                "a.cost": 4.0,
                "b.cost": 7.0,
                "c.cost": 9.0,
                "d.cost": 3.0,
            }
        )
        groups = {(): ["a", "b", "c", "d"]}
        ordinary = BaseComparison(
            result,
            groups,
            metrics=["risk", "cost"],
            ci_metrics=["risk"],
            stats={"risk": np.mean},
            ci_kwargs={"n_resamples": 199, "rng": np.random.default_rng(43)},
        )
        repeated = BaseComparison(
            result,
            groups,
            metrics=["risk", "cost", "risk"],
            ci_metrics=["risk", "risk"],
            stats={"risk": np.mean},
            ci_kwargs={"n_resamples": 199, "rng": np.random.default_rng(43)},
        )
        expected = ci_reference([1.0, 2.0, 8.0, 12.0])
        self.assertEqual(list(repeated.data), ["risk", "cost", "risk_lb", "risk_ub"])
        for column, value in zip(("risk", "risk_lb", "risk_ub"), expected):
            self.assertAlmostEqual(ordinary[column][()], value)
            self.assertAlmostEqual(repeated[column][()], value)
        self.assertEqual(repeated["cost"], ordinary["cost"])

    def test_multiple_ci_metrics_constant_groups_and_array_outputs(self):
        values = np.array([[1.0, 10.0], [2.0, 15.0], [8.0, 25.0], [12.0, 45.0]])
        result = Result({"r" + str(i) + ".vector": row for i, row in enumerate(values)})
        result.update({"r" + str(i) + ".constant": 7.0 for i in range(4)})
        metrics, cis = [], ["vector", "constant"]
        table = BaseComparison(
            result,
            {(): ["r" + str(i) for i in range(4)]},
            metrics=metrics,
            ci_metrics=cis,
            default_stat=np.mean,
            ci_kwargs={"rng": 43, "n_resamples": 199},
        )
        for column, expected in zip(
            ("vector", "vector_lb", "vector_ub"), ci_reference(values)
        ):
            np.testing.assert_allclose(table[column][()], expected)
        for column in ("constant", "constant_lb", "constant_ub"):
            self.assertEqual(table[column][()], 7.0)
        self.assertEqual(metrics, [])
        self.assertEqual(cis, ["vector", "constant"])
        empty = BaseComparison(
            Result(), {}, metrics=[], ci_metrics=["risk"], default_stat=np.mean
        )
        self.assertEqual(empty.data, {"risk": {}, "risk_lb": {}, "risk_ub": {}})

    def test_comparison_and_nested_comparison_expose_ci_only_columns(self):
        inner = make_sample(
            [("a", 1.0), ("a", 2.0), ("a", 8.0), ("b", 4.0), ("b", 4.0)]
        )
        records = {s.name + ".risk": s.p["value"] for s in inner.scenarios()}
        result = Result(records)
        options = {
            "metrics": [],
            "ci_metrics": ["risk"],
            "default_stat": np.mean,
            "ci_kwargs": {"n_resamples": 199, "rng": 43},
        }
        table = Comparison(result, inner, factors=["p.group"], **options)
        outer = make_sample([("outer", 0.0)])
        parent = outer.scen_names()[0]
        nested = NestedComparison(
            Result({parent + "." + key: value for key, value in records.items()}),
            outer,
            [],
            {parent: inner},
            ["p.group"],
            **options,
        )
        for group, values in (("a", [1.0, 2.0, 8.0]), ("b", [4.0, 4.0])):
            for column, expected in zip(
                ("risk", "risk_lb", "risk_ub"), ci_reference(values)
            ):
                self.assertAlmostEqual(table[column][(group,)], expected)
                self.assertAlmostEqual(nested[column][(group,)], expected)
        self.assertEqual(
            list(table.as_table(sort=False).columns), ["risk", "risk_lb", "risk_ub"]
        )

    def test_real_parameter_simulation_produces_ci_only_comparison(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        sample = ParameterSample(domain)
        for value in (1.0, 2.0, 4.0, 7.0):
            sample.add_variable_scenario(2.0, value)
        result, history = propagate.parameter_sample(
            ExampleFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        metric = "tend.classify.xy"
        values = [
            result[scenario.name + "." + metric] for scenario in sample.scenarios()
        ]
        before = copy.deepcopy(result)
        table = Comparison(
            result,
            sample,
            factors=["p.x"],
            metrics=[],
            ci_metrics=[metric],
            default_stat=np.mean,
            ci_kwargs={"n_resamples": 199, "rng": 43},
        )
        for column, expected in zip(
            (metric, metric + "_lb", metric + "_ub"), ci_reference(values)
        ):
            self.assertAlmostEqual(table[column][(2.0,)], expected)
        self.assertEqual(result, before)
        for scenario in sample.scenarios():
            self.assertEqual(len(history[scenario.name + ".s.x"]), 4)


if __name__ == "__main__":
    unittest.main()
