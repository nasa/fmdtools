#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for matching nested comparison samples to their parents.

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

from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import NestedComparison
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


def sample(variable, values, names):
    domain = ParameterDomain(dict)
    domain.add_variable(variable)
    result = ParameterSample(domain)
    for value, name in zip(values, names):
        result.add_variable_scenario(value, name=name)
    return result


def fixture(distinct_outer=False, disjoint_names=False):
    outer = sample("family", ["a", "b" if distinct_outer else "a"], ["first", "second"])
    names = outer.scen_names()
    inner = {
        names[0]: sample("level", [1, 2], ["red", "blue"]),
        names[1]: sample(
            "level", [2, 1], ["green", "gold"] if disjoint_names else ["red", "blue"]
        ),
    }
    records = {}
    expected = {}
    for outer_scen, scores in zip(outer.scenarios(), ([10.0, 20.0], [200.0, 100.0])):
        for inner_scen, score in zip(inner[outer_scen.name].scenarios(), scores):
            records[outer_scen.name + "." + inner_scen.name + ".score"] = score
            group = (outer_scen.p["family"], inner_scen.p["level"])
            expected.setdefault(group, []).append(score)
    return outer, inner, Result(records), expected


class TestNestedComparisonParentage(unittest.TestCase):
    def test_reused_inner_names_are_classified_using_their_own_parent(self):
        outer, inner, result, expected = fixture()
        for method in ("sum", "average"):
            with self.subTest(method=method):
                table = NestedComparison(
                    result,
                    outer,
                    ["p.family"],
                    inner,
                    ["p.level"],
                    metrics=["score"],
                    default_stat=method,
                )
                self.assertEqual(
                    table["score"],
                    {
                        k: sum(v) / (len(v) if method == "average" else 1)
                        for k, v in expected.items()
                    },
                )
                self.assertEqual(table.factors, ["p.family", "p.level"])
                self.assertEqual(set(table.as_table(sort=False).index), set(expected))

    def test_disjoint_inner_samples_merge_instead_of_overwriting_shared_groups(self):
        outer, inner, result, expected = fixture(disjoint_names=True)
        original = copy.deepcopy(result)
        originals = {
            name: copy.deepcopy(scenarios.scenarios())
            for name, scenarios in inner.items()
        }
        for reverse_inner in (False, True):
            for reverse_results in (False, True):
                with self.subTest(
                    reverse_inner=reverse_inner, reverse_results=reverse_results
                ):
                    samps = (
                        dict(reversed(list(inner.items()))) if reverse_inner else inner
                    )
                    records = (
                        Result(dict(reversed(list(result.items()))))
                        if reverse_results
                        else result
                    )
                    table = NestedComparison(
                        records,
                        outer,
                        ["p.family"],
                        samps,
                        ["p.level"],
                        metrics=["score"],
                        default_stat="sum",
                    )
                    self.assertEqual(
                        table["score"], {k: sum(v) for k, v in expected.items()}
                    )
        self.assertEqual(result, original)
        for name, original_scens in originals.items():
            self.assertEqual(
                [s.asdict() for s in inner[name].scenarios()],
                [s.asdict() for s in original_scens],
            )

    def test_only_real_parent_factor_combinations_are_present(self):
        outer = sample("family", ["a", "b"], ["one", "two"])
        first, second = outer.scen_names()
        inner = {
            first: sample("level", [1], ["same"]),
            second: sample("level", [2], ["same"]),
        }
        result = Result({first + ".same_0.score": 10.0, second + ".same_0.score": 20.0})
        table = NestedComparison(
            result,
            outer,
            ["p.family"],
            inner,
            ["p.level"],
            metrics=["score"],
            default_stat="sum",
        )
        self.assertEqual(table["score"], {("a", 1): 10.0, ("b", 2): 20.0})
        subset = NestedComparison(
            result,
            outer,
            ["p.family"],
            {first: inner[first]},
            ["p.level"],
            metrics=["score"],
            default_stat="sum",
        )
        self.assertEqual(subset["score"], {("a", 1): 10.0})

    def test_single_parent_and_empty_factor_lists_keep_expected_totals(self):
        outer, inner, result, _ = fixture()
        first = outer.scen_names()[0]
        for selected in (inner, {first: inner[first]}):
            with self.subTest(parents=list(selected)):
                table = NestedComparison(
                    result,
                    outer,
                    [],
                    selected,
                    [],
                    metrics=["score"],
                    default_stat="sum",
                )
                self.assertEqual(
                    table["score"], {(): 330.0 if len(selected) == 2 else 30.0}
                )
        empty = ParameterSample(ParameterDomain(dict))
        table = NestedComparison(
            result, outer, [], {first: empty}, [], metrics=["score"], default_stat="sum"
        )
        self.assertEqual(table["score"], {})

    def test_real_nested_simulation_uses_only_each_parents_selected_faults(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        outer = ParameterSample(domain)
        outer.add_variable_scenario(2.0, 3.0)
        outer.add_variable_scenario(4.0, 1.0)
        result, _, inner = propagate.nested_sample(
            ExampleFunction(sp={"end_time": 3.0}),
            outer,
            staged=False,
            include_nominal=True,
            showprogress=False,
            faultdomains={"low": (("fault", "examplefunction", "low"), {})},
            faultsamples={"times": (("fault_times", "low", [1.0, 2.0]), {})},
        )
        expected = {}
        for i, scenario in enumerate(outer.scenarios()):
            faults = inner[scenario.name].faultsamples["times"]
            faults.prune_scenarios("time", comparator=np.equal, value=float(i + 1))
            for fault in faults.scenarios():
                expected[(scenario.p["x"], fault.time)] = result[
                    scenario.name + "." + fault.name + ".tend.classify.xy"
                ]
        table = NestedComparison(
            result,
            outer,
            ["p.x"],
            inner,
            ["time"],
            metrics=["tend.classify.xy"],
            default_stat="sum",
        )
        self.assertEqual(table["tend.classify.xy"], expected)
        self.assertEqual(len(expected), 2)


if __name__ == "__main__":
    unittest.main()
