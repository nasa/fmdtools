#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for exact scenario boundaries in comparison groups.

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

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.sim.sample import (
    ParameterDomain,
    ParameterHistSample,
    ParameterResultSample,
)


class TestComparisonGroupScenarioBoundaries(unittest.TestCase):
    """Keep similarly named scenarios out of explicitly selected groups."""

    def make_result(self, result_type):
        data = {}
        for name, value in (
            ("case1", 1.0),
            ("case10", 10.0),
            ("case1_extra", 100.0),
            ("case2", 2.0),
        ):
            data[name + ".loss"] = (
                np.array([value, value + 1]) if result_type is History else value
            )
            data[name + ".time"] = (
                np.array([0.0, 1.0]) if result_type is History else 1.0
            )
        return result_type(data)

    def test_explicit_group_excludes_scenarios_with_the_same_prefix(self):
        for result_type in (Result, History):
            for selection in ("case1", ["case1"], ("case1",), {"case1": None}):
                with self.subTest(
                    result_type=result_type.__name__, selection=selection
                ):
                    result = self.make_result(result_type)
                    original_keys = list(result)
                    group = result.get_comp_groups("loss", selected=selection).selected
                    self.assertIs(type(group), result_type)
                    self.assertEqual(set(group), {"case1.loss", "case1.time"})
                    for key in group:
                        np.testing.assert_array_equal(group[key], result[key])
                    self.assertEqual(list(result), original_keys)

    def test_separate_groups_do_not_overlap_for_prefix_related_names(self):
        result = self.make_result(Result)
        groups = result.get_comp_groups(
            "loss", first=["case1"], tenth=["case10"], other=["case1_extra", "case2"]
        )
        self.assertEqual(set(groups.first), {"case1.loss", "case1.time"})
        self.assertEqual(set(groups.tenth), {"case10.loss", "case10.time"})
        self.assertEqual(
            set(groups.other),
            {"case1_extra.loss", "case1_extra.time", "case2.loss", "case2.time"},
        )
        self.assertFalse(set(groups.first) & set(groups.tenth))
        self.assertFalse(set(groups.first) & set(groups.other))

    def test_hierarchical_scenario_groups_keep_matching_descendants(self):
        result = Result(
            {
                "run1.nominal.loss": 1.0,
                "run1.fault1.loss": 2.0,
                "run1.fault10.loss": 3.0,
                "run10.nominal.loss": 4.0,
                "run1_extra.nominal.loss": 5.0,
            }
        )
        parent = result.get_comp_groups("loss", selected=["run1"]).selected
        self.assertEqual(
            set(parent), {"run1.nominal.loss", "run1.fault1.loss", "run1.fault10.loss"}
        )
        child = result.get_comp_groups("loss", selected=["run1.fault1"]).selected
        self.assertEqual(dict(child), {"run1.fault1.loss": 2.0})

    def test_default_nominal_group_does_not_include_nominal_prefix_scenarios(self):
        for result_type in (Result, History):
            with self.subTest(result_type=result_type.__name__):
                result = result_type(
                    {"nominal.loss": 1.0, "nominal_extra.loss": 9.0, "fault1.loss": 3.0}
                )
                groups = result.get_comp_groups("loss")
                self.assertEqual(set(groups.nominal), {"nominal.loss"})
                self.assertEqual(
                    set(groups["faulty"]), {"nominal_extra.loss", "fault1.loss"}
                )

    def test_default_group_custom_time_and_timer_exclusion_are_preserved(self):
        result = History(
            {
                "case1.loss": [1.0, 2.0],
                "case1.clock": [0.0, 1.0],
                "case1.t.timer.clock": [8.0, 9.0],
                "case10.loss": [10.0, 20.0],
                "case10.clock": [0.0, 1.0],
            }
        )
        expected = {"case1.loss", "case1.clock", "case10.loss", "case10.clock"}
        groups = result.get_comp_groups("loss", time="clock")
        self.assertEqual(set(groups.default), expected)
        explicit_default = result.get_comp_groups(
            "loss", time="clock", all_cases="default"
        )
        self.assertEqual(set(explicit_default.all_cases), expected)
        with self.assertRaisesRegex(Exception, "Invalid comp_groups"):
            result.get_comp_groups("loss", selected=["missing"])

    def test_result_sampling_uses_only_the_selected_scenario(self):
        domain = ParameterDomain(dict)
        domain.add_variable("loss", var_lim=(0.0, 100.0))
        sample = ParameterResultSample(
            self.make_result(Result),
            "loss",
            paramdomain=domain,
            comp_groups={"selected": ["case1"]},
        )
        sample.add_res_reps("selected", ind_seeds=False)
        self.assertEqual([s.p for s in sample.scenarios()], [{"loss": 1.0}])
        self.assertEqual(sample.scenarios()[0].inputparams["rep"], "case1")

    def test_history_sampling_uses_only_the_selected_scenario(self):
        domain = ParameterDomain(dict)
        domain.add_variable("loss", var_lim=(0.0, 100.0))
        sample = ParameterHistSample(
            self.make_result(History),
            "loss",
            paramdomain=domain,
            comp_groups={"selected": ["case1"]},
        )
        sample.add_hist_reps("selected", ind_seeds=False)
        self.assertEqual(
            [s.p for s in sample.scenarios()], [{"loss": 1.0}, {"loss": 2.0}]
        )
        self.assertEqual(
            [s.inputparams["rep"] for s in sample.scenarios()], ["case1", "case1"]
        )


if __name__ == "__main__":
    unittest.main()
