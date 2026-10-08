#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for comparison groups from nested results and histories.

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
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    ParameterDomain,
    ParameterHistSample,
    ParameterResultSample,
    ParameterSample,
)


class TestNestedComparisonInputs(unittest.TestCase):
    def make_data(self, cls):
        result = cls()
        for name, number in [
            ("nominal", 1.0),
            ("fault1", 3.0),
            ("fault10", 7.0),
            ("nominal_extra", 9.0),
        ]:
            result[name + ".s.x"] = (
                np.array([number, number + 1]) if cls is History else number
            )
            result[name + ".time"] = np.array([0.0, 1.0]) if cls is History else 1.0
            result[name + ".t.timer.time"] = (
                np.array([4.0, 5.0]) if cls is History else 5.0
            )
        return result

    def test_flat_nested_and_mixed_inputs_have_identical_default_groups(self):
        for cls in (Result, History):
            flat = self.make_data(cls)
            mixed = cls(
                {
                    "nominal": flat.nest(1)["nominal"],
                    **{k: v for k, v in flat.items() if not k.startswith("nominal.")},
                }
            )
            expected = flat.get_comp_groups("s.x")
            for source in (flat, flat.nest(1), flat.nest(2), flat.nest(3), mixed):
                with self.subTest(cls=cls.__name__, keys=list(source)):
                    before = source.copy()
                    groups = source.get_comp_groups("s.x")
                    self.assertIs(type(groups), cls)
                    self.assertEqual(groups, expected)
                    self.assertEqual(
                        set(groups["nominal"]), {"nominal.s.x", "nominal.time"}
                    )
                    self.assertNotIn("nominal_extra.s.x", groups["nominal"])
                    self.assertEqual(source, before)

    def test_explicit_selection_retains_full_paths_and_scenario_boundaries(self):
        flat = History(
            {
                "run1.fault1.s.x": [1, 2],
                "run1.fault1.time": [0, 1],
                "run1.fault10.s.x": [3, 4],
                "run1.fault10.time": [0, 1],
                "run10.fault1.s.x": [5, 6],
                "run10.fault1.time": [0, 1],
            }
        )
        for groups in (
            {"one": "run1.fault1"},
            {"one": ["run1"]},
            {"one": "default"},
            {"one": ["run1.fault1"], "two": ["run10.fault1"]},
        ):
            expected = flat.get_comp_groups("s.x", **groups)
            for depth in (1, 2, 3, 4):
                with self.subTest(groups=groups, depth=depth):
                    self.assertEqual(
                        flat.nest(depth).get_comp_groups("s.x", **groups), expected
                    )
        with self.assertRaisesRegex(Exception, "Invalid comp_groups"):
            flat.nest(4).get_comp_groups("s.x", one="missing")

    def test_custom_time_and_default_selection_preserve_nested_timer_exclusion(self):
        flat = History(
            {
                "case1.value": [1.0, 2.0],
                "case1.clock": [0.0, 1.0],
                "case1.t.timer.clock": [8.0, 9.0],
                "case2.value": [3.0, 4.0],
                "case2.clock": [0.0, 1.0],
            }
        )
        expected = flat.get_comp_groups("value", time="clock", all_cases="default")
        for source in (flat.nest(1), flat.nest(3)):
            result = source.get_comp_groups("value", time="clock", all_cases="default")
            self.assertEqual(result, expected)
            self.assertNotIn("case1.t.timer.clock", result["all_cases"])

    def test_parameter_resampling_accepts_nested_result_and_history_inputs(self):
        domain = ParameterDomain(dict)
        domain.add_variable("sampled", var_lim=(0.0, 10.0))
        for cls, samplecls in (
            (Result, ParameterResultSample),
            (History, ParameterHistSample),
        ):
            flat = self.make_data(cls)
            for source in (flat, flat.nest(3)):
                with self.subTest(cls=cls.__name__, keys=list(source)):
                    sample = samplecls(
                        source,
                        "s.x",
                        paramdomain=domain,
                        comp_groups={"chosen": ["fault1"]},
                    )
                    if cls is Result:
                        sample.add_res_reps("chosen", ind_seeds=False)
                    else:
                        sample.add_hist_reps("chosen", ind_seeds=False)
                    expected = (
                        [{"sampled": 3.0}]
                        if cls is Result
                        else [{"sampled": 3.0}, {"sampled": 4.0}]
                    )
                    self.assertEqual([s.p for s in sample.scenarios()], expected)
                    self.assertTrue(
                        all(
                            s.inputparams["rep"] == "fault1" for s in sample.scenarios()
                        )
                    )

    def test_real_simulation_outputs_keep_group_metrics_after_nesting(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x")
        sample = ParameterSample(domain, seed=7)
        sample.add_variable_replicates([[1.0], [2.0], [4.0]])
        result, history = propagate.parameter_sample(
            ExampleFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        names = [s.name for s in sample.scenarios()]
        groups = {"first": names[:1], "others": names[1:]}
        for source, value in ((result, "tend.classify.xy"), (history, "s.x")):
            flat = source.get_comp_groups(value, **groups)
            nested = source.nest(4).get_comp_groups(value, **groups)
            self.assertEqual(nested, flat)
            for group in groups:
                np.testing.assert_array_equal(
                    nested[group].get_metric(value, axis=0),
                    flat[group].get_metric(value, axis=0),
                )


if __name__ == "__main__":
    unittest.main()
