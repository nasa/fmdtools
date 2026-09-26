#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for nominal and faulty groups in nested sample outputs.

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
from matplotlib import pyplot as plt

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestNestedNominalComparisonGroups(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def make_result(self, result_type, prefixes=("p0", "p1")):
        data = {}
        for index, prefix in enumerate(prefixes):
            for scenario, offset in (
                ("nominal", 0),
                ("fault", 10),
                ("nominal_fault", 20),
            ):
                value = float(2 * index + offset)
                data[f"{prefix}.{scenario}.metric"] = (
                    np.array([value, value + 1, value + 2])
                    if result_type is History
                    else value
                )
                if result_type is History:
                    data[f"{prefix}.{scenario}.time"] = np.array([10.0, 20.0, 30.0])
        return result_type(data)

    def test_default_groups_partition_nested_scenarios_exactly(self):
        for result_type in (Result, History):
            for nested in (False, True):
                with self.subTest(result_type=result_type.__name__, nested=nested):
                    result = self.make_result(result_type)
                    candidate = result.nest() if nested else result
                    groups = candidate.get_default_comp_groups()
                    self.assertEqual(
                        groups,
                        {
                            "nominal": ["p0.nominal", "p1.nominal"],
                            "faulty": [
                                "p0.fault",
                                "p0.nominal_fault",
                                "p1.fault",
                                "p1.nominal_fault",
                            ],
                        },
                    )
                    selected = result.get_comp_groups("metric")
                    self.assertIsInstance(selected, result_type)
                    self.assertEqual(set(selected), {"nominal", "faulty"})
                    self.assertFalse(set(selected["nominal"]) & set(selected["faulty"]))
                    self.assertEqual(
                        set(selected["nominal"]) | set(selected["faulty"]), set(result)
                    )
                    for group, values in selected.items():
                        for key, value in values.items():
                            self.assertIs(value, result[key])
                            self.assertEqual(
                                key.split(".")[1] == "nominal", group == "nominal"
                            )

    def test_nominal_in_parameter_names_does_not_hide_other_samples(self):
        prefixes = ("nominal_parameters", "non_nominal_parameters")
        result = self.make_result(Result, prefixes)
        groups = result.get_default_comp_groups()
        self.assertEqual(
            groups["nominal"], [prefix + ".nominal" for prefix in prefixes]
        )
        self.assertEqual(
            groups["faulty"],
            [
                prefix + "." + scen
                for prefix in prefixes
                for scen in ("fault", "nominal_fault")
            ],
        )

    def test_no_exact_nominal_scenario_uses_the_default_group(self):
        for result_type in (Result, History):
            with self.subTest(result_type=result_type.__name__):
                result = self.make_result(result_type, ("nominal_parameters", "p1"))
                result = result_type(
                    {
                        key: value
                        for key, value in result.items()
                        if key.split(".")[1] != "nominal"
                    }
                )
                self.assertEqual(
                    result.get_default_comp_groups(), {"default": "default"}
                )
                selected = result.get_comp_groups("metric")
                self.assertEqual(set(selected), {"default"})
                self.assertEqual(set(selected.default), set(result))

    def test_nested_nominal_only_results_do_not_become_faulty(self):
        result = Result({"p0.nominal.metric": 1.0, "p1.nominal.metric": 2.0})
        self.assertEqual(
            result.get_default_comp_groups(),
            {"nominal": ["p0.nominal", "p1.nominal"], "faulty": []},
        )
        grouped = result.get_comp_groups("metric")
        self.assertEqual(set(grouped), {"nominal"})
        self.assertEqual(dict(grouped["nominal"]), dict(result))

    def test_top_level_and_explicit_groups_keep_existing_behavior(self):
        result = Result(
            {"nominal.metric": 1.0, "fault.metric": 2.0, "nominal_fault.metric": 3.0}
        )
        self.assertEqual(
            result.get_default_comp_groups(),
            {"nominal": "nominal", "faulty": ["fault", "nominal_fault"]},
        )
        nested = self.make_result(Result)
        selected = nested.get_comp_groups("metric", chosen=["p1.nominal_fault"])
        self.assertEqual(dict(selected.chosen), {"p1.nominal_fault.metric": 22.0})
        self.assertEqual(Result().get_default_comp_groups(), {"default": "default"})

    def test_default_plots_show_distinct_nominal_and_faulty_statistics(self):
        for aggregation in ("mean_std", "mean_bound", "percentile"):
            with self.subTest(aggregation=aggregation):
                history = self.make_result(History)
                before = history.copy()
                fig, axes = history.plot_line("metric", aggregation=aggregation)
                lines = {line.get_label(): line for line in axes[0].lines}
                self.assertIn("nominal", lines)
                self.assertIn("faulty", lines)
                np.testing.assert_array_equal(
                    lines["nominal"].get_ydata(), [1.0, 2.0, 3.0]
                )
                np.testing.assert_array_equal(
                    lines["faulty"].get_ydata(), [16.0, 17.0, 18.0]
                )
                for name in ("nominal", "faulty"):
                    np.testing.assert_array_equal(
                        lines[name].get_xdata(), [10.0, 20.0, 30.0]
                    )
                fig.canvas.draw()
                for key in history:
                    np.testing.assert_array_equal(history[key], before[key])

    def test_real_nested_sample_separates_reference_and_fault_results(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        parameters = ParameterSample(domain)
        parameters.add_variable_scenario(2.0, 3.0)
        parameters.add_variable_scenario(4.0, 1.0)
        result, history, approaches = propagate.nested_sample(
            model,
            parameters,
            staged=False,
            include_nominal=True,
            showprogress=False,
            faultdomains={"low": (("fault", "examplefunction", "low"), {})},
            faultsamples={"times": (("fault_times", "low", [1.0]), {})},
        )
        self.assertEqual(len(approaches), 2)
        grouped = result.get_comp_groups("xy")
        self.assertEqual(list(grouped["nominal"].values()), [6.0, 12.0])
        self.assertEqual(list(grouped["faulty"].values()), [29.0, 23.0])
        selected = history.get_comp_groups("s.x")
        self.assertEqual(set(selected), {"nominal", "faulty"})
        for name, group in selected.items():
            for key in group:
                self.assertEqual(key.split(".")[1] == "nominal", name == "nominal")
        self.assertEqual(
            set(selected["nominal"]) | set(selected["faulty"]),
            {key for key in history if key.endswith(("s.x", "time"))},
        )


if __name__ == "__main__":
    unittest.main()
