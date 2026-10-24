#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for FMEA weighting and tabular output consistency.

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
import io
import math
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import BaseTab, FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, SampleApproach


class CostMode(Mode):
    fault_minor = Fault(prob=0.2)
    fault_major = Fault(prob=0.1)
    fault_zero = Fault(prob=0.0)


class CostFunction(ExampleFunction):
    container_m = CostMode

    def classify(self, **kwargs):
        return {"cost": self.s.x + self.s.y}


class TestFmeaOutputConsistency(unittest.TestCase):
    def setUp(self):
        self.model = CostFunction("risk", sp={"end_time": 3.0})
        domain = FaultDomain(self.model)
        for fault in ("minor", "major", "zero"):
            domain.add_fault(self.model.name, fault)
        self.sample = FaultSample(domain, def_mdl_phasemap=False)
        self.sample.add_fault_times([1.0, 2.0], weights=[0.25, 0.75])
        self.losses = {
            "minor": (10.0, 100.0),
            "major": (40.0, 80.0),
            "zero": (1000.0, 2000.0),
        }
        self.result = Result(
            {
                scenario.name + ".tend.classify.cost": self.losses[scenario.fault][
                    int(scenario.time) - 1
                ]
                for scenario in self.sample.scenarios()
            }
        )
        # This nominal reference and alternate-time values must not enter FMEA.
        self.result["nominal.tend.classify.cost"] = 9999.0
        for scenario in self.sample.scenarios():
            self.result[scenario.name + ".t1.classify.cost"] = 8888.0

    def test_metric_options_preserve_analytical_default_risk(self):
        # minor: .05*10 + .15*100 = 15.5; major: .025*40 + .075*80 = 7.
        expected = {
            ("risk", "minor"): 15.5,
            ("risk", "major"): 7.0,
            ("risk", "zero"): 0.0,
        }
        for options in (
            {},
            {"round_value": False},
            {"round_value": True},
            {"min_r": 7},
            {"axis": 0},
            {"dtype": np.float64},
            {"round_value": False, "axis": 0, "min_r": 12},
        ):
            for nested in (False, True):
                with self.subTest(options=options, nested=nested):
                    source = self.result.nest() if nested else self.result
                    table = FMEA(source, self.sample, **options)
                    for group, value in expected.items():
                        self.assertAlmostEqual(table["expected_cost"][group], value)
                    self.assertAlmostEqual(sum(table["expected_cost"].values()), 22.5)
                    self.assertEqual(table["sum_cost"][("risk", "minor")], 110.0)
                    self.assertAlmostEqual(
                        table["average_scenario_rate"][("risk", "minor")], 0.1
                    )

    def test_grouped_rows_and_exports_preserve_underlying_risk(self):
        for factors in (("obj", "fault"), ("fault",), ("time",), ("fault", "time"), ()):
            with self.subTest(factors=factors):
                table = FMEA(
                    self.result, self.sample, group_by=factors, round_value=False
                )
                for group, ids in self.sample.get_scen_groups(*factors).items():
                    expected = math.fsum(
                        scenario.rate
                        * self.losses[scenario.fault][int(scenario.time) - 1]
                        for scenario in self.sample.scenarios()
                        if scenario.name in ids
                    )
                    self.assertAlmostEqual(table["expected_cost"][group], expected)
                self.assertAlmostEqual(sum(table["expected_cost"].values()), 22.5)
                frame = table.as_table(sort_by="expected_cost", ascending=True)
                self.assertTrue(frame["expected_cost"].is_monotonic_increasing)
                expected_rows = [
                    {metric: column[group] for metric, column in table.items()}
                    for group in frame.index
                ]
                assert_frame_equal(
                    frame.reset_index(drop=True), pd.DataFrame(expected_rows)
                )
                exported = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
                np.testing.assert_allclose(
                    exported["expected_cost"], frame["expected_cost"]
                )

    def test_empty_tuple_rows_preserve_values_and_column_alignment(self):
        for data, expected in (
            (
                {"cost": {(): 15.5}, "rate": {(): 0.2}},
                pd.DataFrame(
                    {"cost": [15.5], "rate": [0.2]},
                    index=pd.Index([()], tupleize_cols=False),
                ),
            ),
            (
                {
                    "cost": {(): 15.5, ("other",): 7.0},
                    "rate": {("other",): 0.1, ("extra",): 0.3, (): 0.2},
                },
                pd.DataFrame(
                    {"cost": [15.5, 7.0, np.nan], "rate": [0.2, 0.1, 0.3]},
                    index=pd.Index([(), ("other",), ("extra",)], tupleize_cols=False),
                ),
            ),
        ):
            with self.subTest(rows=list(expected.index)):
                table = BaseTab(data)
                before = copy.deepcopy(table.data)
                assert_frame_equal(table.as_table(sort=False), expected)
                self.assertEqual(table.data, before)

    def test_explicit_rate_overrides_and_custom_metrics_keep_their_behavior(self):
        names = self.sample.scen_names()
        explicit = {
            name: rate
            for name, rate in zip(reversed(names), (0.2, 0.1, 0.4, 0.3, 0.6, 0.5))
        }
        source = copy.deepcopy(self.result)
        for name, rate in explicit.items():
            source[name + ".tend.classify.rate"] = rate
        for rates in (None, "rate", explicit, "scenario_rate"):
            with self.subTest(rates=rates):
                table = FMEA(source, self.sample, rates=rates, round_value=False)
                for group, ids in self.sample.get_scen_groups("obj", "fault").items():
                    values = [source[name + ".tend.classify.cost"] for name in ids]
                    if rates is None:
                        expected = sum(values) / len(values)
                    else:
                        mapping = (
                            self.sample.get_scen_values("rate")
                            if rates == "scenario_rate"
                            else explicit
                        )
                        expected = math.fsum(
                            mapping[name] * value for name, value in zip(ids, values)
                        )
                    self.assertAlmostEqual(table["expected_cost"][group], expected)
        # Custom metric selections historically require explicit rates.
        custom = FMEA(source, self.sample, expected_metric="cost", round_value=False)
        self.assertAlmostEqual(custom["expected_cost"][("risk", "minor")], 55.0)

    def test_rounding_and_normalization_are_applied_after_rate_selection(self):
        for scenario in self.sample.scenarios():
            self.result[scenario.name + ".tend.classify.cost"] = 10.12345678
        rounded = FMEA(self.result, self.sample, min_r=3)
        unrounded = FMEA(self.result, self.sample, round_value=False, min_r=3)
        self.assertEqual(rounded["expected_cost"][("risk", "minor")], 2.025)
        self.assertAlmostEqual(
            unrounded["expected_cost"][("risk", "minor")], 2.024691356, places=12
        )
        # Avoid zero-mass groups, whose normalized average is undefined.
        sample = FaultSample(self.sample.faultdomain, def_mdl_phasemap=False)
        sample.add_single_fault_scenario(("risk", "minor"), 1.0, weight=0.25)
        sample.add_single_fault_scenario(("risk", "minor"), 2.0, weight=0.75)
        values = Result(
            {
                scenario.name + ".tend.classify.cost": cost
                for scenario, cost in zip(sample.scenarios(), (10.0, 100.0))
            }
        )
        normalized = FMEA(values, sample, r_norm=True, round_value=False)
        self.assertAlmostEqual(normalized["expected_cost"][("risk", "minor")], 77.5)

    def test_simulation_and_sample_approach_match_direct_result_reduction(self):
        approach = SampleApproach(self.model, def_mdl_phasemap=False)
        approach.faultsamples["risk"] = self.sample
        before = copy.deepcopy(
            [scenario.asdict() for scenario in self.sample.scenarios()]
        )
        for sample in (self.sample, approach):
            for staged in (False, True):
                with self.subTest(sample=type(sample).__name__, staged=staged):
                    result, history = propagate.fault_sample(
                        self.model, sample, staged=staged, showprogress=False
                    )
                    rates = sample.get_scen_values("rate")
                    direct = result.get_metric(
                        "cost",
                        "expected",
                        prefix="tend.classify.",
                        rates={**rates, "nominal": 0.0},
                        round_value=False,
                    )
                    table = FMEA(result, sample, round_value=False)
                    self.assertAlmostEqual(
                        math.fsum(table["expected_cost"].values()), direct, places=13
                    )
                    for scenario in sample.scenarios():
                        np.testing.assert_allclose(
                            result[scenario.name + ".tend.classify.cost"],
                            history[scenario.name + ".s.x"][-1]
                            + history[scenario.name + ".s.y"][-1],
                        )
        self.assertEqual(
            [scenario.asdict() for scenario in self.sample.scenarios()], before
        )
        self.assertFalse(self.model.m.any_faults())

    def test_additional_metrics_and_repeated_calls_do_not_mutate_inputs(self):
        first = self.sample.scenarios()[0].name
        added = Result({first + ".tend.classify.cost": 30.0})
        before, added_before = copy.deepcopy(self.result), copy.deepcopy(added)
        changed = FMEA(self.result, self.sample, add_res=added, round_value=False)
        self.assertAlmostEqual(changed["expected_cost"][("risk", "minor")], 16.5)
        unchanged = FMEA(self.result, self.sample, round_value=False)
        self.assertAlmostEqual(unchanged["expected_cost"][("risk", "minor")], 15.5)
        self.assertEqual(self.result, before)
        self.assertEqual(added, added_before)
        self.assertEqual(list(self.result), list(before))
        empty = FaultSample(self.sample.faultdomain, def_mdl_phasemap=False)
        self.assertEqual(
            FMEA(Result(), empty, round_value=False).data,
            {"average_scenario_rate": {}, "sum_cost": {}, "expected_cost": {}},
        )


if __name__ == "__main__":
    unittest.main()
