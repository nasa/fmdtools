#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for scenario-keyed rate and weight mappings in sample metrics.

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
import itertools
import unittest
from collections import UserDict
from types import MappingProxyType

import numpy as np

from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


def make_sample():
    model = ExampleFunction(sp={"end_time": 4.0})
    domain = FaultDomain(model)
    domain.add_fault(model.name, "low")
    sample = FaultSample(domain, def_mdl_phasemap=False)
    sample.add_fault_times([1.0, 2.0, 4.0], weights=[0.1, 0.2, 0.7])
    return sample


class TestScenarioMetricMappings(unittest.TestCase):
    def test_mapping_order_and_type_do_not_change_weighted_metrics(self):
        sample = make_sample()
        names = sample.scen_names()
        for order in itertools.permutations(range(3)):
            for wrapper in (dict, UserDict, MappingProxyType):
                rates = wrapper({names[i]: [0.1, 0.2, 0.7][i] for i in order})
                with self.subTest(order=order, wrapper=wrapper.__name__):
                    before = dict(rates)
                    self.assertAlmostEqual(
                        sample.get_metric(
                            "time", method="expected", rates=rates, round_value=False
                        ),
                        0.1 + 2 * 0.2 + 4 * 0.7,
                    )
                    self.assertAlmostEqual(
                        sample.get_metric(
                            "time", method="average", weights=rates, round_value=False
                        ),
                        0.1 + 2 * 0.2 + 4 * 0.7,
                    )
                    self.assertEqual(dict(rates), before)

    def test_selection_aligns_only_included_scenarios_and_preserves_sample_order(self):
        sample = make_sample()
        a, b, c = sample.scen_names()
        rates = {"unselected": 500.0, c: 0.75, a: 0.25}
        for selected in ([c, a], (a, c), {a, c}):
            with self.subTest(selected=selected):
                self.assertAlmostEqual(
                    sample.get_metric(
                        "time", ids=selected, method="expected", rates=rates
                    ),
                    0.25 + 4 * 0.75,
                )
                self.assertAlmostEqual(
                    sample.get_metric(
                        "time", ids=selected, method="average", weights=rates
                    ),
                    0.25 + 4 * 0.75,
                )
        with self.assertRaises(KeyError):
            sample.get_metric("time", method="expected", rates=rates)
        self.assertEqual(sample.scen_names(), [a, b, c])
        for argument in ("rates", "weights"):
            with self.subTest(argument=argument):
                with self.assertRaises(KeyError):
                    sample.get_metric("time", ids=[b], **{argument: {a: 1.0}})

    def test_joint_rate_and_weight_mappings_follow_the_same_selected_values(self):
        sample = make_sample()
        a, b, c = sample.scen_names()
        rates = {c: 3.0, a: 2.0, b: 4.0}
        weights = {b: 0.1, a: 0.6, c: 0.3}
        expected = (1 * 2 * 0.6 + 2 * 4 * 0.1 + 4 * 3 * 0.3) / (0.6 + 0.1 + 0.3)
        self.assertAlmostEqual(
            sample.get_metric(
                "time",
                method=np.average,
                rates=rates,
                weights=weights,
                round_value=False,
            ),
            expected,
        )
        expected_normalized = (1 * 2 + 2 * 4 + 4 * 3) / (2 + 4 + 3)
        self.assertAlmostEqual(
            sample.get_metric(
                "time", method="expected", rates=rates, r_norm=True, round_value=False
            ),
            expected_normalized,
        )

    def test_explicit_arrays_scalars_defaults_and_empty_samples_keep_behavior(self):
        sample = make_sample()
        a, _, c = sample.scen_names()
        rates = np.array([0.8, 0.2])
        rates.setflags(write=False)
        self.assertAlmostEqual(
            sample.get_metric("time", ids=[c, a], method="expected", rates=rates),
            1 * 0.8 + 4 * 0.2,
        )
        self.assertAlmostEqual(
            sample.get_metric("time", method="expected", rates=2.0), 14.0
        )
        self.assertAlmostEqual(
            sample.get_metric("time", method="average", round_value=False), 7 / 3
        )
        self.assertEqual(sample.get_metric("time", ids=[], method="sum", rates={}), 0.0)
        empty = FaultSample(sample.faultdomain, def_mdl_phasemap=False)
        self.assertEqual(empty.get_metric("time", method="sum", rates={}), 0.0)
        np.testing.assert_array_equal(rates, [0.8, 0.2])

    def test_fmea_scenario_and_result_metrics_share_name_aligned_rates(self):
        sample = make_sample()
        before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        result, history = propagate.fault_sample(
            sample.faultdomain.mdl, sample, showprogress=False
        )
        table = FMEA(
            result,
            sample,
            group_by=("fault",),
            expected_metric=["scenario_time", "xy"],
            rates="scenario_rate",
            round_value=False,
        )
        expected = sum(s.time * s.rate for s in sample.scenarios())
        self.assertAlmostEqual(table["expected_scenario_time"][("low",)], expected)
        loss = sum(
            result[s.name + ".tend.classify.xy"] * s.rate for s in sample.scenarios()
        )
        self.assertAlmostEqual(table["expected_xy"][("low",)], loss)
        selected = FMEA(
            result,
            sample,
            group_by=("time",),
            expected_metric="scenario_time",
            rates="scenario_rate",
            round_value=False,
        )
        for scenario in sample.scenarios():
            self.assertAlmostEqual(
                selected["expected_scenario_time"][(scenario.time,)],
                scenario.time * scenario.rate,
            )
            self.assertEqual(len(history[scenario.name + ".s.x"]), 5)
        weighted = FMEA(
            result,
            sample,
            group_by=(),
            average_metric="scenario_time",
            weights="scenario_rate",
            round_value=False,
        )
        self.assertAlmostEqual(weighted["average_scenario_time"][()], expected)
        self.assertEqual([s.asdict() for s in sample.scenarios()], before)
        self.assertFalse(sample.faultdomain.mdl.m.any_faults())


if __name__ == "__main__":
    unittest.main()
