#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for nested values in sample metrics and scenario weights.

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
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class GainParameters(Parameter, readonly=True):
    gain: float = 1.0


class NestedParameters(Parameter, readonly=True):
    inner: GainParameters = GainParameters()


class TotalState(State):
    total: np.float64 = 0.0


class GainFunction(Function):
    container_p = NestedParameters
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.p.inner.gain

    def classify(self, **kwargs):
        return {"total": self.s.total}


def make_sample():
    domain = ParameterDomain(NestedParameters)
    domain.add_variable("inner.gain")
    sample = ParameterSample(domain, seed=17)
    for i, (value, weight) in enumerate(zip([0.0, 2.0, 4.0], [0.2, 0.3, 0.5])):
        sample.add_variable_scenario(
            value,
            seed=i,
            sp={"end_time": 3.0},
            weight=weight,
            inputparams={"source": {"index": i}},
        )
    return sample


class TestNestedSampleMetrics(unittest.TestCase):
    def test_scenario_values_follow_nested_fields_and_keep_order(self):
        sample = make_sample()
        names = sample.scen_names()
        before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        for path, expected in [
            ("p.inner.gain", [0.0, 2.0, 4.0]),
            ("r.seed", [0, 1, 2]),
            ("sp.end_time", [3.0, 3.0, 3.0]),
            ("inputparams.source.index", [0, 1, 2]),
        ]:
            with self.subTest(path=path):
                actual = sample.get_scen_values(path)
                self.assertEqual(list(actual), names)
                self.assertEqual(list(actual.values()), expected)
        self.assertEqual([s.asdict() for s in sample.scenarios()], before)

    def test_nested_metrics_honor_selection_weights_and_rates(self):
        sample = make_sample()
        self.assertEqual(sample.get_metric("p.inner.gain"), 2.0)
        self.assertEqual(sample.get_metric("p.inner.gain", method="sum"), 6.0)
        self.assertEqual(sample.get_metric("p.inner.gain", method=np.max), 4.0)
        self.assertEqual(
            sample.get_metric("p.inner.gain", ids=[sample.scen_names()[-1]]), 4.0
        )
        self.assertEqual(
            sample.get_metric(
                "p.inner.gain", ids=sample.scen_names()[::2], method="sum"
            ),
            4.0,
        )
        self.assertAlmostEqual(
            sample.get_metric("p.inner.gain", weights=[0.2, 0.3, 0.5]), 2.6
        )
        self.assertAlmostEqual(
            sample.get_metric("p.inner.gain", method="expected", rates=[0.2, 0.3, 0.5]),
            2.6,
        )

    def test_flat_fields_empty_samples_and_missing_attributes_keep_behavior(self):
        sample = make_sample()
        self.assertEqual(
            list(sample.get_scen_values("name").values()), sample.scen_names()
        )
        self.assertEqual(list(sample.get_scen_values("prob").values()), [0.2, 0.3, 0.5])
        self.assertEqual(sample.get_metric("prob", method="sum"), 1.0)
        with self.assertRaises(AttributeError):
            sample.get_scen_values("unknown")
        self.assertEqual(
            ParameterSample(ParameterDomain(dict)).get_scen_values("p.inner.gain"), {}
        )
        for scenario in sample.scenarios():
            self.assertIs(sample.get_scen_values("p")[scenario.name], scenario.p)

    def test_fmea_can_weight_results_using_nested_scenario_parameters(self):
        sample = make_sample()
        result = Result(
            {s.name + ".tend.classify.cost": 10.0 for s in sample.scenarios()}
        )
        before = copy.deepcopy(result)
        table = FMEA(
            result,
            sample,
            group_by=("p.inner.gain",),
            sum_metric=["cost"],
            expected_metric=["cost"],
            rates="scenario_p.inner.gain",
        )
        self.assertEqual(
            table["expected_cost"], {(0.0,): 0.0, (2.0,): 20.0, (4.0,): 40.0}
        )
        self.assertEqual(table["sum_cost"], {(0.0,): 10.0, (2.0,): 10.0, (4.0,): 10.0})
        self.assertEqual(result, before)

    def test_real_simulation_outputs_match_metrics_of_the_sampled_inputs(self):
        sample = make_sample()
        result, history = propagate.parameter_sample(
            GainFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        gains = sample.get_scen_values("p.inner.gain")
        for name, gain in gains.items():
            self.assertEqual(result[name + ".tend.classify.total"], 3 * gain)
            np.testing.assert_array_equal(
                history[name + ".s.total"], np.arange(4) * gain
            )
        self.assertEqual(
            result.get_metric("total", method="sum"),
            3 * sample.get_metric("p.inner.gain", method="sum"),
        )
        self.assertAlmostEqual(
            result.get_metric(
                "total", method="expected", rates=sample.get_scen_values("prob")
            ),
            3
            * sample.get_metric(
                "p.inner.gain", method="expected", rates=[0.2, 0.3, 0.5]
            ),
        )


if __name__ == "__main__":
    unittest.main()
