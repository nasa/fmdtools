#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for reject invalid history indices before creating parameter scenarios.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
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

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    ParameterDomain,
    ParameterHistSample,
    ParameterResultSample,
)


class InputParameter(Parameter, readonly=True):
    x: float = 1.0


class TotalState(State):
    total: float = 0.0


class HistoryFunction(Function):
    container_p = InputParameter
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.p.x

    def classify(self, **kwargs):
        return {"total": self.s.total}


def history_sample(factory=dict, arrays=True):
    values = np.array([2.0, 5.0, 9.0]) if arrays else [2.0, 5.0, 9.0]
    history = History({"case.signal": values, "case.time": np.array([0.0, 2.0, 4.0])})
    params = ParameterDomain(factory)
    params.add_variable("x")
    return ParameterHistSample(history, "signal", paramdomain=params, seed=7), history


class TestHistorySampleIndices(unittest.TestCase):
    def test_invalid_index_types_raise_instead_of_becoming_parameter_values(self):
        for arrays in (False, True):
            for index in (
                "bad",
                [],
                {},
                object(),
                True,
                np.bool_(False),
                1.0,
                np.float64(1.0),
                np.nan,
            ):
                with self.subTest(arrays=arrays, index=repr(index)):
                    sample, history = history_sample(arrays=arrays)
                    before = copy.deepcopy(history)
                    with self.assertRaisesRegex(TypeError, "integer or None"):
                        sample.get_param_ins(rep="case", t=index)
                    with self.assertRaisesRegex(TypeError, "integer or None"):
                        sample.add_hist_scenario(rep="case", t=index)
                    self.assertEqual(sample.scenarios(), [])
                    for name in history:
                        np.testing.assert_array_equal(history[name], before[name])

    def test_integer_indices_include_zero_negative_and_numpy_values(self):
        for arrays in (False, True):
            for index, expected in (
                (0, 2.0),
                (1, 5.0),
                (-1, 9.0),
                (np.int64(2), 9.0),
                (np.int32(0), 2.0),
            ):
                with self.subTest(arrays=arrays, index=index):
                    sample, _ = history_sample(arrays=arrays)
                    self.assertEqual(
                        sample.get_param_ins(rep="case", t=index), [expected]
                    )
                    sample.add_hist_scenario(rep="case", t=index)
                    scenario = sample.scenarios()[0]
                    self.assertEqual(scenario.p, {"x": expected})
                    self.assertEqual(scenario.inputparams["t"], index)
                    self.assertEqual(scenario.r, {"seed": 7})

    def test_none_preserves_whole_values_and_result_only_sampling(self):
        sample, _ = history_sample()
        np.testing.assert_array_equal(
            sample.get_param_ins(rep="case", t=None)[0], [2.0, 5.0, 9.0]
        )
        domain = ParameterDomain(dict)
        domain.add_variable("x")
        result = ParameterResultSample(
            Result({"case.signal": 7.0}), "signal", paramdomain=domain
        )
        self.assertEqual(result.get_param_ins(rep="case"), [7.0])
        result.add_res_scenario(rep="case")
        self.assertEqual(result.scenarios()[0].p, {"x": 7.0})

    def test_out_of_range_indices_still_raise_without_adding_a_scenario(self):
        for index in (-4, 3):
            sample, _ = history_sample()
            with self.assertRaises(IndexError):
                sample.add_hist_scenario(rep="case", t=index)
            self.assertEqual(sample.scenarios(), [])

    def test_valid_history_parameters_drive_real_simulation_outputs(self):
        sample, history = history_sample(InputParameter)
        sample.add_hist_times("default", "case", ts=[0, 2])
        model = HistoryFunction(sp={"end_time": 3.0})
        result, histories = propagate.parameter_sample(
            model, sample, showprogress=False
        )
        for scenario, value in zip(sample.scenarios(), (2.0, 9.0)):
            self.assertEqual(result[scenario.name + ".tend.classify.total"], 3 * value)
            np.testing.assert_allclose(
                histories[scenario.name + ".s.total"], np.arange(4) * value
            )
        np.testing.assert_array_equal(history["case.signal"], [2.0, 5.0, 9.0])
        self.assertEqual(model.s.total, 0.0)


if __name__ == "__main__":
    unittest.main()
