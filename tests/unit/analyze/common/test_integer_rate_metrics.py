#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for integer-rate arithmetic in weighted risk metrics.

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

from fmdtools.analyze.common import calc_metric, calc_metric_ci, metric_preamble
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class CostState(State):
    cost: np.int64 = 0


class CostMode(Mode):
    fault_low = (1.0,)


class CostFunction(Function):
    container_s = CostState
    container_m = CostMode

    def dynamic_behavior(self):
        self.s.cost = np.int64(100 if self.m.has_fault("low") else 0)

    def classify(self, **kwargs):
        return {"cost": self.s.cost}


class TestIntegerRateMetrics(unittest.TestCase):
    def test_integer_storage_does_not_overflow_before_reduction(self):
        for dtype in (
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ):
            maximum = int(np.iinfo(dtype).max)
            data = np.array([maximum // 2, maximum // 3], dtype=dtype)
            rates = np.array([3, 4], dtype=dtype)
            expected = float(int(data[0]) * 3 + int(data[1]) * 4)
            for method in (np.sum, "expected"):
                with self.subTest(dtype=dtype, method=method):
                    actual = calc_metric(data, method, rates=rates, round_value=False)
                    self.assertAlmostEqual(actual / expected, 1.0)
        self.assertEqual(
            calc_metric(
                [100, 110], "expected", rates=[2, 3], dtype=np.int8, r_dtype=np.int8
            ),
            530.0,
        )

    def test_normalized_large_integer_rates_keep_positive_expected_value(self):
        for dtype, magnitude in ((np.int64, 2**62), (np.uint64, 2**63)):
            with self.subTest(dtype=dtype):
                rates = np.array([magnitude, magnitude], dtype=dtype)
                self.assertEqual(
                    calc_metric([1, 2], "expected", rates=rates, r_norm=True), 1.5
                )
                np.testing.assert_array_equal(
                    metric_preamble([1, 2], rates=rates, r_norm=True), [0.5, 1.0]
                )
        np.testing.assert_array_equal(
            metric_preamble([False, True], rates=np.array([2, 3], dtype=np.int8)),
            [0.0, 3.0],
        )

    def test_broadcast_axes_and_input_ownership_match_float_reference(self):
        data = np.array([[[100, 110], [90, 80]], [[60, 70], [100, 110]]], dtype=np.int8)
        rates = np.array([3, 4], dtype=np.int8)
        data.setflags(write=False)
        rates.setflags(write=False)
        before = data.copy(), rates.copy()
        expected = data.astype(float) * rates[:, None, None].astype(float)
        for axis in (None, 0, 1, (0, 2)):
            with self.subTest(axis=axis):
                np.testing.assert_allclose(
                    calc_metric(data, np.sum, rates=rates, axis=axis),
                    expected.sum(axis=axis),
                )
        np.testing.assert_array_equal(data, before[0])
        np.testing.assert_array_equal(rates, before[1])
        scalar = calc_metric(np.int8(100), np.sum, rates=np.int8(3))
        self.assertEqual(scalar, 300.0)
        self.assertEqual(calc_metric(np.array([], dtype=np.int8), np.sum, rates=3), 0.0)
        raw = metric_preamble(data)
        self.assertEqual(raw.dtype, data.dtype)
        floats = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqual(
            metric_preamble(floats, rates=floats).dtype, np.dtype(np.float32)
        )

    def test_bootstrap_and_named_result_metrics_receive_unwrapped_values(self):
        data = np.array([100, 110, 120, 90], dtype=np.int8)
        rates = np.array([2, 3, 4, 2], dtype=np.int8)
        kwargs = {"method": np.mean, "n_resamples": 99, "random_state": 7}
        actual = calc_metric_ci(data, rates=rates, **kwargs)
        expected = calc_metric_ci(data.astype(float) * rates.astype(float), **kwargs)
        np.testing.assert_allclose(actual, expected)
        result = Result({f"s{i}.cost": value for i, value in enumerate(data)})
        result.update(
            {f"s{i}.rate": value for i, value in reversed(list(enumerate(rates)))}
        )
        before = copy.deepcopy(dict(result))
        self.assertEqual(result.get_metric("cost", "expected", rates="rate"), 1190.0)
        self.assertEqual(dict(result), before)
        history = History(
            {
                f"s{i}.cost": np.array([value, value], dtype=np.int8)
                for i, value in enumerate(data)
            }
        )
        np.testing.assert_array_equal(
            history.get_metric(
                "cost",
                "expected",
                rates={f"s{i}": value for i, value in enumerate(rates)},
                axis=0,
            ),
            [1190.0, 1190.0],
        )

    def test_real_fmea_with_integer_event_counts_preserves_expected_loss(self):
        model = CostFunction(sp={"end_time": 3.0})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "low")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0, 2.0])
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        factors = {s.name: np.int8(3) for s in sample.scenarios()}
        table = FMEA(
            result,
            sample,
            group_by=(),
            expected_metric="cost",
            rates=factors,
            dtype=np.int8,
            r_dtype=np.int8,
        )
        self.assertEqual(table["expected_cost"][()], 600.0)
        for s in sample.scenarios():
            self.assertEqual(history[s.name + ".s.cost"][-1], 100)
        self.assertFalse(model.m.any_faults())


if __name__ == "__main__":
    unittest.main()
