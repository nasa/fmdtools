#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for probability mass at coincident even-sampling nodes.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
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

import numpy as np

from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, sample_times_even


class LossMode(Mode):
    fault_failed = Fault(prob=0.2)


class LossState(State):
    loss: np.float64 = 0.0


class LossFunction(Function):
    container_m = LossMode
    container_s = LossState

    def dynamic_behavior(self):
        self.s.loss = 10.0 if self.m.any_faults() else 0.0

    def classify(self, **kwargs):
        return {"cost": self.s.loss}


def uncombined_nodes(times, count, dt):
    if count + 2 > len(times):
        return list(times)
    array = np.asarray(times)
    nodes = []
    for p in range(1, count + 1):
        candidate = round(float(np.quantile(times, p / (count + 1))) / dt) * dt
        nodes.append(
            candidate
            if candidate in times
            else times[np.argmin(abs(array - candidate))]
        )
    return nodes


class TestEvenSamplingMass(unittest.TestCase):
    def test_collisions_keep_combined_weights_and_first_occurrence_order(self):
        cases = [
            ([0.25, 0.75, 1.25, 1.75], 2, 0.5),
            ([0, 1, 2, 3, 4, 20], 3, 2.0),
            ([0, 1, 2, 3, 4, 5, 100], 5, 2.0),
            ([0.0, 0.1, 0.2, 4.0], 2, 1.0),
        ]
        for (times, count, dt), wrap in itertools.product(
            cases, (list, tuple, np.asarray)
        ):
            with self.subTest(times=times, count=count, wrap=wrap.__name__):
                source = wrap(times)
                before = np.asarray(source).copy()
                raw = uncombined_nodes(times, count, dt)
                expected_times = list(dict.fromkeys(raw))
                expected_weights = [
                    raw.count(time) / len(raw) for time in expected_times
                ]
                actual, weights = sample_times_even(source, count, dt)
                np.testing.assert_array_equal(actual, expected_times)
                np.testing.assert_allclose(weights, expected_weights, rtol=1e-14)
                self.assertEqual(len(actual), len(set(actual)))
                self.assertAlmostEqual(sum(weights), 1.0)
                # Coalescing must not move probability to a different time.
                self.assertAlmostEqual(
                    sum(t * t * w for t, w in zip(actual, weights)),
                    sum(t * t for t in raw) / len(raw),
                )
                np.testing.assert_array_equal(source, before)

    def test_unique_grid_rounding_and_empty_fallback_are_unchanged(self):
        for dt, count in itertools.product((0.25, 1.0, 2.0), (1, 2, 3, 4, 8)):
            times = np.arange(7) * dt
            actual, weights = sample_times_even(times, count, dt)
            expected = uncombined_nodes(times, count, dt)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(
                weights, np.full(len(expected), 1 / len(expected))
            )
        for times in ([], [3.0], [3.0, 4.0]):
            actual, weights = sample_times_even(times, 3)
            self.assertEqual(actual, times)
            self.assertEqual(len(actual), len(weights))

    def test_repeated_input_positions_do_not_overwrite_named_scenarios(self):
        for times, count in [([1.0, 1.0, 2.0], 8), ([1.0, 1.0], 1)]:
            with self.subTest(times=times):
                actual, weights = sample_times_even(times, count)
                expected = list(dict.fromkeys(times))
                self.assertEqual(actual, expected)
                np.testing.assert_allclose(
                    weights, [times.count(t) / len(times) for t in expected]
                )

    def test_named_sample_and_simulated_fmea_keep_all_probability_mass(self):
        model = LossFunction(sp={"end_time": 2.0, "dt": 0.25, "use_local": False})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "failed")
        support = [0.25, 0.75, 1.25, 1.75]
        times, weights = sample_times_even(support, 2, dt=0.5)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times(times, weights)
        before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        result, history = propagate.fault_sample(
            model, sample, showprogress=False, staged=False
        )
        table = FMEA(
            result,
            sample,
            group_by=(),
            expected_metric="cost",
            rates="scenario_rate",
            round_value=False,
        )
        self.assertAlmostEqual(table["expected_cost"][()], 2.0)
        self.assertEqual(len(sample.scenarios()), len(sample.named_scenarios()))
        self.assertAlmostEqual(
            sum(s.rate for s in sample.named_scenarios().values()), 0.2
        )
        for scenario in sample.scenarios():
            self.assertEqual(result[scenario.name + ".tend.classify.cost"], 10.0)
            np.testing.assert_array_equal(
                history[scenario.name + ".s.loss"],
                np.where(history[scenario.name + ".time"] >= scenario.time, 10.0, 0.0),
            )
        self.assertEqual([s.asdict() for s in sample.scenarios()], before)
        self.assertFalse(model.m.any_faults())


if __name__ == "__main__":
    unittest.main()
