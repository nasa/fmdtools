#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for implicit fault sampling within the configured simulation interval.

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

from fmdtools.analyze.phases import PhaseMap
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    JointFaultSample,
    sample_times_even,
    sample_times_quad,
)


class WindowMode(Mode):
    fault_a = Fault(prob=0.2, units="hr")
    fault_b = Fault(prob=0.3, units="hr")


class WindowFunction(Function):
    container_m = WindowMode

    def classify(self, **kwargs):
        return {"failed": float(self.m.any_faults())}


def make_sample(start, end, dt, kind="fault"):
    model = WindowFunction(
        sp={"start_time": start, "end_time": end, "dt": dt, "units": "hr"}
    )
    domain = FaultDomain(model)
    domain.add_fault(model.name, "a")
    if kind == "joint":
        other = FaultDomain(model)
        other.add_fault(model.name, "b")
        return JointFaultSample(domain, other, def_mdl_phasemap=False)
    return FaultSample(domain, def_mdl_phasemap=False)


class TestImplicitSamplingStart(unittest.TestCase):
    def test_all_methods_sample_only_the_requested_interval(self):
        for (start, end, dt), kind in itertools.product(
            [(2.0, 6.0, 1.0), (1.0, 3.0, 0.5), (3.0, 9.0, 1.0)], ("fault", "joint")
        ):
            support = np.arange(start, end + dt / 2, dt)
            for method, args in [
                ("all", ()),
                ("even", (2,)),
                ("quad", ([-0.5, 0.5], [1.0, 3.0])),
            ]:
                with self.subTest(
                    start=start, end=end, dt=dt, kind=kind, method=method
                ):
                    sample = make_sample(start, end, dt, kind)
                    before = copy.deepcopy(sample.faultdomain.mdl.sp.asdict())
                    sample.add_fault_phases(method=method, args=args)
                    if method == "all":
                        times, weights = (
                            support,
                            np.full(len(support), 1 / len(support)),
                        )
                    elif method == "even":
                        times, weights = sample_times_even(support, *args, dt=dt)
                    else:
                        times, weights = sample_times_quad(support, *args)
                    scenarios = sample.scenarios()
                    self.assertTrue(all(start <= s.time <= end for s in scenarios))
                    for fault in sample.faultdomain.faults:
                        chosen = [s for s in scenarios if s.fault == fault[1]]
                        np.testing.assert_array_equal([s.time for s in chosen], times)
                        base_rate = sample.faultdomain.faults[fault].prob * (
                            end - start + dt
                        )
                        np.testing.assert_allclose(
                            [s.rate for s in chosen], base_rate * np.asarray(weights)
                        )
                    self.assertEqual(sample.faultdomain.mdl.sp.asdict(), before)

    def test_single_timestep_and_zero_origin_controls(self):
        for start, end, dt in [(4.0, 4.0, 1.0), (0.0, 4.0, 1.0), (0.0, 1.0, 0.25)]:
            sample = make_sample(start, end, dt)
            sample.add_fault_phases(method="all")
            np.testing.assert_array_equal(
                [s.time for s in sample.scenarios()], np.arange(start, end + dt / 2, dt)
            )
            self.assertAlmostEqual(
                sum(s.rate for s in sample.scenarios()), 0.2 * (end - start + dt)
            )

    def test_explicit_phase_map_still_takes_precedence(self):
        sample = make_sample(2.0, 6.0, 1.0)
        sample.phasemap = PhaseMap({"chosen": [3.0, 4.0]})
        sample.add_fault_phases("chosen", method="all")
        np.testing.assert_array_equal([s.time for s in sample.scenarios()], [3.0, 4.0])
        self.assertAlmostEqual(sum(s.rate for s in sample.scenarios()), 0.2 * 2)

    def test_implicit_and_equivalent_explicit_samples_have_same_times_and_rates(self):
        for start, end, dt in [(2.0, 6.0, 1.0), (1.0, 3.0, 0.5)]:
            for method, args in [("all", ()), ("even", (2,))]:
                with self.subTest(start=start, dt=dt, method=method):
                    implicit = make_sample(start, end, dt)
                    explicit = FaultSample(
                        implicit.faultdomain,
                        phasemap=PhaseMap({"chosen": [start, end]}, dt=dt),
                        def_mdl_phasemap=False,
                    )
                    implicit.add_fault_phases(method=method, args=args)
                    explicit.add_fault_phases("chosen", method=method, args=args)
                    self.assertEqual(
                        [s.time for s in implicit.scenarios()],
                        [s.time for s in explicit.scenarios()],
                    )
                    np.testing.assert_allclose(
                        [s.rate for s in implicit.scenarios()],
                        [s.rate for s in explicit.scenarios()],
                    )

    def test_simulated_fault_times_do_not_precede_the_configured_window(self):
        sample = make_sample(2.0, 4.0, 1.0)
        sample.add_fault_phases(method="all")
        result, history = propagate.fault_sample(
            sample.faultdomain.mdl, sample, showprogress=False
        )
        self.assertEqual(set(sample.get_times()), {2.0, 3.0, 4.0})
        for scenario in sample.scenarios():
            trace = history[scenario.name + ".m.faults.a"]
            times = history[scenario.name + ".time"]
            np.testing.assert_array_equal(trace, times >= scenario.time)
            self.assertEqual(result[scenario.name + ".tend.classify.failed"], 1.0)
        self.assertFalse(sample.faultdomain.mdl.m.any_faults())


if __name__ == "__main__":
    unittest.main()
