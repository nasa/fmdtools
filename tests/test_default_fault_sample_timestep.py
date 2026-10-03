#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for model timesteps in default FaultSample phase maps.

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

from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class RateMode(Mode):
    fault_first = Fault(prob=0.2, units="sec", phases=(("early", 0.25), ("late", 0.75)))
    fault_second = Fault(prob=0.3, units="sec", phases=(("early", 0.6), ("late", 0.4)))
    opermodes = ("early", "late")
    mode: str = "early"


class RateFunction(ExampleFunction):
    container_m = RateMode


def make_domain(dt):
    model = RateFunction(
        "risk",
        sp={
            "end_time": 4 * dt,
            "dt": dt,
            "use_local": False,
            "units": "sec",
            "phases": (("early", 0.0, dt), ("late", 2 * dt, 4 * dt)),
        },
    )
    domain = FaultDomain(model)
    domain.add_faults(("risk", "first"), ("risk", "second"))
    return domain


class TestDefaultFaultSampleTimestep(unittest.TestCase):
    def test_default_phase_map_uses_the_model_grid_and_phase_boundaries(self):
        for dt in (0.25, 0.5, 1.0, 2.0):
            with self.subTest(dt=dt):
                domain = make_domain(dt)
                sample = FaultSample(domain)
                self.assertEqual(sample.phasemap.dt, dt)
                self.assertEqual(sample.phasemap.find_base_phase(2 * dt), "late")
                self.assertEqual(
                    sample.phasemap.calc_samples_in_phases(*(np.arange(5) * dt)),
                    {"early": 2, "late": 3},
                )
                self.assertEqual(sample.phasemap.calc_phase_time("early"), 2 * dt)
                self.assertEqual(sample.phasemap.calc_phase_time("late"), 3 * dt)
                np.testing.assert_array_equal(
                    sample.phasemap.get_phase_times("late"), np.arange(2, 5) * dt
                )

    def test_phase_sampling_preserves_all_model_times_and_expected_rates(self):
        for dt in (0.25, 0.5, 1.0, 2.0):
            for n_joint in (1, 2):
                with self.subTest(dt=dt, n_joint=n_joint):
                    domain = make_domain(dt)
                    sample = FaultSample(domain)
                    explicit = FaultSample(
                        domain, PhaseMap(domain.mdl.sp.phases, dt=dt)
                    )
                    for output in (sample, explicit):
                        output.add_fault_phases("late", method="all", n_joint=n_joint)
                    np.testing.assert_array_equal(
                        sorted(sample.get_times()), np.arange(2, 5) * dt
                    )
                    self.assertEqual(
                        [s.asdict() for s in sample.scenarios()],
                        [s.asdict() for s in explicit.scenarios()],
                    )
                    self.assertEqual(sample.num_scenarios(), 6 if n_joint == 1 else 3)
                    for scenario in sample.scenarios():
                        self.assertEqual(scenario.phase, "late")
                        expected = (
                            (0.2 * 0.75 * dt) * (0.3 * 0.4 * dt)
                            if n_joint == 2
                            else 0.2 * 0.75 * dt
                            if scenario.fault == "first"
                            else 0.3 * 0.4 * dt
                        )
                        self.assertAlmostEqual(scenario.rate, expected, places=14)

    def test_explicit_and_disabled_phase_maps_keep_their_existing_semantics(self):
        domain = make_domain(0.25)
        mapping = PhaseMap({"custom": [0.0, 1.0]}, dt=0.5)
        before = copy.deepcopy(mapping.__dict__)
        sample = FaultSample(domain, phasemap=mapping)
        self.assertIs(sample.phasemap, mapping)
        self.assertEqual(sample.phasemap.dt, 0.5)
        self.assertEqual(mapping.__dict__, before)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        self.assertEqual(sample.phasemap, {})
        sample.add_single_fault_scenario(("risk", "first"), 0.5)
        self.assertEqual(sample.scenarios()[0].phase, "")

    def test_real_staged_and_unstaged_runs_match_an_explicit_model_grid(self):
        for dt in (0.5, 2.0):
            for staged in (False, True):
                with self.subTest(dt=dt, staged=staged):
                    domain = make_domain(dt)
                    sample = FaultSample(domain)
                    reference = FaultSample(
                        domain, PhaseMap(domain.mdl.sp.phases, dt=dt)
                    )
                    for output in (sample, reference):
                        output.add_fault_times([dt, 2 * dt, 4 * dt])
                    result, history = propagate.fault_sample(
                        domain.mdl, sample, staged=staged, showprogress=False
                    )
                    expected_result, expected_history = propagate.fault_sample(
                        domain.mdl, reference, staged=staged, showprogress=False
                    )
                    self.assertEqual(result, expected_result)
                    self.assertEqual(history, expected_history)
                    actual_table = FMEA(
                        result,
                        sample,
                        group_by=("phase", "time"),
                        average_metric=["scenario_rate"],
                        sum_metric=["xy"],
                    )
                    expected_table = FMEA(
                        expected_result,
                        reference,
                        group_by=("phase", "time"),
                        average_metric=["scenario_rate"],
                        sum_metric=["xy"],
                    )
                    self.assertEqual(
                        actual_table["average_scenario_rate"],
                        expected_table["average_scenario_rate"],
                    )
                    self.assertEqual(actual_table["sum_xy"], expected_table["sum_xy"])


if __name__ == "__main__":
    unittest.main()
