#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for phase-aware joint-fault sampling.

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
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.sim.scenario import JointFaultScenario


class PhaseMode(Mode):
    fault_first = Fault(prob=0.2, phases=(("early", 0.25), ("late", 0.75)))
    fault_second = Fault(prob=0.3, phases=(("early", 0.6), ("late", 0.4)))
    opermodes = ("early", "late")
    mode: str = "early"


class PhaseFunction(ExampleFunction):
    container_m = PhaseMode


def make_sample(phasemap=None, use_model_map=False):
    model = PhaseFunction(
        "risk",
        sp={"end_time": 6.0, "phases": (("early", 0.0, 2.0), ("late", 3.0, 6.0))},
    )
    domain = FaultDomain(model)
    domain.add_faults(("risk", "first"), ("risk", "second"))
    return FaultSample(domain, phasemap=phasemap or {}, def_mdl_phasemap=use_model_map)


class TestJointFaultPhaseMap(unittest.TestCase):
    def test_phase_opportunities_reach_all_joint_rate_combinations(self):
        for time, phase, component_rates in (
            (1.0, "early", (0.05, 0.18)),
            (4.0, "late", (0.15, 0.12)),
        ):
            for baserate in ("ind", "max", ("risk", "first")):
                for weight in (0.0, 0.4, 1.0):
                    with self.subTest(time=time, baserate=baserate, weight=weight):
                        mapping = PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]})
                        before = copy.deepcopy(mapping.__dict__)
                        sample = make_sample(mapping)
                        faults = tuple(sample.faultdomain.faults)
                        sample.add_joint_fault_scenario(
                            faults, time, weight=weight, baserate=baserate, p_cond=0.5
                        )
                        actual = sample.scenarios()[0]
                        values = np.array(component_rates) * weight
                        expected = (
                            np.prod(values)
                            if baserate == "ind"
                            else np.max(values)
                            if baserate == "max"
                            else values[0]
                        ) * 0.5
                        self.assertEqual(actual.phase, phase)
                        self.assertAlmostEqual(actual.rate, expected, places=14)
                        direct = JointFaultScenario.from_faults(
                            faults,
                            time,
                            mdl=sample.faultdomain.mdl,
                            phasemap=mapping,
                            weight=weight,
                            baserate=baserate,
                            p_cond=0.5,
                        )
                        self.assertEqual(actual.asdict(), direct.asdict())
                        self.assertEqual(mapping.__dict__, before)

    def test_bulk_and_phase_selected_scenarios_use_the_same_map(self):
        for selected_by_phase in (False, True):
            with self.subTest(selected_by_phase=selected_by_phase):
                mapping = PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]})
                sample = make_sample(mapping)
                if selected_by_phase:
                    sample.add_fault_phases(
                        "late", method="all", n_joint=2, baserate="max"
                    )
                    expected_times, expected_rate, phase = (
                        [3.0, 4.0, 5.0, 6.0],
                        0.15 / 4,
                        "late",
                    )
                else:
                    sample.add_fault_times([0.0, 1.0, 2.0], n_joint=2, baserate="max")
                    expected_times, expected_rate, phase = (
                        [0.0, 1.0, 2.0],
                        0.18 / 3,
                        "early",
                    )
                self.assertEqual(sample.get_times(), expected_times)
                for scenario in sample.scenarios():
                    self.assertEqual(scenario.phase, phase)
                    self.assertAlmostEqual(scenario.rate, expected_rate, places=14)

    def test_model_default_map_and_explicit_no_map_remain_distinct(self):
        for use_model_map in (False, True):
            with self.subTest(use_model_map=use_model_map):
                sample = make_sample(use_model_map=use_model_map)
                faults = tuple(sample.faultdomain.faults)
                sample.add_joint_fault_scenario(faults, 1.0)
                direct = JointFaultScenario.from_faults(
                    faults, 1.0, mdl=sample.faultdomain.mdl, phasemap=sample.phasemap
                )
                self.assertEqual(sample.scenarios()[0].asdict(), direct.asdict())
                self.assertEqual(direct.phase, "early" if use_model_map else "")
                self.assertAlmostEqual(direct.rate, 0.009 if use_model_map else 0.06)

    def test_phase_exposure_time_is_used_for_time_based_fault_rates(self):
        class ExposureMode(Mode):
            fault_first = Fault(prob=0.1, units="sec")
            fault_second = Fault(prob=0.2, units="sec")

        class ExposureFunction(ExampleFunction):
            container_m = ExposureMode

        model = ExposureFunction("risk", sp={"end_time": 6.0, "units": "sec"})
        domain = FaultDomain(model)
        domain.add_faults(("risk", "first"), ("risk", "second"))
        mapping = PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]})
        sample = FaultSample(domain, phasemap=mapping)
        for time, duration in ((1.0, 3.0), (4.0, 4.0)):
            with self.subTest(time=time):
                sample.add_joint_fault_scenario(tuple(domain.faults), time)
                self.assertAlmostEqual(
                    sample.scenarios()[-1].rate, 0.1 * 0.2 * duration**2
                )
                self.assertEqual(
                    sample.scenarios()[-1].phase, "early" if time == 1.0 else "late"
                )

    def test_actual_simulations_and_fmea_preserve_phase_labels_and_results(self):
        sample = make_sample(PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]}))
        sample.add_fault_times([1.0, 4.0], weights=[1.0, 1.0], n_joint=2)
        model = sample.faultdomain.mdl
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        self.assertEqual(
            [scenario.phase for scenario in sample.scenarios()], ["early", "late"]
        )
        for scenario in sample.scenarios():
            with self.subTest(time=scenario.time):
                direct = JointFaultScenario.from_faults(
                    tuple(sample.faultdomain.faults),
                    scenario.time,
                    mdl=model,
                    phasemap=sample.phasemap,
                )
                expected_result, expected_history = Simulation(mdl=model, scen=direct)()
                self.assertEqual(result.get(scenario.name), expected_result)
                self.assertEqual(history.get(scenario.name), expected_history)
        table = FMEA(
            result,
            sample,
            group_by=("phase", "time"),
            average_metric=["scenario_rate"],
            sum_metric=["xy"],
        )
        self.assertEqual(set(table["sum_xy"]), {("early", 1.0), ("late", 4.0)})
        for scenario in sample.scenarios():
            self.assertAlmostEqual(
                table["average_scenario_rate"][(scenario.phase, scenario.time)],
                scenario.rate,
            )


if __name__ == "__main__":
    unittest.main()
