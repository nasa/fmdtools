#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for simulation exposure duration in fault rates.

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

from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.sim.scenario import JointFaultScenario, SingleFaultScenario


class ExposureMode(Mode):
    fault_first = Fault(prob=0.1, units="hr")
    fault_second = Fault(prob=0.2, units="hr")
    fault_per_sim = Fault(prob=0.25, units="sim")


class ExposureFunction(ExampleFunction):
    container_m = ExposureMode


class TestSimulationExposure(unittest.TestCase):
    def model(self, start=0.0, end=3.0, dt=1.0, units="hr"):
        return ExposureFunction(
            "risk",
            sp={
                "start_time": start,
                "end_time": end,
                "dt": dt,
                "units": units,
                "use_local": False,
            },
        )

    def test_positive_exposure_includes_the_final_timestep(self):
        for start, end, dt in (
            (0.0, 3.0, 1.0),
            (2.0, 5.0, 0.5),
            (0.0, 1.0, 1.0),
            (3.0, 3.0, 0.25),
            (0.0, 0.3, 0.1),
        ):
            for units, hours_per_unit in (
                ("hr", 1.0),
                ("min", 1.0 / 60),
                ("sec", 1.0 / 3600),
            ):
                for weight in (0.0, 0.3, 1.0):
                    with self.subTest(
                        start=start, end=end, dt=dt, units=units, weight=weight
                    ):
                        model = self.model(start, end, dt, units)
                        before = model.sp.asdict()
                        duration_hours = (end - start + dt) * hours_per_unit
                        actual = model.get_scen_rate(
                            "risk", "first", start, weight=weight
                        )
                        self.assertAlmostEqual(actual, 0.1 * duration_hours * weight)
                        self.assertGreaterEqual(actual, 0.0)
                        self.assertEqual(model.sp.asdict(), before)

    def test_single_and_joint_scenarios_use_the_same_positive_exposure(self):
        model = self.model()
        faults = (("risk", "first"), ("risk", "second"))
        for weight in (0.25, 1.0):
            for baserate, expected in (
                ("ind", 0.4 * 0.8 * weight**2),
                ("max", 0.8 * weight),
                (faults[0], 0.4 * weight),
            ):
                with self.subTest(weight=weight, baserate=baserate):
                    scen = JointFaultScenario.from_faults(
                        faults,
                        1.0,
                        mdl=model,
                        weight=weight,
                        baserate=baserate,
                        p_cond=0.5,
                    )
                    self.assertAlmostEqual(scen.rate, expected * 0.5)
                    self.assertEqual(scen.times, (1.0,))
            single = SingleFaultScenario.from_fault(
                faults[0], 1.0, mdl=model, weight=weight
            )
            self.assertAlmostEqual(single.rate, 0.4 * weight)

    def test_phase_exposure_and_per_simulation_faults_retain_their_conventions(self):
        for end in (1.0, 3.0):
            model = self.model(end=end)
            phases = PhaseMap({"run": [0.0, end]})
            with self.subTest(end=end):
                with np.errstate(divide="raise", invalid="raise"):
                    self.assertAlmostEqual(
                        model.get_scen_rate("risk", "first", 0.0, phasemap=phases),
                        0.1 * (end + 1.0),
                    )
                self.assertAlmostEqual(
                    model.get_scen_rate("risk", "per_sim", 0.0), 0.25
                )
                self.assertAlmostEqual(
                    model.get_scen_rate(
                        "risk", "per_sim", 0.0, phasemap=phases, weight=0.4
                    ),
                    0.1,
                )

    def test_fault_sampling_pruning_and_fmea_keep_positive_rate_mass(self):
        model = self.model()
        domain = FaultDomain(model)
        domain.add_fault("risk", "first")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0, 2.0], weights=[0.25, 0.75])
        np.testing.assert_allclose(
            [scen.rate for scen in sample.scenarios()], [0.1, 0.3]
        )
        sample.prune_scenarios()
        self.assertEqual(sample.num_scenarios(), 2)
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        table = FMEA(
            result,
            sample,
            group_by=("time",),
            average_metric="scenario_rate",
            expected_metric="xy",
            rates="scenario_rate",
            round_value=False,
        )
        for scen in sample.scenarios():
            key = (scen.times[0],)
            rate = 0.1 if key == (1.0,) else 0.3
            self.assertAlmostEqual(table["average_scenario_rate"][key], rate)
            expected_xy = result.get(scen.name).get("tend.classify.xy")
            self.assertAlmostEqual(table["expected_xy"][key], rate * expected_xy)
            self.assertEqual(len(history.get(scen.name).time), 4)
        self.assertAlmostEqual(sample.get_metric("rate", method="sum"), 0.4)


if __name__ == "__main__":
    unittest.main()
