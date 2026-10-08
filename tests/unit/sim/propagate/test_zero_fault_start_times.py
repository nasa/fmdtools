#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for explicit zero fault-scenario start times.

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
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim.propagate import MultiEventSimulation, Simulation
from fmdtools.sim.sample import BaseSample
from fmdtools.sim.scenario import JointFaultScenario, SingleFaultScenario


FAULTS = (("ex_fxn", "low"), ("ex_fxn2", "short"))


def make_scenario(joint, time, **kwargs):
    if joint:
        return JointFaultScenario.from_faults(FAULTS, time, **kwargs)
    return SingleFaultScenario.from_fault(FAULTS[0], time, **kwargs)


class InitialCheckpointSample(BaseSample):
    """A scenario batch explicitly staged from the initial checkpoint."""

    def __init__(self, scenario):
        self.scenario = scenario

    def scenarios(self):
        return [self.scenario]

    def get_times(self):
        return [0.0]


class TestZeroFaultStartTimes(unittest.TestCase):
    def test_zero_start_is_preserved_for_both_scenario_factories(self):
        for joint in (False, True):
            for injection in (0.0, 2.0, 3.5):
                for start in (0, 0.0, np.int64(0), np.float64(0.0)):
                    with self.subTest(
                        joint=joint,
                        injection=injection,
                        start_type=type(start).__name__,
                    ):
                        scenario = make_scenario(joint, injection, starttime=start)
                        self.assertEqual(scenario.time, 0.0)
                        self.assertEqual(scenario.times, (injection,))
                        self.assertEqual(tuple(scenario.sequence), (injection,))

    def test_none_and_nonzero_start_times_keep_their_existing_meaning(self):
        for joint in (False, True):
            for start in (None, -1.0, 1.0, np.float64(1.5), 2.0):
                with self.subTest(joint=joint, start=start):
                    actual = make_scenario(joint, 2.0, starttime=start)
                    self.assertEqual(actual.time, 2.0 if start is None else start)
                    self.assertEqual(actual.times, (2.0,))
            self.assertEqual(make_scenario(joint, 2.0).time, 2.0)

    def test_rates_phases_names_and_injections_do_not_depend_on_start_override(self):
        model = ExFxnArch(sp={"end_time": 4.0})
        phases = PhaseMap({"early": [0.0, 1.0], "late": [2.0, 4.0]})
        original = copy.deepcopy(phases.phases)
        for joint in (False, True):
            options = {"mdl": model, "phasemap": phases, "weight": 0.4}
            if joint:
                options.update(baserate="max", p_cond=0.3)
            reference = make_scenario(joint, 2.0, **options)
            for start in (0.0, 1.0):
                with self.subTest(joint=joint, start=start):
                    actual = make_scenario(joint, 2.0, starttime=start, **options)
                    expected = reference.asdict()
                    expected["time"] = start
                    self.assertEqual(actual.asdict(), expected)
                    self.assertEqual(actual.copy_with().asdict(), expected)
        self.assertEqual(phases.phases, original)

    def test_actual_staged_runs_select_the_requested_initial_checkpoint(self):
        model = ExFxnArch(sp={"end_time": 4.0})
        for joint in (False, True):
            with self.subTest(joint=joint):
                scenario = make_scenario(joint, 2.0, mdl=model, starttime=0.0)
                sample = InitialCheckpointSample(scenario)
                staged = MultiEventSimulation(
                    mdl=model, samp=sample, showprogress=False
                )
                result, history = staged()
                direct_result, direct_history = Simulation(mdl=model, scen=scenario)()
                self.assertEqual(tuple(staged.mdls), (0.0,))
                self.assertEqual(staged.mdls[0.0].t.time, -0.1)
                self.assertEqual(result.get(scenario.name), direct_result)
                self.assertEqual(history.get(scenario.name), direct_history)
                np.testing.assert_array_equal(
                    history.get(scenario.name).time, np.arange(5.0)
                )
                trace = history.get(scenario.name)["fxns.ex_fxn.m.faults.low"]
                np.testing.assert_array_equal(trace, [False, False, True, True, True])
                self.assertEqual(model.t.time, -0.1)


if __name__ == "__main__":
    unittest.main()
