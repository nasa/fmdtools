#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault-sample times after scenario pruning.

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

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim.propagate import MultiEventSimulation
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    JointFaultSample,
    SampleApproach,
)


class TestPrunedFaultTimes(unittest.TestCase):
    def make_sample(self):
        model = ExampleFunction(sp={"end_time": 5.0})
        domain = FaultDomain(model)
        domain.add_fault("examplefunction", "low")
        return FaultSample(domain, def_mdl_phasemap=False)

    def test_pruning_preserves_only_retained_times_and_scenario_objects(self):
        sample = self.make_sample()
        for time, weight in ((1.0, 0.0), (2.0, 0.5), (3.0, 0.0), (4.0, 1.0)):
            sample.add_single_fault_scenario(
                ("examplefunction", "low"), time, weight=weight
            )
        retained = [sample.scenarios()[1], sample.scenarios()[3]]
        before = [scenario.asdict() for scenario in retained]
        sample.prune_scenarios()
        self.assertEqual(set(sample.get_times()), {2.0, 4.0})
        self.assertEqual(len(sample.scenarios()), 2)
        for actual, original, values in zip(sample.scenarios(), retained, before):
            self.assertIs(actual, original)
            self.assertEqual(actual.asdict(), values)

    def test_shared_injection_time_remains_until_its_last_scenario_is_removed(self):
        model = ExFxnArch(sp={"end_time": 5.0})
        domain = FaultDomain(model)
        domain.add_faults(("ex_fxn", "low"), ("ex_fxn2", "no_charge"))
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_single_fault_scenario(("ex_fxn", "low"), 2.0, weight=0.0)
        sample.add_single_fault_scenario(("ex_fxn2", "no_charge"), 2.0)
        sample.add_single_fault_scenario(("ex_fxn", "low"), 4.0, weight=0.0)
        sample.prune_scenarios()
        self.assertEqual(set(sample.get_times()), {2.0})
        self.assertEqual(len(sample.scenarios()), 1)
        sample.prune_scenarios(scen_var="time", comparator=np.less, value=2.0)
        self.assertEqual(sample.scenarios(), [])
        self.assertEqual(sample.get_times(), [])

    def test_custom_time_predicate_and_repeated_pruning(self):
        sample = self.make_sample()
        sample.add_fault_times([0.5, 1.5, 3.5, 4.5])
        sample.prune_scenarios(scen_var="time", comparator=np.less_equal, value=3.5)
        self.assertEqual(set(sample.get_times()), {0.5, 1.5, 3.5})
        sample.prune_scenarios(scen_var="time", comparator=np.greater, value=1.0)
        self.assertEqual(set(sample.get_times()), {1.5, 3.5})
        sample.prune_scenarios()
        self.assertEqual(set(sample.get_times()), {1.5, 3.5})

    def test_empty_samples_and_additions_after_pruning(self):
        sample = self.make_sample()
        sample.prune_scenarios()
        self.assertEqual(sample.get_times(), [])
        sample.add_fault_times([1.0], weights=[0.0])
        sample.prune_scenarios()
        self.assertEqual(sample.get_times(), [])
        sample.add_fault_times([3.0])
        self.assertEqual(set(sample.get_times()), {3.0})
        sample.prune_scenarios()
        self.assertEqual(set(sample.get_times()), {3.0})

    def test_joint_sample_and_joint_scenarios_retain_only_surviving_times(self):
        model = ExFxnArch(sp={"end_time": 5.0})
        faults = (("ex_fxn", "low"), ("ex_fxn2", "no_charge"))
        domain = FaultDomain(model)
        domain.add_faults(*faults)
        for cls in (FaultSample, JointFaultSample):
            with self.subTest(sample_class=cls.__name__):
                sample = cls(domain, def_mdl_phasemap=False)
                sample.add_joint_fault_scenario(faults, 2.0, weight=0.0)
                sample.add_joint_fault_scenario(faults, 4.0)
                sample.prune_scenarios()
                self.assertEqual(sample.scenarios()[0].times, (4.0,))
                self.assertEqual(set(sample.get_times()), {4.0})

    def test_injection_time_is_distinct_from_an_explicit_earlier_start(self):
        sample = self.make_sample()
        sample.add_single_fault_scenario(("examplefunction", "low"), 2.0, weight=0.0)
        sample.add_single_fault_scenario(("examplefunction", "low"), 4.0, starttime=1.0)
        sample.prune_scenarios()
        self.assertEqual(sample.scenarios()[0].time, 1.0)
        self.assertEqual(sample.scenarios()[0].times, (4.0,))
        self.assertEqual(set(sample.get_times()), {4.0})

    def test_sample_approach_unions_only_surviving_times(self):
        model = ExampleFunction(sp={"end_time": 5.0})
        approach = SampleApproach(model, phasemaps={}, def_mdl_phasemap=False)
        approach.add_faultdomain("low", "fault", "examplefunction", "low")
        approach.add_faultsample(
            "early", "fault_times", "low", [1.0, 2.0], weights=[0.0, 1.0]
        )
        approach.add_faultsample(
            "late", "fault_times", "low", [3.0, 4.0], weights=[1.0, 0.0]
        )
        approach.prune_scenarios()
        self.assertEqual(set(approach.get_times()), {2.0, 3.0})
        approach.prune_scenarios(scen_var="time", comparator=np.less, value=0.0)
        self.assertEqual(approach.get_times(), [])

    def test_staged_propagation_copies_only_retained_times_without_changing_results(
        self,
    ):
        sample = self.make_sample()
        sample.add_fault_times([1.0, 3.0], weights=[0.0, 1.0])
        sample.prune_scenarios()
        model = sample.faultdomain.mdl
        initial_time = model.t.time
        staged = MultiEventSimulation(mdl=model, samp=sample, showprogress=False)
        staged_result, staged_history = staged()
        reference = MultiEventSimulation(
            mdl=model, samp=sample, staged=False, showprogress=False
        )
        direct_result, direct_history = reference()
        self.assertEqual(set(staged.mdls), {3.0})
        self.assertEqual(dict(staged_result), dict(direct_result))
        self.assertEqual(set(staged_history), set(direct_history))
        for key in staged_history:
            np.testing.assert_array_equal(staged_history[key], direct_history[key])
        self.assertEqual(model.t.time, initial_time)

    def test_pruning_nothing_preserves_all_injection_times(self):
        sample = self.make_sample()
        sample.add_fault_times([0.0, 2.0, 4.0])
        original = sample.scenarios()
        sample.prune_scenarios()
        self.assertEqual(set(sample.get_times()), {0.0, 2.0, 4.0})
        self.assertTrue(all(a is b for a, b in zip(original, sample.scenarios())))


if __name__ == "__main__":
    unittest.main()
