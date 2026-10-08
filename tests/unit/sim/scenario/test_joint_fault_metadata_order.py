#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for deterministic joint-fault metadata ordering.

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

from itertools import permutations
import json
import os
import pickle
import subprocess
import sys
import unittest

import numpy as np

from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.sim.scenario import JointFaultScenario, create_scenname


FAULTS = (("zeta", "trip"), ("alpha", "safe"), ("zeta", "low"))


class TestJointFaultMetadataOrder(unittest.TestCase):
    def test_every_input_order_has_the_same_canonical_metadata(self):
        for faults in permutations(FAULTS):
            for base in ("ind", "max", ("alpha", "safe")):
                with self.subTest(faults=faults, base=base):
                    scenario = JointFaultScenario.from_faults(
                        faults, 3.0, weight=0.4, baserate=base, p_cond=0.5
                    )
                    self.assertEqual(scenario.objects, ("alpha", "zeta"))
                    self.assertEqual(scenario.modes, ("low", "safe", "trip"))
                    self.assertEqual(scenario.joint_faults, 3)
                    self.assertEqual(scenario.time, 3.0)
                    self.assertEqual(scenario.times, (3.0,))
                    self.assertEqual(scenario.name, create_scenname(faults, 3.0))
                    expected_faults = {}
                    for obj, mode in faults:
                        expected_faults.setdefault(obj, []).append(mode)
                    self.assertEqual(scenario.sequence[3.0].faults, expected_faults)
                    self.assertAlmostEqual(
                        scenario.rate, 0.5 * 0.4
                    )

    def test_duplicate_components_and_modes_are_deduplicated_without_reordering_injections(
        self,
    ):
        cases = (
            (("b", "same"), ("a", "same")),
            (("b", "two"), ("b", "one")),
            (("plant.z", "trip"), ("plant.a", "low"), ("plant.z", "low")),
            (("only", "single"),),
        )
        for faults in cases:
            with self.subTest(faults=faults):
                scenario = JointFaultScenario.from_faults(faults, 2.0)
                self.assertEqual(
                    scenario.objects, tuple(sorted({f[0] for f in faults}))
                )
                self.assertEqual(scenario.modes, tuple(sorted({f[1] for f in faults})))
                self.assertEqual(scenario.joint_faults, len(faults))
                self.assertEqual(scenario.copy_with().asdict(), scenario.asdict())
                restored = pickle.loads(pickle.dumps(scenario))
                self.assertEqual(restored.asdict(), scenario.asdict())

    def test_metadata_is_stable_in_fresh_interpreters_with_different_hash_seeds(self):
        script = (
            "import json; from fmdtools.sim.scenario import JointFaultScenario; "
            "s=JointFaultScenario.from_faults(" + repr(FAULTS) + ",3.0); "
            "print(json.dumps([s.objects,s.modes]))"
        )
        for seed in ("1", "2", "19"):
            with self.subTest(hash_seed=seed):
                process = subprocess.run(
                    [sys.executable, "-c", script],
                    env={**os.environ, "PYTHONHASHSEED": seed},
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                self.assertEqual(
                    json.loads(process.stdout),
                    [["alpha", "zeta"], ["low", "safe", "trip"]],
                )

    def test_actual_joint_fault_fmea_groups_use_canonical_tuple_keys(self):
        model = ExFxnArch(sp={"end_time": 4.0})
        faults = (("ex_fxn2", "short"), ("ex_fxn", "low"))
        domain = FaultDomain(model)
        domain.add_faults(*faults)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_joint_fault_scenario(faults, 1.0)
        sample.add_joint_fault_scenario(tuple(reversed(faults)), 3.0)
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        original = result.copy()
        table = FMEA(
            result, sample, group_by=("objects", "modes"), sum_metric=["flowval"]
        )
        key = (("ex_fxn", "ex_fxn2"), ("low", "short"))
        self.assertEqual(list(table["sum_flowval"]), [key])
        expected = sum(
            result.get(s.name).get("tend.classify.flowval") for s in sample.scenarios()
        )
        self.assertAlmostEqual(table["sum_flowval"][key], expected)
        self.assertEqual(result, original)
        for scenario in sample.scenarios():
            self.assertEqual(scenario.objects, key[0])
            self.assertEqual(scenario.modes, key[1])
            actual = history.get(scenario.name)
            np.testing.assert_array_equal(
                actual["fxns.ex_fxn.m.faults.low"], actual.time >= scenario.times[0]
            )
            np.testing.assert_array_equal(
                actual["fxns.ex_fxn2.m.faults.short"], actual.time >= scenario.times[0]
            )


if __name__ == "__main__":
    unittest.main()
