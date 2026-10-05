#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for joint-domain construction through SampleApproach.

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
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultSample, JointFaultSample, SampleApproach


class TestApproachJointDomains(unittest.TestCase):
    def make_approach(self):
        model = ExFxnArch(sp={"end_time": 4.0})
        maps = {
            "left_map": PhaseMap({"early": [0.0, 2.0], "late": [3.0, 4.0]}),
            "right_map": PhaseMap({"first": [0.0, 1.0], "second": [2.0, 4.0]}),
        }
        approach = SampleApproach(model, phasemaps=maps)
        approach.add_faultdomain("left", "fault", "ex_fxn", "low")
        approach.add_faultdomain("right", "fault", "ex_fxn2", "low")
        return approach

    def assert_equivalent(self, actual, expected):
        self.assertEqual(
            [s.asdict() for s in actual.scenarios()],
            [s.asdict() for s in expected.scenarios()],
        )
        self.assertEqual(actual.get_times(), expected.get_times())
        self.assertEqual(
            list(actual.faultdomain.faults), list(expected.faultdomain.faults)
        )
        self.assertEqual(actual.phasemap.phases, expected.phasemap.phases)

    def test_domain_names_and_phase_map_forms_match_direct_joint_construction(self):
        for phase_form in (
            "default",
            "none",
            "named",
            "object",
            "dict",
            "tuple",
            "multiple",
        ):
            for order in (("left", "right"), ("right", "left")):
                with self.subTest(phase_form=phase_form, order=order):
                    approach = self.make_approach()
                    phase_map = approach.phasemaps["left_map"]
                    forms = {
                        "default": {},
                        "none": [],
                        "named": "left_map",
                        "object": phase_map,
                        "dict": phase_map.phases,
                        "tuple": (("early", 0.0, 2.0), ("late", 3.0, 4.0)),
                        "multiple": ["left_map", "right_map"],
                    }
                    if phase_form in ("default", "none"):
                        maps = []
                    elif phase_form == "multiple":
                        maps = [phase_map, approach.phasemaps["right_map"]]
                    else:
                        maps = [phase_map]
                    domains = [approach.faultdomains[name] for name in order]
                    expected = JointFaultSample(*domains, phasemaps=maps)
                    expected.add_fault_times(
                        [1.0, 3.0],
                        weights=[0.25, 0.75],
                        n_joint=2,
                        baserate="ind",
                        p_cond=0.5,
                    )
                    names = list(order)
                    argument = forms[phase_form]
                    before_names = names.copy()
                    approach.add_faultsample(
                        "joint",
                        "fault_times",
                        names,
                        [1.0, 3.0],
                        phasemap=argument,
                        weights=[0.25, 0.75],
                        n_joint=2,
                        baserate="ind",
                        p_cond=0.5,
                    )
                    actual = approach.faultsamples["joint"]
                    self.assertIsInstance(actual, JointFaultSample)
                    self.assertIs(actual.faultdomain.mdl, approach.mdl)
                    self.assert_equivalent(actual, expected)
                    self.assertEqual(names, before_names)
                    np.testing.assert_allclose(
                        [s.rate for s in actual.scenarios()],
                        [0.25**2 * 0.5, 0.75**2 * 0.5],
                    )

    def test_single_domain_paths_and_overlapping_domain_entries_remain_consistent(self):
        for names in ("left", ["left"]):
            with self.subTest(names=names):
                approach = self.make_approach()
                approach.add_faultsample("single", "fault_times", names, [1.0, 2.0])
                self.assertIs(type(approach.faultsamples["single"]), FaultSample)
                self.assertEqual(len(approach.scenarios()), 2)
        approach = self.make_approach()
        names = ["left", "right", "left"]
        approach.add_faultsample("joint", "fault_times", names, [2.0], n_joint=2)
        self.assertEqual(len(approach.scenarios()), 1)
        self.assertEqual(len(approach.faultsamples["joint"].faultdomain.faults), 2)
        self.assertEqual(names, ["left", "right", "left"])

    def test_unknown_domains_and_phase_names_do_not_replace_existing_samples(self):
        for domains, phases in (
            (["left", "missing"], {}),
            (["left", "right"], ["left_map", "missing"]),
        ):
            with self.subTest(domains=domains, phases=phases):
                approach = self.make_approach()
                approach.add_faultsample("existing", "fault_times", "left", [1.0])
                existing = approach.faultsamples["existing"]
                with self.assertRaises(KeyError):
                    approach.add_faultsample(
                        "existing",
                        "fault_times",
                        domains,
                        [2.0],
                        phasemap=phases,
                        n_joint=2,
                    )
                self.assertEqual(list(approach.faultsamples), ["existing"])
                self.assertIs(approach.faultsamples["existing"], existing)

    def test_actual_staged_and_unstaged_runs_match_direct_joint_samples(self):
        approach = self.make_approach()
        approach.add_faultsample(
            "joint",
            "fault_times",
            ["left", "right"],
            [1.0, 3.0],
            phasemap=["left_map", "right_map"],
            n_joint=2,
            weights=[0.25, 0.75],
        )
        reference = JointFaultSample(
            approach.faultdomains["left"],
            approach.faultdomains["right"],
            phasemaps=[approach.phasemaps["left_map"], approach.phasemaps["right_map"]],
        )
        reference.add_fault_times([1.0, 3.0], n_joint=2, weights=[0.25, 0.75])
        for staged in (False, True):
            with self.subTest(staged=staged):
                result, history = propagate.fault_sample(
                    approach.mdl, approach, showprogress=False, staged=staged
                )
                expected_result, expected_history = propagate.fault_sample(
                    approach.mdl, reference, showprogress=False, staged=staged
                )
                self.assertEqual(result, expected_result)
                self.assertEqual(history, expected_history)
                self.assertEqual(len(result.nest(1)), 3)


if __name__ == "__main__":
    unittest.main()
