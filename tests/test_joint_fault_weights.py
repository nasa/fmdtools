#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for joint-fault sampling weights.

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
from fmdtools.sim.sample import FaultDomain, FaultSample, JointFaultSample


class TestJointFaultWeights(unittest.TestCase):
    """Keep bulk sampling consistent with the weighted single-scenario API."""

    def make_sample(self, n_joint=2, phasemap=None):
        model = ExFxnArch(sp={"end_time": 6.0})
        domain = FaultDomain(model)
        faults = [("ex_fxn", "low"), ("ex_fxn2", "no_charge"), ("ex_fxn2", "short")]
        domain.add_faults(*faults[:n_joint])
        return FaultSample(domain, phasemap=phasemap, def_mdl_phasemap=False)

    def assert_matches_direct(self, sample, times, weights, **joint_kwargs):
        reference = FaultSample(sample.faultdomain, def_mdl_phasemap=False)
        faults = tuple(sample.faultdomain.faults)
        for time, weight in zip(times, weights):
            reference.add_joint_fault_scenario(
                faults, time, weight=weight, **joint_kwargs
            )
        self.assertEqual(
            [s.asdict() for s in sample.scenarios()],
            [s.asdict() for s in reference.scenarios()],
        )
        self.assertEqual(set(sample.get_times()), set(times))

    def test_explicit_weights_reach_joint_scenarios(self):
        for n_joint in (2, 3):
            for baserate in ("ind", "max", ("ex_fxn", "low")):
                for weights in ([0.25, 0.75], [0.0, 1.0]):
                    with self.subTest(
                        n_joint=n_joint, baserate=baserate, weights=weights
                    ):
                        sample = self.make_sample(n_joint)
                        kwargs = {"baserate": baserate, "p_cond": 0.4}
                        sample.add_fault_times(
                            [1.0, 2.0], weights=weights, n_joint=n_joint, **kwargs
                        )
                        self.assert_matches_direct(
                            sample, [1.0, 2.0], weights, **kwargs
                        )

    def test_phase_derived_weights_reach_joint_scenarios(self):
        for baserate in ("ind", "max"):
            with self.subTest(baserate=baserate):
                phases = PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]})
                sample = self.make_sample(phasemap=phases)
                times = [1.0, 2.0, 4.0]
                sample.add_fault_times(times, n_joint=2, baserate=baserate)
                self.assert_matches_direct(
                    sample, times, [0.5, 0.5, 1.0], baserate=baserate
                )

    def test_phase_sampling_preserves_time_weights(self):
        for method, args in (("even", (2,)), ("all", ())):
            with self.subTest(method=method):
                sample = self.make_sample(phasemap=PhaseMap({"run": [0.0, 6.0]}))
                sample.add_fault_phases(
                    "run", method=method, args=args, n_joint=2, baserate="max"
                )
                times = [s.times[0] for s in sample.scenarios()]
                self.assertEqual(len(times), 2 if method == "even" else 7)
                self.assert_matches_direct(
                    sample, times, [1.0 / len(times)] * len(times), baserate="max"
                )
                self.assertAlmostEqual(sum(s.rate for s in sample.scenarios()), 1.0)

    def test_joint_fault_sample_uses_the_same_weights(self):
        reference = self.make_sample()
        domains = []
        for fault in reference.faultdomain.faults:
            domain = FaultDomain(reference.faultdomain.mdl)
            domain.add_fault(*fault)
            domains.append(domain)
        sample = JointFaultSample(*domains, def_mdl_phasemap=False)
        sample.add_fault_times(
            [1.0, 2.0], weights=[0.25, 0.75], n_joint=2, baserate="max"
        )
        self.assert_matches_direct(sample, [1.0, 2.0], [0.25, 0.75], baserate="max")

    def test_default_weights_and_single_fault_sampling_are_unchanged(self):
        sample = self.make_sample()
        sample.add_fault_times([1.0, 2.0], n_joint=2)
        self.assert_matches_direct(sample, [1.0, 2.0], [1.0, 1.0])
        single = self.make_sample(n_joint=1)
        single.add_fault_times([1.0, 2.0], weights=[0.25, 0.75])
        np.testing.assert_allclose([s.rate for s in single.scenarios()], [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
