#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for sampling and optimization inputs.

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
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    JointFaultSample,
    SampleApproach,
)
from tests.test_default_fault_sample_timestep import make_domain


class TestFaultWeightArrays(unittest.TestCase):
    def assert_samples_equal(self, actual, expected):
        self.assertEqual(
            [s.asdict() for s in actual.scenarios()],
            [s.asdict() for s in expected.scenarios()],
        )
        self.assertEqual(sorted(actual.get_times()), sorted(expected.get_times()))

    def test_weight_arrays_match_list_weights_for_single_and_joint_faults(self):
        for constructor in (list, tuple, np.array):
            for joint in (1, 2):
                for maps in (True, False):
                    with self.subTest(
                        container=constructor.__name__, joint=joint, maps=maps
                    ):
                        domain = make_domain(0.5)
                        weights = constructor([0.0, 0.25, 0.75])
                        sample = FaultSample(domain, def_mdl_phasemap=maps)
                        reference = FaultSample(domain, def_mdl_phasemap=maps)
                        sample.add_fault_times([1.0, 1.5, 2.0], weights, n_joint=joint)
                        reference.add_fault_times(
                            [1.0, 1.5, 2.0], [0.0, 0.25, 0.75], n_joint=joint
                        )
                        self.assert_samples_equal(sample, reference)
                        self.assertTrue(
                            all(
                                s.rate == 0 for s in sample.scenarios() if s.time == 1.0
                            )
                        )
                        np.testing.assert_array_equal(weights, [0.0, 0.25, 0.75])

    def test_single_zero_is_an_explicit_weight_and_not_an_omitted_weight(self):
        for joint in (1, 2):
            for maps in (False, True):
                with self.subTest(joint=joint, maps=maps):
                    domain = make_domain(0.5)
                    sample = FaultSample(domain, def_mdl_phasemap=maps)
                    sample.add_fault_times([1.0], np.array([0.0]), n_joint=joint)
                    self.assertTrue(sample.scenarios())
                    self.assertTrue(all(s.rate == 0 for s in sample.scenarios()))

    def test_empty_arrays_and_none_retain_implicit_weights(self):
        domain = make_domain(0.5)
        for weights in ([], (), np.array([]), None):
            for joint in (1, 2):
                with self.subTest(container=type(weights).__name__, joint=joint):
                    actual, reference = FaultSample(domain), FaultSample(domain)
                    actual.add_fault_times([1.0, 1.5, 2.0], weights, n_joint=joint)
                    reference.add_fault_times([1.0, 1.5, 2.0], n_joint=joint)
                    self.assert_samples_equal(actual, reference)

    def test_readonly_weight_views_pass_through_approach_and_joint_samples(self):
        domain = make_domain(0.5)
        model = domain.mdl
        left, right = FaultDomain(model), FaultDomain(model)
        left.add_fault("risk", "first")
        right.add_fault("risk", "second")
        weights = np.array([0.0, 9.0, 0.25, 9.0, 0.75, 9.0])[::2]
        weights.setflags(write=False)
        mapping = PhaseMap(model.sp.phases, dt=model.sp.dt)
        joint = JointFaultSample(left, right, phasemaps=[mapping])
        reference = JointFaultSample(left, right, phasemaps=[mapping])
        for sample, argument in ((joint, weights), (reference, weights.tolist())):
            sample.add_fault_times([1.0, 1.5, 2.0], argument, n_joint=2, p_cond=0.5)
        self.assert_samples_equal(joint, reference)
        approach = SampleApproach(model, phasemaps={"custom": mapping})
        approach.add_faultdomain("left", "fault", "risk", "first")
        approach.add_faultdomain("right", "fault", "risk", "second")
        approach.add_faultsample(
            "joint",
            "fault_times",
            ["left", "right"],
            [1.0, 1.5, 2.0],
            weights=weights,
            n_joint=2,
            p_cond=0.5,
            phasemap="custom",
        )
        self.assert_samples_equal(approach, reference)
        np.testing.assert_array_equal(weights, [0.0, 0.25, 0.75])

    def test_staged_and_unstaged_simulations_and_tables_match_list_weights(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                domain = make_domain(0.5)
                sample, reference = FaultSample(domain), FaultSample(domain)
                sample.add_fault_times([1.0, 1.5, 2.0], np.array([0.0, 0.25, 0.75]))
                reference.add_fault_times([1.0, 1.5, 2.0], [0.0, 0.25, 0.75])
                actual, history = propagate.fault_sample(
                    domain.mdl, sample, staged=staged, showprogress=False
                )
                expected, ref_history = propagate.fault_sample(
                    domain.mdl, reference, staged=staged, showprogress=False
                )
                self.assertEqual(actual, expected)
                self.assertEqual(history, ref_history)
                tables = [
                    FMEA(
                        res,
                        fs,
                        group_by=("phase", "time"),
                        average_metric=["scenario_rate"],
                        sum_metric=["xy"],
                    )
                    for res, fs in ((actual, sample), (expected, reference))
                ]
                self.assertEqual(
                    tables[0]["average_scenario_rate"],
                    tables[1]["average_scenario_rate"],
                )
                self.assertEqual(tables[0]["sum_xy"], tables[1]["sum_xy"])


if __name__ == "__main__":
    unittest.main()
