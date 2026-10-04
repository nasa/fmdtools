#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for sampling classes, including joint fault sample initialization.

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
import itertools
import unittest

import numpy as np

from fmdtools.analyze.phases import PhaseMap, join_phasemaps
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, JointFaultSample


def make_domains():
    """Return two fresh fault domains on the same example model."""
    model = ExFxnArch(sp={"end_time": 6.0})
    first, second = FaultDomain(model), FaultDomain(model)
    first.add_fault("ex_fxn", "low")
    second.add_fault("ex_fxn2", "no_charge")
    return first, second


def phase_options(mode):
    """Return the phase-map options for a test mode."""
    if mode == "disabled":
        return {"def_mdl_phasemap": False}
    if mode == "joined":
        return {
            "phasemaps": [
                PhaseMap({"first": [0, 4]}),
                PhaseMap({"second": [2, 6]}),
            ]
        }
    if mode == "single":
        return {"phasemaps": [PhaseMap({"first": [0, 4]})]}
    return {}


class TestJointFaultSample(unittest.TestCase):
    """Test JointFaultSample initialization and sampling behavior."""

    def test_joint_sample_starts_empty_with_merged_domains(self):
        for mode in ("default", "disabled", "joined", "single"):
            for domain_count in (1, 2):
                with self.subTest(mode=mode, domain_count=domain_count):
                    domains = make_domains()
                    selected = domains[:domain_count]
                    options = phase_options(mode)
                    sample = JointFaultSample(*selected, **options)
                    self.assertEqual(sample.scenarios(), [])
                    self.assertEqual(sample.get_times(), [])
                    self.assertEqual(sample.num_scenarios(), 0)
                    self.assertEqual(sample.named_scenarios(), {})
                    self.assertIs(sample.faultdomain.mdl, selected[0].mdl)
                    expected_faults = {
                        key: fault
                        for domain in selected
                        for key, fault in domain.faults.items()
                    }
                    self.assertEqual(sample.faultdomain.faults, expected_faults)
                    if mode == "disabled":
                        self.assertFalse(sample.phasemap)
                    elif mode in ("joined", "single"):
                        expected = join_phasemaps(*options["phasemaps"])
                        self.assertEqual(sample.phasemap.phases, expected.phases)
                    else:
                        expected = PhaseMap(selected[0].mdl.sp.phases)
                        self.assertEqual(sample.phasemap.phases, expected.phases)

    def test_sampling_matches_equivalent_regular_fault_sample(self):
        for mode in ("default", "disabled", "joined"):
            for n_joint in (1, 2):
                with self.subTest(mode=mode, n_joint=n_joint):
                    domains = make_domains()
                    options = phase_options(mode)
                    combined = FaultDomain(domains[0].mdl)
                    for domain in domains:
                        combined.faults.update(domain.faults)
                    phasemap = (
                        join_phasemaps(*options["phasemaps"])
                        if "phasemaps" in options
                        else {}
                    )
                    reference = FaultSample(
                        combined,
                        phasemap=phasemap,
                        def_mdl_phasemap=mode != "disabled",
                    )
                    joint = JointFaultSample(*domains, **options)
                    reference.add_fault_times([2, 3], n_joint=n_joint)
                    joint.add_fault_times([2, 3], n_joint=n_joint)
                    self.assertEqual(
                        sorted(joint.get_times()),
                        sorted(reference.get_times()),
                    )
                    self.assertEqual(sorted(joint.get_times()), [2, 3])
                    self.assertEqual(
                        [scen.asdict() for scen in joint.scenarios()],
                        [scen.asdict() for scen in reference.scenarios()],
                    )
                    self.assertEqual(
                        joint.named_scenarios().keys(),
                        reference.named_scenarios().keys(),
                    )

    def test_separate_instances_do_not_share_sampling_state(self):
        domains = make_domains()
        first = JointFaultSample(*domains)
        second = JointFaultSample(*domains)
        original_domains = [dict(domain.faults) for domain in domains]
        first.add_single_fault_scenario(next(iter(domains[0].faults)), 1)
        first.faultdomain.faults.clear()
        self.assertEqual(second.scenarios(), [])
        self.assertEqual(second.get_times(), [])
        self.assertTrue(second.faultdomain.faults)
        self.assertEqual([domain.faults for domain in domains], original_domains)

    def test_empty_domains_have_usable_empty_state(self):
        model = ExFxnArch()
        sample = JointFaultSample(FaultDomain(model), def_mdl_phasemap=False)
        sample.add_fault_times([1, 2])
        self.assertEqual(sample.scenarios(), [])
        self.assertEqual(sample.get_times(), [])
        self.assertFalse(sample.phasemap)

    def test_joint_scenarios_can_be_propagated(self):
        domains = make_domains()
        sample = JointFaultSample(*domains)
        sample.add_fault_times([2], n_joint=2)
        results, histories = propagate.fault_sample(
            domains[0].mdl, sample, showprogress=False
        )
        name = sample.scenarios()[0].name
        self.assertIn(name, results.nest(levels=1))
        self.assertIn(name, histories.nest(levels=1))
        self.assertIn("nominal", histories.nest(levels=1))

    def test_regular_fault_sample_initialization_is_unchanged(self):
        for use_model_phases in (False, True):
            with self.subTest(use_model_phases=use_model_phases):
                domains = make_domains()
                sample = FaultSample(
                    domains[0], def_mdl_phasemap=use_model_phases
                )
                self.assertEqual(sample.scenarios(), [])
                self.assertEqual(sample.get_times(), [])
                self.assertEqual(bool(sample.phasemap), use_model_phases)


class TestFaultSpaceSampleLimit(unittest.TestCase):
    """Treat the requested number of disturbance combinations as a maximum."""

    def make_domain(self, ranges, n="all", seed=42, **kwargs):
        domain = FaultDomain(ExFxnArch())
        domain.add_fault_space(
            "ex_fxn",
            "low",
            copy.deepcopy(ranges),
            n=n,
            seed=seed,
            prefix="space",
            **kwargs,
        )
        return domain

    def test_large_and_equal_limits_keep_the_complete_fault_space(self):
        for ranges, population in (
            ({"s.x": {7.0, 11.0}}, 3),
            ({"s.x": (1.0, 3.0, 3)}, 4),
            ({"s.x": {7.0, 11.0}, "s.y": {-1.0, 2.0}}, 9),
            ({}, 1),
        ):
            complete = self.make_domain(ranges)
            for cap in (population, population + 1, 100, "all"):
                with self.subTest(ranges=ranges, cap=cap):
                    actual = self.make_domain(ranges, n=cap)
                    self.assertEqual(
                        {key: fault.asdict() for key, fault in actual.faults.items()},
                        {key: fault.asdict() for key, fault in complete.faults.items()},
                    )
                    self.assertEqual(len(actual.faults), population - 1)
                    for key, fault in actual.faults.items():
                        self.assertTrue(key[2].startswith("space"))
                        self.assertAlmostEqual(fault.prob, 1.0 / population)

    def test_smaller_limits_preserve_seeded_selection_and_probabilities(self):
        ranges = {"s.x": {7.0, 11.0}, "s.y": {-1.0, 2.0}}
        possible = {
            tuple(
                (state, value)
                for state, value in zip(ranges, values)
                if value is not None
            )
            for values in itertools.product((7.0, 11.0, None), (-1.0, 2.0, None))
        } - {()}
        for cap in (1, 2, 5):
            for seed in (0, 42):
                for kwargs in ({}, {"prob": 0.2, "cost": 17.0}):
                    with self.subTest(cap=cap, seed=seed, kwargs=kwargs):
                        first = self.make_domain(ranges, cap, seed, **kwargs)
                        repeated = self.make_domain(ranges, cap, seed, **kwargs)
                        self.assertEqual(
                            {
                                key: fault.asdict()
                                for key, fault in first.faults.items()
                            },
                            {
                                key: fault.asdict()
                                for key, fault in repeated.faults.items()
                            },
                        )
                        # The sampled all-nominal combination creates no fault.
                        self.assertIn(len(first.faults), (cap - 1, cap))
                        self.assertEqual(
                            len(
                                {fault.disturbances for fault in first.faults.values()}
                            ),
                            len(first.faults),
                        )
                        for fault in first.faults.values():
                            self.assertIn(fault.disturbances, possible)
                            self.assertAlmostEqual(
                                fault.prob, kwargs.get("prob", 1 / cap)
                            )
                            self.assertEqual(fault.cost, kwargs.get("cost", 0.0))

    def test_larger_limit_propagates_every_available_disturbance(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                model = ExampleFunction(sp={"end_time": 4.0})
                domain = FaultDomain(model)
                domain.add_fault_space(
                    "examplefunction", "low", {"s.x": {7.0, 11.0}}, n=20
                )
                sample = FaultSample(domain)
                sample.add_fault_times([1.0, 3.0])
                self.assertEqual(len(sample.named_scenarios()), 4)
                result, history = propagate.fault_sample(
                    model, sample, staged=staged, showprogress=False
                )
                weighted_total = 0.0
                for scenario in sample.scenarios():
                    time = scenario.times[0]
                    injection = scenario.sequence[time].faults["examplefunction"]["low"]
                    value = dict(injection["disturbances"])["s.x"]
                    self.assertIn(value, (7.0, 11.0))
                    self.assertAlmostEqual(scenario.rate, 1.0 / 6)
                    expected_x = [float(i) if i < time else value for i in range(5)]
                    actual = history.get(scenario.name)
                    np.testing.assert_array_equal(actual.get("s.x"), expected_x)
                    outcome = value + 3.0 * (5.0 - time)
                    self.assertEqual(
                        result.get(scenario.name).get("tend.classify.xy"), outcome
                    )
                    weighted_total += scenario.rate * outcome
                self.assertAlmostEqual(weighted_total, 12.0)
                self.assertEqual(model.s.x, 0.0)
                self.assertEqual(model.m.faults, set())


if __name__ == "__main__":
    unittest.main()
