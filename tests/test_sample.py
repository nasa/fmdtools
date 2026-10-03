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

import unittest

import numpy as np

from fmdtools.analyze.phases import PhaseMap, join_phasemaps
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    JointFaultSample,
    sample_times_even,
)


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
                sample = FaultSample(domains[0], def_mdl_phasemap=use_model_phases)
                self.assertEqual(sample.scenarios(), [])
                self.assertEqual(sample.get_times(), [])
                self.assertEqual(bool(sample.phasemap), use_model_phases)


class TestEvenSamplingSupport(unittest.TestCase):
    """Even samples stay on the supplied support, including disconnected modes."""

    def test_gapped_shifted_and_sparse_supports_never_create_unavailable_times(self):
        cases = (
            ([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], 3, 1.0),
            ([0.0, 0.5, 1.0, 5.0, 5.5, 6.0], 3, 0.5),
            ([0.25, 1.25, 2.25, 3.25, 4.25], 2, 1.0),
            ([0.0, 1.0, 10.0, 11.0, 100.0, 101.0], 3, 1.0),
            ([0.0, 0.1, 0.7, 0.8, 4.2, 4.3], 3, 0.1),
        )
        for values, count, dt in cases:
            for wrap in (list, tuple, np.asarray):
                with self.subTest(values=values, container=wrap.__name__, dt=dt):
                    times = wrap(values)
                    before = np.asarray(times).copy()
                    sampled, weights = sample_times_even(times, count, dt=dt)
                    self.assertEqual(len(sampled), count)
                    self.assertTrue(set(sampled).issubset(values))
                    self.assertEqual(len(set(sampled)), len(sampled))
                    self.assertAlmostEqual(sum(weights), 1.0)
                    np.testing.assert_array_equal(weights, np.full(count, 1.0 / count))
                    np.testing.assert_array_equal(times, before)

    def test_existing_uniform_grid_rounding_and_all_times_fallback_are_unchanged(self):
        for count in (1, 2, 3, 4, 6):
            for dt in (1.0, 0.5, 2.0):
                with self.subTest(count=count, dt=dt):
                    times = np.arange(7) * dt
                    if count + 2 > len(times):
                        expected = times
                    else:
                        expected = [
                            round(np.quantile(times, p / (count + 1)) / dt) * dt
                            for p in range(1, count + 1)
                        ]
                    actual, weights = sample_times_even(times, count, dt=dt)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertAlmostEqual(sum(weights), 1.0)
        integer_grid, _ = sample_times_even([0, 1, 2, 3, 4], 2)
        self.assertEqual(integer_grid, [1.0, 3.0])
        self.assertTrue(all(type(value) is float for value in integer_grid))
        for times in ([], [3.0], [3.0, 4.0]):
            with self.subTest(times=times):
                actual, weights = sample_times_even(times, 3)
                self.assertEqual(actual, times)
                self.assertEqual(len(weights), len(times))

    def test_modephase_sampling_keeps_total_probability_out_of_excluded_modes(self):
        for include_gap in (False, True):
            with self.subTest(include_gap=include_gap):
                model = ExampleFunction(sp={"end_time": 12.0})
                domain = FaultDomain(model)
                domain.add_fault(model.name, "no_charge")
                phases = {"standby": [0.0, 2.0], "standby1": [10.0, 12.0]}
                modephases = {"standby": {"standby", "standby1"}}
                if include_gap:
                    phases["charge"] = [3.0, 9.0]
                    modephases["charge"] = {"charge"}
                phase_map = PhaseMap(phases, modephases)
                sample = FaultSample(domain, phasemap=phase_map)
                sample.add_fault_phases("standby", method="even", args=(3,))
                for scenario in sample.scenarios():
                    self.assertEqual(
                        phase_map.find_base_phase(scenario.time), "standby"
                    )
                    self.assertGreater(scenario.rate, 0.0)
                self.assertAlmostEqual(
                    sum(s.rate for s in sample.scenarios()),
                    model.m.get_fault("no_charge").prob,
                    places=15,
                )
                self.assertEqual(sorted(sample.get_times()), [1.0, 2.0, 11.0])

    def test_disconnected_phase_samples_run_staged_and_unstaged(self):
        model = ExampleFunction(sp={"end_time": 12.0})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "no_charge")
        phase_map = PhaseMap(
            {"standby": [0.0, 2.0], "standby1": [10.0, 12.0]},
            {"standby": {"standby", "standby1"}},
        )
        sample = FaultSample(domain, phasemap=phase_map)
        sample.add_fault_phases("standby", method="even", args=(3,))
        reference = FaultSample(domain, phasemap=phase_map)
        reference.add_fault_times([1.0, 2.0, 11.0], weights=[1.0 / 3] * 3)
        for staged in (False, True):
            with self.subTest(staged=staged):
                actual_result, actual_history = propagate.fault_sample(
                    model, sample, staged=staged, showprogress=False
                )
                expected_result, expected_history = propagate.fault_sample(
                    model, reference, staged=staged, showprogress=False
                )
                self.assertEqual(actual_result, expected_result)
                self.assertEqual(actual_history, expected_history)
                for scenario in sample.scenarios():
                    history = actual_history.get(scenario.name)
                    self.assertEqual(history.get_fault_time(), int(scenario.time))
                self.assertEqual(model.s.x, 0.0)


if __name__ == "__main__":
    unittest.main()
