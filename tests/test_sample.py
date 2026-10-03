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
    sample_times_quad,
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


class TestQuadratureSampleMass(unittest.TestCase):
    """Nodes snapping to one injection time retain their combined probability."""

    def test_coincident_nodes_combine_weights_in_first_occurrence_order(self):
        times = [0.0, 1.0, 2.0, 3.0]
        nodes = [0.9, -0.9, -0.8, 0.8]
        weights = [1.0, 2.0, 3.0, 4.0]
        for wrap in (list, tuple, np.asarray):
            with self.subTest(container=wrap.__name__):
                before = [np.array(x) for x in (times, nodes, weights)]
                actual_times, actual_weights = sample_times_quad(
                    wrap(times), wrap(nodes), wrap(weights)
                )
                np.testing.assert_array_equal(actual_times, [3.0, 0.0])
                np.testing.assert_allclose(actual_weights, [0.5, 0.5], rtol=1e-14)
                for previous, actual in zip(before, (times, nodes, weights)):
                    np.testing.assert_array_equal(previous, actual)

    def test_legendre_rules_preserve_discretized_integrals_and_probability_mass(self):
        for count in (2, 3, 5, 10, 16):
            for offset, spacing in ((0.0, 1.0), (0.25, 0.5)):
                with self.subTest(count=count, offset=offset, spacing=spacing):
                    times = offset + np.arange(count) * spacing
                    nodes, weights = np.polynomial.legendre.leggauss(count)
                    normalized = weights / weights.sum()
                    raw = [
                        times[
                            np.argmin(
                                np.abs(times - np.quantile(times, (node + 1) / 2))
                            )
                        ]
                        for node in nodes
                    ]
                    selected, merged = sample_times_quad(times, nodes, weights)
                    self.assertEqual(len(selected), len(set(selected)))
                    self.assertEqual(set(selected), set(raw))
                    self.assertAlmostEqual(sum(merged), 1.0, places=14)
                    for t, weight in zip(selected, merged):
                        self.assertAlmostEqual(
                            weight,
                            sum(w for raw_t, w in zip(raw, normalized) if raw_t == t),
                            places=14,
                        )
                    for values in (
                        lambda x: np.ones_like(x),
                        lambda x: x,
                        lambda x: x**2,
                    ):
                        self.assertAlmostEqual(
                            np.dot(merged, values(np.asarray(selected))),
                            np.dot(normalized, values(np.asarray(raw))),
                            places=12,
                        )

    def test_distinct_nodes_and_existing_length_validation_are_preserved(self):
        selected, weights = sample_times_quad([0, 1, 2, 3, 4], [-0.5, 0.5], [1, 3])
        self.assertEqual(selected, [1, 3])
        np.testing.assert_array_equal(weights, [0.25, 0.75])
        selected, weights = sample_times_quad([2], [0.0], [2.0])
        self.assertEqual(selected, [2])
        self.assertEqual(weights, [1.0])
        with self.assertRaisesRegex(Exception, "Nodes length"):
            sample_times_quad([1, 2], [-1.0, 0.0, 1.0], [1.0, 1.0, 1.0])

    def test_fault_sample_names_do_not_discard_quadrature_probability(self):
        model = ExampleFunction(sp={"end_time": 9.0})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "no_charge")
        sample = FaultSample(domain, phasemap=PhaseMap({"standby": [0.0, 9.0]}))
        nodes, weights = np.polynomial.legendre.leggauss(10)
        sample.add_fault_phases("standby", method="quad", args=(nodes, weights))
        self.assertEqual(sample.num_scenarios(), 8)
        self.assertEqual(len(sample.named_scenarios()), sample.num_scenarios())
        self.assertEqual(len(sample.get_times()), sample.num_scenarios())
        probability = model.m.get_fault("no_charge").prob
        self.assertAlmostEqual(
            sum(s.rate for s in sample.named_scenarios().values()),
            probability,
            places=15,
        )
        expected_at_one = probability * (weights[1] + weights[2]) / weights.sum()
        at_one = [s for s in sample.scenarios() if s.time == 1.0]
        self.assertEqual(len(at_one), 1)
        self.assertAlmostEqual(at_one[0].rate, expected_at_one, places=15)

    def test_staged_and_unstaged_simulations_keep_the_weighted_quadrature_outcome(self):
        model = ExampleFunction(sp={"end_time": 9.0})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "no_charge")
        sample = FaultSample(domain, phasemap=PhaseMap({"standby": [0.0, 9.0]}))
        nodes, weights = np.polynomial.legendre.leggauss(10)
        sample.add_fault_phases("standby", method="quad", args=(nodes, weights))
        raw_times = np.array([0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.0, 9.0])
        nominal_steps = np.maximum(raw_times - 1.0, 0.0)
        costs = nominal_steps * model.p.x + (9.0 - nominal_steps) * model.p.y
        expected = model.m.get_fault("no_charge").prob * np.dot(
            weights / weights.sum(), costs
        )
        reference_history = None
        for staged in (False, True):
            with self.subTest(staged=staged):
                result, history = propagate.fault_sample(
                    model, sample, staged=staged, showprogress=False
                )
                self.assertEqual(len(result.nest(levels=1)), sample.num_scenarios() + 1)
                actual = sum(
                    s.rate * result.get(s.name + ".tend.classify.xy")
                    for s in sample.named_scenarios().values()
                )
                self.assertAlmostEqual(actual, expected, places=13)
                if reference_history is not None:
                    self.assertEqual(history, reference_history)
                reference_history = history
                for scenario in sample.scenarios():
                    hist = history.get(scenario.name)
                    self.assertEqual(hist.get_fault_time(), int(scenario.time))
                self.assertEqual(model.s.x, 0.0)


if __name__ == "__main__":
    unittest.main()
