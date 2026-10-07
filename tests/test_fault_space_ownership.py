#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for reusable disturbance-space specifications.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
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
from types import MappingProxyType

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    ParameterDomain,
    ParameterSample,
)


def make_domain(specification, **kwargs):
    domain = FaultDomain(ExampleFunction(sp={"end_time": 3.0}))
    domain.add_fault_space("examplefunction", "low", specification, **kwargs)
    return domain


def normalized_definitions(domain):
    return {key: fault.asdict() for key, fault in domain.faults.items()}


class TestFaultSpaceOwnership(unittest.TestCase):
    def test_sets_and_range_tuples_are_not_modified_or_replaced(self):
        shared = {0.0, 5.0}
        choices = {"s.x": shared, "s.y": (1.0, 2.0, 2)}
        before = copy.deepcopy(choices)
        for limit in ("all", 3, 50):
            with self.subTest(limit=limit):
                domain = make_domain(choices, n=limit)
                self.assertTrue(domain.faults)
                self.assertEqual(choices, before)
                self.assertIs(choices["s.x"], shared)
                self.assertIs(type(choices["s.y"]), tuple)
                self.assertEqual(shared, {0.0, 5.0})

    def test_mapping_proxy_and_shared_set_aliases_keep_their_ownership(self):
        options = {2.0, 8.0}
        backing = {"s.x": options, "s.y": options}
        view = MappingProxyType(backing)
        domain = make_domain(view)
        self.assertEqual(options, {2.0, 8.0})
        self.assertIs(backing["s.x"], backing["s.y"])
        observed = {fault.disturbances for fault in domain.faults.values()}
        expected = {
            tuple(
                (key, value) for key, value in zip(backing, pair) if value is not None
            )
            for pair in itertools.product((2.0, 8.0, None), repeat=2)
        } - {()}
        self.assertEqual(observed, expected)
        for fault in domain.faults.values():
            self.assertAlmostEqual(fault.prob, 1 / 9)

    def test_reusing_a_specification_keeps_seeded_selection_and_probabilities(self):
        specification = {"s.x": {0.0, 2.0, 5.0}, "s.y": (1.0, 3.0, 3)}
        original = copy.deepcopy(specification)
        for seed, limit in itertools.product((0, 7, 17), ("all", 4)):
            with self.subTest(seed=seed, limit=limit):
                first = make_domain(
                    specification, n=limit, seed=seed, prob=0.02, prefix="severity_"
                )
                second = make_domain(
                    specification, n=limit, seed=seed, prob=0.02, prefix="severity_"
                )
                control = make_domain(
                    copy.deepcopy(original),
                    n=limit,
                    seed=seed,
                    prob=0.02,
                    prefix="severity_",
                )
                self.assertEqual(
                    normalized_definitions(first), normalized_definitions(second)
                )
                self.assertEqual(
                    normalized_definitions(first), normalized_definitions(control)
                )
                self.assertEqual(specification, original)
                for fault in first.faults.values():
                    self.assertEqual(fault.prob, 0.02)

    def test_a_failed_request_does_not_contaminate_the_specification(self):
        for faultname in ("missing",):
            specification = {"s.x": {1.0, 2.0}, "s.y": (0.0, 2.0, 3)}
            before = copy.deepcopy(specification)
            with self.assertRaises(Exception):
                domain = FaultDomain(ExampleFunction())
                domain.add_fault_space("examplefunction", faultname, specification)
            self.assertEqual(specification, before)
        self.assertEqual(make_domain({}).faults, {})

    def test_shared_parameter_constraints_remain_finite_after_fault_sampling(self):
        parameter_domain = ParameterDomain(ExampleParameter)
        parameter_domain.add_variable("x", var_set={1.0, 3.0})
        specification = {"s.x": parameter_domain.variables["x"]}
        make_domain(specification)
        self.assertEqual(parameter_domain.variables["x"], {1.0, 3.0})
        sample = ParameterSample(parameter_domain)
        sample.add_variable_ranges()
        self.assertEqual({s.p["x"] for s in sample.scenarios()}, {1.0, 3.0})
        self.assertTrue(all(np.isfinite(s.p["x"]) for s in sample.scenarios()))

    def test_repeated_fault_simulations_preserve_configurations_and_results(self):
        specification = {"s.x": {0.0, 5.0}, "s.y": (1.0, 2.0, 2)}
        before = copy.deepcopy(specification)
        saved_result = saved_history = None
        for _ in range(2):
            domain = make_domain(specification)
            sample = FaultSample(domain, def_mdl_phasemap=False)
            sample.add_fault_times([1.0])
            result, history = propagate.fault_sample(
                domain.mdl, sample, showprogress=False
            )
            self.assertEqual(len(sample.scenarios()), 8)
            self.assertAlmostEqual(sum(s.rate for s in sample.scenarios()), 8 / 9)
            if saved_result is not None:
                self.assertEqual(result, saved_result)
                self.assertEqual(history, saved_history)
            saved_result, saved_history = result, history
            self.assertEqual(specification, before)
            self.assertFalse(domain.mdl.m.any_faults())


if __name__ == "__main__":
    unittest.main()
