#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for overrides of preconstructed Fault definitions.

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
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class ObjectMode(Mode):
    fault_stuck: Fault = Fault(prob=0.2, cost=10.0, disturbances=(("s.position", 2.0),))
    fault_tuple = (0.2, 10.0, (), (("s.position", 2.0),), "sim")
    fault_dict = {"prob": 0.2, "cost": 10.0, "disturbances": {"s.position": 2.0}}


class Position(State):
    position: np.float64 = 5.0


class ObjectFunction(Function):
    container_m = ObjectMode
    container_s = Position

    def dynamic_behavior(self):
        if not self.m.has_fault("stuck"):
            self.s.position += 1.0

    def classify(self, **kwargs):
        return {"cost": self.s.position}


class PreScaledMode(Mode):
    failrate = 0.5
    fault_stuck: Fault = Fault(prob=0.2, failrate=0.5, cost=10.0)


class TestFaultObjectOverrides(unittest.TestCase):
    def test_each_explicit_field_override_is_used_and_source_is_unchanged(self):
        mode = ObjectMode()
        original = mode.get_fault("stuck")
        before = copy.deepcopy(original.asdict())
        cases = [
            {"prob": 0.0},
            {"prob": 0.75},
            {"cost": 0.0},
            {"cost": 80.0},
            {"phases": {"run": 0.3}},
            {"phases": ()},
            {"units": "min"},
            {"disturbances": {"s.position": 0.0}},
            {"disturbances": ()},
            {
                "prob": 0.8,
                "cost": 3.0,
                "units": "hr",
                "phases": {"run": 0.5},
                "disturbances": {"s.position": 7.0},
            },
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                supplied = copy.deepcopy(kwargs)
                actual = mode.get_fault("stuck", **kwargs)
                expected = Fault(**{**before, **kwargs})
                self.assertEqual(actual.asdict(), expected.asdict())
                self.assertIsNot(actual, original)
                self.assertEqual(original.asdict(), before)
                self.assertEqual(kwargs, supplied)
        self.assertIs(mode.get_fault("stuck"), original)

    def test_object_argument_and_subclass_are_preserved(self):
        class SpecializedFault(Fault):
            pass

        fault = SpecializedFault(prob=0.1, cost=2.0)
        mode = ObjectMode()
        actual = mode.get_fault(fault, prob=0.7)
        self.assertIsInstance(actual, SpecializedFault)
        self.assertEqual(actual.prob, 0.7)
        self.assertEqual(actual.cost, 2.0)
        self.assertEqual(fault.prob, 0.1)
        self.assertIs(mode.get_fault(fault), fault)

    def test_no_implicit_double_scaling_and_explicit_failrate_is_honored(self):
        mode = PreScaledMode()
        original = mode.get_fault("stuck")
        self.assertAlmostEqual(original.prob, 0.1)
        self.assertIs(mode.get_fault("stuck"), original)
        self.assertAlmostEqual(mode.get_fault("stuck", cost=8.0).prob, 0.1)
        self.assertAlmostEqual(mode.get_fault("stuck", failrate=0.25).prob, 0.025)
        self.assertAlmostEqual(
            mode.get_fault("stuck", prob=0.8, failrate=0.25).prob, 0.2
        )
        self.assertAlmostEqual(original.prob, 0.1)

    def test_tuple_and_dictionary_definitions_retain_equivalent_overrides(self):
        mode = ObjectMode()
        for field in ("stuck", "tuple", "dict"):
            with self.subTest(field=field):
                actual = mode.get_fault(
                    field, prob=0.4, cost=6.0, disturbances={"s.position": 9.0}
                )
                expected = Fault(prob=0.4, cost=6.0, disturbances={"s.position": 9.0})
                self.assertEqual(actual.asdict(), expected.asdict())
                self.assertEqual(mode.get_fault(field).prob, 0.2)

    def test_block_scenario_rate_uses_overridden_probability_units_and_opportunity(
        self,
    ):
        model = ObjectFunction(sp={"end_time": 3.0, "units": "hr"})
        rate = model.get_scen_rate(
            model.name, "stuck", 1.0, prob=0.03, units="hr", weight=0.5
        )
        self.assertAlmostEqual(rate, 0.03 * 4 * 0.5)
        phases = PhaseMap({"run": [0.0, 1.0], "idle": [2.0, 3.0]})
        rate = model.get_scen_rate(
            model.name,
            "stuck",
            1.0,
            prob=0.03,
            units="hr",
            phases={"run": 0.25},
            phasemap=phases,
            weight=0.5,
        )
        self.assertAlmostEqual(rate, 0.03 * 2 * 0.25 * 0.5)
        self.assertEqual(model.get_scen_rate(model.name, "stuck", 1.0, prob=0.0), 0.0)
        self.assertEqual(model.m.get_fault("stuck").prob, 0.2)

    def test_parametric_faults_reach_simulation_histories_and_expected_loss(self):
        model = ObjectFunction(sp={"end_time": 3.0})
        before = model.m.get_fault("stuck").asdict()
        domain = FaultDomain(model)
        choices = (("zero", 0.0, 0.1), ("high", 10.0, 0.3))
        for label, position, probability in choices:
            domain.add_fault(
                model.name,
                "stuck",
                label,
                prob=probability,
                disturbances={"s.position": position},
            )
        for (_, position, probability), fault in zip(choices, domain.faults.values()):
            self.assertEqual(dict(fault.disturbances), {"s.position": position})
            self.assertEqual(fault.prob, probability)
        total_expected_loss = 0.0
        for (_, position, probability), definition in zip(
            choices, domain.faults.values()
        ):
            # Configure a fresh model with the returned immutable definition.
            # In-place assignment to a read-only Fault is a separate operation.
            trial = ObjectFunction(m={"fault_stuck": definition}, sp={"end_time": 3.0})
            trial_domain = FaultDomain(trial)
            trial_domain.add_fault(trial.name, "stuck")
            sample = FaultSample(trial_domain, def_mdl_phasemap=False)
            sample.add_fault_times([1.0])
            result, history = propagate.fault_sample(trial, sample, showprogress=False)
            scenario = sample.scenarios()[0]
            self.assertEqual(scenario.rate, probability)
            self.assertEqual(result[scenario.name + ".tend.classify.cost"], position)
            np.testing.assert_array_equal(
                history[scenario.name + ".s.position"],
                [5.0, position, position, position],
            )
            table = FMEA(
                result,
                sample,
                group_by=(),
                expected_metric="cost",
                rates="scenario_rate",
                round_value=False,
            )
            self.assertAlmostEqual(table["expected_cost"][()], position * probability)
            total_expected_loss += table["expected_cost"][()]
        self.assertAlmostEqual(total_expected_loss, 3.0)
        self.assertEqual(model.m.get_fault("stuck").asdict(), before)
        self.assertEqual(model.s.position, 5.0)


if __name__ == "__main__":
    unittest.main()
