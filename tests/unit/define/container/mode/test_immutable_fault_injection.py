#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for injection updates of read-only Fault definitions.

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

from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class FrozenMode(Mode):
    fault_stuck: Fault = Fault(prob=0.2, cost=10.0, disturbances=(("s.position", 2.0),))
    fault_off: Fault = Fault(prob=0.1, cost=30.0)


class ExclusiveFrozenMode(FrozenMode):
    exclusive = True
    mode: str = "nominal"


class Position(State):
    position: np.float64 = 5.0


class FrozenFunction(Function):
    container_m = FrozenMode
    container_s = Position

    def dynamic_behavior(self):
        if not self.m.has_fault("stuck"):
            self.s.position += 1.0

    def classify(self, **kwargs):
        return {"cost": self.s.position}


class MutableMode(Mode):
    fault_stuck: dict = {"prob": 0.2, "cost": 10.0, "disturbances": {"s.position": 2.0}}


class TestImmutableFaultInjection(unittest.TestCase):
    def test_partial_dictionary_updates_preserve_unmentioned_fields(self):
        for updates in (
            {"cost": 80.0},
            {"prob": 0.0},
            {"disturbances": {"s.position": 0.0}},
            {"phases": {"run": 0.5}, "units": "hr"},
            {},
        ):
            for mode_type in (FrozenMode, ExclusiveFrozenMode):
                with self.subTest(updates=updates, mode_type=mode_type.__name__):
                    mode = mode_type()
                    original = mode.get_fault("stuck")
                    before = original.asdict()
                    params = copy.deepcopy(updates)
                    mode.add_fault({"stuck": updates})
                    expected = Fault(**{**before, **updates})
                    self.assertEqual(
                        mode.get_fault("stuck").asdict(), expected.asdict()
                    )
                    self.assertIsNot(mode.get_fault("stuck"), original)
                    self.assertEqual(original.asdict(), before)
                    self.assertEqual(updates, params)
                    self.assertEqual(mode.faults, {"stuck"})
                    if mode.exclusive:
                        self.assertEqual(mode.mode, "stuck")
                    with self.assertRaises(AttributeError):
                        mode.get_fault("stuck").cost = 99.0

    def test_complete_fault_and_tuple_inputs_replace_values_without_aliasing(self):
        supplied = Fault(prob=0.7, cost=3.0, disturbances=(("s.position", 10.0),))
        for definition in (supplied, tuple(supplied)):
            with self.subTest(definition=definition):
                mode = FrozenMode()
                before = mode.get_fault("stuck").asdict()
                mode.add_fault({"stuck": definition})
                self.assertEqual(mode.get_fault("stuck").asdict(), supplied.asdict())
                self.assertIsNot(mode.get_fault("stuck"), supplied)
                self.assertEqual(FrozenMode().get_fault("stuck").asdict(), before)
                clone = mode.copy()
                clone.add_fault({"stuck": {"cost": 88.0}})
                self.assertEqual(mode.get_fault("stuck").cost, 3.0)
                self.assertEqual(clone.get_fault("stuck").cost, 88.0)

    def test_multiple_updates_and_plain_name_controls(self):
        mode = FrozenMode()
        baseline = mode.get_faults()
        mode.add_fault({"stuck": {"cost": 40.0}, "off": {"prob": 0.9}})
        self.assertEqual(mode.faults, {"stuck", "off"})
        self.assertEqual(mode.get_fault("stuck").cost, 40.0)
        self.assertEqual(mode.get_fault("off").prob, 0.9)
        mode.add_fault({"stuck": {"disturbances": {"s.position": 9.0}}})
        self.assertEqual(mode.get_fault("stuck").cost, 40.0)
        self.assertEqual(
            dict(mode.get_fault("stuck").disturbances), {"s.position": 9.0}
        )
        plain = FrozenMode()
        original = plain.get_fault("stuck")
        plain.add_fault("stuck")
        self.assertIs(plain.get_fault("stuck"), original)
        with self.assertRaisesRegex(Exception, "not defined"):
            plain.add_fault({"missing": {}})
        mutable = MutableMode()
        mutable.add_fault({"stuck": {"cost": 12.0}})
        self.assertEqual(mutable.fault_stuck, {"cost": 12.0})
        self.assertEqual(FrozenMode().get_fault("stuck"), baseline["stuck"])

    def test_sequence_injects_two_immutable_definition_updates(self):
        model = FrozenFunction(sp={"end_time": 3.0})
        first = {"disturbances": {"s.position": 0.0}, "cost": 20.0}
        second = Fault(prob=0.4, cost=30.0, disturbances=(("s.position", 10.0),))
        faults = {
            1.0: {model.name: {"stuck": first}},
            2.0: {model.name: {"stuck": second}},
        }
        before = copy.deepcopy(faults)
        result, history = propagate.sequence(model, faultseq=faults, showprogress=False)
        np.testing.assert_array_equal(
            history["sequence.s.position"], [5.0, 0.0, 10.0, 10.0]
        )
        self.assertEqual(result["sequence.tend.classify.cost"], 10.0)
        self.assertEqual(faults, before)
        self.assertEqual(model.s.position, 5.0)
        self.assertEqual(
            dict(model.m.get_fault("stuck").disturbances), {"s.position": 2.0}
        )

    def test_parametric_fault_sample_uses_each_immutable_variant(self):
        model = FrozenFunction(sp={"end_time": 3.0})
        domain = FaultDomain(model)
        choices = [("zero", 0.0, 0.2), ("high", 10.0, 0.2)]
        for label, position, prob in choices:
            domain.add_fault(model.name, "stuck", label)
            domain.faults[(model.name, "stuck", label)] = Fault(
                prob=prob, cost=10.0, disturbances=(("s.position", position),)
            )
        definitions = {k: v.asdict() for k, v in domain.faults.items()}
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0])
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        for scenario, (_, position, prob) in zip(sample.scenarios(), choices):
            self.assertAlmostEqual(scenario.rate, prob)
            np.testing.assert_array_equal(
                history[scenario.name + ".s.position"],
                [5.0, position, position, position],
            )
            self.assertEqual(result[scenario.name + ".tend.classify.cost"], position)
        table = FMEA(
            result,
            sample,
            group_by=(),
            expected_metric="cost",
            rates="scenario_rate",
            round_value=False,
        )
        self.assertAlmostEqual(table["expected_cost"][()], 2.0)
        self.assertEqual({k: v.asdict() for k, v in domain.faults.items()}, definitions)
        self.assertFalse(model.m.any_faults())
        self.assertEqual(model.m.get_fault("stuck").prob, 0.2)


if __name__ == "__main__":
    unittest.main()
