#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for atomic changes to fault states and rate definitions.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate


class ExclusiveRiskMode(Mode):
    exclusive = True
    opermodes = ("nominal", "active")
    mode: str = "nominal"

    fault_sensor: Fault = Fault(prob=0.05, cost=10.0)
    fault_power: Fault = Fault(prob=0.10, cost=100.0)


class ConcurrentRiskMode(ExclusiveRiskMode):
    exclusive = False


class RiskState(State):
    loss: float = 0.0


class RiskFunction(Function):
    container_m = ExclusiveRiskMode
    container_s = RiskState

    def dynamic_behavior(self):
        self.s.loss = sum(self.m.get_fault(name).cost for name in self.m.faults)

    def classify(self, **kwargs):
        return {"loss": self.s.loss}


class TestFaultTransitionAtomicity(unittest.TestCase):
    """Invalid changes must not create phantom faults or alter future risk."""

    def assert_unchanged(self, mode, before):
        self.assertEqual(mode.mode, before[0])
        self.assertEqual(mode.faults, before[1])
        self.assertEqual(mode.get_fault("sensor").asdict(), before[2])
        self.assertEqual(mode.get_fault("power").asdict(), before[3])

    @staticmethod
    def snapshot(mode):
        return (
            mode.mode,
            mode.faults.copy(),
            mode.get_fault("sensor").asdict(),
            mode.get_fault("power").asdict(),
        )

    def test_exclusive_rejects_multiple_faults_before_mutating_definitions(self):
        variants = (
            ("sensor", "power"),
            (["sensor", "power"],),
            ({"sensor": {"prob": 0.90}, "power": {"prob": 0.70}},),
        )
        for args in variants:
            for already_active in (False, True):
                with self.subTest(args=args, already_active=already_active):
                    mode = ExclusiveRiskMode()
                    if already_active:
                        mode.add_fault("sensor")
                    original = self.snapshot(mode)
                    with self.assertRaisesRegex(Exception, "Multiple fault modes"):
                        mode.add_fault(*args)
                    self.assert_unchanged(mode, original)

    def test_exclusive_does_not_accumulate_one_fault_after_another(self):
        mode = ExclusiveRiskMode()
        mode.add_fault("sensor")
        original = self.snapshot(mode)

        with self.assertRaisesRegex(Exception, "exclusive"):
            mode.add_fault("power")
        self.assert_unchanged(mode, original)

        with self.assertRaisesRegex(Exception, "exclusive"):
            mode.add_fault({"power": {"prob": 0.8, "cost": 20.0}})
        self.assert_unchanged(mode, original)

        # Existing transitions still work: the same fault can be updated, and
        # to_fault explicitly replaces the active fault.
        mode.add_fault({"sensor": {"prob": 0.25}})
        self.assertEqual(mode.faults, {"sensor"})
        self.assertEqual(mode.mode, "sensor")
        self.assertAlmostEqual(mode.get_fault("sensor").prob, 0.25)
        mode.to_fault("power")
        self.assertEqual(mode.faults, {"power"})
        self.assertEqual(mode.mode, "power")

    def test_bad_rate_override_does_not_leave_ghost_faults(self):
        # Constructing Fault(prob="invalid") raises before insertion. A valid
        # first update must not be applied if a later update is invalid.
        mode = ConcurrentRiskMode()
        initial = self.snapshot(mode)
        with self.assertRaises((TypeError, ValueError)):
            mode.add_fault(
                {
                    "sensor": {"prob": 0.9},
                    "power": {"prob": "invalid"},
                }
            )
        self.assert_unchanged(mode, initial)

    def test_failed_removal_and_replacement_preserve_exclusive_state(self):
        mode = ExclusiveRiskMode()
        mode.add_fault("sensor")
        original = self.snapshot(mode)
        with self.assertRaisesRegex(Exception, "Unclear which operational mode"):
            mode.remove_fault("sensor")
        self.assert_unchanged(mode, original)
        with self.assertRaises(KeyError):
            mode.replace_fault("not_active", "power")
        self.assert_unchanged(mode, original)

        mode.replace_fault("sensor", "sensor")
        self.assert_unchanged(mode, original)
        mode.replace_fault("sensor", "power")
        self.assertEqual(mode.faults, {"power"})
        self.assertEqual(mode.mode, "power")
        mode.remove_fault("power", opermode="nominal")
        self.assertFalse(mode.any_faults())
        self.assertEqual(mode.mode, "nominal")

    def test_invalid_injection_does_not_change_nominal_simulation_risk(self):
        model = RiskFunction(sp={"end_time": 2.0})
        expected_sensor_rate = model.get_scen_rate(model.name, "sensor", 1.0)
        self.assertAlmostEqual(expected_sensor_rate, 0.05)
        with self.assertRaisesRegex(Exception, "Multiple fault modes"):
            model.m.add_fault({"sensor": {"prob": 0.9}, "power": {"prob": 0.8}})
        self.assertFalse(model.m.any_faults())
        self.assertEqual(model.m.mode, "nominal")
        self.assertEqual(model.get_scen_rate(model.name, "sensor", 1.0), 0.05)
        result, history = propagate.nominal(model, showprogress=False)
        self.assertEqual(result["tend.classify.loss"], 0.0)
        self.assertFalse(history["m.faults.sensor"].any())
        self.assertFalse(history["m.faults.power"].any())


if __name__ == "__main__":
    unittest.main()
