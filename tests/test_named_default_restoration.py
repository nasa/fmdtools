#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for model calculation correctness.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class DistinctDefaults(State):
    x: np.float64 = 1.0
    y: np.float64 = 2.0
    z: np.float64 = 3.0


class ChildDefaults(State):
    payload: dict = {"values": [7, 8]}


class MixedDefaults(State):
    number: np.float64 = 1.0
    label: str = "ready"
    values: np.array = np.array([3.0, 4.0])
    child: ChildDefaults = ChildDefaults()


class RecoveryFunction(Function):
    container_s = DistinctDefaults

    def dynamic_behavior(self):
        if self.t.time == 1.0:
            self.s.y = 20.0
        elif self.t.time == 2.0:
            self.s.to_default("y")

    def classify(self, **kwargs):
        return {"y": self.s.y}


class TestNamedDefaultRestoration(unittest.TestCase):
    def test_subsets_and_reordered_fields_use_their_own_defaults(self):
        defaults = {"x": 1.0, "y": 2.0, "z": 3.0}
        for count in (1, 2, 3):
            for fields in itertools.permutations(defaults, count):
                with self.subTest(fields=fields):
                    state = DistinctDefaults(91.0, 92.0, 93.0)
                    expected = dict(zip(defaults, (91.0, 92.0, 93.0)))
                    expected.update({field: defaults[field] for field in fields})
                    state.to_default(*fields)
                    self.assertEqual(state.asdict(), expected)

    def test_heterogeneous_defaults_are_selected_without_type_coercion(self):
        for fields in (
            ("label",),
            ("values",),
            ("child",),
            ("child", "label"),
            ("values", "number"),
        ):
            with self.subTest(fields=fields):
                state = MixedDefaults(
                    number=9.0,
                    label="changed",
                    values=np.array([90.0, 80.0]),
                    child={"payload": {"values": [99]}},
                )
                before_child = state.child
                state.to_default(*fields)
                self.assertEqual(state.number, 1.0 if "number" in fields else 9.0)
                self.assertEqual(
                    state.label, "ready" if "label" in fields else "changed"
                )
                np.testing.assert_array_equal(
                    state.values, [3.0, 4.0] if "values" in fields else [90.0, 80.0]
                )
                self.assertEqual(
                    state.child.payload,
                    {"values": [7, 8]} if "child" in fields else {"values": [99]},
                )
                self.assertIs(state.child, before_child)

    def test_restored_mutable_defaults_remain_independent(self):
        snapshot = copy.deepcopy(MixedDefaults.__defaults__)
        first, second = MixedDefaults(), MixedDefaults()
        first.to_default("child", "values")
        second.to_default("values", "child")
        first.values[0] = 100.0
        first.child.payload["values"].append(100)
        np.testing.assert_array_equal(second.values, snapshot["values"])
        self.assertEqual(second.child.payload, snapshot["child"].payload)
        np.testing.assert_array_equal(
            MixedDefaults.__defaults__["values"], snapshot["values"]
        )
        self.assertEqual(
            MixedDefaults.__defaults__["child"].payload, snapshot["child"].payload
        )
        first.to_default("child", "values")
        np.testing.assert_array_equal(first.values, snapshot["values"])
        self.assertEqual(first.child.payload, snapshot["child"].payload)

    def test_complete_default_restoration_and_duplicate_fields_remain_supported(self):
        for fields in ((), ("all",), ("z", "z", "x")):
            with self.subTest(fields=fields):
                state = DistinctDefaults(91.0, 92.0, 93.0)
                state.to_default(*fields)
                expected = {
                    "x": 1.0,
                    "y": 92.0 if fields == ("z", "z", "x") else 2.0,
                    "z": 3.0,
                }
                self.assertEqual(state.asdict(), expected)

    def test_recovery_simulation_restores_the_selected_state(self):
        result, history = Simulation(mdl=RecoveryFunction(sp={"end_time": 3.0}))()
        np.testing.assert_array_equal(history["s.y"], [2.0, 20.0, 2.0, 2.0])
        np.testing.assert_array_equal(history["s.x"], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(history["s.z"], [3.0, 3.0, 3.0, 3.0])
        self.assertEqual(result["tend.classify.y"], 2.0)


if __name__ == "__main__":
    unittest.main()
