#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for state compare.

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

import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import ExampleState, State
from fmdtools.sim.propagate import Simulation


class ArrayState(State):
    first: np.array = np.arange(6).reshape(2, 3)
    second: np.array = np.ones((2, 3))
    matches: bool = True


class ComparisonFunction(Function):
    container_s = ArrayState

    def dynamic_behavior(self):
        self.s.matches = self.s.same(first=np.arange(6).reshape(2, 3))
        self.s.first[0, 0] += 1

    def classify(self, **kwargs):
        return {"matches": self.s.matches}


class TestMultidimensionalStateComparison(unittest.TestCase):
    def test_positional_and_named_comparisons_reduce_all_dimensions(self):
        for shape in ((2, 3), (2, 2, 3), (2, 1, 2), (2, 0)):
            state = ArrayState()
            state.first = np.arange(np.prod(shape)).reshape(shape)
            expected = state.first.copy()
            with self.subTest(shape=shape):
                self.assertIs(state.same(first=expected), True)
                self.assertIs(state.same([expected], "first"), True)
                if expected.size:
                    expected.flat[-1] += 1
                    self.assertIs(state.same(first=expected), False)
                    self.assertIs(state.same([expected], "first"), False)
                np.testing.assert_array_equal(
                    state.first, np.arange(np.prod(shape)).reshape(shape)
                )

    def test_multiple_array_fields_detect_differences_in_each_field(self):
        state = ArrayState()
        first, second = state.first.copy(), state.second.copy()
        self.assertIs(state.same(first=first, second=second), True)
        self.assertIs(state.same([first, second], "first", "second"), True)
        for field in ("first", "second"):
            values = {"first": first.copy(), "second": second.copy()}
            values[field][-1, -1] += 1
            with self.subTest(field=field):
                self.assertIs(state.same(**values), False)

    def test_scalar_vector_boolean_and_nan_behavior_is_preserved(self):
        scalar = ExampleState(x=1.0, y=2.0)
        self.assertIs(scalar.same([1.0, 2.0], "x", "y"), True)
        self.assertIs(scalar.same(x=0.0, y=2.0), False)
        state = ArrayState()
        for values in (np.array([1, 2, 3]), np.array([[True, False], [False, True]])):
            state.first = values.copy()
            with self.subTest(values=values.tolist()):
                self.assertIs(state.same(first=values), True)
        state.first = np.array([[1.0, np.nan], [2.0, 3.0]])
        self.assertIs(state.same(first=state.first.copy()), False)
        with self.assertRaisesRegex(Exception, "Cannot use args and kwargs"):
            state.same([1.0], "first", first=1.0)

    def test_real_simulation_uses_array_equality_as_a_state_condition(self):
        result, history = Simulation(mdl=ComparisonFunction(sp={"end_time": 3.0}))()
        np.testing.assert_array_equal(history["s.matches"], [True, True, False, False])
        np.testing.assert_array_equal(history["s.first"][:, 0, 0], [0, 1, 2, 3])
        self.assertFalse(result["tend.classify.matches"])


if __name__ == "__main__":
    unittest.main()
