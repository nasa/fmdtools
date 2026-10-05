#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for bounded array-state updates.

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
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import ExampleState, State
from fmdtools.sim.propagate import Simulation


class ArrayState(State):
    value: np.array = np.array([[0.0, 4.0], [2.0, 1.0]])
    other: np.array = np.array([0.0, 0.0])


class BoundedFunction(Function):
    container_s = ArrayState

    def dynamic_behavior(self):
        self.s.inc(
            value=(
                np.array([[2.0, -2.0], [0.0, 1.0]]),
                np.array([[3.0, -1.0], [-99.0, 4.0]]),
            )
        )

    def classify(self, **kwargs):
        return {"total": self.s.value.sum()}


def scalar_reference(current, step, bound):
    values, steps, bounds = np.broadcast_arrays(current, step, bound)
    expected = np.empty(values.shape, dtype=np.result_type(values, steps, bounds))
    for index in np.ndindex(values.shape):
        candidate = values[index] + steps[index]
        sign = np.sign(steps[index])
        expected[index] = (
            candidate if sign * candidate <= sign * bounds[index] else bounds[index]
        )
    return expected


class TestBoundedArrayIncrements(unittest.TestCase):
    def test_every_element_uses_its_own_increment_direction_and_bound(self):
        for shape in ((4,), (2, 2), (2, 1, 2)):
            for dtype in (np.int64, np.float32, np.float64):
                for step, bound in (
                    (2, 3),
                    (-2, -1),
                    (0, -99),
                    (
                        np.array([2, -2, 0, 1]).reshape(shape),
                        np.array([3, -1, -99, 4]).reshape(shape),
                    ),
                ):
                    with self.subTest(shape=shape, dtype=dtype, step=np.shape(step)):
                        state = ArrayState()
                        state.value = np.array([0, 4, 2, 1], dtype=dtype).reshape(shape)
                        original = state.value.copy()
                        expected = scalar_reference(original, step, bound)
                        state.inc(value=(step, bound))
                        np.testing.assert_array_equal(state.value, expected)
                        self.assertEqual(state.value.shape, shape)

    def test_broadcast_bounds_readonly_views_and_multiple_fields(self):
        current = np.arange(12.0).reshape(3, 4)[:, ::2]
        steps = np.array([[2.0], [-2.0], [0.0]])
        bounds = np.array([3.0, 4.0])
        current.setflags(write=False)
        steps.setflags(write=False)
        bounds.setflags(write=False)
        before = current.copy(), steps.copy(), bounds.copy()
        state = ArrayState(value=current)
        state.inc(value=(steps, bounds), other=(1.0, 0.5))
        np.testing.assert_array_equal(state.value, scalar_reference(*before))
        np.testing.assert_array_equal(state.other, [0.5, 0.5])
        for actual, expected in zip((current, steps, bounds), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.shares_memory(state.value, current))

    def test_empty_zero_dimensional_and_unbounded_updates_remain_supported(self):
        for shape in ((), (0,), (2, 0), (1,)):
            state = ArrayState()
            state.value = np.full(shape, 4.0)
            state.inc(value=(2.0, 5.0))
            np.testing.assert_array_equal(state.value, np.full(shape, 5.0))
            self.assertEqual(np.shape(state.value), shape)
            state.inc(value=-1.0)
            np.testing.assert_array_equal(state.value, np.full(shape, 4.0))
        scalar = ExampleState(x=1.0, y=4.0)
        scalar.inc(x=(3.0, 2.0), y=(-3.0, 2.0))
        self.assertEqual((scalar.x, scalar.y), (2.0, 2.0))
        scalar.inc(x=(0.0, -99.0))
        self.assertEqual(scalar.x, 2.0)
        self.assertIs(type(scalar.x), np.float64)
        with self.assertRaisesRegex(Exception, "not a property"):
            scalar.inc(absent=(1.0, 2.0))

    def test_invalid_broadcast_does_not_replace_the_existing_state(self):
        state = ArrayState()
        before = state.value.copy()
        with self.assertRaises(ValueError):
            state.inc(value=(np.ones(3), 4.0))
        np.testing.assert_array_equal(state.value, before)

    def test_simulation_stops_each_component_at_its_limit(self):
        model = BoundedFunction(sp={"end_time": 4.0})
        before = copy.deepcopy(model.s.value)
        result, history = Simulation(mdl=model)()
        expected = [before]
        step = np.array([[2.0, -2.0], [0.0, 1.0]])
        limit = np.array([[3.0, -1.0], [-99.0, 4.0]])
        for _ in range(4):
            expected.append(scalar_reference(expected[-1], step, limit))
        np.testing.assert_array_equal(history["s.value"], expected)
        np.testing.assert_array_equal(history.time, np.arange(5))
        self.assertEqual(result["tend.classify.total"], expected[-1].sum())
        np.testing.assert_array_equal(model.s.value, before)


if __name__ == "__main__":
    unittest.main()
