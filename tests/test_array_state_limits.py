#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for array-valued state limits.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import ExampleState, State
from fmdtools.sim.propagate import Simulation


class BoundedState(State):
    values: np.array = np.array([[0.0, 1.0], [2.0, 3.0]])
    other: np.array = np.array([0.0, 1.0])


class BoundedFunction(Function):
    container_s = BoundedState

    def dynamic_behavior(self):
        self.s.values += np.array([[-2.0, 1.0], [2.0, -1.0]])
        self.s.limit(values=(0.0, 4.0))

    def classify(self, **kwargs):
        return {"total": self.s.values.sum()}


class TestArrayStateLimits(unittest.TestCase):
    def test_limits_apply_to_every_element_without_collapsing_dimensions(self):
        for dtype in (np.int16, np.int64, np.float32, np.float64):
            for shape in ((4,), (2, 2), (2, 1, 2), (1, 4, 1)):
                with self.subTest(dtype=dtype, shape=shape):
                    value = np.array([-2, 0, 2, 5], dtype=dtype).reshape(shape)
                    state = BoundedState(values=value)
                    result = state.limit(values=(0, 3))
                    self.assertIsNone(result)
                    np.testing.assert_array_equal(
                        state.values, np.array([0, 0, 2, 3]).reshape(shape)
                    )
                    self.assertEqual(state.values.shape, shape)
                    np.testing.assert_array_equal(
                        value, np.array([-2, 0, 2, 5]).reshape(shape)
                    )

    def test_bounds_broadcast_over_readonly_noncontiguous_state_arrays(self):
        source = np.arange(-6.0, 6.0).reshape(3, 4)[:, ::2]
        source.setflags(write=False)
        lower = np.array([[-4.0], [0.0], [4.0]])
        upper = np.array([1.0, 5.0])
        original = source.copy(), lower.copy(), upper.copy()
        expected = np.array(
            [
                [min(upper[j], max(lower[i, 0], source[i, j])) for j in range(2)]
                for i in range(3)
            ]
        )
        state = BoundedState()
        state.values = source
        state.limit(values=(lower, upper))
        np.testing.assert_array_equal(state.values, expected)
        self.assertFalse(np.shares_memory(source, state.values))
        for actual, before in zip((source, lower, upper), original):
            np.testing.assert_array_equal(actual, before)

    def test_multiple_empty_and_singleton_fields_are_limited_independently(self):
        for shape in ((0,), (2, 0), (0, 2, 3), (1,)):
            with self.subTest(shape=shape):
                state = BoundedState(
                    values=np.zeros(shape), other=np.array([-4.0, 8.0])
                )
                state.limit(values=(1.0, 2.0), other=(-1.0, 3.0))
                np.testing.assert_array_equal(state.values, np.ones(shape))
                np.testing.assert_array_equal(state.other, [-1.0, 3.0])
                self.assertEqual(state.values.shape, shape)

    def test_scalar_nonfinite_and_reversed_interval_behavior_is_preserved(self):
        for value in (-np.inf, -2.0, 0.0, 5.0, np.inf, np.nan):
            for bounds in ((0.0, 3.0), (3.0, 0.0)):
                with self.subTest(value=value, bounds=bounds):
                    state = ExampleState(x=value)
                    expected = np.min([bounds[1], np.max([bounds[0], value])])
                    state.limit(x=bounds)
                    np.testing.assert_equal(state.x, expected)
        state = BoundedState(values=np.array([[np.nan, np.inf], [-np.inf, 2.0]]))
        state.limit(values=(0.0, 3.0))
        np.testing.assert_equal(state.values, [[np.nan, 3.0], [0.0, 2.0]])
        with self.assertRaisesRegex(Exception, "not a property"):
            state.limit(missing=(0.0, 1.0))
        with self.assertRaisesRegex(Exception, "Invalid state values"):
            state.limit(values=(np.zeros(3), np.ones(3)))

    def test_bounded_simulation_retains_the_expected_state_history(self):
        model = BoundedFunction(sp={"end_time": 4.0})
        initial = model.s.values.copy()
        result, history = Simulation(mdl=model)()
        expected = [initial]
        change = np.array([[-2.0, 1.0], [2.0, -1.0]])
        for _ in range(4):
            advanced = expected[-1] + change
            expected.append(
                np.array([[min(4.0, max(0.0, x)) for x in row] for row in advanced])
            )
        np.testing.assert_array_equal(history["s.values"], expected)
        self.assertEqual(result["tend.classify.total"], expected[-1].sum())
        np.testing.assert_array_equal(model.s.values, initial)


if __name__ == "__main__":
    unittest.main()
