#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for rounding array-valued states to a resolution.

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


class RoundedState(State):
    value: np.array = np.zeros((2, 2))
    other: np.array = np.array([1.25, -1.25])


class RoundedFunction(Function):
    container_s = RoundedState

    def dynamic_behavior(self):
        self.s.inc(value=np.array([[0.16, -0.16], [0.31, -0.31]]))
        self.s.roundto(value=np.array([[0.1], [0.25]]))

    def classify(self, **kwargs):
        return {"energy": np.square(self.s.value).sum()}


def scalar_reference(current, resolution):
    values, steps = np.broadcast_arrays(current, resolution)
    out = np.empty(values.shape)
    for index in np.ndindex(values.shape):
        out[index] = np.round(round(values[index] / steps[index]) * steps[index], 7)
    return out


class TestArrayStateRounding(unittest.TestCase):
    def test_vectors_matrices_and_higher_dimensions_match_scalar_rounding(self):
        for shape in ((6,), (2, 3), (1, 2, 3)):
            for dtype in (np.float32, np.float64, np.int64):
                for step in (0.1, 0.5, 2.0):
                    values = np.array(
                        [-2.5, -1.25, -0.5, 0.5, 1.25, 2.5], dtype=dtype
                    ).reshape(shape)
                    with self.subTest(shape=shape, dtype=dtype, step=step):
                        state = RoundedState(value=values)
                        before = values.copy()
                        state.roundto(value=step)
                        np.testing.assert_allclose(
                            state.value,
                            scalar_reference(before, step),
                            rtol=1e-7,
                            atol=1e-7,
                        )
                        self.assertEqual(state.value.shape, shape)
                        np.testing.assert_array_equal(values, before)

    def test_broadcast_resolutions_readonly_views_and_multiple_fields(self):
        values = np.array([[0.16, 0.26, 0.36, 0.46], [-0.16, -0.26, -0.36, -0.46]])[
            :, ::2
        ]
        steps = np.array([[0.1], [0.25]])
        values.setflags(write=False)
        steps.setflags(write=False)
        saved = values.copy(), steps.copy()
        state = RoundedState(value=values)
        state.roundto(value=steps, other=0.5)
        np.testing.assert_array_equal(state.value, scalar_reference(*saved))
        np.testing.assert_array_equal(state.other, [1.0, -1.0])
        np.testing.assert_array_equal(values, saved[0])
        np.testing.assert_array_equal(steps, saved[1])
        self.assertFalse(np.shares_memory(state.value, values))

    def test_empty_and_zero_dimensional_arrays_retain_their_shapes(self):
        for shape in ((0,), (2, 0), (), (1,)):
            with self.subTest(shape=shape):
                state = RoundedState(value=np.full(shape, 1.25))
                state.roundto(value=0.5)
                self.assertEqual(np.shape(state.value), shape)
                np.testing.assert_array_equal(state.value, np.full(shape, 1.0))

    def test_scalar_ties_and_invalid_broadcast_preserve_existing_behavior(self):
        for value in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1.7585):
            for step in (0.1, 0.5, 1.0):
                with self.subTest(value=value, step=step):
                    state = ExampleState(x=value)
                    state.roundto(x=step)
                    self.assertEqual(
                        state.x, np.round(round(np.float64(value) / step) * step, 7)
                    )
                    self.assertIs(type(state.x), np.float64)
        state = RoundedState()
        before = state.value.copy()
        with self.assertRaises(ValueError):
            state.roundto(value=np.ones(3))
        np.testing.assert_array_equal(state.value, before)

    def test_simulation_quantizes_each_component_and_retains_history(self):
        model = RoundedFunction(sp={"end_time": 4.0})
        result, history = Simulation(mdl=model)()
        expected = [np.zeros((2, 2))]
        increments = np.array([[0.16, -0.16], [0.31, -0.31]])
        steps = np.array([[0.1], [0.25]])
        for _ in range(4):
            expected.append(scalar_reference(expected[-1] + increments, steps))
        np.testing.assert_array_equal(history["s.value"], expected)
        np.testing.assert_array_equal(history.time, np.arange(5))
        self.assertAlmostEqual(
            result["tend.classify.energy"], np.square(expected[-1]).sum()
        )
        np.testing.assert_array_equal(model.s.value, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
