#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for filtering array-valued result and history differences.

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

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class MatrixState(State):
    matrix: np.array = np.zeros((2, 2))


class MatrixFunction(Function):
    container_s = MatrixState

    def dynamic_behavior(self):
        self.s.matrix += np.array([[1.0, 2.0], [3.0, 4.0]])

    def classify(self, **kwargs):
        return {"matrix": self.s.matrix.copy(), "total": np.sum(self.s.matrix)}


class TestArrayDifferences(unittest.TestCase):
    def assert_unchanged(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_numeric_differences_are_filtered_across_all_dimensions(self):
        for cls in (Result, History):
            for shape in ((), (3,), (2, 3), (2, 1, 2)):
                for dtype in (np.int64, np.float32, np.float64):
                    with self.subTest(cls=cls.__name__, shape=shape, dtype=dtype):
                        first = np.zeros(shape, dtype=dtype)
                        second = first.copy()
                        second.flat[-1] = 2
                        left = cls({"same": first.copy(), "changed": first})
                        right = cls({"same": first.copy(), "changed": second})
                        before = copy.deepcopy(left), copy.deepcopy(right)
                        actual = left.get_different(right)
                        self.assertIs(type(actual), cls)
                        self.assertEqual(list(actual), ["changed"])
                        np.testing.assert_array_equal(actual["changed"], first - second)
                        self.assertEqual(np.shape(actual["changed"]), shape)
                        self.assertEqual(actual["changed"].dtype, np.dtype(dtype))
                        self.assert_unchanged(left, before[0])
                        self.assert_unchanged(right, before[1])

    def test_boolean_and_string_differences_retain_the_existing_representation(self):
        for cls in (Result, History):
            for dtype in (bool, str):
                with self.subTest(cls=cls.__name__, dtype=dtype):
                    if dtype is bool:
                        left = np.array([[True, False], [True, False]])
                        right = np.array([[False, True], [True, False]])
                        expected = np.array([[1, -1], [0, 0]], dtype=np.int32)
                    else:
                        left = np.array([["run", "idle"], ["run", "idle"]])
                        right = np.array([["idle", "run"], ["run", "idle"]])
                        expected = np.array([[True, True], [False, False]])
                    first = cls({"same": left.copy(), "changed": left})
                    second = cls({"same": left.copy(), "changed": right})
                    difference = first.get_different(second)
                    self.assertEqual(list(difference), ["changed"])
                    np.testing.assert_array_equal(difference["changed"], expected)
                    self.assertEqual(difference["changed"].dtype, expected.dtype)

    def test_empty_and_identical_arrays_do_not_appear_as_differences(self):
        for cls in (Result, History):
            for shape in ((0,), (2, 0, 3), (1,), (2, 3)):
                with self.subTest(cls=cls.__name__, shape=shape):
                    value = cls({"metric": np.zeros(shape)})
                    self.assertEqual(dict(value.get_different(value.copy())), {})
            self.assertEqual(dict(cls().get_different(cls())), {})

    def test_scalar_metadata_and_nonfinite_difference_semantics_are_preserved(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                first = cls({"number": 3.0, "flag": True, "label": "old", "same": 7})
                second = cls({"number": 1.0, "flag": False, "label": "new", "same": 7})
                difference = first.get_different(second)
                self.assertEqual(
                    dict(difference), {"number": 2.0, "flag": 1, "label": True}
                )
                values = cls(
                    {
                        "missing": np.array([0.0, np.nan]),
                        "infinite": np.array([0.0, np.inf]),
                    }
                )
                other = cls({key: np.zeros(2) for key in values})
                filtered = values.get_different(other)
                self.assertEqual(set(filtered), set(values))
                self.assertTrue(np.isnan(filtered["missing"][-1]))
                self.assertTrue(np.isposinf(filtered["infinite"][-1]))

    def test_noncontiguous_readonly_inputs_and_returned_arrays_are_independent(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                source = np.arange(24.0).reshape(4, 6)[:, ::2]
                source.flags.writeable = False
                left = cls({"metric": source})
                right = cls({"metric": np.zeros_like(source)})
                before = source.copy()
                result = left.get_different(right)
                result["metric"][0, 0] = 99.0
                np.testing.assert_array_equal(source, before)
                np.testing.assert_array_equal(right["metric"], 0.0)

    def test_real_matrix_state_simulations_can_be_compared(self):
        first = MatrixFunction(sp={"end_time": 3.0})
        second = MatrixFunction(sp={"end_time": 3.0}, s={"matrix": np.ones((2, 2))})
        first_result, first_history = Simulation(mdl=first)()
        second_result, second_history = Simulation(mdl=second)()
        result = first_result.get_different(second_result)
        self.assertEqual(set(result), {"tend.classify.matrix", "tend.classify.total"})
        np.testing.assert_array_equal(result["tend.classify.matrix"], -np.ones((2, 2)))
        self.assertEqual(result["tend.classify.total"], -4.0)
        history = first_history.get_different(second_history)
        self.assertEqual(set(history), {"s.matrix"})
        np.testing.assert_array_equal(
            history["s.matrix"], -np.ones((len(first_history.time), 2, 2))
        )
        self.assertEqual(dict(first_history.get_different(first_history.copy())), {})
        np.testing.assert_array_equal(first.s.matrix, np.zeros((2, 2)))
        np.testing.assert_array_equal(second.s.matrix, np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
