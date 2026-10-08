#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for equality of array-valued results and histories.

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
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class MatrixState(State):
    values: np.array = np.zeros((2, 3))


class MatrixFunction(Function):
    container_s = MatrixState

    def dynamic_behavior(self):
        self.s.values += np.arange(6).reshape(2, 3) + 1

    def classify(self, **kwargs):
        return {"matrix": self.s.values.copy()}


class TestArrayResultEquality(unittest.TestCase):
    def assert_comparison(self, left, right, expected):
        """Check boolean results and complementary inequality in both orders."""
        for first, second in ((left, right), (right, left)):
            self.assertIs(first == second, expected)
            self.assertIs(first != second, not expected)

    def test_arrays_of_every_dimension_have_boolean_equality(self):
        for cls in (Result, History):
            for shape in ((), (1,), (3,), (2, 3), (2, 1, 3), (0,), (2, 0)):
                for dtype in (np.int64, np.float32, np.float64, np.bool_, "U4"):
                    with self.subTest(cls=cls.__name__, shape=shape, dtype=dtype):
                        values = np.full(shape, 1, dtype=dtype)
                        before = values.copy()
                        left = cls({"value": values})
                        right = cls({"value": values.copy()})
                        self.assert_comparison(left, right, True)
                        if values.size:
                            right["value"].flat[-1] = 0
                            self.assert_comparison(left, right, False)
                        np.testing.assert_array_equal(values, before)

    def test_different_shapes_are_not_equal_or_broadcast(self):
        for cls in (Result, History):
            for shapes in (
                ((1,), (3,)),
                ((3,), (2, 3)),
                ((2, 3), (3, 2)),
                ((0,), (2, 0)),
                ((), (1,)),
                ((2,), (3,)),
            ):
                with self.subTest(cls=cls.__name__, shapes=shapes):
                    left, right = (cls(value=np.ones(shape)) for shape in shapes)
                    self.assert_comparison(left, right, False)

    def test_array_and_python_values_compare_symmetrically(self):
        for cls in (Result, History):
            for python_value in (
                2.0,
                [1, 2],
                [[1, 2], [3, 4]],
                ("on", "off"),
                [],
                [[], []],
            ):
                with self.subTest(cls=cls.__name__, value=python_value):
                    array = np.array(python_value)
                    self.assert_comparison(
                        cls(value=array), cls(value=python_value), True
                    )

    def test_noncontiguous_read_only_and_nested_values_are_unchanged(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                values = np.arange(24).reshape(4, 6)[:, ::2]
                values.flags.writeable = False
                left = cls({"outer": cls({"value": values}), "label": "trial"})
                right = cls({"label": "trial", "outer": cls({"value": values.copy()})})
                self.assert_comparison(left, right, True)
                right["outer"]["value"][2, 1] = -1
                self.assert_comparison(left, right, False)
                self.assertFalse(values.flags.writeable)
                np.testing.assert_array_equal(
                    values, np.arange(24).reshape(4, 6)[:, ::2]
                )

    def test_existing_scalar_list_key_and_exact_nonfinite_rules_are_preserved(self):
        for cls in (Result, History):
            for left, right, expected in (
                ({"a": 1, "b": "ready"}, {"b": "ready", "a": 1.0}, True),
                ({"a": [1, 2]}, {"a": [1, 2]}, True),
                ({"a": [1, 2]}, {"a": [1, 3]}, False),
                ({"a": 1}, {"b": 1}, False),
                ({"a": 1}, {"a": 1, "b": 2}, False),
                ({}, {}, True),
                (
                    {"a": np.array([1.0, np.inf, -np.inf])},
                    {"a": np.array([1.0, np.inf, -np.inf])},
                    True,
                ),
                ({"a": np.array([np.nan])}, {"a": np.array([np.nan])}, False),
                (
                    {"a": np.array([1.0])},
                    {"a": np.array([np.nextafter(1.0, 2.0)])},
                    False,
                ),
            ):
                with self.subTest(cls=cls.__name__, left=left, right=right):
                    self.assert_comparison(cls(left), cls(right), expected)

    def test_real_matrix_simulations_and_serialized_outputs_compare(self):
        model = MatrixFunction(sp={"end_time": 3.0})
        first = Simulation(mdl=model)()
        second = Simulation(mdl=model)()
        for original, repeated in zip(first, second):
            with self.subTest(cls=type(original).__name__):
                self.assert_comparison(original, repeated, True)
                self.assert_comparison(original, original.copy(), True)
                before = copy.deepcopy(original)
                with TemporaryDirectory() as directory:
                    for filetype in ("npz", "json"):
                        path = Path(directory) / ("output." + filetype)
                        original.save(str(path))
                        restored = type(original).load(str(path))
                        self.assert_comparison(original, restored, True)
                key = None
                for candidate, value in repeated.items():
                    if np.ndim(value) > 1 and np.size(value):
                        key = candidate
                        break
                self.assertIsNotNone(key)
                repeated[key].flat[-1] += 1
                self.assert_comparison(original, repeated, False)
                self.assert_comparison(original, before, True)
        np.testing.assert_array_equal(model.s.values, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
