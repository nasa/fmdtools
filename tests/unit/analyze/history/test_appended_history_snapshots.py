#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent values in appended histories.

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
from types import SimpleNamespace
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class SnapshotState(State):
    vector: np.array = np.zeros((2, 3))


class SnapshotFunction(Function):
    __slots__ = ("recorded",)
    container_s = SnapshotState

    def init_block(self, **kwargs):
        self.recorded = self.s.create_hist(track="all")

    def dynamic_behavior(self):
        self.s.vector[:] += np.arange(6).reshape(2, 3) + 1
        self.recorded.log(self.s, self.t.t_ind)

    def classify(self, **kwargs):
        return {"snapshots": np.array(self.recorded.vector)}


class TestAppendedHistorySnapshots(unittest.TestCase):
    def test_initial_arrays_and_lists_are_independent_of_the_source(self):
        for value in ([1, 2], [[1], [2]], np.array([1, 2]), np.ones((2, 3))):
            with self.subTest(value_type=type(value).__name__, shape=np.shape(value)):
                obj = SimpleNamespace(value=value)
                before = copy.deepcopy(value)
                history = History()
                history.init_att("value", value, track="all")
                self.assertIsInstance(history.value, list)
                self.assertIsNot(history.value[0], value)
                if isinstance(value, np.ndarray):
                    value[...] = 9
                else:
                    value[0] = 9
                np.testing.assert_array_equal(history.value[0], before)
                self.assertIs(obj.value, value)

    def test_reused_arrays_keep_every_logged_value_and_its_shape(self):
        for shape in ((), (3,), (2, 3), (0,), (2, 0)):
            for dtype in (np.int32, np.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    obj = SimpleNamespace(value=np.zeros(shape, dtype=dtype))
                    history = History()
                    history.init_att("value", obj.value, track="all")
                    for index in range(1, 4):
                        obj.value[...] = index
                        history.log(obj, index)
                    obj.value[...] = 99
                    self.assertEqual(len(history.value), 4)
                    for index, recorded in enumerate(history.value):
                        np.testing.assert_array_equal(
                            recorded, np.full(shape, index, dtype=dtype)
                        )
                        self.assertEqual(recorded.shape, shape)
                        self.assertEqual(recorded.dtype, dtype)
                        self.assertFalse(np.shares_memory(recorded, obj.value))
                    if shape and np.prod(shape):
                        history.value[1][...] = -10
                        np.testing.assert_array_equal(
                            history.value[2], np.full(shape, 2)
                        )

    def test_nested_mutables_are_copied_in_each_appended_value(self):
        for kind in ("list", "tuple", "object_array"):
            with self.subTest(kind=kind):
                shared = {"samples": [0]}
                value = [shared, shared]
                if kind == "tuple":
                    value = tuple(value)
                elif kind == "object_array":
                    value = np.array(value, dtype=object)
                obj = SimpleNamespace(value=value)
                history = History()
                history.init_att("value", value, track="all")
                for index in (1, 2):
                    shared["samples"][0] = index
                    history.log(obj, index)
                shared["samples"][0] = 99
                for index, recorded in enumerate(history.value):
                    self.assertEqual(recorded[0]["samples"], [index])
                    self.assertIs(recorded[0], recorded[1])
                    self.assertIsNot(recorded[0], shared)
                history.value[1][0]["samples"].append(8)
                self.assertEqual(history.value[2][0]["samples"], [2])

    def test_nested_container_history_and_scalar_controls_keep_their_layout(self):
        state = SnapshotState()
        history = state.create_hist()
        for index in (1, 2):
            state.vector[...] = index
            history.log(state, index)
        np.testing.assert_array_equal(
            history.vector, [np.full((2, 3), i) for i in range(3)]
        )
        obj = SimpleNamespace(
            nested={"vector": np.array([1.0, 2.0])}, count=1, label="a"
        )
        history = History()
        for name in ("nested", "count", "label"):
            history.init_att(name, getattr(obj, name), track="all")
        history.init_att("ignored", [5], track=())
        obj.nested["vector"][...] = 3
        obj.count, obj.label = 2, "b"
        history.log(obj, 1)
        obj.nested["vector"][...] = 9
        self.assertIsInstance(history.nested, History)
        np.testing.assert_array_equal(history.nested.vector, [[1.0, 2.0], [3.0, 3.0]])
        self.assertEqual(history.count, [1, 2])
        self.assertEqual(history.label, ["a", "b"])
        self.assertNotIn("ignored", history)

    def test_preallocated_arrays_and_noncontiguous_inputs_keep_their_values(self):
        original = np.arange(24).reshape(4, 6)
        obj = SimpleNamespace(value=original[:, ::2])
        original_copy = original.copy()
        for timerange in (None, [0, 1]):
            with self.subTest(preallocated=timerange is not None):
                history = History()
                history.init_att("value", obj.value, timerange=timerange, track="all")
                history.log(obj, 1)
                for recorded in history.value:
                    np.testing.assert_array_equal(recorded, original_copy[:, ::2])
                np.testing.assert_array_equal(original, original_copy)
        obj.value.flags.writeable = False
        history = History()
        history.init_att("value", obj.value, track="all")
        history.log(obj, 1)
        np.testing.assert_array_equal(history.value, [obj.value, obj.value])
        self.assertFalse(obj.value.flags.writeable)

    def test_real_simulation_retains_initial_and_intermediate_snapshots(self):
        model = SnapshotFunction(sp={"end_time": 3.0})
        result, history = Simulation(mdl=model)()
        expected = np.arange(4)[:, None, None] * (np.arange(6).reshape(2, 3) + 1)
        np.testing.assert_array_equal(result["tend.classify.snapshots"], expected)
        np.testing.assert_array_equal(history["s.vector"], expected)
        self.assertEqual(len(model.recorded.vector), 1)
        np.testing.assert_array_equal(model.s.vector, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
