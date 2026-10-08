#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent object-valued history copies.

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

from fmdtools.analyze.history import History
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class EventState(State):
    events: set = set()


class EventFunction(Function):
    container_s = EventState

    def dynamic_behavior(self):
        self.s.events.add(int(self.t.time))

    def classify(self, **kwargs):
        return {"count": len(self.s.events)}


class TestObjectHistoryCopies(unittest.TestCase):
    def test_object_arrays_copy_nested_mutables_in_every_dimension(self):
        for shape in ((), (1,), (2, 2), (0,), (2, 0)):
            with self.subTest(shape=shape):
                values = np.empty(shape, dtype=object)
                for index in np.ndindex(shape):
                    values[index] = {"seen": [1]}
                source = History({"events": values})
                clone = source.copy()
                self.assertIs(type(clone), History)
                self.assertEqual(clone.events.shape, shape)
                self.assertEqual(clone.events.dtype, values.dtype)
                self.assertFalse(np.shares_memory(clone.events, values))
                for index in np.ndindex(shape):
                    self.assertIsNot(clone.events[index], values[index])
                    clone.events[index]["seen"].append(2)
                    self.assertEqual(values[index]["seen"], [1])
                    values[index]["seen"].append(3)
                    self.assertEqual(clone.events[index]["seen"], [1, 2])

    def test_shared_objects_within_an_entry_remain_shared_only_in_the_copy(self):
        shared = {"items": []}
        values = np.empty(2, dtype=object)
        values[:] = [shared, shared]
        clone = History(events=values).copy()
        self.assertIs(clone.events[0], clone.events[1])
        self.assertIsNot(clone.events[0], shared)
        clone.events[0]["items"].append("copy")
        self.assertEqual(shared, {"items": []})
        self.assertEqual(clone.events[1]["items"], ["copy"])

    def test_structured_object_fields_and_nested_histories_are_independent(self):
        values = np.empty(2, dtype=[("time", "i4"), ("payload", object)])
        values["time"] = [0, 1]
        values["payload"] = [{"seen": [1]}, {"seen": [2]}]
        source = History(plant=History(events=values), time=np.arange(2))
        clone = source.copy()
        self.assertIsNot(clone.plant, source.plant)
        self.assertEqual(clone.plant.events.dtype, values.dtype)
        clone.plant.events["payload"][0]["seen"].append(99)
        clone.plant.events["time"][0] = 5
        self.assertEqual(values["payload"][0], {"seen": [1]})
        np.testing.assert_array_equal(values["time"], [0, 1])

    def test_list_backed_objects_keep_existing_array_conversion(self):
        for values in ([{"x": [1]}, {"x": [2]}], [{1}, {2}]):
            with self.subTest(values=values):
                source = History(events=values)
                clone = source.copy()
                self.assertIs(type(clone.events), np.ndarray)
                self.assertEqual(clone.events.dtype, object)
                self.assertIsNot(clone.events[0], values[0])
                if isinstance(values[0], set):
                    clone.events[0].add(99)
                    self.assertEqual(values[0], {1})
                else:
                    clone.events[0]["x"].append(99)
                    self.assertEqual(values[0], {"x": [1]})

    def test_numeric_strings_lists_and_readonly_views_keep_copy_behavior(self):
        for values in (
            np.array(2.0),
            np.arange(12).reshape(3, 4)[:, ::2],
            np.array(["a", "b"]),
            [1.0, 2.0],
            np.empty((2, 0)),
        ):
            with self.subTest(shape=np.shape(values), dtype=np.asarray(values).dtype):
                if isinstance(values, np.ndarray):
                    values.flags.writeable = False
                expected = np.copy(values)
                clone = History(data_values=values).copy()
                np.testing.assert_array_equal(clone.data_values, expected)
                self.assertEqual(clone.data_values.dtype, expected.dtype)
                self.assertTrue(clone.data_values.flags.writeable)
                self.assertFalse(np.shares_memory(clone.data_values, values))
                if clone.data_values.size and np.issubdtype(expected.dtype, np.number):
                    clone.data_values.flat[0] = 99
                    np.testing.assert_array_equal(values, expected)

    def test_real_simulation_copy_and_cut_do_not_modify_recorded_events(self):
        _, history = Simulation(mdl=EventFunction(sp={"end_time": 3.0}))()
        expected = [set(), {1}, {1, 2}, {1, 2, 3}]
        self.assertEqual(history["s.events"].tolist(), expected)
        clone = history.copy()
        clone["s.events"][1].add(99)
        self.assertEqual(history["s.events"].tolist(), expected)
        cut = history.cut(end_ind=2, start_ind=1, newcopy=True)
        cut["s.events"][0].add(100)
        self.assertEqual(history["s.events"].tolist(), expected)
        np.testing.assert_array_equal(cut.time, [1.0, 2.0])
        history["s.events"][3].add(101)
        self.assertEqual(clone["s.events"][3], {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
