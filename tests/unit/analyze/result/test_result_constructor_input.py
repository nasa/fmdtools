#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for non-mutating result construction.

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
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation


class DerivedResult(Result):
    """Exercise the existing Result subclass constructor path."""


class TestResultConstructorInput(unittest.TestCase):
    def test_keyword_overrides_only_change_the_new_mapping(self):
        for source_type in (Result, History, DerivedResult):
            for target_type in (Result, History, DerivedResult):
                with self.subTest(
                    source=source_type.__name__, target=target_type.__name__
                ):
                    original = source_type({"cost": 2.0, "label": "original"})
                    before = copy.deepcopy(original)
                    created = target_type(original, cost=7.0, added=3.0)
                    self.assertEqual(original, before)
                    self.assertEqual(list(original), ["cost", "label"])
                    self.assertIs(type(created), target_type)
                    self.assertIsNot(created.data, original.data)
                    self.assertEqual(
                        dict(created), {"cost": 7.0, "label": "original", "added": 3.0}
                    )
                    self.assertEqual(list(created), ["cost", "label", "added"])
                    del created["label"]
                    self.assertIn("label", original)
                    original["later"] = 11
                    self.assertNotIn("later", created)

    def test_empty_sources_and_construction_without_overrides(self):
        for cls in (Result, History):
            for contents in ({}, {"value": 2}):
                with self.subTest(cls=cls.__name__, contents=contents):
                    source = cls(contents)
                    copied = cls(source)
                    amended = cls(source, added=None)
                    self.assertEqual(dict(source), contents)
                    self.assertEqual(copied, source)
                    self.assertIsNot(copied.data, source.data)
                    self.assertIn("added", amended)
                    self.assertIsNone(amended["added"])
                    self.assertNotIn("added", source)

    def test_constructor_retains_existing_shallow_value_semantics(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                values = np.arange(6).reshape(2, 3)
                nested = cls({"value": [1, 2]})
                original = cls(values=values, nested=nested)
                new_value = {"label": "new"}
                created = cls(original, metadata=new_value)
                self.assertIs(created["values"], original["values"])
                self.assertIs(created["nested"], original["nested"])
                self.assertIs(created["metadata"], new_value)
                self.assertNotIn("metadata", original)
                # Constructors remain shallow; independent value copies are separate APIs.
                created["values"][0, 0] = 99
                self.assertEqual(original["values"][0, 0], 99)

    def test_dictionary_and_keyword_only_inputs_keep_existing_behavior(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__):
                original = {"value": 2, "nested": {"amount": 3}}
                before = copy.deepcopy(original)
                created = cls(original, value=4)
                self.assertEqual(original, before)
                self.assertEqual(created["value"], 4)
                self.assertIsInstance(created["nested"], cls)
                self.assertEqual(created["nested"]["amount"], 3)
                self.assertEqual(dict(cls(value=2)), {"value": 2})
                self.assertEqual(dict(cls()), {})
                with self.assertRaisesRegex(Exception, "Invalid mapping"):
                    cls(3)

    def test_actual_simulation_outputs_can_be_annotated_without_modifying_sources(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        result, history = Simulation(mdl=model)()
        for output, key in ((result, "tend.classify.xy"), (history, "s.x")):
            with self.subTest(cls=type(output).__name__):
                original = copy.deepcopy(output)
                replacement = np.full_like(output[key], -100)
                created = type(output)(output, **{key: replacement, "reviewed": True})
                self.assertEqual(output, original)
                self.assertNotIn("reviewed", output)
                np.testing.assert_array_equal(created[key], replacement)
                self.assertTrue(created["reviewed"])
        repeated_result, repeated_history = Simulation(mdl=model)()
        self.assertEqual(result, repeated_result)
        self.assertEqual(history, repeated_history)


if __name__ == "__main__":
    unittest.main()
