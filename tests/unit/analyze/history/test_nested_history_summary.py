#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for summaries of nested simulation histories.

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

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate


class TestNestedHistorySummary(unittest.TestCase):
    def make_history(self):
        return History(
            {
                "fxn.s.x": np.array([0.0, 2.0, 1.0]),
                "fxn.s.y": np.array([4.0, 1.0, 3.0]),
                "fxn2.s.x": np.array([8.0, 6.0, 7.0]),
                "time": np.arange(3),
            }
        )

    def test_nested_and_flat_histories_have_identical_scalar_summaries(self):
        flat = self.make_history()
        for levels in (1, 2, 3):
            for operator in (np.max, np.min, np.sum, np.mean):
                with self.subTest(levels=levels, operator=operator.__name__):
                    nested = flat.nest(levels=levels)
                    summary = nested.get_summary(operator=operator)
                    self.assertIs(type(summary), Result)
                    self.assertEqual(list(summary), list(flat))
                    for key, value in flat.items():
                        self.assertEqual(summary[key], operator(value))
                    self.assertEqual(summary, flat.get_summary(operator=operator))

    def test_selected_paths_preserve_order_and_do_not_select_related_names(self):
        flat = self.make_history()
        for history in (flat, flat.nest()):
            for fields in (
                ("fxn.s.x",),
                ("fxn.s.y", "time", "fxn.s.x"),
                ("fxn.s.x", "missing", "fxn2.s.x"),
                ("fxn.s.x", "fxn.s.x"),
                ("missing",),
            ):
                with self.subTest(fields=fields, nested=history is not flat):
                    expected = {key: np.max(flat[key]) for key in fields if key in flat}
                    self.assertEqual(dict(history.get_summary(*fields)), expected)
                    self.assertEqual(list(history.get_summary(*fields)), list(expected))

    def test_array_and_boolean_leaves_are_reduced_with_the_requested_operator(self):
        flat = History(
            {
                "a.matrix": np.arange(12).reshape(3, 2, 2),
                "a.flag": np.array([False, True, False]),
                "time": np.arange(3),
            }
        )
        nested = flat.nest()
        for operator in (np.max, np.sum, np.mean, np.any):
            with self.subTest(operator=operator.__name__):
                result = nested.get_summary(operator=operator)
                for key in flat:
                    self.assertEqual(result[key], operator(flat[key]))
        self.assertEqual(dict(History().get_summary()), {})
        self.assertEqual(dict(History(a=History()).get_summary()), {})

    def test_operator_receives_leaves_without_changing_nested_input(self):
        flat = self.make_history()
        nested = flat.nest()
        before = nested.copy()
        seen = []

        def span(values):
            seen.append(np.asarray(values).shape)
            return np.max(values) - np.min(values)

        result = nested.get_summary("fxn.s.x", "fxn.s.y", operator=span)
        self.assertEqual(dict(result), {"fxn.s.x": 2.0, "fxn.s.y": 3.0})
        self.assertEqual(seen, [(3,), (3,)])
        self.assertEqual(nested, before)
        self.assertEqual(nested.flatten(), flat)

    def test_real_nominal_and_faulty_histories_keep_all_requested_peak_values(self):
        model = ExampleFunction(sp={"end_time": 4.0})
        _, history = propagate.one_fault(
            model, model.name, "low", time=2.0, showprogress=False
        )
        flat = history.flatten()
        faulty = next(
            key.split(".")[0] for key in flat if not key.startswith("nominal.")
        )
        fields = ("nominal.s.x", faulty + ".s.x", faulty + ".s.y", faulty + ".time")
        self.assertTrue(all(field in flat for field in fields))
        expected = {field: np.max(flat[field]) for field in fields}
        for source in (flat.nest(), flat.nest(levels=1)):
            self.assertEqual(dict(source.get_summary(*fields)), expected)
        self.assertNotEqual(expected["nominal.s.x"], expected[faulty + ".s.x"])


if __name__ == "__main__":
    unittest.main()
