#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for custom separators when retaining nesting prefixes.

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
from fmdtools.define.base import nest_dict


class TestNestingCustomSeparators(unittest.TestCase):
    def test_skipped_prefix_uses_the_requested_separator(self):
        for separator in ("/", "::", "|", " ", "."):
            for skip in (0, 1, 2, 5):
                for result_type in (dict, Result, History):
                    with self.subTest(
                        separator=separator, skip=skip, result_type=result_type.__name__
                    ):
                        first = np.array([1.0, 2.0])
                        second = np.array([3.0, 4.0])
                        third = np.array([5.0, 6.0])
                        data = result_type(
                            {
                                separator.join(parts): value
                                for parts, value in (
                                    (("sample", "first", "x"), first),
                                    (("sample", "first", "y"), second),
                                    (("sample", "second", "x"), third),
                                )
                            }
                        )
                        original = copy.deepcopy(data)
                        output = nest_dict(data, separator=separator, skip=skip)
                        self.assertIs(type(output), result_type)
                        if skip == 0:
                            self.assertIs(output["sample"]["first"]["x"], first)
                            self.assertIs(output["sample"]["first"]["y"], second)
                            self.assertIs(output["sample"]["second"]["x"], third)
                        elif skip == 1:
                            self.assertIs(
                                output[separator.join(("sample", "first"))]["x"], first
                            )
                            self.assertIs(
                                output[separator.join(("sample", "first"))]["y"], second
                            )
                            self.assertIs(
                                output[separator.join(("sample", "second"))]["x"], third
                            )
                        else:
                            self.assertEqual(list(output), list(data))
                            for key, value in data.items():
                                self.assertIs(output[key], value)
                        for key in data:
                            np.testing.assert_array_equal(data[key], original[key])

    def test_depth_limit_and_unrelated_prefixes_keep_all_values(self):
        for separator in ("/", "::", "."):
            with self.subTest(separator=separator):
                keys = [
                    ("case1", "run", "state", "x"),
                    ("case10", "run", "state", "x"),
                    ("case1_extra", "run", "state", "y"),
                ]
                data = {separator.join(key): i for i, key in enumerate(keys)}
                nested = nest_dict(data, levels=1, separator=separator, skip=1)
                self.assertEqual(
                    nested,
                    {
                        separator.join(key[:2]): {separator.join(key[2:]): i}
                        for i, key in enumerate(keys)
                    },
                )

    def test_public_result_and_history_nesting_support_the_same_options(self):
        for result_type in (Result, History):
            for separator in ("/", "::"):
                with self.subTest(
                    result_type=result_type.__name__, separator=separator
                ):
                    data = result_type(
                        {
                            separator.join(("run", "fault", "value")): np.array(
                                [2.0, 3.0]
                            ),
                            separator.join(("run", "nominal", "value")): np.array(
                                [0.0, 1.0]
                            ),
                        }
                    )
                    output = data.nest(separator=separator, skip=1)
                    self.assertEqual(
                        list(output),
                        [
                            separator.join(("run", "fault")),
                            separator.join(("run", "nominal")),
                        ],
                    )
                    for key, value in data.items():
                        self.assertIs(
                            output[separator.join(key.split(separator)[:2])]["value"],
                            value,
                        )

    def test_default_dotted_paths_retain_their_existing_layout(self):
        data = {"p0.nominal.x": 1, "p0.fault.x": 2, "p1.nominal.x": 3}
        self.assertEqual(
            nest_dict(data, levels=1, skip=1),
            {"p0.nominal": {"x": 1}, "p0.fault": {"x": 2}, "p1.nominal": {"x": 3}},
        )
        self.assertEqual(
            nest_dict(data),
            {
                "p0": {"nominal": {"x": 1}, "fault": {"x": 2}},
                "p1": {"nominal": {"x": 3}},
            },
        )

    def test_empty_and_unsplit_keys_retain_existing_behavior(self):
        for result_type in (dict, Result, History):
            with self.subTest(result_type=result_type.__name__):
                self.assertEqual(
                    dict(nest_dict(result_type(), separator="/", skip=1)), {}
                )
                data = result_type({"standalone": np.array([1.0])})
                self.assertIs(
                    nest_dict(data, separator="/", skip=1)["standalone"],
                    data["standalone"],
                )


if __name__ == "__main__":
    unittest.main()
