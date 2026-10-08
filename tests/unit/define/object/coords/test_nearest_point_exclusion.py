#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for nearest.

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

from fmdtools.define.object.coords import ExampleCoords


class TestNearestPointExclusion(unittest.TestCase):
    def test_exclusion_applies_when_containing_point_is_a_candidate(self):
        coords = ExampleCoords()
        for query in ((10.0, 10.0), (10.4, 9.8), (0.0, 0.0), (90.0, 90.0)):
            with self.subTest(query=query):
                rounded = coords.to_gridpoint(*query)
                candidates = [p for p in coords.pts if not np.array_equal(p, rounded)]
                expected = min(
                    candidates, key=lambda p: np.linalg.norm(p - np.array(query))
                )
                actual = coords.find_closest(*query, "pts", include_pt=False)
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.array_equal(actual, rounded))
                np.testing.assert_array_equal(
                    coords.find_closest(*query, "pts"), rounded
                )

    def test_other_points_on_the_same_row_or_column_remain_eligible(self):
        coords = ExampleCoords()
        coords.st[:] = 0.0
        coords.set(10.0, 0.0, "st", 1.0)
        coords.set(0.0, 10.0, "st", 1.0)
        for value, comparator in ((1.0, np.equal), (0.0, np.greater)):
            with self.subTest(comparator=comparator.__name__):
                candidates = coords.find_all_prop("st", value, comparator)
                expected = min(candidates, key=lambda p: np.linalg.norm(p))
                actual = coords.find_closest(
                    0.0, 0.0, "st", include_pt=False, value=value, comparator=comparator
                )
                np.testing.assert_array_equal(actual, expected)

    def test_collections_and_properties_use_the_same_nearest_point_rule(self):
        coords = ExampleCoords()
        original = coords.high_v.copy()
        for include in (False, True):
            for query in ((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)):
                with self.subTest(include=include, query=query):
                    rounded = coords.to_gridpoint(*query)
                    candidates = [
                        p for p in original if include or not np.array_equal(p, rounded)
                    ]
                    expected = min(
                        candidates, key=lambda p: np.linalg.norm(p - np.array(query))
                    )
                    np.testing.assert_array_equal(
                        coords.find_closest(*query, "high_v", include_pt=include),
                        expected,
                    )
        np.testing.assert_array_equal(coords.high_v, original)

    def test_excluding_the_only_matching_point_does_not_return_it(self):
        coords = ExampleCoords()
        coords.st[:] = 0.0
        coords.set(10.0, 10.0, "st", 1.0)
        with self.assertRaises(ValueError):
            coords.find_closest(10.0, 10.0, "st", value=1.0, include_pt=False)
        np.testing.assert_array_equal(
            coords.find_closest(10.0, 10.0, "st", value=1.0), [10.0, 10.0]
        )


if __name__ == "__main__":
    unittest.main()
