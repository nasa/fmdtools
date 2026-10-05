#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for numerical correctness.

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

from fmdtools.define.object.coords import Coords


class ValueGrid(Coords):
    state_value = (float, 0.0)


class TestCoordinateCellBoundaries(unittest.TestCase):
    def test_in_range_boundary_queries_never_produce_out_of_bounds_indices(self):
        for nx, ny in ((1, 1), (2, 3), (3, 4), (4, 2), (9, 10)):
            for blocksize in (0.1, 0.5, 10.0):
                coords = ValueGrid(
                    p={"x_size": nx, "y_size": ny, "blocksize": blocksize}
                )
                for axis, count in enumerate((nx, ny)):
                    boundaries = (np.arange(count) + 0.5) * blocksize
                    values = np.concatenate(
                        [
                            np.arange(count) * blocksize,
                            boundaries,
                            np.nextafter(boundaries, -np.inf),
                            np.nextafter(boundaries, np.inf),
                        ]
                    )
                    for value in values:
                        point = [0.0, 0.0]
                        point[axis] = value
                        if not coords.in_range(*point):
                            continue
                        with self.subTest(
                            shape=(nx, ny), blocksize=blocksize, axis=axis, value=value
                        ):
                            expected = min(round(float(value) / blocksize), count - 1)
                            index = coords.to_index(*point)
                            self.assertEqual(index[axis], expected)
                            self.assertEqual(index[1 - axis], 0)
                            self.assertTrue(all(isinstance(i, int) for i in index))
                            np.testing.assert_array_equal(
                                coords.to_gridpoint(*point), coords.grid[index]
                            )

    def test_upper_edges_remain_accessible_for_even_and_odd_grid_sizes(self):
        for nx, ny in ((1, 2), (2, 1), (2, 2), (3, 4), (4, 5), (10, 10)):
            coords = ValueGrid(p={"x_size": nx, "y_size": ny, "blocksize": 2.0})
            upper = (2.0 * nx - 1.0, 2.0 * ny - 1.0)
            with self.subTest(shape=(nx, ny)):
                self.assertTrue(coords.in_range(*upper))
                self.assertEqual(coords.to_index(*upper), (nx - 1, ny - 1))
                coords.set(*upper, "value", 7.0)
                self.assertEqual(coords.get(*upper, "value"), 7.0)
                expected = np.zeros((nx, ny))
                expected[-1, -1] = 7.0
                np.testing.assert_array_equal(coords.value, expected)
                np.testing.assert_array_equal(
                    coords.find_all(value=(7.0, np.equal)), [coords.grid[-1, -1]]
                )

    def test_midpoint_writes_preserve_existing_nearest_even_behavior(self):
        coords = ValueGrid(p={"x_size": 4, "y_size": 4, "blocksize": 10.0})
        for x, y, expected in (
            (5.0, 5.0, (0, 0)),
            (15.0, 15.0, (2, 2)),
            (25.0, 15.0, (2, 2)),
            (15.0, 25.0, (2, 2)),
        ):
            with self.subTest(point=(x, y)):
                coords.value[:] = 0.0
                coords.set(x, y, "value", 12.0)
                self.assertEqual(coords.value[expected], 12.0)
                self.assertEqual(np.count_nonzero(coords.value), 1)

    def test_outside_policy_is_unchanged_at_both_edges(self):
        coords = ValueGrid(p={"x_size": 4, "y_size": 3, "blocksize": 10.0})
        for point in (
            (-5.0, 0.0),
            (0.0, -5.0),
            (np.nextafter(35.0, np.inf), 0.0),
            (0.0, np.nextafter(25.0, np.inf)),
            (np.inf, 0.0),
            (0.0, np.nan),
        ):
            with self.subTest(point=point):
                self.assertFalse(coords.in_range(*point))
                with self.assertRaisesRegex(Exception, "Outside bounds"):
                    coords.to_index(*point)
                self.assertEqual(coords.get(*point, "value", outside=-1.0), -1.0)
        self.assertEqual(coords.to_index(np.nextafter(-5.0, np.inf), 0.0), (0, 0))

    def test_neighbors_use_the_correct_cell_at_midpoints_and_outer_corners(self):
        coords = ValueGrid(p={"x_size": 4, "y_size": 4, "blocksize": 10.0})
        np.testing.assert_array_equal(
            coords.get_neighbors(15.0, 15.0, direction="right"), [[30.0, 20.0]]
        )
        self.assertEqual(coords.get_neighbors(35.0, 35.0, direction="right"), [])
        np.testing.assert_array_equal(
            coords.get_neighbors(35.0, 35.0, direction="left"), [[20.0, 30.0]]
        )


if __name__ == "__main__":
    unittest.main()
