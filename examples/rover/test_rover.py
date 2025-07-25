#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

from tests.common import CommonTests
from examples.rover.optimization.search_rover import line_dist, line_dist_faster
import unittest


class RoverTests(unittest.TestCase, CommonTests):
    def test_obj_values(self):

        testvalues = [[1.0, 0.5, 0.0], [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]
        for testvalue in testvalues:
            dist_int, enddist_int, endpt_int = line_dist_faster(testvalue)
            dist, enddist, endpt = line_dist(testvalue)

            self.assertEqual(dist, dist_int)
            self.assertEqual(enddist, enddist_int)
            self.assertEqual(endpt, endpt_int)


if __name__ == '__main__':
    unittest.main()
