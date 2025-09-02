#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of the multiflow demo model.

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

from examples.multiflow_demo.multiflow_demo import ExModel

from fmdtools.sim import propagate

import unittest
import numpy as np



class define_Tests(unittest.TestCase):
    def setUp(self):
        self.mdl = ExModel()
        self.endresults, self.mdlhist = propagate.nominal(self.mdl)

    def test_multiflow_passing(self):
        """Check that location copied such that the global version aren't modified
        but the local ones are."""
        np.testing.assert_array_equal(self.mdlhist.flows.location.s.x,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.s.y,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_1.s.x,
                                      [i for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_1.s.y,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_2.s.x,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_2.s.y,
                                      [i for i in range(11)])

    def test_multiflow_combination(self):
        """Check that communications combined such that both Movers have iterating x-y
        values."""
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_1.s.x,
                                      [i for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_1.s.y,
                                      [i for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_2.s.x,
                                      [i for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_2.s.y,
                                      [i for i in range(11)])
        # check that coordinator parses communiations from each Mover
        x1 = self.mdlhist.flows.communications.coordinator.mover_1.s.x
        np.testing.assert_array_equal(x1, [i for i in range(11)])
        y1 = self.mdlhist.flows.communications.coordinator.mover_2.s.y
        np.testing.assert_array_equal(y1, [i for i in range(11)])

    def test_mutliflow_copying(self):
        """Check that multiflow copies as expected."""
        self.mdl.flows["communications"].mover_1.s.x = 25
        self.mdl.flows["communications"].mover_1.send(["mover_2", "coordinator"])
        self.assertEqual(self.mdl.flows["communications"].fxns["coordinator"]["in"],
                         {"mover_1": ()})

        self.mdl.flows["communications"].coordinator.receive()
        self.assertEqual(self.mdl.flows["communications"].fxns["mover_1"]["out"].s.x,
                         25)
        cx = self.mdl.flows["communications"].fxns["coordinator"]["internal"].mover_1.s.x
        self.assertEqual(cx, 25)
        self.assertEqual(self.mdl.flows["communications"].fxns["mover_2"]["in"],
                         {"mover_1": ()})

        # copies should keep in/out dicts in place
        mdl2 = self.mdl.copy()
        self.assertEqual(mdl2.flows["communications"].fxns["mover_1"]["out"].s.x, 25)
        self.assertEqual(mdl2.flows["communications"].fxns["mover_2"]["in"],
                         {"mover_1": ()})
        cx = self.mdl.flows["communications"].fxns["coordinator"]["internal"].mover_1.s.x
        self.assertEqual(cx, 25)


if __name__ == '__main__':
    unittest.main()