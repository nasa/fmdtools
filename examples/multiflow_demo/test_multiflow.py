# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:45:26 2023

@author: dhulse
"""
import unittest
import numpy as np
from examples.multiflow_demo.multiflow_demo import TestModel
from fmdtools.sim import propagate


class define_Tests(unittest.TestCase):
    def setUp(self):
        self.mdl = TestModel()
        self.endresults, self.mdlhist = propagate.nominal(self.mdl)

    def test_multiflow_passing(self):
        """Check that location copied such that the global version aren't modified
        but the local ones are."""
        np.testing.assert_array_equal(self.mdlhist.flows.location.s.x,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.s.y,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_1.s.x,
                                      [i+1.0 for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_1.s.y,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_2.s.x,
                                      np.zeros(11))
        np.testing.assert_array_equal(self.mdlhist.flows.location.mover_2.s.y,
                                      [i+1.0 for i in range(11)])

    def test_multiflow_combination(self):
        """Check that communications combined such that both Movers have iterating x-y
        values."""
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_1.s.x,
                                      [i+1.0 for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_1.s.y,
                                      [i+1.0 for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_2.s.x,
                                      [i+1.0 for i in range(11)])
        np.testing.assert_array_equal(self.mdlhist.flows.communications.mover_2.s.y,
                                      [i+1.0 for i in range(11)])
        # check that coordinator parses communiations from each Mover
        x1 = self.mdlhist.flows.communications.coordinator.mover_1.s.x
        np.testing.assert_array_equal(x1, [i+1.0 for i in range(11)])
        y1 = self.mdlhist.flows.communications.coordinator.mover_2.s.y
        np.testing.assert_array_equal(y1, [i+1.0 for i in range(11)])

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