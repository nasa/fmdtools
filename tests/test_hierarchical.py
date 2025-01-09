# -*- coding: utf-8 -*-
"""
This module tests the example of hierarchical modelling in which a function contains a
function architecture, which in turn contains a function.
"""

from fmdtools.define.container.state import ExampleState
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import FunctionArchitecture, ExFxnArch

from fmdtools.sim import propagate

import unittest


class OverFxn(Function):
    """Test function containing a FunctionArchitecture."""
    __slots__ = ('fa')
    arch_fa = ExFxnArch
    container_s = ExampleState
    container_m = Mode

    def dynamic_behavior(self, time):
        self.s.assign(self.fa.flows['exf'].s)


class define_Tests(unittest.TestCase):
    def setUp(self):
        self.mdl = OverFxn()

    def test_propagation(self):
        """Check that contained functionarch propagates as expected."""
        self.mdl.propagate(1.0)
        # should match values from functionarch propagation, see docstrings
        self.assertEqual(self.mdl.s.x, 4.0)
        self.mdl.propagate(2.0)
        self.assertEqual(self.mdl.s.x, 10.0)

    def test_fault_injection(self):
        """Check that faults get injected into contained functionarch."""
        self.mdl.inject_faults({'overfxn': {'ex_fxn': 'no_charge'}})
        self.assertEqual(self.mdl.m.faults, {'ex_fxn_no_charge'})
        self.assertEqual(self.mdl.fa.fxns['ex_fxn'].m.mode, "no_charge")

    def test_top_level_injection(self):
        """Check that faults also get injected into top level."""
        self.mdl.inject_faults("test_fault")
        self.assertEqual(self.mdl.m.faults, {'test_fault'})

    def test_prop_method(self):
        """Check that faults get injected using propagate.one_fault"""
        res, hist = propagate.one_fault(self.mdl,
                                        "overfxn", "ex_fxn_no_charge", time=5.0)
        self.assertFalse(hist.faulty.m.faults.ex_fxn_no_charge[4])
        self.assertTrue(hist.faulty.m.faults.ex_fxn_no_charge[5])
        self.assertFalse(hist.faulty.fa.fxns.ex_fxn.m.faults.no_charge[4])
        self.assertTrue(hist.faulty.fa.fxns.ex_fxn.m.faults.no_charge[5])


if __name__ == '__main__':

    oa = OverFxn()
    res, hist = propagate.one_fault(oa, "overfxn", "ex_fxn_no_charge", time=5.0)
    #works:
    hist.faulty.m.faults.ex_fxn_no_charge
    #doesn't work:
    hist.faulty.fa.fxns.ex_fxn.m.faults.no_charge

    # oa.inject_faults(['ex_fxn_no_charge'])
    # oa.inject_faults({'overfxn': {'ex_fxn': 'no_charge'}})

    unittest.main()
    
    # test manual propagation
    
    # test propagation method(s)

