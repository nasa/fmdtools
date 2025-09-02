# -*- coding: utf-8 -*-
"""
This module tests the example of hierarchical modelling in which a function contains a
function architecture, which in turn contains a function.
"""

from fmdtools.define.container.state import ExampleState
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import ExFxnArch

from fmdtools.sim import propagate

import unittest


class OverFxn(Function):
    """Test function containing a FunctionArchitecture."""

    arch_fa = ExFxnArch
    container_s = ExampleState
    container_m = Mode

    def dynamic_behavior(self):
        self.s.assign(self.fa.flows['exf'].s)


class define_Tests(unittest.TestCase):
    def setUp(self):
        self.mdl = OverFxn()

    def test_propagation(self):
        """Check that contained functionarch propagates as expected."""
        self.mdl(time=1.0)
        # should match values from functionarch propagation, see docstrings
        self.assertEqual(self.mdl.s.x, 2.0)
        self.mdl(time=2.0)
        self.assertEqual(self.mdl.s.x, 6.0)

    def test_fault_injection(self):
        """Check that faults get injected into contained functionarch."""
        self.mdl.inject_faults({'overfxn.fa.fxns.ex_fxn': 'no_charge'})
        self.assertTrue(self.mdl.m.sub_faults)
        self.assertEqual(self.mdl.fa.fxns['ex_fxn'].m.mode, "no_charge")

    def test_prop_method(self):
        """Check that faults get injected using propagate.one_fault"""
        res, hist = propagate.one_fault(self.mdl,
                                        "overfxn.fa.fxns.ex_fxn", "no_charge", time=5.0)
        self.assertFalse(hist.faulty.m.sub_faults[4])
        self.assertTrue(hist.faulty.m.sub_faults[5])
        self.assertFalse(hist.faulty.fa.fxns.ex_fxn.m.faults.no_charge[4])
        self.assertTrue(hist.faulty.fa.fxns.ex_fxn.m.faults.no_charge[5])

    def test_arg_passdown(self):
        """Check that function args get passed to functionarch."""
        oa = OverFxn(fa={'p': {'x': 3.0}})
        self.assertEqual(oa.fa.p.x, 3.0)
        oa2 = oa.new()
        self.assertEqual(oa2.fa.p.x, 3.0)


if __name__ == '__main__':
    oa = OverFxn(fa={'p': {'x': 3.0}})

    oa = OverFxn()
    res, hist = propagate.one_fault(oa, "overfxn.fa.fxns.ex_fxn", "no_charge", time=5.0)


    # oa.inject_faults(['ex_fxn_no_charge'])
    # oa.inject_faults({'overfxn': {'ex_fxn': 'no_charge'}})

    unittest.main()
