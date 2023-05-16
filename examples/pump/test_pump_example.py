# -*- coding: utf-8 -*-
"""
Below demonstrates simple unit-tests of model constructs in fmdtools using Python's built-in unitttest module.

Unit testing can help ensure that model constructs actually embody the intended
dynamic behavior, and help when error-checking a large model. 

The use-cases shown are:
    -Testing function initialization (test_initialization)
    -Testing single-timestep function behavior (ImportEE)
    -Testing multi-timestep function behavior (test_condfaults_dynamic)
    -Testing a model
"""

from fmdtools.define.flow import Flow
import fmdtools.analyze as an
import fmdtools.sim.propagate as propagate

from ex_pump import ImportEE, Water, MoveWat, Pump, Electricity

import unittest


class ImportEE_Tests(unittest.TestCase):
    """Tests the single-timestep behaviors in ImportEE."""
    def setUp(self):
        """Instantiate Function and Connected Flows for Tests"""
        self.ee_1 = Electricity('ee1', s={'current':1.0, 'voltage':1.0}) #The connected flow, ee_1, is a generic flow (i.e., doesn't instantiate a custom class)
        self.import_ee = ImportEE('ImportEE', flows={"ee_1":self.ee_1})      #Syntax for instantiating function block
    def test_initialization(self):
        """Tests that the connections, faults, etc instantiated in the initialization are correct"""
        self.assertIs(self.ee_1, self.import_ee.ee_out)
        self.assertIn('no_v', self.import_ee.m.faultmodes)
        self.assertIn('inf_v', self.import_ee.m.faultmodes)
    def test_condfaults_hi(self):
        """Tests the conditional fault behavior under high current - should result in no_v fault"""
        self.ee_1.s.current = 20
        self.import_ee.condfaults(1)
        self.assertTrue(self.import_ee.m.has_fault('no_v'))
    def test_condfaults_nom(self):
        """Tests the conditional fault behavior under no current - should not result in any faults"""
        self.ee_1.s.current = 1
        self.import_ee.condfaults(1)
        self.assertFalse(self.import_ee.m.any_faults())
    def test_behave_nom(self):
        """Tests the nominal behavior - voltage should be 500"""
        self.import_ee.behavior(1)
        self.assertEqual(self.ee_1.s.voltage, 500.0)
    def test_behave_inf_v(self):
        """Tests the inf_v behavior - voltage should be 500*100"""
        self.import_ee.m.add_fault('inf_v')
        self.import_ee.behavior(1)
        self.assertEqual(self.ee_1.s.voltage, 50000.0)
    def test_behave_no_v(self):
        """Tests the no_v behavior - voltage should be 0.0"""
        self.import_ee.m.add_fault('no_v')
        self.import_ee.behavior(1)
        self.assertEqual(self.ee_1.s.voltage, 0.0) 
        
class MoveWat_Tests(unittest.TestCase):
    """Tests the multi-timestep behaviors in MoveWat."""
    def setUp(self):
        """Instantiate Function and Connected Flows for Tests"""
        
        self.move_wat = MoveWat('ImportWat', p={'delay':10}, t={'dt':1.0}) #test with delay=10--can be tested w- others
    def test_initialization(self):
        """Tests that the connections, faults, etc instantiated in the initialization are correct"""
        self.assertIn('mech_break', self.move_wat.m.faultmodes)
        self.assertIn('short', self.move_wat.m.faultmodes)
        self.assertEqual(self.move_wat.s.eff, 1.0)
    def test_nom(self):
        self.move_wat.behavior(1.0)
        self.assertEqual(self.move_wat.ee_in.s.current, 0.002)
        self.move_wat.behavior(1.0)
        self.assertEqual(self.move_wat.wat_in.s.pressure, 0.02)
        self.assertEqual(self.move_wat.wat_in.s.flowrate, 0.0006)
    def test_condfaults_dynamic(self):
        """ tests the dynamic behavior of the conditional fault which occurs when the pump is blocked,
            resulting in a mechanical failure after a delay of 10"""
        self.move_wat.wat_out.s.area=0.0001
        for t in range(0, 20):
            self.move_wat.condfaults(t)
            self.move_wat.behavior(t)
            if t<10: self.assertFalse(self.move_wat.m.has_fault('mech_break')) # checks each time-step to ensure the fault has not been added until the delay
        self.assertTrue(self.move_wat.m.has_fault('mech_break'))
        self.assertEqual(self.move_wat.s.eff, 0.0)
        self.assertEqual(self.move_wat.ee_in.s.current, 0.0004)
        self.assertEqual(self.move_wat.wat_out.s.flowrate, 0.0)
        self.assertEqual(self.move_wat.wat_in.s.flowrate, 0.0)

class Integration_Tests(unittest.TestCase):
    """Tests the integrated simulation of components"""
    def setUp(self):
        self.mdl=Pump()
    def test_nominal_results(self):
        """Tests the output of the model when integrated in the nominal scenario"""
        endresult, mdlhist=propagate.nominal(self.mdl, protect=False, track='all')
        modes, modeprops = self.mdl.return_faultmodes()
        self.assertFalse(modes) # does it have any fault modes in the nominal scenario?
        for t in range(1,self.mdl.sp.times[-1]): # are the values of the function/flow states what we wanted?
            if t<5 or t>=50:
                self.assertEqual(mdlhist.flows.sig_1.s.power[t], 0.0)
                self.assertEqual(mdlhist.flows.ee_1.s.current[t], 0.0)
                self.assertEqual(mdlhist.flows.wat_1.s.flowrate[t], 0.0)
                self.assertEqual(mdlhist.flows.wat_2.s.flowrate[t], 0.0)
            else:
                self.assertEqual(mdlhist.flows.sig_1.s.power[t], 1.0)
                self.assertEqual(mdlhist.flows.ee_1.s.current[t], 10.0)
                self.assertEqual(mdlhist.flows.wat_1.s.flowrate[t], 0.3)
                self.assertEqual(mdlhist.flows.wat_2.s.flowrate[t], 0.3)
            self.assertEqual(mdlhist.flows.ee_1.s.voltage[t], 500.0)
            self.assertEqual(mdlhist.fxns.move_water.s.eff[t], 1.0)
    def test_blockage_results(self):
        """Tests the output of the model when integrated in a faulty scenario"""
        endfaults, mdlhist=propagate.one_fault(self.mdl, 'export_water','block', time=10, desired_result='endfaults', track='all')
        self.assertIn('move_water', endfaults['endfaults'])
        self.assertIn('export_water', endfaults['endfaults'])
        for t in range(1,self.mdl.sp.times[-1]): # are the values of the function/flow states what we wanted?
            if t<5 or t>=50:
                self.assertEqual(mdlhist.faulty.flows.sig_1.s.power[t], 0.0)
                self.assertEqual(mdlhist.faulty.flows.ee_1.s.current[t], 0.0)
                self.assertEqual(mdlhist.faulty.flows.wat_1.s.flowrate[t], 0.0)
                self.assertEqual(mdlhist.faulty.flows.wat_2.s.flowrate[t], 0.0)
            elif t<10: # should see faulty behavior at t=10
                self.assertEqual(mdlhist.faulty.flows.sig_1.s.power[t], 1.0)
                self.assertEqual(mdlhist.faulty.flows.ee_1.s.current[t], 10.0)
                self.assertEqual(mdlhist.faulty.flows.wat_1.s.flowrate[t], 0.3)
                self.assertEqual(mdlhist.faulty.flows.wat_2.s.flowrate[t], 0.3)
                self.assertEqual(mdlhist.faulty.fxns.move_water.s.eff[t], 1.0)
            elif t<20: #at t=20, the conditional damage occurs (depending on the delay)
                self.assertEqual(mdlhist.faulty.flows.sig_1.s.power[t], 1.0)
                self.assertEqual(mdlhist.faulty.flows.ee_1.s.current[t], 13.0)
                self.assertEqual(mdlhist.faulty.flows.wat_1.s.flowrate[t], 0.003)
                self.assertEqual(mdlhist.faulty.flows.wat_2.s.flowrate[t], 0.003)
                self.assertTrue(mdlhist.faulty.fxns.export_water.m.faults.block[t])
            else:
                self.assertEqual(mdlhist.faulty.fxns.move_water.s.eff[t], 0.0)
                self.assertTrue(mdlhist.faulty.fxns.export_water.m.faults.block[t])
                self.assertTrue(mdlhist.faulty.fxns.move_water.m.faults.mech_break[t])
                self.assertEqual(mdlhist.faulty.flows.ee_1.s.current[t], 0.2)
                self.assertEqual(mdlhist.faulty.flows.wat_1.s.flowrate[t], 0.0)
                self.assertEqual(mdlhist.faulty.flows.wat_2.s.flowrate[t], 0.0)

            self.assertEqual(mdlhist.faulty.flows.ee_1.s.voltage[t], 500.0)
    def test_blockage_static(self):
        """Checks state of the model itself at a particular time-step. Useful when the model has states which are not recorded."""
        for t in range(0, 10): self.mdl.propagate(t)     #simulate time up until t=10
        
        self.mdl.propagate(10, {'move_water': 'mech_break'})  #instantiate fault at time
        self.assertTrue(self.mdl.fxns['move_water'].m.has_fault('mech_break'))         #check model properties
        self.assertEqual(self.mdl.flows['ee_1'].s.current, 0.2)
        self.assertEqual(self.mdl.flows['wat_1'].s.flowrate, 0.0)
        self.assertEqual(self.mdl.flows['wat_2'].s.flowrate, 0.0)
                

if __name__ == '__main__':
    unittest.main()
        