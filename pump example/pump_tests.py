# -*- coding: utf-8 -*-
"""
Below demonstrates simple unit-tests of model constructs in fmdtools using Python's built-in unitttest module.

The use-cases shown are:
    -Testing function initialization (test_initialization)
    -Testing single-timestep function behavior (ImportEE)
    -Testing multi-timestep function behavior (test_condfaults_dynamic)
    
"""

import sys
# for use in development - makes sure git version is used instead of pip-installed version
paths = sys.path
if paths[1]!='../':
    sys.path=[sys.path[0]] + ['../'] + paths


from fmdtools.modeldef import *
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as propagate

from ex_pump import *

import unittest


class ImportEE_Tests(unittest.TestCase):
    """Tests the single-timestep behaviors in ImportEE."""
    def setUp(self):
        """Instantiate Function and Connected Flows for Tests"""
        self.EE_1 = Flow({'current':1.0, 'voltage':1.0},'EE_1') #The connected flow, EE_1, is a generic flow (i.e., doesn't instantiate a custom class)
        self.Import_EE = ImportEE('ImportEE', [self.EE_1])      #Syntax for instantiating function block
    def test_initialization(self):
        """Tests that the connections, faults, etc instantiated in the initialization are correct"""
        self.assertIs(self.EE_1, self.Import_EE.EEout)
        self.assertIn('no_v', self.Import_EE.faultmodes)
        self.assertIn('inf_v', self.Import_EE.faultmodes)
    def test_condfaults_hi(self):
        """Tests the conditional fault behavior under high current - should result in no_v fault"""
        self.EE_1.current = 20
        self.Import_EE.condfaults(1)
        self.assertTrue(self.Import_EE.has_fault('no_v'))
    def test_condfaults_nom(self):
        """Tests the conditional fault behavior under no current - should not result in any faults"""
        self.EE_1.current = 1
        self.Import_EE.condfaults(1)
        self.assertFalse(self.Import_EE.any_faults())
    def test_behave_nom(self):
        """Tests the nominal behavior - voltage should be 500"""
        self.Import_EE.behavior(1)
        self.assertEqual(self.EE_1.voltage, 500.0)
    def test_behave_inf_v(self):
        """Tests the inf_v behavior - voltage should be 500*100"""
        self.Import_EE.add_fault('inf_v')
        self.Import_EE.behavior(1)
        self.assertEqual(self.EE_1.voltage, 50000.0)
    def test_behave_no_v(self):
        """Tests the no_v behavior - voltage should be 0.0"""
        self.Import_EE.add_fault('no_v')
        self.Import_EE.behavior(1)
        self.assertEqual(self.EE_1.voltage, 0.0) 
        
class MoveWat_Tests(unittest.TestCase):
    """Tests the multi-timestep behaviors in MoveWat."""
    def setUp(self):
        """Instantiate Function and Connected Flows for Tests"""
        self.EE_1 = Flow({'current':1.0, 'voltage':500.0},'EE_1')
        self.Sig_1 = Flow({'power':1.0},'Sig_1')
        self.Wat_1 = Water()
        self.Wat_2 = Water()
        self.Move_Wat = MoveWat('ImportWat', [self.EE_1, self.Sig_1, self.Wat_1, self.Wat_2], 10) #test with delay=10--can be tested w- others
    def test_initialization(self):
        """Tests that the connections, faults, etc instantiated in the initialization are correct"""
        self.assertIs(self.EE_1, self.Move_Wat.EEin)
        self.assertIs(self.Wat_1, self.Move_Wat.Watin)
        self.assertIs(self.Wat_2, self.Move_Wat.Watout)
        self.assertIs(self.Sig_1, self.Move_Wat.Sigin)
        self.assertIn('mech_break', self.Move_Wat.faultmodes)
        self.assertIn('short', self.Move_Wat.faultmodes)
        self.assertEqual(self.Move_Wat.eff, 1.0)
    def test_nom(self):
        self.Move_Wat.behavior(1)
        self.assertEqual(self.Move_Wat.EEin.current, 1.0)
        self.assertEqual(self.Move_Wat.Watin.pressure, 10)
        self.assertEqual(self.Move_Wat.Watin.flowrate, 0.3)
    def test_condfaults_dynamic(self):
        """ tests the dynamic behavior of the conditional fault which occurs when the pump is blocked,
            resulting in a mechanical failure after a delay of 10"""
        self.Wat_2.area=0.0001
        for t in range(0, 20):
            self.Move_Wat.condfaults(t)
            self.Move_Wat.behavior(t)
            if t<=10: self.assertFalse(self.Move_Wat.has_fault('mech_break')) # checks each time-step to ensure the fault has not been added until the delay
        self.assertTrue(self.Move_Wat.has_fault('mech_break'))
        self.assertEqual(self.Move_Wat.eff, 0.0)
        self.assertEqual(self.EE_1.current, 0.2)
        self.assertEqual(self.Wat_2.flowrate, 0.0)
        self.assertEqual(self.Wat_1.flowrate, 0.0)
        

if __name__ == '__main__':
    unittest.main()
        