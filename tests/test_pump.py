# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:49:13 2021

@author: dhulse
"""
import unittest
import sys, os
sys.path.insert(1, os.path.join('..'))
from example_pump.ex_pump import Pump
from fmdtools.faultsim import propagate
import fmdtools.resultdisp as rd
from fmdtools.modeldef import SampleApproach
from CommonTests import CommonTests
import numpy as np

class PumpTests(unittest.TestCase, CommonTests):
    def setUp(self):
        self.default_mdl = Pump()
        self.water_mdl = Pump(params={'cost':{'water'}, 'delay':10, 'units':'hrs'})
    def test_dynamic_prop_values(self):
        """Test that given fault times result in the expected water/value loss"""
        faulttimes = [10,20,30]
        for faulttime in faulttimes:
            endresults, resgraph, mdlhist = propagate.one_fault(self.water_mdl, "MoveWater", "mech_break", time=faulttime)
            expected_wcost = self.expected_water_cost(faulttime)
            self.assertAlmostEqual(expected_wcost, endresults['classification']['cost'])
    def test_dynamic_prop_values_2(self):
        """Test that the delayed fault behavior occurs at the time specified"""
        delays = [0, 1, 5, 10]
        for delay in delays:
            mdl = Pump(params={'cost':{'water'}, 'delay':delay, 'units':'hrs'})
            endresults, resgraph, mdlhist = propagate.one_fault(mdl, 'ExportWater', 'block', time=25)
            has_fault_at_time = mdlhist['faulty']['functions']['MoveWater']['faults']['mech_break'][25+delay]
            self.assertEqual(has_fault_at_time, 1)
            has_fault_before_time = mdlhist['faulty']['functions']['MoveWater']['faults']['mech_break'][25+delay-1]
            self.assertEqual(has_fault_before_time, 0)
    def test_app_prop_values(self):
        """Test that the delayed fault behavior occurs at the time specified (with approach)"""
        approach = SampleApproach(self.water_mdl, faults=[('MoveWater','mech_break')], phases=['on'],defaultsamp={'samp':'evenspacing','numpts':5})
        endclasses, mdlhists = propagate.approach(self.water_mdl, approach)
        for scen in approach.scenlist:
            name = scen['properties']['name']
            exp_wcost = self.expected_water_cost(scen['properties']['time'])
            self.assertAlmostEqual(exp_wcost, endclasses[name]['cost'])
    def expected_water_cost(self, faulttime):
        return (50-faulttime) * 0.3*750 
    def test_model_copy_same(self):
        mdl = Pump()
        mdl2 = Pump()
        inj_times = [10,20,30,40]
        self.check_model_copy_same(mdl, mdl2, inj_times, 30, max_time=55)
    def test_model_copy_different(self):
        mdl = Pump(); inj_times = [10,20,30,40]
        self.check_model_copy_different(mdl, inj_times, max_time=55)
    def test_model_reset(self):
        mdl = Pump(); inj_times= [10,20,30,40]
        mdl_reset = Pump()
        self.check_model_reset(mdl, mdl_reset, inj_times, max_time=55)
        
        

            

if __name__ == '__main__':
    unittest.main()
    
    
