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
import numpy as np

class pump_tests(unittest.TestCase):
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
        approach = SampleApproach(self.water_mdl, faults=[('MoveWater','mech_break')], phases=['on'],defaultsamp={'samp':'evenspacing','numpts':5})
        endclasses, mdlhists = propagate.approach(self.water_mdl, approach)
        for scen in approach.scenlist:
            name = scen['properties']['name']
            exp_wcost = self.expected_water_cost(scen['properties']['time'])
            self.assertAlmostEqual(exp_wcost, endclasses[name]['cost'])
    def expected_water_cost(self, faulttime):
        return (50-faulttime) * 0.3*750 
    def test_model_copy_same(self):
        """ Tests to see that two models have the same states and that a copied model
        has the same states as the others given the same inputs"""
        mdl = Pump()
        mdl2 = Pump()
        faultscens = [{fname: [*f.faultmodes][0]} for fname, f in mdl.fxns.items()]
        for faultscen in faultscens:
            for inj_time in [10,20,30,40]:
                for t in range(55):
                    if t==inj_time:   scen=faultscen
                    else:       scen={}
                    propagate.propagate(mdl,scen,t)       
                    propagate.propagate(mdl2,scen,t) 
                    self.check_same_model(mdl, mdl2)
                    if t==30: mdl_copy = mdl.copy()
                    if t>30: 
                        propagate.propagate(mdl_copy, scen, t)
                        self.check_same_model(mdl, mdl_copy)
    def test_model_copy_different(self):
        """ Tests to see that a copied model has different states from the model
        it was copied from after fault injection/etc """
        mdl = Pump()
        faultscens = [{fname: [*f.faultmodes][0]} for fname, f in mdl.fxns.items()] 
        for faultscen in faultscens:
            for inj_time in [10,20,30,40]:
                for t in range(55):
                    propagate.propagate(mdl,{},t)       
                    if t==inj_time: mdl_copy = mdl.copy()
                    if t>inj_time: 
                        propagate.propagate(mdl_copy, faultscen, t)
                        self.check_diff_model(mdl, mdl_copy)
    def check_same_model(self, mdl, mdl2):
        for flname, fl in mdl.flows.items():
            for state in fl._states:
                self.assertEqual(getattr(fl, state), getattr(mdl2.flows[flname], state))
        for fxnname, fxn in mdl.fxns.items():
            for state in fxn._states:
                self.assertEqual(getattr(fxn, state), getattr(mdl2.fxns[fxnname], state))
            self.assertEqual(fxn.faults, mdl2.fxns[fxnname].faults)
    def check_diff_model(self, mdl, mdl2):
        same=1
        for flname, fl in mdl.flows.items():
            for state in fl._states:
                if getattr(fl, state)==getattr(mdl2.flows[flname], state): same = same*1
                else:                                                       same=0
        for fxnname, fxn in mdl.fxns.items():
            for state in fxn._states:
                if getattr(fxn, state)== getattr(mdl2.fxns[fxnname], state): same= same*1
                else:                                                       same=0
            if fxn.faults==mdl2.fxns[fxnname].faults:                   same=same*1
            else:                                                       same=0
        if same==1:
            a=1
        self.assertEqual(same,0)
            

if __name__ == '__main__':
    unittest.main()
    
    
