# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:51:57 2021

@author: dhulse
"""
import unittest
import sys, os
sys.path.insert(1, os.path.join('..'))
from example_tank.tank_model import Tank
from fmdtools.faultsim import propagate
import fmdtools.resultdisp as rd
from fmdtools.modeldef import SampleApproach, NominalApproach
from CommonTests import CommonTests
import numpy as np


class TankTests(unittest.TestCase, CommonTests):
    def test_model_copy_same(self):
        self.check_model_copy_same(Tank(),Tank(), [5,10,15], 10, max_time=20)
    def test_model_copy_different(self):
        self.check_model_copy_different(Tank(), [5,10,15], max_time=20)
    def test_model_reset(self):
        mdl = Tank(); mdl2 = Tank()
        self.check_model_reset(mdl, mdl2, [5,10,15], max_time=20)
    def test_comp_mode_inj(self):
        """ Tests that component modes injected in functions end up in their respective
        components."""
        mdl = Tank()
        compmodes = [mode for c in mdl.fxns['Human'].components.values() for mode in c.faultmodes]
        compname = {mode:cname for cname,c in mdl.fxns['Human'].components.items() for mode in c.faultmodes}
        for compmode in compmodes:
            mdl = Tank()
            scen = {'Human': compmode}
            propagate.propagate(mdl,scen,1)
            self.assertIn(compmode, mdl.fxns['Human'].faults)
            self.assertIn(compmode, mdl.fxns['Human'].components[compname[compmode]].faults)
    def test_different_components(self):
        """ Tests that model copies have different components"""
        mdl=Tank()
        mdl_copy = mdl.copy()
        for compname, comp, in mdl.fxns['Human'].components.items():
            self.assertNotEqual(mdl_copy.fxns['Human'].components[compname],comp)
            self.assertNotEqual(mdl_copy.fxns['Human'].components[compname].__hash__(),comp.__hash__())
    def test_epc_math(self):
        """Spot check of epc math work in human error calculation"""
        mdl=Tank()
        ratecalc = 0.02 * ((4-1)*0.1+1)* ((4-1)*0.6+1)* ((1.1-1)*0.9+1)
        self.assertEqual(mdl.fxns['Human'].components['look'].failrate, ratecalc)
        
if __name__ == '__main__':
    unittest.main()
    
    mdl = Tank()
    scen = {'Human': 'NotDetected'}
    propagate.propagate(mdl,scen,1)    