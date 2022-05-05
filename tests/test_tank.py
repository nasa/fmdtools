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
    def setUp(self):
        self.mdl = Tank()
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
    def test_save_load_nominal(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_onerun(self.mdl, "tank_mdlhist"+extension, "tank_endclass"+extension, 'nominal')
    def test_save_load_onefault(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_onerun(self.mdl, "tank_mdlhist"+extension, "tank_endclass"+extension, 'one_fault', faultscen=('Import_Water', 'stuck', 5))
    def test_save_load_multfault(self):
        for extension in [".pkl",".csv",".json"]:
            faultscen = {5:{"Import_Water": ['stuck']},10:{"Store_Water":["Leak"]}}
            self.check_save_load_onerun(self.mdl, "tank_mdlhist"+extension, "tank_endclass"+extension, 'mult_fault', faultscen =faultscen )
    def test_save_load_singlefaults(self):
        self.check_save_load_approach(self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", 'single_faults')
        self.check_save_load_approach(self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", 'single_faults')
        self.check_save_load_approach(self.mdl, "tank_mdlhists.json", "tank_endclasses.json", 'single_faults')
    def test_save_load_singlefaults_indiv(self):
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "pkl", 'single_faults')
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "csv", 'single_faults')
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "json", 'single_faults')
    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", 'nominal_approach', app=app)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", 'nominal_approach', app=app)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.json", "tank_endclasses.json", 'nominal_approach', app=app)
    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "pkl", 'nominal_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "csv", 'nominal_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "json", 'nominal_approach', app=app)
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", 'nested_approach', app=app)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", 'nested_approach', app=app)
        self.check_save_load_approach(self.mdl, "tank_mdlhists.json", "tank_endclasses.json", 'nested_approach', app=app)
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "pkl", 'nested_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "csv", 'nested_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "tank_mdlhists", "tank_endclasses", "json", 'nested_approach', app=app)
    def test_save_load_approach(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach(self.mdl,"tank_mdlhists.pkl", "tank_endclasses.pkl", 'approach', app=app)
        self.check_save_load_approach(self.mdl,"tank_mdlhists.csv", "tank_endclasses.csv", 'approach', app=app)
        self.check_save_load_approach(self.mdl,"tank_mdlhists.json", "tank_endclasses.json", 'approach', app=app)
    def test_save_load_approach_indiv(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach_indiv(self.mdl,"tank_mdlhists", "tank_endclasses", "pkl", 'approach', app=app)
        self.check_save_load_approach_indiv(self.mdl,"tank_mdlhists", "tank_endclasses", "csv", 'approach', app=app)
        self.check_save_load_approach_indiv(self.mdl,"tank_mdlhists", "tank_endclasses", "json", 'approach', app=app)

if __name__ == '__main__':
    unittest.main()
    
    mdl = Tank()
    scen = {'Human': 'NotDetected'}
    propagate.propagate(mdl,scen,1)    