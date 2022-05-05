# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:58:33 2022

@author: dhulse
"""
from ex_pump import Pump
import sys, os
sys.path.append(os.path.join('..'))


from fmdtools.modeldef import *
from tests.CommonTests import CommonTests
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as propagate





import unittest
import numpy as np

class SaveLoadTests(unittest.TestCase, CommonTests):
    #maxDiff=None
    def setUp(self):
        """Instantiate Function and Connected Flows for Tests"""
        self.mdl = Pump()
        
    def test_one_run_pickle(self):
        if os.path.exists("single_fault.pkl"): os.remove("single_fault.pkl")
        
        endresults, resgraph, mdlhist=propagate.one_fault(self.mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})
        mdlhist_flattened = rd.process.flatten_hist(mdlhist)
        rd.process.save_result(mdlhist, "single_fault.pkl")
        mdlhist_saved = rd.process.load_result("single_fault.pkl")
        mdlhist_saved_flattened = rd.process.flatten_hist(mdlhist_saved)
        
        self.assertCountEqual([*mdlhist_flattened.keys()], [*mdlhist_saved_flattened.keys()])
        
        for hist_key in mdlhist_flattened: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist_flattened[hist_key],mdlhist_saved_flattened[hist_key])
            
        mdlhist_flattened['faulty', 'time'][0]=100
        self.assertNotEqual(mdlhist_flattened['faulty', 'time'][0], mdlhist_saved_flattened['faulty', 'time'][0])
        
        os.remove("single_fault.pkl")
        
    def test_one_run_csv(self):
        if os.path.exists("single_fault.csv"): os.remove("single_fault.csv")
        endresults, resgraph, mdlhist=propagate.one_fault(self.mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})
        mdlhist_flattened = rd.process.flatten_hist(mdlhist)
        
        rd.process.save_result(mdlhist, "single_fault.csv")
        mdlhist_saved = rd.process.load_result("single_fault.csv")
        mdlhist_saved_flattened = rd.process.flatten_hist(mdlhist_saved)
        
        self.assertCountEqual([*mdlhist_flattened.keys()], [*mdlhist_saved_flattened.keys()])
        for hist_key in mdlhist_flattened: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist_flattened[hist_key],mdlhist_saved_flattened[hist_key])
        os.remove("single_fault.csv")
    def test_one_run_json(self):
        if os.path.exists("single_fault.json"): os.remove("single_fault.json")
        endresults, resgraph, mdlhist=propagate.one_fault(self.mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})
        mdlhist_flattened = rd.process.flatten_hist(mdlhist)
        
        rd.process.save_result(mdlhist, "single_fault.json")
        mdlhist_saved = rd.process.load_result("single_fault.json")
        mdlhist_saved_flattened = rd.process.flatten_hist(mdlhist_saved)
        
        self.assertCountEqual([*mdlhist_flattened.keys()], [*mdlhist_saved_flattened.keys()])
        for hist_key in mdlhist_flattened: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist_flattened[hist_key],mdlhist_saved_flattened[hist_key])
        os.remove("single_fault.json")
    def test_save_load_nominal(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_nominal(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension)
    def test_save_load_singlefaults(self):
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl")
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv")
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.json", "pump_endclasses.json")
    def test_save_load_singlefaults_indiv(self):
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl")
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv")
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json")
    def test_save_load_approach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nominal_approach(self.mdl, app, "pump_mdlhists.pkl", "pump_endclasses.pkl")
        self.check_save_load_nominal_approach(self.mdl, app, "pump_mdlhists.csv", "pump_endclasses.csv")
        self.check_save_load_nominal_approach(self.mdl, app, "pump_mdlhists.json", "pump_endclasses.json")
    def test_save_load_approach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nominal_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "pkl")
        self.check_save_load_nominal_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "csv")
        self.check_save_load_nominal_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "json")
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nested_approach(self.mdl, app, "pump_mdlhists.pkl", "pump_endclasses.pkl")
        self.check_save_load_nested_approach(self.mdl, app, "pump_mdlhists.csv", "pump_endclasses.csv")
        self.check_save_load_nested_approach(self.mdl, app, "pump_mdlhists.json", "pump_endclasses.json")
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nested_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "pkl")
        self.check_save_load_nested_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "csv")
        self.check_save_load_nested_approach_indiv(self.mdl, app, "pump_mdlhists", "pump_endclasses", "json")
        
# nominal approach

# resilience approach

# nested approach

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(SaveLoadTests("test_save_load_nestedapproach_indiv"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    #unittest.main()
        