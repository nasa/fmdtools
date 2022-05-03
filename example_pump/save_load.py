# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:58:33 2022

@author: dhulse
"""
from ex_pump import Pump
import sys, os
sys.path.append(os.path.join('..'))


from fmdtools.modeldef import *
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as propagate



mdl = Pump()

# nominal, single-fault, mult-fault
endresults, resgraph, mdlhist=propagate.one_fault(mdl, 'ExportWater','block', time=20, staged=False, run_stochastic=True, modelparams={'seed':10})


# pickle, CSV, json
mdlhist_flattened = rd.process.flatten_hist(mdlhist)

mdlhist_renested = rd.process.nest_flattened_hist(mdlhist_flattened)

#os.remove("single_fault.csv")
if os.path.exists("single_fault.json"): os.remove("single_fault.json")
rd.process.save_result(mdlhist, "single_fault.json")
mdlhist_loaded = rd.process.load_result("single_fault.json", renest_dict=False)
mdlhist_loaded_nest = rd.process.load_result("single_fault.json")

#mdlhist_saved['faulty']['time'][0]=100
#mdlhist['faulty']['time']==mdlhist_saved['faulty']['time']


import unittest
import numpy as np

class SaveLoadTests(unittest.TestCase):
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
        
        
# nominal approach

# resilience approach

# nested approach

if __name__ == '__main__':
    unittest.main()
        