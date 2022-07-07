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
from fmdtools.modeldef import SampleApproach, check_pickleability, NominalApproach
from CommonTests import CommonTests
import numpy as np

class PumpTests(unittest.TestCase, CommonTests):
    def setUp(self):
        self.default_mdl = Pump()
        self.mdl = Pump()
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
        endclasses, mdlhists = propagate.approach(self.water_mdl, approach, showprogress=False)
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
    def test_approach_cost_calc(self):
        """Test that the (linear) resilience loss function is perfectly approximated 
        using the given sampling methods"""
        mdl = Pump(params={'cost':{'ee', 'repair', 'water'}, 'delay':0})
        app_full = SampleApproach(mdl, defaultsamp={'samp':'fullint'})
        full_util=exp_cost_quant(app_full,mdl)
        
        app_multipt = SampleApproach(mdl, defaultsamp={'samp':'evenspacing', 'numpts':3})
        multipt_util=exp_cost_quant(app_multipt,mdl)
        self.assertAlmostEqual(full_util, multipt_util)
        app_center = SampleApproach(mdl, defaultsamp={'samp':'evenspacing', 'numpts':1})
        center_util=exp_cost_quant(app_center,mdl)
        self.assertAlmostEqual(full_util, center_util)
        from scipy import integrate
        nodes, weights = integrate._quadrature._cached_roots_legendre(3)
        app_quad = SampleApproach(mdl, defaultsamp={'samp':'quadrature', 'quad':{'nodes':nodes,'weights':weights}})
        quad_util=exp_cost_quant(app_quad,mdl)
        self.assertAlmostEqual(full_util, quad_util)
    def test_approach_parallelism(self):
        """Test whether the pump simulates the same when simulated using parallel or staged options"""
        app = SampleApproach(self.default_mdl)
        self.check_approach_parallelism(self.default_mdl, app)
        app1 = SampleApproach(self.default_mdl, defaultsamp={'samp':'evenspacing','numpts':4})
        self.check_approach_parallelism(self.default_mdl, app1)
    def test_approach_pruning(self):
        """Tests that sample approach pruning places points in the center of their
        respective intervals for linear resilience loss functions."""
        mdl = Pump(params={'cost':{'ee', 'repair', 'water'}, 'delay':0})
        app_full = SampleApproach(mdl, defaultsamp={'samp':'fullint'})
        app_center = SampleApproach(mdl, defaultsamp={'samp':'evenspacing', 'numpts':1})
        endclasses, mdlhists = propagate.approach(mdl, app_full, showprogress=False)
        self.assertNotEqual(app_full.times, app_center.times)
        app_full.prune_scenarios(endclasses)
        self.assertEqual(app_full.times, app_center.times)
    def test_pickleability(self):
        unpickleable = check_pickleability(Pump(), verbose=False)
        self.assertTrue(unpickleable==[])
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
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'nominal')
    def test_save_load_onefault(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'one_fault', faultscen=('ExportWater', 'block', 25))
    def test_save_load_multfault(self):
        for extension in [".pkl",".csv",".json"]:
            faultscen = {10:{"ExportWater": ['block']},20:{"MoveWater":["short"]}}
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'mult_fault', faultscen =faultscen )
    def test_save_load_singlefaults(self):
        self.check_save_load_approach(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl", 'single_faults')
        self.check_save_load_approach(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv", 'single_faults')
        self.check_save_load_approach(self.mdl, "pump_mdlhists.json", "pump_endclasses.json", 'single_faults')
    def test_save_load_singlefaults_indiv(self):
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl", 'single_faults')
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv", 'single_faults')
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json", 'single_faults')
    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl", 'nominal_approach', app=app)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv", 'nominal_approach', app=app)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.json", "pump_endclasses.json", 'nominal_approach', app=app)
    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl", 'nominal_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv", 'nominal_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json", 'nominal_approach', app=app)
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl", 'nested_approach', app=app)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv", 'nested_approach', app=app)
        self.check_save_load_approach(self.mdl, "pump_mdlhists.json", "pump_endclasses.json", 'nested_approach', app=app)
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl", 'nested_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv", 'nested_approach', app=app)
        self.check_save_load_approach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json", 'nested_approach', app=app)
    def test_save_load_approach(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.pkl", "pump_endclasses.pkl", 'approach', app=app)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.csv", "pump_endclasses.csv", 'approach', app=app)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.json", "pump_endclasses.json", 'approach', app=app)
    def test_save_load_approach_indiv(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "pkl", 'approach', app=app)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "csv", 'approach', app=app)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "json", 'approach', app=app)
    def test_fmea_options(self):
        app = SampleApproach(self.water_mdl, faults=[('MoveWater','mech_break')], phases=['on'],defaultsamp={'samp':'evenspacing','numpts':5})
        endclasses, mdlhists = propagate.approach(self.water_mdl, app, showprogress=False)
        self.check_same_fmea(app, endclasses, self.water_mdl)
        
        app2 = SampleApproach(self.mdl, defaultsamp={'samp':'evenspacing','numpts':2})
        endclasses2, mdlhists2 = propagate.approach(self.mdl, app2, showprogress=False)
        self.check_same_fmea(app2, endclasses2, self.mdl)
        
def exp_cost_quant(approach, mdl):
    """ Calculates the expected cost of faults over a given sampling approach 
    on the given model"""
    endclasses, mdlhists = propagate.approach(mdl, approach, showprogress=False)
    reshists, diffs, summaries = rd.process.hists(mdlhists)
    fmea = rd.tabulate.summfmea(endclasses, approach)
    util=sum(fmea['expected cost'])
    return util

if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(PumpTests("test_fmea_options"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    
    
