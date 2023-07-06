# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:49:13 2021

@author: dhulse
"""
import os
import unittest
from examples.pump.ex_pump import Pump
from fmdtools.sim import propagate
import fmdtools.analyze as an
from fmdtools.define.common import check_pickleability
from fmdtools.sim.approach import SampleApproach, NominalApproach
from tests.common import CommonTests
import numpy as np
from fmdtools.analyze.result import load, load_folder, Result, History


class PumpTests(unittest.TestCase, CommonTests):
    """Overall test structure for Pump model"""
    def setUp(self):
        self.default_mdl = Pump()
        self.mdl = Pump()
        self.water_mdl = Pump(p={'cost':('water',), 'delay':10})
    def test_value_setting(self):
        statenames = ['import_ee.m.mode', 'sig_1.s.power', 'move_water.s.eff']
        newvalues = ['newmode', 20, 0.1]
        self.check_var_setting(self.mdl, statenames, newvalues)
    def test_value_setting_dict(self):
        dict_to_check ={'import_ee.m.mode':'thismode', 'wat_2.s.area':0.0}
        self.check_var_setting_dict(self.mdl, dict_to_check)
    def test_dynamic_prop_values(self):
        """Test that given fault times result in the expected water/value loss"""
        faulttimes = [10,20,30]
        for faulttime in faulttimes:
            endresults, mdlhist = propagate.one_fault(self.water_mdl, "move_water", "mech_break", time=faulttime)
            expected_wcost = self.expected_water_cost(faulttime)
            self.assertAlmostEqual(expected_wcost, endresults.endclass.cost)
    def test_dynamic_prop_values_2(self):
        """Test that the delayed fault behavior occurs at the time specified"""
        delays = [0, 1, 5, 10]
        for delay in delays:
            mdl = Pump(p={'cost':('water',), 'delay':delay})
            endresults, mdlhist = propagate.one_fault(mdl, 'export_water', 'block', time=25, track="all")
            has_fault_at_time = mdlhist.faulty.fxns.move_water.m.faults.mech_break[25+delay]
            self.assertEqual(has_fault_at_time, 1)
            has_fault_before_time = mdlhist.faulty.fxns.move_water.m.faults.mech_break[25+delay-1]
            self.assertEqual(has_fault_before_time, 0)
    def test_app_prop_values(self):
        """Test that the delayed fault behavior occurs at the time specified (with approach)"""
        approach = SampleApproach(self.water_mdl, faults=[('move_water','mech_break')], phases=['on'],defaultsamp={'samp':'evenspacing','numpts':5})
        endclasses, mdlhists = propagate.approach(self.water_mdl, approach, showprogress=False)
        for scen in approach.scenlist:
            exp_wcost = self.expected_water_cost(scen.time)
            self.assertAlmostEqual(exp_wcost, endclasses.get(scen.name).endclass.cost)
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
        mdl = Pump(p={'cost':('ee', 'repair', 'water'), 'delay':0})
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
        mdl = Pump(p={'cost':('ee', 'repair', 'water'), 'delay':0})
        app_full = SampleApproach(mdl, phases= ["on"], defaultsamp={'samp':'fullint'})
        app_center = SampleApproach(mdl, phases=["on"], defaultsamp={'samp':'evenspacing', 'numpts':1})
        endclasses, mdlhists = propagate.approach(mdl, app_full, showprogress=False)
        self.assertNotEqual(app_full.times, app_center.times)
        app_full.prune_scenarios(endclasses)
        self.assertEqual(app_full.times, app_center.times)
    def test_pickleability(self):
        unpickleable = check_pickleability(Pump(), verbose=False)
        self.assertTrue(unpickleable==[])
    def test_one_run_pickle(self):
        if os.path.exists("single_fault.pkl"): os.remove("single_fault.pkl")
        
        endresults, mdlhist=propagate.one_fault(self.mdl, 'export_water','block', time=20, staged=False, run_stochastic=True, sp={'seed':10})
        
        mdlhist.save("single_fault.pkl")
        mdlhist_saved = load("single_fault.pkl", renest_dict=False)
        self.assertCountEqual([*mdlhist.keys()], [*mdlhist_saved.keys()])
        
        for hist_key in mdlhist: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist[hist_key],mdlhist_saved[hist_key])
            
        mdlhist.faulty.time[0]=100
        self.assertNotEqual(mdlhist.faulty.time[0], mdlhist_saved.faulty.time[0])
        
        os.remove("single_fault.pkl")
        
    def test_one_run_csv(self):
        if os.path.exists("single_fault.csv"): os.remove("single_fault.csv")
        endresults, mdlhist=propagate.one_fault(self.mdl, 'export_water','block', time=20, staged=False, run_stochastic=True, sp={'seed':10})
        mdlhist.save("single_fault.csv")
        mdlhist_saved = load("single_fault.csv", renest_dict=False, Rclass=History)
        self.assertCountEqual([*mdlhist.keys()], [*mdlhist_saved.keys()])
        
        for hist_key in mdlhist: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist[hist_key],mdlhist_saved[hist_key])
        os.remove("single_fault.csv")
    def test_one_run_json(self):
        if os.path.exists("single_fault.json"): os.remove("single_fault.json")
        endresults, mdlhist=propagate.one_fault(self.mdl, 'export_water','block', time=20, staged=False, run_stochastic=True, sp={'seed':10})
        mdlhist.save("single_fault.json")
        
        mdlhist_saved = load("single_fault.json", Rclass=History)
        
        self.assertCountEqual([*mdlhist.keys()], [*mdlhist_saved.keys()])
        for hist_key in mdlhist: # test to see that all values of the arrays in the hist are the same
            np.testing.assert_array_equal(mdlhist[hist_key],mdlhist_saved[hist_key])
        os.remove("single_fault.json")
    def test_save_load_nominal(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'nominal')
    def test_save_load_onefault(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'one_fault', faultscen=('export_water', 'block', 25))
    def test_save_load_multfault(self):
        for extension in [".pkl",".csv",".json"]:
            faultscen = {10:{"export_water": ['block']},20:{"move_water":["short"]}}
            self.check_save_load_onerun(self.mdl, "pump_mdlhist"+extension, "pump_endclass"+extension, 'sequence', faultscen =faultscen )
    def test_save_load_singlefaults(self):
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl")
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv")
        self.check_save_load_singlefaults(self.mdl, "pump_mdlhists.json", "pump_endclasses.json")
    def test_save_load_singlefaults_indiv(self):
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl")
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv")
        self.check_save_load_singlefaults_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json")
    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nomapproach(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl", app=app)
        self.check_save_load_nomapproach(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv", app=app)
        self.check_save_load_nomapproach(self.mdl, "pump_mdlhists.json", "pump_endclasses.json", app=app)
    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nomapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl", app=app)
        self.check_save_load_nomapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv", app=app)
        self.check_save_load_nomapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json",app=app)
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nestapproach(self.mdl, "pump_mdlhists.pkl", "pump_endclasses.pkl", app=app)
        self.check_save_load_nestapproach(self.mdl, "pump_mdlhists.csv", "pump_endclasses.csv", app=app)
        self.check_save_load_nestapproach(self.mdl, "pump_mdlhists.json", "pump_endclasses.json", app=app)
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nestapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "pkl", app=app)
        self.check_save_load_nestapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "csv", app=app)
        self.check_save_load_nestapproach_indiv(self.mdl, "pump_mdlhists", "pump_endclasses", "json", app=app)
    def test_save_load_approach(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.pkl", "pump_endclasses.pkl", app=app)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.csv", "pump_endclasses.csv", app=app)
        self.check_save_load_approach(self.mdl,"pump_mdlhists.json", "pump_endclasses.json", app=app)
    def test_save_load_approach_indiv(self):
        app = SampleApproach(self.mdl)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "pkl", app=app)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "csv", app=app)
        self.check_save_load_approach_indiv(self.mdl,"pump_mdlhists", "pump_endclasses", "json", app=app)
    def test_fmea_options(self):
        app = SampleApproach(self.water_mdl, faults=[('move_water','mech_break')], phases=['on'],defaultsamp={'samp':'evenspacing','numpts':5})
        endclasses, mdlhists = propagate.approach(self.water_mdl, app, showprogress=False)
        self.check_same_fmea(app, endclasses, self.water_mdl)
        
        app2 = SampleApproach(self.mdl, defaultsamp={'samp':'evenspacing','numpts':2})
        endclasses2, mdlhists2 = propagate.approach(self.mdl, app2, showprogress=False)
        self.check_same_fmea(app2, endclasses2, self.mdl)
        
def exp_cost_quant(approach, mdl):
    """ Calculates the expected cost of faults over a given sampling approach 
    on the given model"""
    result, mdlhists = propagate.approach(mdl, approach, showprogress=False)
    fmea = an.tabulate.summfmea(result, approach)
    util=sum(fmea['expected cost'])
    return util

if __name__ == '__main__':
    unittest.main()
    
    #suite = unittest.TestSuite()
    #suite.addTest(PumpTests("test_approach_cost_calc"))
    #suite.addTest(PumpTests("test_value_setting_dict"))
    #suite.addTest(PumpTests("test_one_run_csv"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    
    
