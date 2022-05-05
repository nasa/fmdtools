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
from fmdtools.modeldef import SampleApproach, check_pickleability
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
        from multiprocessing import Pool
        endclasses, mdlhists = propagate.approach(self.default_mdl, app, showprogress=False,pool=False)
        endclasses_staged, mdlhist_staged = propagate.approach(self.default_mdl, app, showprogress=False,pool=False, staged=True)
        self.assertEqual([*endclasses.values()], [*endclasses_staged.values()])
        endclasses_par, mdlhists_par = propagate.approach(self.default_mdl, app, showprogress=False,pool=Pool(4), staged=False)
        self.assertEqual([*endclasses.values()], [*endclasses_par.values()])
        endclasses_staged_par, mdlhists_staged_par = propagate.approach(self.default_mdl, app, showprogress=False,pool=Pool(4), staged=True)
        self.assertEqual([*endclasses.values()], [*endclasses_staged_par.values()])
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
    def test_save_load_nominal(self):
        for extension in [".pkl",".csv",".json"]:
            self.check_save_load_nominal(self.default_mdl, "pump_mdlhists"+extension, "pump_endclasses"+extension)
        
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
    
    
