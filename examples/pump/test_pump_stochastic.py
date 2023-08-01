# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 08:52:58 2021

@author: dhulse
"""
import unittest
from examples.pump.pump_stochastic import Pump, paramfunc
from fmdtools.sim import propagate
from fmdtools.sim.approach import NominalApproach
from tests.common import CommonTests
import numpy as np
import multiprocessing as mp

class StochasticPumpTests(unittest.TestCase, CommonTests):
    maxDiff=None
    def setUp(self):
        self.mdl = Pump()
    def test_stochastic_pdf(self):
        """Tests that (1) track_pdf option runs and (2) gives repeated 
        probability density results under the same seed(s)"""
        testvals = [23.427857638009993,
                    28.879844891816045,
                    0.0009482614961342181,
                    10.9180526929105,
                    0.02603536236159078,
                    1.9736111095116995,
                    31.579024323385077,
                    0.016549021807197088,
                    6.303651266980214]
        for i in range(1,10):
            self.mdl.update_seed(i)
            self.mdl.propagate(i, run_stochastic='track_pdf')
            pd = self.mdl.return_probdens()
            #print(pd)
            self.assertAlmostEqual(pd, testvals[i-1])
    def test_run_safety(self):
        """ Tests that two models with the same seed will run the same and produce the same results"""
        for seed in [1, 10, 209840]:
            mdl = Pump(r = {'seed':seed})
            endresults_1, mdlhist_1 = propagate.nominal(mdl, run_stochastic=True)
            endclasses_1, mdlhists_1 = propagate.single_faults(mdl, run_stochastic=True, showprogress=False)
            if seed==None: 
                seed=mdl.r.seed
            mdl2 = Pump(r = {'seed':seed})
            endresults_2,  mdlhist_2 = propagate.nominal(mdl2, run_stochastic=True)
            endclasses_2, mdlhists_2 = propagate.single_faults(mdl2, run_stochastic=True, showprogress=False)
            self.assertTrue(all(mdlhist_1.fxns.move_water.s.eff==mdlhist_2.fxns.move_water.s.eff))
            for val in mdlhists_1:
                self.assertTrue(all(mdlhists_1.get(val)==mdlhists_2.get(val)))
    def test_set_seeds(self):
        for seed in [1, 10, 209840]:
            mdl = Pump(r = {'seed':seed})
            mdl2 = Pump()
            mdl2.update_seed(seed)
            self.assertEqual(seed, mdl.r.seed, mdl2.r.seed)
    def test_run_approach(self):
         mdl = Pump()
         nomapp = NominalApproach()
         nomapp.add_seed_replicates('default',1000)
         endresults, mdlhists = propagate.nominal_approach(mdl, nomapp, showprogress=False, run_stochastic=True, track={'fxns':{'move_water':"r"}}, desired_result={})
         ave_effs=[]; std_effs=[]
         for scen in nomapp.scenarios:
             ave_effs.append(np.mean(mdlhists.get(scen).fxns.move_water.r.s.eff))
             std_effs.append(np.std(mdlhists.get(scen).fxns.move_water.r.s.eff))
         ave_eff = np.mean(ave_effs); std_eff = np.mean(std_effs)
         self.assertAlmostEqual(ave_eff, mdl.fxns['move_water'].r.s.eff_update[1][0], 2) # test means
         self.assertLess(abs(std_eff-mdl.fxns['move_water'].r.s.eff_update[1][1]), 0.05)
         
    def test_model_copy_same(self):
        self.check_model_copy_same(Pump(), Pump(), [10,20,30], 25, max_time=55, run_stochastic=True)
    def test_model_copy_different(self):
        self.check_model_copy_different(Pump(), [10,20,30], max_time=55, run_stochastic=True)
    def test_model_reset(self):
        mdl = Pump(); mdl2 = Pump(); mdl2.r.seed=mdl.r.seed
        self.check_model_reset(mdl, mdl2, [10,20,30], max_time=55, run_stochastic=True)
    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nomapproach(self.mdl, "stochpump_mdlhists.pkl", "stochpump_endclasses.pkl", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nomapproach(self.mdl, "stochpump_mdlhists.csv", "stochpump_endclasses.csv", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nomapproach(self.mdl, "stochpump_mdlhists.json", "stochpump_endclasses.json", app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nomapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "pkl", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nomapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "csv", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nomapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "json", app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nestapproach(self.mdl, "stochpump_mdlhists.pkl", "stochpump_endclasses.pkl", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nestapproach(self.mdl, "stochpump_mdlhists.csv", "stochpump_endclasses.csv", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nestapproach(self.mdl, "stochpump_mdlhists.json", "stochpump_endclasses.json", app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nestapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "pkl", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nestapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "csv", app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_nestapproach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "json", app=app, run_stochastic=True, pool=mp.Pool(4))

def check_nominal_vals():
    """tests nominal_vals_1d"""
    from fmdtools import analyze as an
    mdl = Pump()
    app = NominalApproach()
    app.add_seed_replicates('test_seeds', 100)
    endres, mdlhists = propagate.nominal_approach(mdl, app, run_stochastic=True)
    endres['test_seeds_1.endclass.cost'] = 10.0
    #an.plot.nominal_vals_1d(app, endres, 'r.seed', metric="nonsense")
    title = "should show at least one red line over range of seeds"
    an.plot.nominal_vals_1d(app, endres, 'r.seed', metric='cost', faultalpha=1.0,
                            title=title)
    title = "should show single line over nonexistent range"
    an.plot.nominal_vals_1d(app, endres, 'nonsense', title=title)

def check_nominal_2d():
    """tests nominal_vals_1d"""
    from fmdtools import analyze as an
    mdl = Pump()
    app = NominalApproach()
    app.add_param_replicates(paramfunc, 'delay_1', 10, (1))
    app.add_param_replicates(paramfunc, 'delay_10', 10, (10))
    app.add_param_replicates(paramfunc, 'delay_10', 20, (15))
    
    endres, mdlhists = propagate.nominal_approach(mdl, app, run_stochastic=True)
    endres['delay_10_20.endclass.cost'] = 10.0
    title = "should show at least one red x over range of seeds, probs, and delay=1, 10"
    an.plot.nominal_vals_2d(app, endres, 'r.seed', 'p.delay',
                            metric="cost", title=title)
    
    an.plot.nominal_vals_3d(app, endres, 'r.seed', 'p.delay', 'prob',
                            metric="cost", title=title)

    

if __name__ == '__main__':
    #suite = unittest.TestSuite()
    #suite.addTest(StochasticPumpTests("test_run_safety"))
    #suite.addTest(StochasticPumpTests("test_run_approach"))
    
    #suite.addTest(StochasticPumpTests("test_save_load_nominalapproach"))
    #suite.addTest(StochasticPumpTests("test_save_load_nominalapproach_indiv"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    #unittest.main()
    
    check_nominal_2d()
    
    #check_nominal_vals()