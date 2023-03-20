# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 08:52:58 2021

@author: dhulse
"""
import unittest
from example_pump.pump_stochastic import Pump
from fmdtools.sim import propagate
import fmdtools.analyze as an
from fmdtools.define import SampleApproach, NominalApproach
from CommonTests import CommonTests
import numpy as np
import multiprocessing as mp

class StochasticPumpTests(unittest.TestCase, CommonTests):
    maxDiff=None
    def setUp(self):
        self.mdl = Pump()
    def test_stochastic_pdf(self):
        """Tests that (1) track_pdf option runs and (2) gives repeated probability density results under the same seed(s)"""
        testvals = [6.32173873679189, 10.56678004828189, 37.27832255997974, 15.163623336729625,  0.10687727708471972, 5.313965836539551, 54.568223188266394, 33.79026902541023, 7.636851629723572]
        for i in range(1,10):
            self.mdl.update_seed(i)
            self.mdl.propagate(i, run_stochastic='track_pdf')
            pd = self.mdl.return_probdens()
            self.assertAlmostEqual(pd, testvals[i-1])
    def test_run_safety(self):
        """ Tests that two models with the same seed will run the same and produce the same results"""
        for seed in [None, 1, 10, 209840]:
            mdl = Pump(modelparams = {'phases':{'start':[0,4], 'on':[5, 49], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':seed})
            seed0 = mdl.seed
            endresults_1, mdlhist_1=propagate.nominal(mdl, run_stochastic=True)
            endclasses_1, mdlhists_1=propagate.single_faults(mdl, run_stochastic=True, showprogress=False)
            if seed==None: seed=mdl.seed
            mdl2 = Pump(modelparams = {'phases':{'start':[0,4], 'on':[5, 49], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1,'seed':seed})
            seed1 = mdl2.seed
            endresults_2,  mdlhist_2=propagate.nominal(mdl2, run_stochastic=True)
            endclasses_2, mdlhists_2=propagate.single_faults(mdl2, run_stochastic=True, showprogress=False)
            seed2 = mdl.seed
            self.assertEqual(seed0, seed1,seed2)
            self.assertTrue(all(mdlhist_1['functions']['MoveWater']['eff']==mdlhist_2['functions']['MoveWater']['eff']))
            for scen in mdlhists_1:
                self.assertTrue(all(mdlhists_1[scen]['functions']['MoveWater']['eff']==mdlhists_2[scen]['functions']['MoveWater']['eff']))
    def test_run_approach(self):
         mdl = Pump()
         nomapp = NominalApproach()
         nomapp.add_seed_replicates('default',1000)
         endresults, mdlhists = propagate.nominal_approach(mdl, nomapp, showprogress=False, run_stochastic=True)
         ave_effs=[]; std_effs=[]
         for scen in mdlhists:
             ave_effs.append(np.mean(mdlhists[scen]['functions']['MoveWater']['eff']))
             std_effs.append(np.std(mdlhists[scen]['functions']['MoveWater']['eff']))
         ave_eff = np.mean(ave_effs); std_eff = np.mean(std_effs)
         self.assertAlmostEqual(ave_eff, mdl.fxns['MoveWater']._rng_params['eff'][2][0], 2) # test means
         self.assertLess(abs(std_eff-mdl.fxns['MoveWater']._rng_params['eff'][2][1]), 0.05)
         
    def test_model_copy_same(self):
        self.check_model_copy_same(Pump(), Pump(), [10,20,30], 25, max_time=55, run_stochastic=True)
    def test_model_copy_different(self):
        self.check_model_copy_different(Pump(), [10,20,30], max_time=55, run_stochastic=True)
    def test_model_reset(self):
        mdl = Pump(); mdl2 = Pump(); mdl2.seed=mdl.seed
        self.check_model_reset(mdl, mdl2, [10,20,30], max_time=55, run_stochastic=True)
    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.pkl", "stochpump_endclasses.pkl", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.csv", "stochpump_endclasses.csv", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.json", "stochpump_endclasses.json", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "pkl", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "csv", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "json", 'nominal_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.pkl", "stochpump_endclasses.pkl", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.csv", "stochpump_endclasses.csv", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach(self.mdl, "stochpump_mdlhists.json", "stochpump_endclasses.json", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "pkl", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "csv", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))
        self.check_save_load_approach_indiv(self.mdl, "stochpump_mdlhists", "stochpump_endclasses", "json", 'nested_approach', app=app, run_stochastic=True, pool=mp.Pool(4))

if __name__ == '__main__':
    #suite = unittest.TestSuite()
    #suite.addTest(StochasticPumpTests("test_stochastic_pdf"))
    #suite.addTest(StochasticPumpTests("test_save_load_nominalapproach"))
    #suite.addTest(StochasticPumpTests("test_save_load_nominalapproach_indiv"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    unittest.main()
