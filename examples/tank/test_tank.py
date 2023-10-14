# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:51:57 2021

@author: dhulse
"""
import unittest
from examples.tank.tank_model import Tank
from fmdtools.sim import propagate
from fmdtools.sim.approach import NominalApproach
from fmdtools.sim.sample import FaultDomain, FaultSample
from tests.common import CommonTests


class TankTests(unittest.TestCase, CommonTests):
    def setUp(self):
        self.mdl = Tank()
        self.fd = FaultDomain(self.mdl)
        self.fd.add_all()
        self.fs = FaultSample(self.fd)
        self.fs.add_fault_phases()
        self.fs1 = FaultSample(self.fd)
        self.fs1.add_fault_phases(args=(5,))

    def test_model_copy_same(self):
        self.check_model_copy_same(Tank(), Tank(), [5, 10, 15], 10, max_time=20)

    def test_model_copy_different(self):
        self.check_model_copy_different(Tank(), [5, 10, 15], max_time=20)

    def test_tank_copy_args(self):
        mdl_cop = self.mdl.copy()
        mdl_cop_2 = mdl_cop.copy()

        assert self.mdl.fxns['human']._args_aa == mdl_cop.fxns['human']._args_aa
        assert self.mdl.fxns['human']._args_aa == mdl_cop_2.fxns['human']._args_aa

        self.assertEqual(self.mdl.fxns['human'].aa.actions['detect'].duration, 2)
        self.assertEqual(mdl_cop.fxns['human'].aa.actions['detect'].duration, 2)
        self.assertEqual(mdl_cop_2.fxns['human'].aa.actions['detect'].duration, 2)

    def test_approach(self):
        endresults, mdlhists = propagate.approach(
            self.mdl, self.fs, track="all", showprogress=False)
        for scen in self.fs.scenarios():
            seq = scen.sequence
            name = scen.name
            endresult, mdlhist = propagate.sequence(self.mdl, seq=seq)
            faulthist = mdlhist.faulty
            self.check_same_hist(faulthist, mdlhists.get(name), "approach")

    def test_model_reset(self):
        mdl = Tank()
        mdl2 = Tank()
        self.check_model_reset(mdl, mdl2, [5, 10, 15], max_time=20)

    def test_approach_parallelism_notrack(self):
        """Test whether the pump simulates the same when simulated using parallel or
        staged options"""
        self.check_approach_parallelism(self.mdl, self.fs, track="default")

    def test_approach_parallelism_0(self):
        """Test whether the pump simulates the same when simulated using parallel or
        staged options"""
        self.check_approach_parallelism(self.mdl, self.fs)

    def test_approach_parallelism_1(self):
        self.check_approach_parallelism(self.mdl, self.fs1)

    def test_comp_mode_inj(self):
        """ Tests that action modes injected in functions end up in their respective
        actions."""
        mdl = Tank()
        amodes = [aname+"_"+mode for aname, a in mdl.fxns['human'].aa.actions.items()
                  for mode in a.m.faultmodes]
        fmodes = [*mdl.fxns['human'].m.faultmodes.keys()]
        self.assertListEqual(amodes, fmodes)

        anames = {mode: aname for aname, a in mdl.fxns['human'].aa.actions.items()
                  for mode in a.m.faultmodes}
        for amode, aname in anames.items():
            mdl = Tank()
            scen = {'human': aname+"_"+amode}
            mdl.propagate(1, fxnfaults=scen)
            self.assertIn(aname+"_"+amode, mdl.fxns['human'].m.faults)
            self.assertIn(amode, mdl.fxns['human'].aa.actions[aname].m.faults)

    def test_different_components(self):
        """ Tests that model copies have different components"""
        mdl = Tank()
        mdl_copy = mdl.copy()
        for aname, act, in mdl.fxns['human'].aa.actions.items():
            self.assertNotEqual(mdl_copy.fxns['human'].aa.actions[aname], act)
            self.assertNotEqual(
                mdl_copy.fxns['human'].aa.actions[aname].__hash__(), act.__hash__())

    def test_local_tstep(self):
        """ Tests running the model with a different local timestep in the
        Store_Liquid function"""
        mdl_global = Tank(sp={'phases': (('na', 0, 0),
                                         ('operation', 1, 20)),
                              'times': (0, 5, 10, 15, 20), 'dt': 1.0,
                              'units': 'min', 'use_local': False})
        _, mdlhist_global = propagate.one_fault(
            mdl_global, 'store_water', 'leak', time=2)
        mdlhist_global = mdlhist_global.flatten()

        mdl_loc_low = Tank(p={'reacttime': 2, 'store_tstep': 0.1})
        _, mdlhist_loc_low = propagate.one_fault(mdl_loc_low,
                                                 'store_water', 'leak', time=2)
        mdlhist_loc_low = mdlhist_loc_low.flatten()

        self.compare_results(mdlhist_global, mdlhist_loc_low)

        mdl_loc_high = Tank(p={'reacttime': 2, 'store_tstep': 3.0})
        _, mdlhist_loc_high = propagate.one_fault(
            mdl_loc_high, 'store_water', 'leak', time=2)
        mdlhist_loc_high = mdlhist_loc_high.flatten()
        for i in [2, 5, 8, 12]:
            slice_global = mdlhist_global.get_slice(i)
            slice_loc_high = mdlhist_loc_high.get_slice(i)
            self.compare_results(slice_global, slice_loc_high)

    def test_epc_math(self):
        """Spot check of epc math work in human error calculation"""
        mdl = Tank()
        ratecalc = 0.02 * ((4-1)*0.1+1) * ((4-1)*0.6+1) * ((1.1-1)*0.9+1)
        self.assertEqual(mdl.fxns['human'].aa.actions['look'].m.failrate, ratecalc)

    def test_save_load_nominal(self):
        for extension in [".pkl", ".csv", ".json"]:
            self.check_save_load_onerun(self.mdl,
                                        "tank_mdlhist"+extension,
                                        "tank_endclass"+extension,
                                        'nominal')

    def test_save_load_onefault(self):
        for extension in [".pkl", ".csv", ".json"]:
            self.check_save_load_onerun(self.mdl,
                                        "tank_mdlhist"+extension,
                                        "tank_endclass"+extension,
                                        'one_fault',
                                        faultscen=('import_water', 'stuck', 5))

    def test_save_load_multfault(self):
        for extension in [".pkl", ".csv", ".json"]:
            faultscen = {5: {"import_water": ['stuck']}, 10: {"store_water": ["leak"]}}
            self.check_save_load_onerun(self.mdl,
                                        "tank_mdlhist"+extension,
                                        "tank_endclass"+extension,
                                        'sequence',
                                        faultscen=faultscen)

    def test_save_load_singlefaults(self):
        self.check_save_load_singlefaults(self.mdl,
                                          "tank_mdlhists.pkl",
                                          "tank_endclasses.pkl")
        self.check_save_load_singlefaults(self.mdl,
                                          "tank_mdlhists.csv",
                                          "tank_endclasses.csv")
        self.check_save_load_singlefaults(self.mdl,
                                          "tank_mdlhists.json",
                                          "tank_endclasses.json")

    def test_save_load_singlefaults_indiv(self):
        indiv_names = ("tank_mdlhists", "tank_endclasses")
        self.check_save_load_singlefaults_indiv(self.mdl, *indiv_names, "pkl")
        self.check_save_load_singlefaults_indiv(self.mdl, *indiv_names, "csv")
        self.check_save_load_singlefaults_indiv(self.mdl, *indiv_names, "json")

    def test_save_load_nominalapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nomapproach(
            self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", app=app)
        self.check_save_load_nomapproach(
            self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", app=app)
        self.check_save_load_nomapproach(
            self.mdl, "tank_mdlhists.json", "tank_endclasses.json", app=app)

    def test_save_load_nominalapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        indiv_names = ("tank_mdlhists", "tank_endclasses")
        self.check_save_load_nomapproach_indiv(self.mdl, *indiv_names, "pkl", app=app)
        self.check_save_load_nomapproach_indiv(self.mdl, *indiv_names, "csv", app=app)
        self.check_save_load_nomapproach_indiv(self.mdl, *indiv_names, "json", app=app)

    def test_save_load_nestedapproach(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        self.check_save_load_nestapproach(
            self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", app=app)
        self.check_save_load_nestapproach(
            self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", app=app)
        self.check_save_load_nestapproach(
            self.mdl, "tank_mdlhists.json", "tank_endclasses.json", app=app)

    def test_save_load_nestedapproach_indiv(self):
        app = NominalApproach()
        app.add_seed_replicates("replicates", 10)
        indiv_names = ("tank_mdlhists", "tank_endclasses")
        self.check_save_load_nestapproach_indiv(self.mdl, *indiv_names, "pkl", app=app)
        self.check_save_load_nestapproach_indiv(self.mdl, *indiv_names, "csv", app=app)
        self.check_save_load_nestapproach_indiv(self.mdl, *indiv_names, "json", app=app)

    def test_save_load_approach(self):
        self.check_save_load_approach(
            self.mdl, "tank_mdlhists.pkl", "tank_endclasses.pkl", app=self.fs)
        self.check_save_load_approach(
            self.mdl, "tank_mdlhists.csv", "tank_endclasses.csv", app=self.fs)
        self.check_save_load_approach(
            self.mdl, "tank_mdlhists.json", "tank_endclasses.json", app=self.fs)

    def test_save_load_approach_indiv(self):
        indiv_names = ("tank_mdlhists", "tank_endclasses")
        self.check_save_load_approach_indiv(self.mdl, *indiv_names, "pkl", app=self.fs)
        self.check_save_load_approach_indiv(self.mdl, *indiv_names, "csv", app=self.fs)
        self.check_save_load_approach_indiv(self.mdl, *indiv_names, "json", app=self.fs)


def check_parallel():
    """Informal test setup for checking that parallel execution is working/consistent"""
    mdl = Tank()
    fd = FaultDomain(mdl)
    fd.add_all()
    app = FaultSample(fd)
    app.add_fault_phases(args=(4,))
    import multiprocessing as mp
    print("normal")
    endclasses, mdlhists = propagate.approach(
        mdl, app, showprogress=False, track='all', staged=True)
    print("staged")
    endclasses_staged, mdlhists_staged = propagate.approach(
        mdl, app, showprogress=False, track='all', staged=True)

    assert endclasses == endclasses_staged
    print("parallel")
    endclasses_par, mdlhists_par = propagate.approach(
        mdl, app, showprogress=False, pool=mp.Pool(4), staged=False, track='all')

    assert endclasses == endclasses_par
    print("staged-parallel")
    endclasses_par_staged, mdlhists_par_staged = propagate.approach(
        mdl, app, showprogress=False, pool=mp.Pool(4), staged=True, track='all')
    print("staged-parallel")
    endclasses_par_staged, mdlhists_par_staged = propagate.approach(
        mdl, app, showprogress=False, pool=mp.Pool(4), staged=True, track='all')

    mc_diff = mdlhists.get_different(mdlhists_par_staged)
    ec_diff = endclasses.get_different(endclasses_par_staged)

    mc_diff.guide_water_out_leak_t0p0.flows.wat_in_2.s.effort

    #mc_diff.guide_water_in_leak_t0p0.flows.wat_in_2.s.effort

    mc_diff.human_detect_false_low_t16p0.fxns.human.aa.active_actions[16]

    assert endclasses == endclasses_par_staged


if __name__ == '__main__':
    
    #suite = unittest.TestSuite()
    #suite.addTest(TankTests("test_local_tstep"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    
    #suite = unittest.TestSuite()
    #suite.addTest(TankTests("test_save_load_approach"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    
    #suite = unittest.TestSuite()
    #suite.addTest(TankTests("test_approach_parallelism_notrack"))
    #suite.addTest(TankTests("test_approach_parallelism_0"))
    #suite.addTest(TankTests("test_approach_parallelism_1"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    
    unittest.main()
    
    #mdl = Tank()
    #scen = {'human': 'NotDetected'}
    #mdl.propagate(scen,1)    
