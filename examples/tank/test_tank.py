#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of the tank model.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from examples.tank.tank_model import Tank

from tests.common import CommonTests

from fmdtools.sim import propagate as prop
from fmdtools.sim.sample import FaultDomain, FaultSample, ParameterSample

import unittest


class TankTests(unittest.TestCase, CommonTests):
    def setUp(self):
        self.maxDiff = None
        self.mdl = Tank()
        self.fd = FaultDomain(self.mdl)
        self.fd.add_all()
        self.fs = FaultSample(self.fd)
        self.fs.add_fault_phases()
        self.fs1 = FaultSample(self.fd)
        self.fs1.add_fault_phases(args=(5,))
        self.ps = ParameterSample()
        self.ps.add_variable_replicates([], replicates=10)
        self.faultdomains = {'fd': (('all', ), {})}
        self.faultsamples = {'fs': (('fault_phases', 'fd'), {})}

    def test_model_copy_same(self):
        self.check_model_copy_same(Tank(), Tank(), [5, 10, 15], 10, max_time=20)

    def test_model_copy_different(self):
        self.check_model_copy_different(Tank(), [5, 10, 15], max_time=20)

    def test_tank_copy_args(self):
        mdl_cop = self.mdl.copy()
        mdl_cop_2 = mdl_cop.copy()

        self.assertEqual(self.mdl.fxns['human'].aa.acts['detect'].duration, 2)
        self.assertEqual(mdl_cop.fxns['human'].aa.acts['detect'].duration, 2)
        self.assertEqual(mdl_cop_2.fxns['human'].aa.acts['detect'].duration, 2)

    def test_approach(self):
        res, hists = prop.fault_sample(self.mdl, self.fs,
                                      track="all", showprogress=False)
        for scen in self.fs.scenarios():
            seq = scen.sequence
            name = scen.name
            res, hist = prop.sequence(self.mdl, seq=seq)
            faulthist = hist.faulty
            self.check_same_hist(faulthist, hists.get(name), "approach")

    @unittest.skip('Reset not fully implemented yet and unused throughout.')
    def test_model_reset(self):
        mdl = Tank()
        mdl2 = Tank()
        self.check_model_reset(mdl, mdl2, [5, 10, 15], max_time=20)

    def test_approach_parallelism_notrack(self):
        """Test whether the pump simulates the same when simulated using parallel or
        staged options."""
        self.check_fs_parallel(self.mdl, self.fs, track="default")

    def test_approach_parallelism_0(self):
        """Test whether the pump simulates the same when simulated using parallel or
        staged options."""
        self.check_fs_parallel(self.mdl, self.fs)

    def test_approach_parallelism_1(self):
        self.check_fs_parallel(self.mdl, self.fs1)

    def test_comp_mode_inj(self):
        """ Tests that action modes injected in functions end up in their respective
        actions."""
        mdl = Tank()
        amodes = [aname+"_"+mode for aname, a in mdl.fxns['human'].aa.acts.items()
                  for mode in a.m.faultmodes]
        fmodes = [*mdl.fxns['human'].m.faultmodes.keys()]
        self.assertListEqual(amodes, fmodes)

        anames = {mode: aname for aname, a in mdl.fxns['human'].aa.acts.items()
                  for mode in a.m.faultmodes}
        for amode, aname in anames.items():
            mdl = Tank()
            scen = {'human': aname+"_"+amode}
            mdl.propagate(1, fxnfaults=scen)
            self.assertIn(aname+"_"+amode, mdl.fxns['human'].m.faults)
            self.assertIn(amode, mdl.fxns['human'].aa.acts[aname].m.faults)

    def test_different_components(self):
        """Tests that model copies have different components."""
        mdl = Tank()
        mdl_copy = mdl.copy()
        for aname, act, in mdl.fxns['human'].aa.acts.items():
            self.assertNotEqual(mdl_copy.fxns['human'].aa.acts[aname], act)
            self.assertNotEqual(
                mdl_copy.fxns['human'].aa.acts[aname].__hash__(), act.__hash__())

    def test_local_tstep(self):
        """Tests running the model with a different local timestep in the
        Store_Liquid function."""
        mdl_global = Tank(sp={'phases': (('na', 0, 0),
                                         ('operation', 1, 20)),
                              'end_time': 20.0, 'dt': 1.0,
                              'units': 'min', 'use_local': False})
        _, hist_global = prop.one_fault(mdl_global, 'store_water', 'leak', time=2)
        hist_global = hist_global.flatten()

        mdl_loc_low = Tank(p={'reacttime': 2, 'store_tstep': 0.1})
        _, hist_loc_low = prop.one_fault(mdl_loc_low, 'store_water', 'leak', time=2)
        hist_loc_low = hist_loc_low.flatten()

        self.compare_results(hist_global, hist_loc_low)

        mdl_loc_high = Tank(p={'reacttime': 2, 'store_tstep': 3.0})
        _, hist_loc_high = prop.one_fault(mdl_loc_high, 'store_water', 'leak', time=2)
        hist_loc_high = hist_loc_high.flatten()
        for i in [2, 5, 8, 12]:
            slice_global = hist_global.get_slice(i)
            slice_loc_high = hist_loc_high.get_slice(i)
            self.compare_results(slice_global, slice_loc_high)

    def test_epc_math(self):
        """Spot check of epc math work in human error calculation"""
        mdl = Tank()
        ratecalc = 0.02 * ((4-1)*0.1+1) * ((4-1)*0.6+1) * ((1.1-1)*0.9+1)
        self.assertEqual(mdl.fxns['human'].aa.acts['look'].m.failrate, ratecalc)

    def test_save_load_nominal(self):
        for extension in [".npz", ".csv", ".json"]:
            fname = "tank_hist"+extension, "tank_res"+extension
            self.check_onerun_save(self.mdl, 'nominal', *fname)

    def test_save_load_onefault(self):
        for extension in [".npz", ".csv", ".json"]:
            faultscen = ('import_water', 'stuck', 5)
            fname = "tank_hist"+extension, "tank_res"+extension
            self.check_onerun_save(self.mdl, 'one_fault', *fname, faultscen=faultscen)

    def test_save_load_multfault(self):
        for extension in [".npz", ".csv", ".json"]:
            faultscen = {5: {"import_water": ['stuck']}, 10: {"store_water": ["leak"]}}
            fname = "tank_hist"+extension, "tank_res"+extension
            self.check_onerun_save(self.mdl, 'sequence', *fname, faultscen=faultscen)

    def test_save_load_singlefaults(self):
        self.check_sf_save(self.mdl, "tank_res.npz", "tank_hists.npz")
        self.check_sf_save(self.mdl, "tank_res.csv", "tank_hists.csv")
        self.check_sf_save(self.mdl, "tank_res.json", "tank_hists.json")

    def test_singlefaults_isave(self):
        indiv_names = ("tank_res", "tank_hist")
        self.check_sf_isave(self.mdl, *indiv_names, "npz")
        self.check_sf_isave(self.mdl, *indiv_names, "csv")
        self.check_sf_isave(self.mdl, *indiv_names, "json")

    def test_param_sample_save(self):
        self.check_ps_save(self.mdl, self.ps, "tank_res.npz", "tank_hists.npz")
        self.check_ps_save(self.mdl, self.ps, "tank_res.csv", "tank_hists.csv")
        self.check_ps_save(self.mdl, self.ps, "tank_res.json", "tank_hists.json")

    def test_param_sample_save(self):
        indiv_names = ("tank_res", "tank_hist")
        self.check_ps_isave(self.mdl, self.ps, *indiv_names, "npz")
        self.check_ps_isave(self.mdl, self.ps, *indiv_names, "csv")
        self.check_ps_isave(self.mdl, self.ps, *indiv_names, "json")

    def test_nested_sample_save(self):
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "tank_res.npz", "tank_hists.npz")
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "tank_res.csv", "tank_hists.csv")
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "tank_res.json", "tank_hists.json")

    def test_nested_sample_isave(self):
        indiv_names = ("tank_res", "tank_hist")
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *indiv_names, "npz")
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *indiv_names, "csv")
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *indiv_names, "json")

    def test_fault_sample_save(self):
        self.check_fs_save(self.mdl, self.fs, "tank_res.npz", "tank_hists.npz")
        self.check_fs_save(self.mdl, self.fs, "tank_res.csv", "tank_hists.csv")
        self.check_fs_save(self.mdl, self.fs, "tank_res.json", "tank_hists.json")

    def test_fault_sample_isave(self):
        indiv_names = ("tank_res", "tank_hist")
        self.check_fs_isave(self.mdl, self.fs, *indiv_names, "npz")
        self.check_fs_isave(self.mdl, self.fs, *indiv_names, "csv")
        self.check_fs_isave(self.mdl, self.fs, *indiv_names, "json")


def check_parallel():
    """Informal test setup for checking that parallel execution is consistent."""
    mdl = Tank()
    fd = FaultDomain(mdl)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_phases(args=(4,))
    import multiprocessing as mp
    print("normal")
    res, hist = prop.fault_sample(mdl, fs, showprogress=False, track='all', staged=True)
    print("staged")
    res_stage, hist_stage = prop.fault_sample(mdl, fs, showprogress=False,
                                              track='all', staged=True)

    assert res == res_stage
    print("parallel")
    res_par, hist_par = prop.fault_sample(mdl, fs, showprogress=False, pool=mp.Pool(4),
                                          staged=False, track='all')

    assert res == res_par
    print("staged-parallel")
    res_par_staged, hist_par_staged = prop.fault_sample(mdl, fs,showprogress=False,
                                                        pool=mp.Pool(4), staged=True,
                                                        track='all')

    hist_diff = hist.get_different(hist_par_staged)
    res_diff = res.get_different(res_par_staged)

    hist_diff.guide_water_out_leak_t0p0.flows.wat_in_2.s.effort

    #mc_diff.guide_water_in_leak_t0p0.flows.wat_in_2.s.effort

    hist_diff.human_detect_false_low_t16p0.fxns.human.aa.active_actions[16]

    assert res == res_par_staged


if __name__ == '__main__':
    import sys
    sys.path.append("../..")
    # NOTE: reset expected not to work since args are no longer being saved

    # suite = unittest.TestSuite()
    # suite.addTest(TankTests("test_model_reset"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    # suite = unittest.TestSuite()
    # suite.addTest(TankTests("test_model_copy_same"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

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
