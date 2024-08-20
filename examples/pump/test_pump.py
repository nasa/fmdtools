#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of the pump model.

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


from examples.pump.ex_pump import Pump, PumpParam
from examples.pump.pump_indiv import MoveWatDynamic

from tests.common import CommonTests

from fmdtools.define.object.base import check_pickleability
from fmdtools.sim import propagate as prop
from fmdtools.sim.sample import ParameterDomain, ParameterSample
from fmdtools.sim.sample import FaultDomain, FaultSample, ParameterSample
from fmdtools.analyze.result import load
from fmdtools.analyze.history import History
from fmdtools.analyze import tabulate

import os
import unittest
import numpy as np


class PumpTests(unittest.TestCase, CommonTests):
    """Overall test structure for Pump model"""

    def setUp(self):
        self.default_mdl = Pump()
        self.mdl = Pump()
        self.water_mdl = Pump(p={'cost': ('water',), 'delay': 10})
        self.fd = FaultDomain(self.mdl)
        self.fd.add_all()
        self.fs = FaultSample(self.fd)
        self.fs.add_fault_phases()
        self.faultdomains = {'fd': (('all', ), {})}
        self.faultsamples = {'fs': (('fault_phases', 'fd'), {})}
        self.ps = ParameterSample()
        self.ps.add_variable_replicates([], replicates=10)
        self.filenames = ("pump_res", "pump_hist")

    def test_hist_tracking_setup(self):
        """Test that tracking args set up history keys as expected."""
        # default track tracks wat_2.s.flowrate, ee_1.s.current, i.on, i.finished
        mdl_def = Pump()
        self.assertEqual(len(mdl_def.h.keys()), 4)
        def_keys = {'flows.ee_1.s.current', 'flows.wat_2.s.flowrate',
                    'i.finished', 'i.on'}
        self.assertEqual(set(mdl_def.h.keys()), def_keys)
        # sending track arguments should overwrite
        track_arg = {'flows': {'ee_1': 'all', "wat_1": {'s': ('flowrate',)}}}
        to_track = ['flows.ee_1.s.current', 'flows.ee_1.s.voltage',
                    'flows.wat_1.s.flowrate']
        mdl_cust = Pump(track=track_arg)
        self.assertEqual(set(mdl_cust.h.keys()), set(to_track))
        # at the very least, tracking all should mean there are a lot more keys
        mdl_all = Pump(track='all')
        self.assertGreater(len(mdl_all.h.keys()), 25)

    def test_param_sample(self):
        pd = ParameterDomain(PumpParam)
        pd.add_variable("delay")
        pd.add_constant("cost", ('repair', 'water'))
        ps = ParameterSample(pd)
        ps.add_variable_replicates([], replicates=10)
        # test that 10 replicates = 10 unique seeds
        self.assertEqual(len(set([p.r['seed'] for p in ps.scenarios()])), 10)
        # test that all have delay of 10 (same params)
        self.assertEqual(set([p.p['delay'] for p in ps.scenarios()]), {10})

    def test_value_setting(self):
        statenames = ['sig_1.s.power', 'move_water.s.eff']
        newvalues = [20, 0.1]
        self.check_var_setting(self.mdl, statenames, newvalues)

    def test_value_setting_dict(self):
        dict_to_check = {'wat_2.s.area': 0.0}
        self.check_var_setting_dict(self.mdl, dict_to_check)

    def test_dynamic_prop_values(self):
        """Test that given fault times result in the expected water/value loss"""
        faulttimes = [10, 20, 30]
        for faulttime in faulttimes:
            res, hist = prop.one_fault(self.water_mdl, "move_water", "mech_break",
                                       time=faulttime)
            expected_wcost = self.expected_water_cost(faulttime)
            self.assertAlmostEqual(expected_wcost, res.endclass.cost)

    def test_dynamic_prop_values_2(self):
        """Test that the delayed fault behavior occurs at the time specified"""
        delays = [0, 1, 5, 10]
        for delay in delays:
            mdl = Pump(p={'cost': ('water',), 'delay': delay}, track='all')
            res, hist = prop.one_fault(mdl, 'export_water', 'block', time=25)
            fault_at_time = hist.faulty.fxns.move_water.m.faults.mech_break[25+delay]
            self.assertEqual(fault_at_time, 1)
            fault_bef_time = hist.faulty.fxns.move_water.m.faults.mech_break[25+delay-1]
            self.assertEqual(fault_bef_time, 0)

    def test_app_prop_values(self):
        """Test that delayed fault behavior occurs at the time specified by sample."""
        fd = FaultDomain(self.mdl)
        fd.add_fault('move_water', 'mech_break')
        fs = FaultSample(fd)
        fs.add_fault_phases("on", args=(5,))

        res, hist = prop.fault_sample(self.water_mdl, fs, showprogress=False)
        for scen in fs.scenarios():
            exp_wcost = self.expected_water_cost(scen.time)
            self.assertAlmostEqual(exp_wcost, res.get(scen.name).endclass.cost)

    def expected_water_cost(self, faulttime):
        return (50 - faulttime) * 0.3 * 750

    def test_model_copy_same(self):
        inj_times = [10, 20, 30, 40]
        self.check_model_copy_same(self.mdl, Pump(), inj_times, 30, max_time=55)

    def test_model_copy_different(self):
        inj_times = [10, 20, 30, 40]
        self.check_model_copy_different(self.mdl, inj_times, max_time=55)

    @unittest.skip('Reset not fully implemented yet and unused throughout.')
    def test_model_reset(self):
        inj_times = [10, 20, 30, 40]
        self.check_model_reset(self.mdl, Pump(), inj_times, max_time=55)

    def test_approach_cost_calc(self):
        """Test that the (linear) resilience loss function is perfectly approximated
        using the given sampling methods"""
        mdl = Pump(p={'cost': ('ee', 'repair', 'water'), 'delay': 0})
        fs_full = FaultSample(self.fd)
        fs_full.add_fault_phases(method='all')
        full_util = exp_cost_quant(fs_full, mdl)

        fs_multipt = FaultSample(self.fd)
        fs_multipt.add_fault_phases(args=(3,))
        multipt_util = exp_cost_quant(fs_multipt, mdl)
        self.assertAlmostEqual(full_util, multipt_util, places=2)

        fs_center = FaultSample(self.fd)
        fs_center.add_fault_phases(args=(1,))
        center_util = exp_cost_quant(fs_center, mdl)
        self.assertAlmostEqual(full_util, center_util, places=2)
        from scipy import integrate
        nodes, weights = integrate._quadrature._cached_roots_legendre(3)
        fs_quad = FaultSample(self.fd)
        fs_quad.add_fault_phases(method='quad', args=(nodes, weights))
        quad_util = exp_cost_quant(fs_quad, mdl)
        self.assertAlmostEqual(full_util, quad_util, places=2)

    def test_approach_parallelism(self):
        """Test whether the pump simulates the same when simulated using parallel or
        staged options"""
        self.check_fs_parallel(self.default_mdl, self.fs)
        fs = FaultSample(self.fd)
        fs.add_fault_phases(args=(4,))
        self.check_fs_parallel(self.default_mdl, fs)

    def test_pickleability(self):
        unpickleable = check_pickleability(Pump(), verbose=False)
        self.assertTrue(unpickleable == [])

    def test_one_run_pickle(self):
        if os.path.exists("single_fault.npz"):
            os.remove("single_fault.npz")

        res, hist = prop.one_fault(self.mdl, 'export_water', 'block', time=20,
                                   staged=False, run_stochastic=True, sp={'seed': 10})

        hist.save("single_fault.npz")
        hist_saved = load("single_fault.npz", renest_dict=False)
        self.assertCountEqual([*hist.keys()], [*hist_saved.keys()])
        # test to see that all values of the arrays in the hist are the same
        for hist_key in hist:
            np.testing.assert_array_equal(hist[hist_key], hist_saved[hist_key])

        hist.faulty.time[0] = 100
        self.assertNotEqual(hist.faulty.time[0], hist_saved.faulty.time[0])

        os.remove("single_fault.npz")

    def test_one_run_csv(self):
        if os.path.exists("single_fault.csv"):
            os.remove("single_fault.csv")
        res, hist = prop.one_fault(self.mdl, 'export_water', 'block', time=20,
                                   staged=False, run_stochastic=True, sp={'seed': 10})
        hist.save("single_fault.csv")
        hist_saved = load("single_fault.csv", renest_dict=False, Rclass=History)
        self.assertCountEqual([*hist.keys()], [*hist_saved.keys()])
        # test to see that all values of the arrays in the hist are the same
        for hist_key in hist:
            np.testing.assert_array_equal(hist[hist_key], hist_saved[hist_key])
        os.remove("single_fault.csv")

    def test_one_run_json(self):
        if os.path.exists("single_fault.json"):
            os.remove("single_fault.json")

        res, hist = prop.one_fault(self.mdl, 'export_water', 'block', time=20,
                                   staged=False, run_stochastic=True, sp={'seed': 10})
        hist.save("single_fault.json")
        hist_saved = load("single_fault.json", Rclass=History)

        self.assertCountEqual([*hist.keys()], [*hist_saved.keys()])
        # test to see that all values of the arrays in the hist are the same
        for hist_key in hist:
            np.testing.assert_array_equal(hist[hist_key], hist_saved[hist_key])
        os.remove("single_fault.json")

    def test_nominal_save(self):
        for ext in [".npz", ".csv", ".json"]:
            fnames = "pump_res" + ext, "pump_hist" + ext
            self.check_onerun_save(self.mdl, "nominal", *fnames)

    def test_onefault_save(self):
        faultscen = ('export_water', 'block', 25)
        for ext in [".npz", ".csv", ".json"]:
            fnames = "pump_res" + ext, "pump_hist" + ext
            self.check_onerun_save(self.mdl, 'one_fault', *fnames, faultscen=faultscen)

    def test_save_load_multfault(self):
        faultscen = {10: {"export_water": ['block']}, 20: {"move_water": ["short"]}}
        for ext in [".npz", ".csv", ".json"]:
            fnames = "pump_res" + ext, "pump_hist" + ext
            self.check_onerun_save(self.mdl, 'sequence', *fnames, faultscen=faultscen)

    def test_single_faults_save(self):
        self.check_sf_save(self.mdl, "pump_res.npz", "pump_hist.npz")
        self.check_sf_save(self.mdl, "pump_res.csv", "pump_hist.csv",)
        self.check_sf_save(self.mdl, "pump_res.json", "pump_hist.json")

    def test_single_faults_isave(self):
        self.check_sf_isave(self.mdl, *self.filenames, "npz")
        self.check_sf_isave(self.mdl, *self.filenames, "csv")
        self.check_sf_isave(self.mdl, *self.filenames, "json")

    def test_param_sample_save(self):
        self.check_ps_save(self.mdl, self.ps, "pump_res.npz", "pump_hist.npz")
        self.check_ps_save(self.mdl, self.ps, "pump_res.csv", "pump_hist.csv")
        self.check_ps_save(self.mdl, self.ps, "pump_res.json", "pump_hist.json")

    def test_param_sample_isave(self):
        self.check_ps_isave(self.mdl, self.ps, *self.filenames, "npz")
        self.check_ps_isave(self.mdl, self.ps, *self.filenames, "csv")
        self.check_ps_isave(self.mdl, self.ps, *self.filenames, "json")

    def test_nested_sample_save(self):
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "pump_res.npz", "pump_hist.npz")
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "pump_res.csv", "pump_hist.csv")
        self.check_ns_save(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                           "pump_res.json", "pump_hist.json")

    def test_nested_sample_isave(self):
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *self.filenames, "npz")
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *self.filenames, "csv")
        self.check_ns_isave(self.mdl, self.ps, self.faultdomains, self.faultsamples,
                            *self.filenames, "json")

    def test_fault_sample_save(self):
        self.check_fs_save(self.mdl, self.fs, "pump_res.npz", "pump_hist.npz")
        self.check_fs_save(self.mdl, self.fs, "pump_res.csv", "pump_hist.csv")
        self.check_fs_save(self.mdl, self.fs, "pump_res.json", "pump_hist.json")

    def test_fault_sample_isave(self):
        self.check_fs_isave(self.mdl, self.fs, *self.filenames, "npz")
        self.check_fs_isave(self.mdl, self.fs, *self.filenames, "csv")
        self.check_fs_isave(self.mdl, self.fs, *self.filenames, "json")

    def test_fmea_options(self):
        fd = FaultDomain(self.mdl)
        fd.add_fault('move_water', 'mech_break')
        fs = FaultSample(fd)
        fs.add_fault_phases("on", args=(5,))

        ec, mdlhists = prop.fault_sample(self.water_mdl, fs, showprogress=False)

        fs2 = FaultSample(fd)
        fs2.add_fault_phases()
        ec2, hist2 = prop.fault_sample(self.mdl, fs2, showprogress=False)


def exp_cost_quant(fs, mdl):
    """ Calculate expected cost of faults over a faultsample for the model."""

    result, mdlhists = prop.fault_sample(mdl, fs, showprogress=False)
    fmea = tabulate.FMEA(result, fs)
    util = fmea.as_table()['expected_cost'].sum()
    return util


class IndivPumpTests(unittest.TestCase):
    """Unit tests for individual pump model."""

    def setUp(self):
        self.mdl = MoveWatDynamic()

    def test_mutable_setup(self):
        """Check that non-default state carries through to simulation."""
        mdl_diff = MoveWatDynamic(s={'eff': 2.0})
        self.assertEqual(mdl_diff.s.eff, 2.0)
        res, hist = prop.nominal(mdl_diff, showprogress=False, warn_faults=False)
        # should sim with eff = 2.0
        self.assertEqual(hist.s.eff[0], 2.0)
        # after it turns on at t=5, should break at t=6 (due to delay)
        self.assertEqual(hist.m.faults.mech_break[6], True)

if __name__ == '__main__':
    unittest.main()

    # suite = unittest.TestSuite()
    # suite.addTest(IndivPumpTests("test_mutable_setup"))
    # suite.addTest(PumpTests("test_model_copy_same"))
    # suite.addTest(PumpTests("test_value_setting_dict"))
    # suite.addTest(PumpTests("test_one_run_csv"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
