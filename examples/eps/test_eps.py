#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of the EPS Model.

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
from examples.eps.eps import EPS

from tests.common import CommonTests

from fmdtools.sim import propagate as prop
from fmdtools.sim.sample import FaultDomain, FaultSample
from fmdtools.define.object.base import check_pickleability

import unittest
import numpy as np
import math

class epsTests(unittest.TestCase, CommonTests):
    def setUp(self):
        self.mdl = EPS()
        self.fd = FaultDomain(self.mdl)
        self.fd.add_all()

    def test_backward_fault_prop_1(self):
        """Tests that defined fault cases that require reverse propagation propagate
        backwards through the graph as expected - distributor short leads to empty battery
        """
        res, hist = prop.one_fault(self.mdl, "distribute_ee", "short",
                                   to_return="faults")
        self.assertEqual(res.get_faulty().tend.faults,
                         ['store_ee.no_storage', 'distribute_ee.short'])

    def test_backward_fault_prop_2(self):
        """Tests that defined fault cases that require reverse propagation propagate
        backwards through the graph as expected - motor short leads to distributor short
        """
        res, hist = prop.one_fault(self.mdl, "ee_to_me", "short",
                                   to_return="faults")
        self.assertEqual(res.get_faulty().tend.faults,
                         ['store_ee.no_storage', 'distribute_ee.short', 'ee_to_me.short'])

    def test_all_faults(self):
        """Some basic tests for propagating lists of faults in the model--
        that histories have length 1, endresults have >0 costs, and total costs are higher
        than repairs"""
        mdl = self.mdl
        res, hist = prop.single_faults(mdl, showprogress=False)
        actual_num_faults = np.sum([len(f.m.get_all_faultnames()) for f in mdl.fxns.values()
                                    if hasattr(f, 'm')])
        self.assertEqual(len(res.nest(1)), actual_num_faults + 1)
        hist_len_is_1 = all([len(v) == 1 for v in hist.values()])
        self.assertTrue(hist_len_is_1)  # all histories have length 1
        all_have_costs = all([v > 0 for k, v in res.get_values('.cost').items()
                              if 'nominal' not in k])
        # all endresults have positive costs
        self.assertTrue(all_have_costs)
        repcosts = np.sum([np.sum([m["cost"] for m in f.m.get_faults().values()])
                           for f in mdl.fxns.values() if hasattr(f, 'm')])
        # fault costs higher than if it was just repairs
        total_simcosts = sum([v for v in res.get_values('.cost').values()])
        self.assertGreater(total_simcosts, repcosts)

    def test_fault_app(self):
        """Tests that the expected number of scenarios are generated for a given
        approach"""
        tot_faults = [len(f.m.get_all_faultnames()) for f in self.mdl.fxns.values()
                      if hasattr(f, 'm')]
        actual_num_faults = int(np.sum(tot_faults))
        for n_joint in [2, 3, actual_num_faults]:
            fs = FaultSample(self.fd)
            fs.add_fault_phases(n_joint=n_joint)
            # tests the length
            self.assertEqual(len(fs.scenarios()), math.comb(actual_num_faults, n_joint))

    def test_pickleability(self):
        unpickleable = check_pickleability(self.mdl, verbose=False)
        self.assertTrue(unpickleable == [])

    def test_nominal_saving(self):
        for extension in [".npz", ".csv", ".json"]:
            resname = "eps_hist" + extension
            histname = "eps_res" + extension
            self.check_onerun_save(self.mdl, 'nominal', resname, histname)

    def test_save_load_onefault(self):
        faultscen = ("store_ee", "no_storage", 0)
        for extension in [".npz", ".csv", ".json"]:
            resname = "eps_hist" + extension
            histname = "eps_res" + extension
            self.check_onerun_save(self.mdl, 'one_fault', resname, histname,
                                   faultscen=faultscen)

    def test_multfault_saving(self):
        faultscen = {0: {"store_ee": ["no_storage"], "distribute_ee": "short"}}
        for extension in [".npz", ".csv", ".json"]:
            resname = "eps_hist" + extension
            histname = "eps_res" + extension
            self.check_onerun_save(self.mdl, "sequence", resname, histname,
                                   faultscen=faultscen)

    def test_save_load_singlefaults(self):
        self.check_sf_save(self.mdl, "eps_res.npz", "eps_hist.npz")
        self.check_sf_save(self.mdl, "eps_res.csv", "eps_hist.csv")
        self.check_sf_save(self.mdl, "eps_res.json", "eps_hist.json")


if __name__ == "__main__":
    import sys
    sys.path.append("../..")

    # suite = unittest.TestSuite()
    # suite.addTest(epsTests("test_fault_app"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    unittest.main()

    # mdl = EPS()


    # endresults, resgraph, mdlhist = propagate.one_fault(mdl, 'Distribute_EE', 'short')

    # mdl = EPS()
    # endresults, resgraph, mdlhist = propagate.one_fault(mdl, 'EE_to_ME', 'short')

    # mdl_nom = EPS()
    # endresults_nom, resgraph_nom, mdlhist_nom = propagate.nominal(mdl_nom)

    # results, hists = propagate.single_faults(mdl)
