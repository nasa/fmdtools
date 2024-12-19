#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

from examples.multirotor.drone_mdl_rural import Drone, DroneParam

from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import ParameterSimProblem, SingleFaultScenarioProblem
from fmdtools.sim.search import DisturbanceProblem
import fmdtools.sim.propagate as propagate

import numpy as np
import unittest

bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']


def bat_var_map(x):
    return (bats[int(x)], )


linarchs = ['quad', 'hex', 'oct']


def line_arch_map(x):
    return (linarchs[int(x)], )


expd1 = ParameterDomain(DroneParam)
expd1.add_variable("phys_param.bat", var_map=bat_var_map, var_lim=(0, 3))
expd1.add_variable("phys_param.linearch", var_map=line_arch_map, var_lim=(0, 2))


ex_soc_opt = ParameterSimProblem(Drone(track=None), expd1, 'nominal')
ex_soc_opt.add_result_objective("f1", "store_ee.s.soc", time=16)
ex_soc_opt.add_result_constraint("g1", "store_ee.s.soc", time=16)


class DroneParameterTests(unittest.TestCase):
    def test_param_domain_1(self):
        """Test that mapped parameters become expected output parameters."""
        p_0_0 = expd1(0, 0)
        self.assertEqual(p_0_0.phys_param.bat, 'monolithic')
        self.assertEqual(p_0_0.phys_param.linearch, 'quad')

    def test_param_domain_2(self):
        """Test that mapped parameters become expected output parameters."""
        p_1_2 = expd1(1, 2)
        self.assertEqual(p_1_2.phys_param.bat, 'series-split')
        self.assertEqual(p_1_2.phys_param.linearch, 'oct')

    def test_set_constraints_1(self):
        """Test that mapped parameters result in expected set constraints."""
        sc = expd1.get_set_constraints(0, 0)
        self.assertFalse(sc[0])
        self.assertFalse(sc[1])

    def test_set_constraints_2(self):
        """Test that mapped parameters result in expected set constraints."""
        sc1 = expd1.get_set_constraints(4, 1)
        self.assertTrue(sc1[0])
        self.assertFalse(sc1[1])

    def test_sim_mdl(self):
        """Test that Problem tracking options are used (only needed hist/res gotten)."""
        res, hist = ex_soc_opt.sim_mdl(1, 1)
        self.assertEqual(len(res), 1)
        self.assertTrue('t16p0.store_ee.s.soc' in res)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist.time[-1], 16.0)

    def test_objectives(self):
        """Test that increasing architecture weight decreases soc objective."""
        oldobj = np.inf
        for i in range(2):
            obj = ex_soc_opt.f1(0, i)
            self.assertLess(obj, oldobj)
            oldobj = obj


mdl = Drone()

sp = SingleFaultScenarioProblem(mdl, ("affect_dof", "rf_propwarp"),
                                sim_start=2.0, track=None)
sp.add_result_objective("f1", "plan_path.t.time", time=15)
sp.add_result_objective("f2", "dofs.s.x", time=15)


class DroneScenarioTest(unittest.TestCase):
    """Tests to check that SingleFaultScenarioProblem works as expected."""
    time = 4

    def setUp(self):
        res, self.hist = propagate.nominal(mdl)

    def test_scenprob_results(self):
        """Test that the objective lines up with expected results at the given time."""
        # fault should cause the final x (f5) to match the nominal x at the fault time
        x_nominal = self.hist.flows.dofs.s.x[self.time]
        x_failure = sp.f2(self.time)
        self.assertEqual(x_nominal, x_failure)
        # time objective should be at 15
        f_time = sp.f1(self.time)
        self.assertEqual(f_time, 15.0)


class DroneScenarioTest2(DroneScenarioTest):
    time = 5


class DroneScenarioTest3(DroneScenarioTest):
    time = 6


sp2 = DisturbanceProblem(mdl, 5.0, "store_ee.ca.comps.s1p1.s.soc", track=None)
sp2.add_result_objective("f1", "store_ee.s.soc", time=10)
sp2.add_result_objective("f2", "store_ee.s.soc", time=5)


class DroneDisturbanceTest(unittest.TestCase):

    def test_disturbance_set(self):
        """Test that disturbance objectives set and call properly."""
        # make sure the disturbance is set (note that we correct by amt b/c of use)
        soc_set = 10.0
        amt = mdl.fxns['store_ee'].ca.comps['s1p1'].p.amt
        soc_expected = soc_set - 100/amt
        soc_res = sp2.f2(soc_set)
        self.assertEqual(soc_expected, soc_res)

        # make sure the disturbance wasn't set permanently
        soc_later = sp2.f1(soc_set)
        self.assertLess(soc_later, soc_res)






if __name__ == '__main__':
    unittest.main()
