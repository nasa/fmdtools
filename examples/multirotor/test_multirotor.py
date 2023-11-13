# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:51:18 2022

@author: dhulse
"""
import unittest
from examples.multirotor.drone_mdl_rural import Drone, DroneParam
from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import ParameterProblem
import numpy as np

bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']

def bat_var_map(x):
    return (bats[x], )

linarchs = ['quad', 'hex', 'oct']

def line_arch_map(x):
    return (linarchs[x], )

expd1 = ParameterDomain(DroneParam)
expd1.add_variable("phys_param.bat", var_map=bat_var_map, var_lim=(0, 3))
expd1.add_variable("phys_param.linearch", var_map=line_arch_map, var_lim=(0, 2))


ex_soc_opt = ParameterProblem(Drone(), expd1, 'nominal', track=None)
ex_soc_opt.add_result_objective("f1", "store_ee.s.soc", time=16)



class DroneTests(unittest.TestCase):
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
        res, hist = ex_soc_opt.sim_mdl(1,1)
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


if __name__ == '__main__':
    unittest.main()

