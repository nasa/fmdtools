# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:51:18 2022

@author: dhulse
"""
import unittest
from examples.multirotor.drone_mdl_opt import Drone, opt_prob, x_to_rcost, x_to_ocost
import multiprocessing as mp

class DroneTests(unittest.TestCase):
    def test_interface_values(self):
        
        testvalues = [[0,0, 50, 0,0], [0,2, 100, 1,1],[2,2, 150, 1,1]]
        # NOTE: because there is a fault in the nominal sim that triggers the resilience policy
        # the value [2,2, 50, 0,0] will give inconsistent results, since the operational model
        # doesn't have a consistent resilience policy in that case
        for testvalue in testvalues:
        
            rcost_manual = x_to_rcost(testvalue[:2], [testvalue[2]], testvalue[3:], faultmodes='StoreEE')
            rcost_int = opt_prob.cr(testvalue)
            self.assertAlmostEqual(rcost_manual, rcost_int)
    def test_sim_types(self):
        
        testvalue= [0,2, 100, 1,1]
        rcost_manual = x_to_rcost(testvalue[:2], [testvalue[2]], testvalue[3:], faultmodes='StoreEE')
        
        opt_prob.update_sim_options("rcost", staged=True)
        rcost_int = opt_prob.cr(testvalue)
        self.assertAlmostEqual(rcost_manual, rcost_int)
        
        opt_prob.update_sim_options("rcost", staged=True, pool=mp.Pool(4))
        rcost_int = opt_prob.cr(testvalue)
        self.assertAlmostEqual(rcost_manual, rcost_int)
        
        opt_prob.update_sim_options("rcost", staged=False, pool=mp.Pool(4))
        rcost_int = opt_prob.cr(testvalue)
        self.assertAlmostEqual(rcost_manual, rcost_int)
        

if __name__ == '__main__':
    unittest.main()
        