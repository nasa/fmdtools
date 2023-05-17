# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:26:19 2022

@author: dhulse
"""
import unittest
from examples.rover.optimization.search_rover import line_dist, line_dist_faster
from tests.common import CommonTests
import multiprocessing as mp

class RoverTests(unittest.TestCase, CommonTests):
    def test_obj_values(self):
        
        testvalues = [[1.0,0.5,0.0],[0.0,0.0,0.0], [1.0,1.0,1.0], [0.5,0.5,0.5]]
        for testvalue in testvalues:
            dist_int,enddist_int, endpt_int = line_dist_faster(testvalue)
            dist,enddist, endpt = line_dist(testvalue)
            
            self.assertEqual(dist, dist_int)
            self.assertEqual(enddist, enddist_int)
            self.assertEqual(endpt, endpt_int)
            
if __name__ == '__main__':
    unittest.main()
        