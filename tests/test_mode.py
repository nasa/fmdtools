# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:17:29 2023

@author: dhulse
"""
import unittest

from fmdtools.define.mode import Mode

class StoreEnergyMode(Mode):
    faultparams = {"no_charge":(1e-5, {'standby':1.0}, 100),
                   "short":(1e-5, {'supply':1.0}, 100)}
    opermodes = ("supply","charge","standby")
    exclusive = True
    key_phases_by = "self"
    mode: str = "standby"

class ModeTests(unittest.TestCase):
    def setUp(self):
        self.mode = StoreEnergyMode()
    def test_mode_generation(self):
        self.assertTrue(self.mode.mode=='standby')
        self.assertFalse(self.mode.any_faults())
        self.assertTrue("no_charge" in self.mode.faultmodes)
            
if __name__ == '__main__':
    unittest.main()
        
