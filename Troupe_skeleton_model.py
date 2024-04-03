#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:05:24 2024

@author: smbaye
"""


from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import GenericFxn

class Rover(FunctionArchitecture):
    __slots__=()
    default_sp={}
    def init_architecture(self, **kwargs):
        #Flows
        self.add_flow("location_pose")
        self.add_flow("environment")
        self.add_flow("electrical_energy")
        self.add_flow("map")
        self.add_flow("user_input")
        #Functions
        self.add_fxn("navigation", GenericFxn, "location_pose", "electrical_energy", "environment", "user_input")
        self.add_fxn("sensing", GenericFxn, "location_pose", "electrical_energy")
        self.add_fxn("mapping", GenericFxn, "environment", "electrical_energy", "map")
        self.add_fxn("communication", GenericFxn, "map", "electrical_energy", "user_input")
        self.add_fxn("power_supply", GenericFxn, "electrical_energy")
mdl = Rover()