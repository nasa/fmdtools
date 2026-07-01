"""
Testing model

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

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.flow.multiflow import Flow
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
import random


import numpy as np

class WaterState(State):
    rate: float = 0.0
    total_flow: float = 0.0

class Water(Flow):
    container_s = WaterState

class WasteState(State):
    size: float = 0.0

class Waste(Flow):
    container_s = WasteState

class Flusher(Function):
    flow_water = Water
    flow_waste = Waste
    default_sp = {'end_time': 10}
    def dynamic_behavior(self):
        self.water.s.rate = 1.0
        if (self.waste.s.size <= 19):
            self.waste.s.size = 0
        self.water.s.total_flow += self.water.s.rate * self.t.dt

class BathroomUser(Function):
    flow_waste = Waste
    flow_water = Water
    def dynamic_behavior(self):
        self.waste.s.size += random.randint(1,20)
        if(self.waste.s.size > 19):
            self.water.s.rate = 0

class Bathroom(FunctionArchitecture):
    def init_architecture(self, **kwargs):
        self.add_flow("water", Water)
        self.add_flow("waste", Waste)
        self.add_fxn("flusher", Flusher, "water", "waste")
        self.add_fxn("bathroom_user", BathroomUser, "water", "waste")
    
        

mdl = Bathroom(sp={'end_time': 15}, track = "all")
res, hist = propagate.nominal(mdl)
print("water rate:")
print(hist.flows.water.s.rate)
print("waste size:")
print(hist.flows.waste.s.size)



"""fs = Flusher(sp={'end_time': 15}, track="all")
print(fs)

res, hist = propagate.nominal(fs)

print(hist)
print(hist.water.s.total_flow)
print(hist.water.s.rate)"""