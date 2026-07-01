"""
UAS Rescue Model

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

#using SI units

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.flow.multiflow import Flow
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.object.coords import Coords
from fmdtools.define.object.coords import CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.object.geom import GeomPoint
from fmdtools.define.object.geom import GeomParameter
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.sim import propagate
from typing import ClassVar

import random
import numpy as np

#States

class DroneState(State):
    battery: float = 100 #percent

class BuoysMode(Mode):
    opmodes = ("deflated", "inflated")

class CommsMode(Mode):
    opmodes = ("distress", "standby")

class SwimmerMode(Mode):
    opmodes = ("safe", "drowning", "dead")

class DroneMode(Mode):
    opmodes = ("connected", "unconnected")

class DronePilotMode(Mode):
    opmodes = ("patrol", "rescue")

#Flows

class Comms(CommsFlow):
    container_m = CommsMode

class Buoys(Flow):
    container_m = BuoysMode

#Functions

class Drone(Function):
    container_s = DroneState
    flow_comms = Comms

class FDNY(Function):
    flow_comms = Comms
    def dynamic_behavior(self):
        if(self.comms.m.mode == "distress"):
            if(self.s.x != self.comms.s.rescue_x):
                self.s.x += self.t.dt*(self.comms.s.rescue_x-self.s.x)/abs(self.comms.s.rescue_x-self.s.x)
            if(self.s.z != self.comms.s.rescue_x):
                self.s.z += self.t.dt*(self.comms.s.rescue_z-self.s.z)/abs(self.comms.s.rescue_z-self.s.z)
            if(abs(self.comms.s.rescue_x-self.s.x) < 2 and abs(self.comms.s.rescue_x-self.s.x) < 2):
                self.comms.m.mode = "standby"


class SwimmerPointParam(GeomParameter):
    coordinates: tuple = (1.0, 1.0)
    buffer_on: float = 1.0

class SwimmerGeomPoint(GeomPoint):
    container_p = SwimmerPointParam

class ExGeomArch(GeomArchitecture):
    def init_architecture(self, **kwargs):
        self.add_geom("swimmer_point", SwimmerGeomPoint)

class SwimmerState(State):
    weight: float = 64 #kg

class Swimmer(Function):
    container_s = SwimmerState
    container_m = SwimmerMode
    flow_comms = Comms
    container_ga = SwimmerGeomPoint
    def dynamic_behavior(self):
        if(self.m.mode == "drowning"):
            self.comms.m.mode = "distress"

class CommsState(State):
    point_rescue_point = SwimmerGeomPoint
            
class BeachCoordsParam(CoordsParam):
    x_size: ClassVar[int] = 2000
    y_size: ClassVar[int] = 350
    blocksize: ClassVar[float] = 1
    gapwidth: ClassVar[float] = 0.0
        
class BeachCoords(Coords):   
    __slots__ = ("p", "grid", "pts")
    container_p = BeachCoordsParam
    point_start = (0.0, 0.0)

    feature_depth = (int, 0) #represents whether a point is beach, shore, ocean

    collection_beach = ("depth", 0, np.equal) # collection of points with depth = 0 (beach)
    collection_shore = ("depth", 1, np.equal) # collction of points with depth = 1 (shore)
    collection_ocean = ("depth", 2, np.equal) # collction of points with depth = 2 (ocean)

    def init_properties(self, **kwargs):
        shore_points = [[float(x), float(y)] for x in range(0, 200) for y in range(50, 200)]
        ocean_points = [[float(x), float(y)] for x in range(0,200) for y in range(200,350)]
        self.set_pts(shore_points, "depth", 1.0) 
        self.set_pts(ocean_points, "depth", 2.0)

class Beach(Environment):
    coords_c = BeachCoords

beach = BeachCoords()
swimmer = Swimmer()

print(beach.depth[0,200])
swimmer.ga.p.coordinates = (0.0, 1.0)
print(swimmer.ga.p.coordinates)

