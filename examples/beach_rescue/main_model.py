#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Flows (e.g., Ground, AirSpace, etc) used in wildfire response model.

Includes classes for defining the fire map (fuels, bases, etc) and its behavior.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.analyze.common import setup_plot, consolidate_legend

import numpy as np
import random

class BeachMapParam(CoordsParam):
    """
    Parameter defining the fire map.

    Parameters
    ----------
    x_size: int
        Number of grid cells in the x. 
    y_size: int
        Number of grid cells in the y. 
    blocksize: float
        Size of grid cels. 
    base_locations: tuple
        Locations in the grid to put a base.
    map_type: str
        Type of map, specified as "uniform-xx","split-xx-yy", or "xx-yy-zz", where
        xx, yy, and zz are "grass", "forest", or "scrub" fuels
    """

    x_size: int = 20
    y_size: int = 10
    blocksize: float = 500 #500m
    base_locations: tuple = ((0.0, 0.0))
    num_strikes: int = 1
    map_type: str = "uniform-sand"


class HazardState(State):
    location: tuple = (0,0)

class Hazard(Function):
    container_s = HazardState
    def drowning(self):
        x = random.randint(0,9600)
        y = random.randint(166,4800)
        self.s.location = (x, y)

class DroneState(State):
    location: tuple = (0,0)
class DroneMode(Mode):
    mode = "standby"


        

class BeachMap(Coords):
    container_p = BeachMapParam
    feature_terrain: tuple = (float, 0.0)  # 0.0 = ocean, 1.0 = sand
    feature_strike: tuple = (bool, False)
    feature_base: tuple = (bool, False)
    point_start: tuple = (4800, 0)
    point_hazard: tuple = (1000, 0)
    point_drone: tuple = (0,0)

    def update_objects(self, pointHazard, pointDrone):
        self.hazard = pointHazard
        self.drone = pointDrone
    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        # set entire grid as ocean by default (already 0.0)
        # set bottom rows as sand
        land_y = 166  # one row up from 0
        for pt in self.pts:
            if pt[1] <= land_y:
                self.set(*pt, "terrain", 1.0)


class BeachEnvironment(Environment): #type of flow
    """Environment flow containing the BeachMap for use in a larger model."""

    coords_c = BeachMap

    def prop_time(self, tstep=1.0):
        """Propagate a timestep of fire behavior."""
        self.c.prop_fire(tstep=tstep)

class Drone(Function):
    container_m = DroneMode
    container_s = DroneState
    def go_to_hazard(self, Location):
        self.s.location = Location
    def dynamic_behavior():
        if(self.m.mode == "active"):
            go_to_hazard((0,0))
