#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:05:24 2024

@author: smbaye
"""


from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.environment import Environment
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.container.mode import Mode
from fmdtools.define.container.state import State
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.flow.base import Flow

# DEFINE PARAMETERS

class MapParams (CoordsParam):
    """
    Define the map parameters (default is 100x100 grid where each point is 0.1m).

    Features/Collections
    -------
    occupied: feature
        places in the map that are occupied and the rover can't reach
    explored: feature
        areas of the map where the rover has exploed
    occupied: feature
        places where people are (landing would be dangerous)
    start: point
        where the drone starts
    all_occupied: collection
        all points that are occupied
    all_explored: collection
        all points that are explored
    """
    x_size: int = 100
    y_size: int = 100
    blocksize: float = 0.1


    num_occupied: int = 3000
    feature_explored: tuple = (bool, False)
    feature_occupied: tuple = (bool, False)
    point_start: tuple = (0, 0)
    collect_all_explored: tuple = ("explored", True)
    collect_all_occupied: tuple = ('occupied', True)

class Map(Coords):
    """Define the map. occupied and unoccupied points."""

    container_p = MapParams

    def init_properties(self, *args, **kwargs):
        """Randomly allocate the occupied points"""
        self.set_rand_pts('occupied', True, self.p.num_occupied, pts=self.pts[1:-1])

class EnvironmentState(State):
    """
    States relating the rover with its environment.

    Fields
    -------
    safe: bool
        weather the rover is in the safe zone (>0.3m from occupied points)
    allowed: bool
        whether the drone is above an allowed grid location
    """
    safe: bool = True
    

class RoverEnvironment(Map):
    """ Rover environment with occupiable and unoccupiable spaces."""
    container_p = MapParams
    coords_c = Map
    container_s = EnvironmentState

    

# class Rover(FunctionArchitecture):
#     __slots__=()
#     default_sp={}
#     def init_architecture(self, **kwargs):
#         #Flows
#         self.add_flow("location_pose")
#         self.add_flow("environment")
#         self.add_flow("electrical_energy")
#         self.add_flow("map")
#         self.add_flow("user_input")
#         #Functions
#         self.add_fxn("navigation", GenericFxn, "location_pose", "electrical_energy", "environment", "user_input")
#         self.add_fxn("sensing", GenericFxn, "location_pose", "electrical_energy")
#         self.add_fxn("mapping", GenericFxn, "environment", "electrical_energy", "map")
#         self.add_fxn("communication", GenericFxn, "map", "electrical_energy", "user_input")
#         self.add_fxn("power_supply", GenericFxn, "electrical_energy")
# mdl = Rover()

if __name__ == "__main__":
    x = Map()
    x.show_collection('all_occupied', label = False)
    x.show_property('occupied',)
    