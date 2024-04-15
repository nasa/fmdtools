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
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.analyze.graph import FunctionArchitectureGraph, FunctionArchitectureFlowGraph
from fmdtools.analyze.graph import FunctionArchitectureFxnGraph, FunctionArchitectureTypeGraph

# DEFINE PARAMETERS

class EnvironmentParams (CoordsParam):
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

    num_occupied: int = 1000
    feature_explored: tuple = (bool, False)
    feature_occupied: tuple = (bool, False)
    point_start: tuple = (0, 0)
    collect_all_explored: tuple = ("explored", True)
    collect_all_occupied: tuple = ('occupied', True)

class GroundControlParams(Parameter):
    """
    Define the parameter inputs from ground control at various timesteps.

    Features/Collections
    -------
    destination: tuple
        coordinate for the destination
    alternate_destination: tuple
        coordinate for an alternate destination in case primary destination is not reachable
    """

    destination: tuple = (9.9, 9.9)
    alternate_desitnation: tuple = (9.9, 8)
#DEFINE FLOWS

class EnvironmentGrid(Coords):
    """Define the map. occupied and unoccupied points."""

    container_p = EnvironmentParams

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
    

class RoverEnvironment(Environment):
    """ Rover environment with occupiable and unoccupiable spaces."""
    container_p = EnvironmentParams
    coords_c = EnvironmentGrid
    container_s = EnvironmentState


class EEState(State):
    """Electricity state (voltage v and amperage a)."""

    v: float = 0.0
    a: float = 0.0

class EE(Flow):
    """Electricity flow."""

    __slots__ = ()
    container_s = EEState

class CommsState(State):
    '''Communication variables between the rover and ground control
    
    Feilds
    ------
    end_point: tuple 
        destination point
    new_desitnation: bool
        if a new destination is possible
    map: RoverEnvironment
        areas mapped by the rover
    comms_to_ground: str
        communication from the rover to ground control
    '''
    end_point: tuple = (0,0)
    new_destination: bool = True
    map: RoverEnvironment = {}

class Comms(Flow):
    '''Communications flow'''
    __slots__ = ()
    container_s = CommsState

class LocationPosState(State):
    '''Rover location and Pose
    
    Feilds
    ------
    curr_point: tuple 
        current_location
    velocity: list
        velocity vector with i and j compoenents 
    '''
    curr_point: tuple = (0,0)
    velocity: list = (0,0)

class LocationPos(MultiFlow):
    '''Flow for the Locationa and Pose of the Rover '''
    __slots__ = ()
    container_s = LocationPosState

    
# DEFINE FUNCTIONS
class SupplyPowerState(State):
    """
    State of power.

    Fields
    ------
    charge: float
        State of charge (percentage).
    power: float
        Power output (percentage of soc).
    """

    charge: float = 100.0
    power: float = 0.0

class SupplyPowerMode(Mode):
    """
    Possible modes for Supply Power function.

    Modes
    -------
    no_charge : Fault
        Battery is out of charge.
    short: Fault
        There is a short.
    supply: Mode
        supply power
    charge: Mode
        charge battery
    standby: Mode
        power supply is in stand by
    off: Mode
        power supply is off
    """
    fm_args = ("no_charge", "short")
    opermodes = ("off", "supply", "charge", "standby")
    mode: str =  "off"

class SupplyPower(Function):
    """Rover power supply."""

    __slots__ = ("ee", "comms")
    container_s = SupplyPowerState
    container_m = SupplyPowerMode
    flow_ee = EE
    flow_comms = Comms


    
class CommunicateMode(Mode):
    fm_args = ("packet_not_sent", "loss_from_ground", "wrong_packet_sent")

class Communicate(Function):
    __slots__ = ("ee", "comms")
    container_m = CommunicateMode
    flow_ee = EE
    flow_comms = Comms

class SenseMode(Mode):
    fm_args = ("too_late",)

class Sense(Function):
    __slots__ = ("ee", "location_pose", "environment")
    container_m = SenseMode
    flow_ee = EE
    flow_location_pose = LocationPos
    flow_environment = RoverEnvironment

class MapMode(Mode):
    fm_args = ("no_obstacle_detection", "wrong_obstacle_detection", "ghose_obstacle_detection", "no_own_map")

class Map(Function):
    __slots__ = ("ee", "comms", "environment")
    container_m = MapMode
    flow_ee = EE
    flow_comms = Comms
    flow_environment = RoverEnvironment

class NavigateMode(Mode):
    fm_args = ("no_obstacle_detection", "wrong_obstacle_detection", "ghose_obstacle_detection", "no_own_map")

class Navigate(Function):
    __slots__ = ("ee", "comms", "environment", "location_pose")
    container_m = NavigateMode
    flow_ee = EE
    flow_comms = Comms
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPos

class Rover(FunctionArchitecture):
    __slots__=()
    default_sp = dict(end_time=150,
                    phases=(("start", 0, 30), ("end", 31, 150)))
    def init_architecture(self, **kwargs):
        #Flows
        self.add_flow("location_pose", LocationPos)
        self.add_flow("environment", RoverEnvironment)
        self.add_flow("ee", EE)
        self.add_flow("comms", Comms)
        #Functions
        self.add_fxn("communicate", Communicate, "ee", "comms")
        self.add_fxn("supply_power", SupplyPower, "ee", "comms") 
        self.add_fxn("map", Map, "environment", "ee") 
        self.add_fxn("sense", Sense, "location_pose", "ee")
        self.add_fxn("navigate", Navigate, "location_pose", "ee", "environment", "comms")
        self.add_fxn("communicate", Communicate, "ee", "comms")
        


if __name__ == "__main__":
    mdl = Rover()
    x = EnvironmentGrid()
    x.show_collection('all_occupied', label = False)
    x.show('explored', collections = {'all_occupied':{'label': False}})
    x.show_property('occupied')
    mtg = FunctionArchitectureTypeGraph(mdl)
    fig, ax = mtg.draw()
    dot = mtg.draw_graphviz()