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
import fmdtools.sim.propagate as prop

from shapely import Point, LineString, MultiLineString, Polygon
from shapely.ops import split

# DEFINE PARAMETERS

class EnvironmentParams (CoordsParam):
    """
    Define the map parameters (default is 1000mX1000m grid where each point is 0.1m).

    Fields
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
    blocksize: float = 0.3

    num_occupied: int = 50
    state_explored: tuple = (bool, False)
    state_ghose_objects: tuple = (bool, False)
    feature_occupied: tuple = (bool, False)
    point_base: tuple = (5.1, 5.1)
    collect_all_occupied: tuple = ('occupied', True)

class GroundControlParams(Parameter):
    """
    Define the parameter inputs from ground control.

    Fields
    -------
    destination: tuple
        coordinate for the destination
    speed: float
        maximum speed of the rover
    

    """

    destination: tuple = (25, 25)
    speed: float = 0.1
    
class MissionParams(Parameter):
    """
    Define the parameter inputs re;ated to the mission.

    Fields
    -------
    loc_pos_error: float
        maximum error in location position esitmation. default is 10%
    num_waypoints: int
        number of waitpoints per 5 meters. between 1 and 16. default is 5.
        i.e., 1 waypoint per meter at least.
    vision_distance: float
        lidar/camare visual distance
    
    """
    loc_pos_error: float = 0.1
    num_waypoints: int = 5
    vision_range: float = 5.0
    
class RoverParams(Parameter):
    '''
    Overall Rover Parameters
    '''
    environment: EnvironmentParams = EnvironmentParams()
    ground_control: GroundControlParams = GroundControlParams()
    mission: MissionParams = MissionParams()
    
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
    power_draw: bool = False

class EE(Flow):
    """Electricity flow."""

    __slots__ = ()
    container_s = EEState

# class CommsState(State):
#     '''Communication variables between the rover and ground control
    
#     Feilds
#     ------
#     end_point: tuple 
#         destination point
#     new_desitnation: bool
#         if a new destination is possible
#     power: bool
#         turn on rover
#     map: RoverEnvironment
#         areas mapped by the rover
#     comms_to_ground: str
#         communication from the rover to ground control
#     '''
#     end_point: tuple = (0,0)
#     power: bool = False
#     map: RoverEnvironment = {}

# class Comms(Flow):
#     '''Communications flow'''
#     __slots__ = ()
#     container_s = CommsState

class LocationPoseState(State):
    '''Rover location and Pose
    
    Feilds
    ------
    curr_x: float 
        x coordinate of the current location
    curr_y: float
        y coordinate of the current location
    velocity: list
        velocity vector with i and j compoenents 
    '''
    curr_x: float = 5.0
    curr_y: float = 5.0
    velocity: list = (0,0)

class LocationPose(MultiFlow):
    '''Flow for the Locationa and Pose of the Rover '''
    __slots__ = ()
    container_s = LocationPoseState

    
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
    standby: Mode
        power supply is in stand by
    """
    fm_args = ("no_charge", "short")
    opermodes = ( "supply", "standby")
    mode: str =  "standby"

class SupplyPower(Function):
    """Rover power supply."""

    __slots__ = ("ee")
    container_s = SupplyPowerState
    container_m = SupplyPowerMode
    flow_ee = EE

    def static_behavior(self, time):
        """Determine power use based on mode."""
        if self.m.in_mode("standby"):
            self.standby_power()
        elif self.m.in_mode("supply"):
            self.supply_power()
        elif self.m.in_mode("short"):
            self.short_power()
        elif self.m.in_mode("no_charge"):
            self.no_charge_power()

        self.power_usage()
        if self.m.in_mode("short"):
            self.short_power_usage()

    def dynamic_behavior(self, time):
        """power usage over time."""
        self.s.inc(charge=-self.s.power / 100)
        self.s.limit(charge=(0, 100))

    def short_power(self):
        """Power in case of a short has normal voltage."""
        self.ee.s.v = 12

    def no_charge_power(self):
        """Battery is out of charge."""
        self.ee.s.v = 0
        
    def standby_power(self):
        """during standby the supplied current is reduced."""
        self.ee.s.put(v=12, a=0.1)
        if self.ee.s.power_draw == True:
            self.m.set_mode("supply")

    def supply_power(self):
        """Power supply is in supply mode."""
        if self.s.charge > 0:
            self.ee.s.v = 12.0
        else:
            self.m.set_mode("no_charge")
        if self.ee.s.power_draw == False:
            self.m.set_mode("standby")

    def power_usage(self):
        """Calculate the power usage in general."""
        self.s.power = self.ee.s.mul("v", "a") 


    def short_power_usage(self):
        """Calculate power usage when there is a short (calculated as double)."""
        self.s.power = self.s.power * 2
        if self.s.charge == 0:
            self.m.set_mode("no_charge")
        if self.ee.s.power_draw == False:
            self.m.set_mode("standby")


    
# class CommunicateMode(Mode):
#     fm_args = ("loss_of_communication")

# class Communicate(Function):
#     __slots__ = ("ee_comms", "comms")
#     container_m = CommunicateMode
#     container_p = GroundControlParams
#     flow_comms = Comms
#     flow_ee_comms = EE
    
#     def dynamic_behavior (self, time):
#         if self.ee_comms.s.v > 0:
#             self.ee.comms.s.a = 1
#             if self.m.any_faults() is False:
#                 self.comms.s.end_point = self.p.destination
                
            



class MapMode(Mode):
    fm_args = ("no_obstacle_detection", "ghose_obstacle_detection", "no_feed", "no_self_map")

class Map(Function):
    __slots__ = ("ee", "location_pose", "environment")
    container_m = MapMode
    container_p = MissionParams
    flow_ee = EE
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPose
    
    def init_block(self, **kwargs):
        self.perc_environment = self.environment.create_comms(self.name)
    
    def dynamic_behavior (self, time):
        explored_pts = list()
        if self.ee.s.v == 12:
            self.ee.s.a = 0.1
            if not self.m.has_fault("no_self_map"):
                self.perc_environment.update(to_update="local", to_get="global")
                explored_pts = self.perc_environment.c.find_all('explored', True)
                self.perc_environment.c.set_pts(explored_pts, 'explored', False)

            if self.m.in_mode('no_feed') is False:
                self.ee.s.power_draw = True
                self.ee.s.a += 1
                vision_cone = Point(self.location_pose.s.curr_x, self.location_pose.s.curr_y).buffer(self.p.vision_range)
                unexplored_pts = self.perc_environment.c.find_all('explored', False)
                for i in unexplored_pts:
                    if vision_cone.contains(Point(i[0],i[1])):
                        explored_pts.append(i)
                self.environment.c.set_pts(explored_pts, 'explored', True)

            else:
                self.ee.s.power_draw = False
                self.ee.s.a = 0
        else:
            self.m.set_mode ('no_feed')
        
        return

     

class SenseMode(Mode):
    fm_args = ("too_late",)

class Sense(Function):
    __slots__ = ("ee", "location_pose", "environment")
    container_m = SenseMode
    flow_ee = EE
    flow_location_pose = LocationPose
    flow_environment = RoverEnvironment
    
    def dynamic_behavior (self, time):
        return

class NavigateMode(Mode):
    fm_args = ("no_obstacle_detection", "wrong_obstacle_detection", "ghose_obstacle_detection", "no_own_map")

class Navigate(Function):
    __slots__ = ("ee", "environment", "location_pose")
    container_m = NavigateMode
    flow_ee = EE
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPose
    
    def dynamic_behavior (self, time):
        return

class Rover(FunctionArchitecture):
    __slots__=()
    container_p = RoverParams
    default_sp = dict(end_time=150,
                    phases=(("start", 0, 30), ("end", 31, 150)))
    def init_architecture(self, **kwargs):
        #Flows
        self.add_flow("location_pose", LocationPose)
        self.add_flow("environment", RoverEnvironment, p=self.p.environment)
        #self.add_flow("ee_comms", EE)
        self.add_flow("ee", EE)
        #self.add_flow("comms", Comms)
        #Functions
        self.add_fxn("supply_power", SupplyPower, "ee") 
        #self.add_fxn("communicate", Communicate, "ee_comms", "comms")
        self.add_fxn("map", Map, "environment", "ee", "location_pose", p = self.p.mission) 
        self.add_fxn("sense", Sense, "location_pose", "ee")
        self.add_fxn("navigate", Navigate, "location_pose", "ee", "environment")

        


if __name__ == "__main__":
    function = SupplyPower()
    
    mdl = Rover()
    mdl.flows['environment'].c.show_collection('all_occupied', label = True)
    #mdl.flows['environment'].c.show('explored', collections = {'all_occupied':{'label': False}})
    #mdl.flows['environment'].c.show_property('occupied')
    mtg = FunctionArchitectureTypeGraph(mdl)
    fig, ax = mtg.draw()
    dot = mtg.draw_graphviz()
    
    ec1, hist = prop.nominal(mdl)
    # hist.flows.environment.c.show('explored', collections = {'all_occupied':{'label': False}})
    # hist.flows.environment.c.show_property('occupied')