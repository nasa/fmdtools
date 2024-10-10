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
from fmdtools.define.container.time import Time
from fmdtools.define.container.rand import Rand
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.analyze.graph.architecture import (
    FunctionArchitectureGraph,
    FunctionArchitectureFlowGraph,
)
from fmdtools.analyze.graph.architecture import (
    FunctionArchitectureFxnGraph,
    FunctionArchitectureTypeGraph,
)
import fmdtools.sim.propagate as prop
import numpy as np

from shapely import Point, LineString, MultiLineString, Polygon
from shapely.ops import split
import matplotlib.pyplot as plt
from shapely.plotting import plot_line, plot_points

plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\mmohame2\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'

# DEFINE PARAMETERS


class GroundControlParams(Parameter):
    """
    Define the parameter inputs from ground control.

    Fields
    -------
    destination: tuple
        coordinate for the destination
    dest_buffer: float
        accaptable buffer for the end point
    speed: float
        maximum speed of the rover


    """

    destination: tuple = (25, 25)
    dest_buffer: float = 0.3
    speed: float = 0.1


class EnvironmentParams(CoordsParam):
    """
    Define the map parameters (default is 1000mX1000m grid where each point 
   is 0.1m).

    Fields
    -------
    x_size: int
        number of blocks in x direction
    y_size: int
        number of blocks in y direction
    blocksize: float
        size of each block
    operator_params: GroundControlParams
        takes in the destination point from ground control
    num_occupied: int
        number of occupied grid points to initiate the map with
    occupied: feature
        places in the map that are occupied and the rover can't reach
    explored: state
        areas of the map where the rover has exploed
    perecieved_objects: state
        objects that are percieved and mapped
    perecieved_end_zone: feature
        acceptable end zone corresponding to the desitination point
    start: base
        base location
    all_occupied: collection
        all points that are occupied
    all_endzone: collection
        all points that are in th end zone
    """

    x_size: int = 100
    y_size: int = 100
    blocksize: float = 0.3
    operator_params: GroundControlParams = GroundControlParams()

    num_occupied: int = 50
    state_explored: tuple = (bool, False)
    state_perceived_objects: tuple = (bool, False)
    state_rover_path: tuple = (bool, False)
    feature_occupied: tuple = (bool, False)
    feature_end_zone: tuple = (bool, False)
    point_base: tuple = (5.1, 5.1)
    collect_all_occupied: tuple = ("occupied", True)
    collect_all_endzone: tuple = ("end_zone", True)


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
    vision_range: float
        lidar/camare visual distance
    max_ghost_points: int
        maximum number of ghost obstacles the simulation will create in case of
        ghost obstacle fault is present for mapping function. Default is 1.
    max_sense_delay: int
        maximum number of time steps a sensing delay is to occur. For delayed
        sensing fault, sensing will be delayed beyond this value.
    sense_malfunc_rate: float
        error rate when the sensor malfunctions
    """

    loc_pos_error: float = 0.1
    num_waypoints: int = 5
    max_sense_delay: int = 4
    sense_malfunc_rate: float = 0.3
    vision_range: float = 5.0
    max_ghost_points: int = 1


class RoverParams(Parameter):
    """
    Overall Rover Parameters
    
    Fields
    -------
    environment: EnvironmentParams
        parameters relating to the rover environment (map)
    ground_control: GroundControlParams 
        paramters that are passed on by the ground control
    mission: MissionParams 
        paramters that are related to the mission and simulation
    """
    environment: EnvironmentParams = EnvironmentParams()
    ground_control: GroundControlParams = GroundControlParams()
    mission: MissionParams = MissionParams()


# DEFINE FLOWS


class EnvironmentGrid(Coords):
    """Define the map. occupied and unoccupied points."""

    container_p = EnvironmentParams

    def init_properties(self, *args, **kwargs):
        """Randomly allocate the occupied points and create an end zone based
            on the desination point        
        """
        self.set_rand_pts("occupied", True,
                          self.p.num_occupied, pts=self.pts[1:-1])
        end_zone = Point(
            self.p.operator_params.destination[0], 
            self.p.operator_params.destination[1]
        ).buffer(self.p.operator_params.dest_buffer)
        for i in self.pts:
            if end_zone.contains(Point(i[0], i[1])):
                self.set(i[0], i[1], "end_zone", True)


class EnvironmentState(State):
    """
    States relating the rover with its environment.

    Fields
    -------
    safe: bool
        weather the rover is in the safe zone (>0.3m from occupied points)
    """

    safe: bool = True


class RoverEnvironment(Environment):
    """Rover environment with occupiable and unoccupiable spaces."""

    container_p = RoverParams
    coords_c = EnvironmentGrid
    container_s = EnvironmentState

    def at_finish(self, curr_x, curr_y):
        """determine if the rover is at the end zone"""
        return self.c.in_area(curr_x, curr_y, "all_endzone")
    
    def check_collision(self, curr_x, curr_y):
        """determine if the rover has hit an obstacle"""
        return self.c.get(curr_x, curr_y, 'occupied')

class EEState(State):
    """States relating to the power supply.
    
    Feilds
    -------
    v: float
        voltage
    a: float
        amparage
    power_draw: bool
        Indicates weather components are drawing power from the power supply
    """
    v: float = 12.0
    a: float = 0.0
    power_draw: bool = True


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
    """Rover location and Pose

    Feilds
    ------
    curr_x: float
        x coordinate of the current location
    curr_y: float
        y coordinate of the current location
    pose_angle: float
        current pose measured from the positive x-axis
    velocity: list
        velocity vector with i and j compoenents

    """

    curr_x: float = 5.1
    curr_y: float = 5.1
    heading_angle: float = 0.0
    velocity: list = [0, 0]
 


class LocationPose(MultiFlow):
    """Flow for the Locationa and Pose of the Rover"""

    __slots__ = ()
    container_s = LocationPoseState

class SenseDataState(State):
    '''
    location: bool
        indicates if the Sense function has sensed the location
    pose: bool
        indicates if the Sense function has sensed the pose
    malfunc_location: bool
        indicates if sense is mulfunctioning for location sensing
    malfunc_pose: bool
        indicates if sense is mulfunctioning for pose sensing
    delay: int
        time delay in sensing
    '''

    location: bool = False
    pose: bool = False
    malfunc_location: bool = False
    malfunc_pose: bool = False
    delay: int = 0
    
class SenseData(Flow):
    '''flow that passes sensing related information'''
    
    __slots__ = ()
    container_s = SenseDataState
    
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
    opermodes = ("supply", "standby")
    mode: str = "standby"


class SupplyPower(Function):
    """Rover power supply."""

    __slots__ = "ee"
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


class MapState(State):
    '''state for the map function. Tracks the obstacles that are within the
    vision cone'''
    obstacles_inrange: list = []


class MapMode(Mode):
    '''Modes for the mapping functions
    
    -------------------
    on: mode
        mapping function is on
    off: mode
        mapping function is off
    no_obstacle_detection: fault
        some objects are not detected
    ghost_obstacle_detection: fault
        Objects that are not present are detected
    no_feed: fault
        there is no camera/lidar feed
    no_self_map: fault
        already mapped areas are no longer available
    '''
    fm_args = (
        "no_obstacle_detection",
        "ghost_obstacle_detection",
        "no_feed",
        "no_self_map",
    )
    opermodes = ("on", "off")
    mode: str = "off"


class Map(Function):
    ''' Mapping Function
    
        percieves the environment based on the visible region
    '''
    __slots__ = ("ee", "location_pose", "environment")
    container_m = MapMode
    container_p = MissionParams
    container_r = Rand
    container_s = MapState
    flow_ee = EE
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPose

    def dynamic_behavior(self, time):
        ''' all nominal and faulty perceptions behaviors'''
        if self.ee.s.v == 12:
            self.m.set_mode("on")
            self.ee.s.a = 0.1
            
            #All percieved points are set to unexplored when 
            #no_self_map fault is present
            if self.m.has_fault("no_self_map"):
                explored_pts = self.environment.c.find_all_prop(
                    "explored", True)
                self.environment.c.set_pts(explored_pts, "explored", False)
                self.environment.c.set_pts(
                    explored_pts, "perceived_objects", False)
                self.s.obstacles_inrange = list()
            
            # perception occurs when video/lidar is present
            if not self.m.has_fault("no_feed"):
                clear_pts = list()
                obstacle_pts = list()
                # perception draws power from the power supply
                self.ee.s.power_draw = True
                self.ee.s.a += 1
                
                #initial points of interests are determined
                vision_cone = Point(
                    self.location_pose.s.curr_x, self.location_pose.s.curr_y
                ).buffer(self.p.vision_range)
                unexplored_pts = self.environment.c.find_all_prop(
                    "explored", False)
                explored_pts = self.check_in_cone(unexplored_pts, vision_cone)
                
                #check if the tracked obstacles from previous time step are
                #within the new vision_cone of interest
                if len(self.s.obstacles_inrange) > 0:
                    self.s.obstacles_inrange = self.check_in_cone(
                        self.s.obstacles_inrange, vision_cone
                    )
                
                # distinguish occupied and onoccupied points
                for i in explored_pts:
                    if self.environment.c.get(i[0], i[1], "occupied"):
                        obstacle_pts.append(i)
                    else:
                        clear_pts.append(i)
                
                #no_obstacle_detection fault behavior: does not percieve 
                #a random number of obstacles
                if self.m.has_fault("no_obstacle_detection"):
                    indexes = self.find_rand_points(obstacle_pts)
                    for i in sorted(indexes, reverse=True):
                        del obstacle_pts[i]
                        
                #ghost_obstacle_detection fault behavior: percieves a random 
                #number of obstacles that do not exist
                if self.m.has_fault("ghost_obstacle_detection"):
                    indexes = self.find_rand_points(
                        clear_pts, self.p.max_ghost_points)
                    for i in indexes:
                        obstacle_pts.append(clear_pts[i])
                
                # find the blids spots for percieved obstacles
                investigate_obstacles = self.s.obstacles_inrange + obstacle_pts
                if len(investigate_obstacles) > 0:
                    vision_cone = self.find_blind_spots(
                        vision_cone, investigate_obstacles
                    )
                
                # determine the visible points based on the vision cone with 
                # blind spots. The buffers are set to cushion error and for 
                # better visualization in the grid points
                visible_obstacle_pts = self.check_in_cone(
                    obstacle_pts, vision_cone, 
                    self.environment.c.p.blocksize / 10
                )
                visible_explored_pts = self.check_in_cone(
                    explored_pts, vision_cone, 
                    -self.environment.c.p.blocksize / 3
                )
                

                # add the visible obstacles to the tracked obstacle list
                self.s.obstacles_inrange = self.s.obstacles_inrange + visible_obstacle_pts
                
                # update the environment to with the percieved information
                self.environment.c.set_pts(
                    visible_obstacle_pts, "perceived_objects", True
                )
                self.environment.c.set_pts(
                    visible_obstacle_pts, "explored", True)
                self.environment.c.set_pts(
                    visible_explored_pts, "explored", True)

            else:
                # when there is no_feed the mapping does not draw power
                self.ee.s.power_draw = False
                self.ee.s.a = 0
        else:
            # mapping function is shuf off when there is no power supply
            self.m.set_mode("off")
            self.ee.s.power_draw = False
            self.ee.s.a = 0
        return

    def check_in_cone(self, pts, cone, buffer=0):
        '''checks if given points are within the cone (including the buffer)
           and returns the points that are within the cone. Boundaries exluded
        '''
        cone_pts = list()
        if buffer != 0:
            cone = cone.buffer(buffer)
        for i in pts:
            if cone.contains(Point(i[0], i[1])):
                cone_pts.append(i)
        return cone_pts

    def find_blind_spots(self, vision_cone, perceived_obstacles):
        '''removes blind spots that are caused by obstacles from the vision cone
        and returns the new cone.'''
        for i in perceived_obstacles:
            if vision_cone.contains(Point(i[0], i[1])):
                inside_pt = Point(self.location_pose.s.curr_x, self.location_pose.s.curr_y)
                delta = self.environment.c.p.blocksize / 2
                if (
                    self.location_pose.s.curr_x <= i[0]
                    and self.location_pose.s.curr_y <= i[1]
                ):
                    point1 = np.array([i[0] - delta, i[1] + delta])
                    point2 = np.array([i[0] + delta, i[1] - delta])
                    splitter = self.calc_splitter(point1, point2)
                    vision_cone = self.split_cone(
                        vision_cone, splitter, inside_pt)
                elif (
                    self.location_pose.s.curr_x >= i[0]
                    and self.location_pose.s.curr_y <= i[1]
                ):
                    point1 = np.array([i[0] - delta, i[1] - delta])
                    point2 = np.array([i[0] + delta, i[1] + delta])
                    splitter = self.calc_splitter(point1, point2)
                    vision_cone = self.split_cone(
                        vision_cone, splitter, inside_pt)
                elif (
                    self.location_pose.s.curr_x >= i[0]
                    and self.location_pose.s.curr_y >= i[1]
                ):
                    point1 = np.array([i[0] - delta, i[1] + delta])
                    point2 = np.array([i[0] + delta, i[1] - delta])
                    splitter = self.calc_splitter(point1, point2)
                    vision_cone = self.split_cone(
                        vision_cone, splitter, inside_pt)
                else:
                    point1 = np.array([i[0] - delta, i[1] - delta])
                    point2 = np.array([i[0] + delta, i[1] + delta])
                    splitter = self.calc_splitter(point1, point2)
                    vision_cone = self.split_cone(
                        vision_cone, splitter, inside_pt)
        return vision_cone

    def calc_splitter(self, point1, point2):
        '''determines the splitter for the vision cone, when it needs to 
        be split'''
        point3 = self.calc_blind_coords(point1)
        point4 = self.calc_blind_coords(point2)
        return LineString([point3, point1, point2, point4])

    def split_cone(self, vision_cone, splitter, inside_pt):
        '''splits the vision cone and returns the chunk that is of interest'''
        split_cone = split(vision_cone, splitter)
        geoms = [*split_cone.geoms]
        for geom in geoms:
            if geom.contains(inside_pt):
                vision_cone = geom
        return vision_cone

    def calc_blind_coords(self, Point):
        '''given a point, calculates the entexsion of that point that is at
        the endge of the vision cone'''
        norm_coords = np.array(
            [
                Point[0] - self.location_pose.s.curr_x,
                Point[1] - self.location_pose.s.curr_y,
            ]
        )
        new_norm_coords = norm_coords * (
            self.p.vision_range * 2 / np.linalg.norm(norm_coords)
        )
        new_point = np.array(
            [
                new_norm_coords[0] + self.location_pose.s.curr_x,
                new_norm_coords[1] + self.location_pose.s.curr_y,
            ]
        )
        return new_point

    def find_rand_points(self, grid_pts, max_pts=None):
        '''given grib points and maximum numbers of points to select, 
        randomly picks indexes from the grid pts'''
        if max_pts == None or len(grid_pts) <= max_pts:
            if len(grid_pts) > 0:
                max_pts = self.r.rng.integers(1, len(grid_pts), endpoint=True)
            else:
                max_pts = 0
        else:
            max_pts = self.r.rng.integers(0, max_pts, endpoint=True)
        index_of_interest = self.r.rng.choice(
            range(len(grid_pts)), max_pts, replace=False
        )
        return index_of_interest
    
class SenseTime(Time):
    timernames = ('nominal', 'too_late')

class SenseMode(Mode):
    fm_args = ("too_late", 'position_est_malfunc', 'position_est_loss', 
               'pose_est_malfunc', 'pose_est_loss')


class Sense(Function):
    __slots__ = ("ee", "sense_data", "environment", "location_pose")
    container_p = MissionParams
    container_m = SenseMode
    container_t = SenseTime
    container_r = Rand
    flow_ee = EE
    flow_sense_data = SenseData
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPose

    def dynamic_behavior(self, time):
        if self.ee.s.v == 12:
            power_draw_indicator = True
            if self.m.has_fault('too_late'):
                if self.t.too_late.indicate_standby():
                    self.sense_data.s.delay = self.r.rng.integers(self.p.max_sense_delay, self.p.max_sense_delay * 3, endpoint=True)
                    self.update_sensed('both', False)
                    if self.environment.c.get(self.location_pose.s.curr_x, self.location_pose.s.curr_y, 'explored'):
                        self.t.too_late.set_timer(self.sense_data.s.delay)
                    else:
                        power_draw_indicator = False
                elif self.t.too_late.indicate_complete():
                    self.assign_sense_data()
                    self.t.too_late.reset()
                else:
                    self.t.too_late.inc()
            else:
                if self.t.nominal.indicate_standby():
                    self.sense_data.s.delay = self.r.rng.integers(0, self.p.max_sense_delay, endpoint=True)
                    self.update_sensed('both', False)
                    if self.environment.c.get(self.location_pose.s.curr_x, self.location_pose.s.curr_y, 'explored'):
                        self.t.nominal.set_timer(self.sense_data.s.delay)
                    else:
                        power_draw_indicator = False
                elif self.t.nominal.indicate_complete():
                    self.assign_sense_data()
                    self.t.nominal.reset()
                else:
                    self.t.nominal.inc()   
                    
            if self.m.has_fault ('pose_est_loss', 'position_est_loss') or power_draw_indicator == False:
                if self.ee.s.power_draw == False:
                    self.ee.s.a = 0  
                else:
                    self.ee.s.a += 0
            else:
                self.ee.s.power_draw = True
                self.ee.s.a += 1

    
    def update_sensed (self, variable, value):
        if variable == 'both':
            self.sense_data.s.location = value
            self.sense_data.s.pose = value
        elif variable == 'location':
            self.sense_data.s.location = variable
        else:
            self.sense_data.s.pose = variable
            
    def assign_sense_data(self):
        self.update_sensed('both', True)
        self.sense_data.s.malfunc_location = False
        self.sense_data.s.malfunc_pose = False
        if self.m.has_fault ('pose_est_loss'):
            self.update_sensed('pose', False)
        if self.m.has_fault ('position_est_loss'):
            self.update_sensed('location', False)
        if self.m.has_fault('pose_est_malfunc'):
            self.sense_data.s.malfunc_location = True
        if self.m.has_fault('position_est_malfunc'):
            self.sense_data.s.malfunc_pose = True

            
class NavigateMode(Mode):
    fm_args = (
        "steering_stuck",
        "poor_wheel_alignment_right",
        "poor_wheel_alignment_left",
        "no_throttle",
        "waypoint_calc_malfunction"
        "path_planning_malfunction"
        "no_path_planning"
        "no_waypoint"
    )
    opermodes = ('idle', 'drive')
    mode: str = 'idle'


class Navigate(Function):
    __slots__ = ("ee", "environment", "location_pose", "location_pose_est", "sense_data")
    container_m = NavigateMode
    container_p = RoverParams
    flow_ee = EE
    flow_environment = RoverEnvironment
    flow_location_pose = LocationPose
    flow_sense_data = SenseData
    
    def init_block(self, **kwargs):
        self.location_pose_est = self.location_pose.create_local(self.name)

    def dynamic_behavior(self, time):
        if self.ee.s.v == 12:
            if not self.environment.c.in_range (self.p.ground_control.destination[0], self.p.ground_control.destination[1]):
                raise Exception ("Desitination is out of bounds or the defined map. Maximun x value is " + str(self.p.environment.x_size * self.p.environment.blocksize)
                                 + "Maximum y value is " + str(self.p.environment.y_size * self.p.environment.blocksize) + ".")
            if self.environment.c.get(self.p.ground_control.destination[0], self.p.ground_control.destination[1], 'explored'):
                if self.environment.c.get(self.p.ground_control.destination[0], self.p.ground_control.destination[1], 'occupied'):
                    self.m.set_mode('idle')
                else:
                    self.m.set_mode('drive')
            else:
                self.m.set_mode('drive')
            
            if self.m.in_mode('idle'):
                self.ee.s.a += 0.1
            else:
                delta_i = self.p.ground_control.destination[0] - \
                    self.location_pose.s.curr_x
                delta_j = self.p.ground_control.destination[1] - \
                    self.location_pose.s.curr_y
                magnitude = np.sqrt(delta_i**2 + delta_j**2)
                unit_i = delta_i / magnitude
                unit_j = delta_j / magnitude
                dist_x = unit_i * self.p.ground_control.speed * \
                self.t.dt
                dist_y = unit_j * self.p.ground_control.speed * \
                self.t.dt
                self.location_pose.s.curr_x = self.location_pose.s.curr_x + dist_x
                self.location_pose.s.curr_y = self.location_pose.s.curr_y + dist_y

        self.environment.c.set(self.location_pose.s.curr_x, self.location_pose.s.curr_y, 'rover_path', True)
        return


class Rover(FunctionArchitecture):
    __slots__ = ()
    container_p = RoverParams
    default_sp = dict(
        end_time=300,
        phases=(("start", 0, 30), ("end", 31, 150)),
        end_condition="indicate_finished",
        dt = 1.0
    )

    def init_architecture(self, **kwargs):
        # Flows
        self.add_flow("location_pose", LocationPose)
        self.add_flow("environment", RoverEnvironment, p=self.p)
        self.add_flow("sense_data", SenseData)
        self.add_flow("ee", EE)
        # self.add_flow("comms", Comms)
        # Functions
        self.add_fxn("supply_power", SupplyPower, "ee")
        # self.add_fxn("communicate", Communicate, "ee_comms", "comms")
        self.add_fxn("map", Map, "environment", "ee",
                     "location_pose", p=self.p.mission)
        self.add_fxn("sense", Sense, "sense_data", "ee")
        self.add_fxn(
            "navigate", Navigate, "location_pose", "ee", "environment", "sense_data", p=self.p
        )

    def indicate_finished(self, time):
        """indicates if the mission has ended"""
        end = False
        if (
            self.flows["environment"].at_finish(
                self.flows["location_pose"].s.curr_x,
                self.flows["location_pose"].s.curr_y,
            )
            or self.flows["environment"].check_collision(
                self.flows["location_pose"].s.curr_x,
                self.flows["location_pose"].s.curr_y,
                        )
            or (time > 5 and self.fxns["supply_power"].m.in_mode("standby"))
            or self.fxns["supply_power"].m.in_mode("no_charge")
        ):
            end = True

        return end

    def find_classification(self, scen, mdlhists):
        """
        calculates metrics that need to be tracked for the simulation

        Returns
        ----------
            at_finish: bool
                is true if the rover has reached the end point
            classification: str
                mission status (nominal, faulty, or incomplete)
            num_explored_obstacles: int
                number of obstacles in the explored area,
            num_perceived_obstacles: int
                number of perceived obstacles in the explored area
            num_false_positive_objects: int
                number of ghost obstacles detected
            num_false_negetive_objects: int
                number of actual obstacles not detected
--------------------------------------------------------------
            safety breaches - safety margin is 20 cm.
            Safety Dangerzone - margin 
        """

        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish = True

        # mission is incomplete if the rovr is not at the end point
        if not self.flows["environment"].at_finish(
            self.flows["location_pose"].s.curr_x, 
            self.flows["location_pose"].s.curr_y
        ):
            classification = "incomplete mission"
            at_finish = False

        # mission is fault if any fault modes are present
        if any(modes):
            classification = classification + " faulty"

        # missing is nominal in no fault modes are present
        if not classification:
            classification = "nominal mission"

        objects_explored = self.flows["environment"].c.find_all(
            explored=(True, np.equal), occupied=(True, np.equal)
        )
        perceived_objects = self.flows["environment"].c.find_all(
            perceived_objects=(True, np.equal)
        )
        
        false_negetive_objects = self.flows["environment"].c.find_all(
            perceived_objects=(False, np.equal), occupied=(True, np.equal))
        
        false_positive_objects = self.flows["environment"].c.find_all(
            perceived_objects=(True, np.equal), occupied=(False, np.equal))

        return {
            "classification": classification,
            "at_finish": at_finish,
            "num_obstacles_in_explored_area": len(objects_explored),
            "num_false_positive_obstacles": len(false_positive_objects),
            "num_false_negetive_obstacles": len(false_negetive_objects),
            "num_perceived_obstacles": len(perceived_objects),
        }


if __name__ == "__main__":
    function = SupplyPower()

    mdl = Rover()
    # mdl.flows['environment'].c.show_collection('all_occupied', label = False)
    # mdl.flows['environment'].c.show('explored', collections = {'all_occupied':{'label': False}})
    # mdl.flows['environment'].c.show_property('occupied')
    mtg = FunctionArchitectureTypeGraph(mdl)
    fig, ax = mtg.draw()
    dot = mtg.draw_graphviz()

    # ec1, hist = prop.nominal(mdl)
    # mdl.flows['environment'].c.assign_from(hist.flows.environment.c, len(hist.time)-1, 'explored')
    # mdl.flows['environment'].c.assign_from(hist.flows.environment.c, len(hist.time)-1, 'perceived_objects')

    ex2, hist2 = prop.one_fault(
        mdl, "map", "no_self_map", time=10, run_stochastic=True, seed=50
    )
    mdl.flows["environment"].c.assign_from(
        hist2.faulty.flows.environment.c, len(
            hist2.faulty.time) - 1, "explored"
    )
    mdl.flows["environment"].c.assign_from(
        hist2.faulty.flows.environment.c,
        len(hist2.faulty.time) - 1,
        "perceived_objects"
    )
    mdl.flows["environment"].c.show(
        {"explored": {}},
        collections={"all_occupied": {"label": False},
                      "all_endzone": {"label": False}},
        alpha=0.5, coll_overlay=False,
    )
    # mdl.flows['environment'].c.show_property('occupied')

    mdl.flows["environment"].c.show(
        {
            "explored": {"color": "blue", "alpha": 0.3},
            "perceived_objects": {"color": "yellow", "alpha": 0.6},
            "rover_path": {"color": "black", "alpha": 0.6},
        },
        collections={
            "all_occupied": {"label": False, "color": "red"},
            "all_endzone": {"label": False, "color": "green"},
        }, coll_overlay=False,
        linewidth=0.0,
    )
    
    animation = mdl.flows["environment"].c.animate(
        hist2.faulty.flows.environment.c, properties ={"explored": {"color": "blue", "alpha": 0.3},
    "perceived_objects": {"color": "yellow", "alpha": 0.6}, "rover_path": {"color": "black", "alpha": 0.6}
    },
    collections={
    "all_occupied": {"label": False, "color": "red"},
    "all_endzone": {"label": False, "color": "green"}}, coll_overlay=False, linewidth =0.0)
    
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
    #animation.save('Troupe_faulty_ghost_obstacle_detection.mp4', progress_callback=progress_callback)
    animation.save('Troupe_faulty_no_self_map.mp4', progress_callback=progress_callback)
    
    # animation = mdl.flows["environment"].c.animate(
    #     hist2.nominal.flows.environment.c, properties ={"explored": {"color": "blue", "alpha": 0.3},
    # "perceived_objects": {"color": "yellow", "alpha": 0.6}, "rover_path": {"color": "black", "alpha": 0.6}
    # },
    # collections={
    # "all_occupied": {"label": False, "color": "red"},
    # "all_endzone": {"label": False, "color": "green"}}, coll_overlay=False, linewidth =0.0)
    # animation.save('Troupe_nominal.mp4', progress_callback=progress_callback)
    
