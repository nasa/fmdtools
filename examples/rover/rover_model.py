#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base rover model.

Functions:
    - Power
    - Operator
    - Communications
    - Perception
    - Plan Path
    - Override
    - Drive

Flows:
    - Communications
    - Override Communications
    - Ground
    - Position Signal
    - Video
    - Electrical Energy
    - Position
    - Auto Control
    - Manual Control
    - Fault Signals
    - Switch Signals

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

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.flow.base import Flow
from fmdtools.define.object.geom import PointParam, GeomPoint, LineParam, GeomLine
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.environment import Environment
import fmdtools.sim.propagate as prop

import itertools
import numpy as np
import multiprocessing as mp
from shapely import Point


class DegParam(Parameter, readonly=True):
    """Parameters for rover degradation."""

    friction: float = 0.0
    drift: float = 0.0


class GroundParam(Parameter):
    """
    Parameter defining line for rover to follow.

    Fields
    ------
    linetype: str
        line type (sine or turn)
    amp: float
        sine amplitude
    period: float
        sine period
    radius: float
        turn radius
    x_start: float
        turn starting x-value
    y_end: float
        turn y-distance (after turn/radius) before end
    x_min: float
        minimum x-value for line generation (sine or turn)
    x_max: float
        maximum x-value for line generation (sine)
    x_res: float
        resolution for line generation
    """

    linetype: str = 'sine'
    linetype_set = ("sine", "turn")
    amp: float = 1.0
    period: float = 2 * np.pi
    radius: float = 20.0
    x_start: float = 10.0
    y_end: float = 10.0
    x_min: float = 0.0
    x_max: float = 30.0
    x_res: float = 0.1
    path_buffer_on: float = 0.2
    path_buffer_poor: float = 0.3
    path_buffer_near: float = 0.4
    dest_buffer_on: float = 1.0
    dest_buffer_near: float = 2.0

    def gen_ls_sine(self):
        """Generate coordinates in sine environment."""
        ls = tuple([[x, sin_func(x, self.amp, self.period)]
                    for x in np.arange(self.x_min, self.x_max, self.x_res)])
        return ls

    def gen_ls_turn(self):
        """Generate line coordinates in turn environment."""
        ls = [[x, turn_func(x, self.radius, self.x_start)]
              for x in np.arange(self.x_min, self.radius+self.x_start, self.x_res)]
        return tuple(ls)

    def gen_ls(self):
        """Generate line coordinates."""
        if self.linetype == 'sine':
            return self.gen_ls_sine()
        elif self.linetype == 'turn':
            return self.gen_ls_turn()
        else:
            raise Exception("Invalid line time: "+self.linetype)


class ResCorrection(Parameter):
    """
    Correction parameters given faulty states.

    Fields
    ------
    ub_f: float
        upper bound of friction
    lb_f: float
        lower bound of friction
    ub_t: float
        upper bound of transfer
    lb_t: float
        lower bound of transfer
    ub_d: float
        upper bound of drift
    lb_d: float
        lower bound of drift
    cor_d: float
        correction factor for drift 
        (if drift is present, takes values between -1 and 1)
    cor_t: float
        correction factor for transfer 
        (if transfer is low or hgih: takes values between -1 and 1)
    cor_f: float
        correction for friction (if out of bounds)
    """

    ub_f: float = 10.0
    lb_f: float = -1.0
    ub_t: float = 10.0
    lb_t: float = 0.0
    ub_d: float = 2.0
    lb_d: float = -2.0
    cor_d: float = 0.0
    cor_t: float = 0.0
    cor_f: float = 0.0

class DestParam(PointParam):
    """
    Parameter defining start and end points.

    Has a 1.0-m buffer for being 'on' the location and a 2.0-m buffer for being 'near'
    the location.
    """

    x: float = 0.0
    y: float = 0.0
    buffer_on: float = 1.0
    buffer_near: float = 2.0


def sin_func(x, amp, period):
    """Sine line function generator."""
    return amp * np.sin(x * 2 * np.pi / period)


class PathParam(LineParam):
    """Parameter defining the path."""

    xys: tuple = tuple([[x, sin_func(x, 1, 1)] for x in np.arange(0, 100, 1)])
    buffer_on: float = 0.2
    buffer_poor: float = 0.3
    buffer_near: float = 0.4


class RoverParam(Parameter):
    """Parameters for rover."""

    ground: GroundParam = GroundParam()
    correction: ResCorrection = ResCorrection()
    degradation: DegParam = DegParam()
    drive_modes: dict = {"mode_args": "set"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, strict_immutability=False, **kwargs)


def turn_func(x, radius, start):
    """Turn line function generator."""
    if x >= start + radius:
        raise Exception("x "+str(x)+" outside range 0-"+str(radius+start))
    elif x >= start:
        return radius - np.sqrt(radius**2 - (x - start) ** 2)
    elif x < start:
        return 0.0
    elif x < 0.0:
        raise Exception("x="+str(x)+" <0.0")


class DestState(State):
    """State defining whether rover is on or near point."""

    on: bool = False
    near: bool = False


class Dest(GeomPoint):
    """Start/end/point."""

    container_p = DestParam
    container_s = DestState


class GroundGeomArch(GeomArchitecture):
    """Geometry of rover environment--start, end, and line."""

    container_p = GroundParam

    def init_architecture(self, **kwargs):
        """Initialize geometry with line and start/end points."""
        ls = self.p.gen_ls()
        self.add_line('line', PathLine, p={'xys': ls,
                                           'buffer_on': self.p.path_buffer_on,
                                           'buffer_poor': self.p.path_buffer_poor,
                                           'buffer_near': self.p.path_buffer_near})
        self.add_point('start', Dest, p={'x': ls[0][0], 'y': ls[0][1],
                                         'buffer_on': self.p.dest_buffer_on,
                                         'buffer_near': self.p.dest_buffer_near})
        self.add_point('end', Dest, p={'x': ls[-1][0], 'y': ls[-1][1],
                                       'buffer_on': self.p.dest_buffer_on,
                                       'buffer_near': self.p.dest_buffer_near})


class GroundState(State):
    """state defining if rover is in bound"""
    in_bound: bool = True

class Ground(Environment):
    """Ground environment of the rover."""

    arch_ga = GroundGeomArch
    container_p = GroundParam
    container_s = GroundState

    def on_course(self, pos_state):
        """Check if the rover is on_course (i.e., within the 'on' buffer)."""
        return self.ga.lines['line'].at(pos_state.get('x', 'y'), 'on')

    def at_end(self, pos_state):
        """Check if the rover is at the end point, accounting for the 'on' buffer)."""
        return self.ga.points['end'].at(pos_state.get("x", "y"), 'on')

    def end_dist(self, pos_state):
        """Calculate minimum distance between the end point and the rover."""
        return self.ga.points['end'].on.distance(Point(pos_state.get("x", "y")))

    def max_line_dist(self, pos_hist):
        """Calculate the maximum distance the rover has deviated from its path."""
        xhist = pos_hist.x
        yhist = pos_hist.y
        max_dist = 0.0
        for i, x in enumerate(xhist):
            dist = self.ga.lines['line'].shape.distance(Point(x, yhist[i]))
            if dist > max_dist:
                max_dist = dist
        return max_dist


class PathLine(GeomLine):
    """Rover Line."""

    container_p = PathParam


class PosState(State):
    """
    Rover position states.

    Fields
    ------
    x: float
        x-position
    y: float
        y-position
    vel: float
        rover straightline velocity
    ux: float
        rover x-direction (positive if forward, negative if backward)
    uy: float
        rover y-direction (positive if forward, negative if backward)
    """

    x: float = 0.0
    y: float = 0.0
    vel: float = 0.0
    ux: float = 1.0
    uy: float = 0.0


class Pos(Flow):
    """Rover position/velocity flow."""

    __slots__ = ()
    container_s = PosState


class EEState(State):
    """Electricity state (voltage v and amperage a)."""

    v: float = 0.0
    a: float = 0.0


class EE(Flow):
    """Electricity flow."""

    __slots__ = ()
    container_s = EEState


class VideoState(State):
    """
    State of the rover video field.

    Fields
    ------
    lin_ux: float
        unit vector indicating the x direction of the line
    lin_uy: float
        unit vector indicating the y direction of the line
    lin_dx: float
        distance to the line in the x
    lin_dy: float
        distance to the lin in the y
    quality: float
        quality of feed
    """

    lin_ux: float = 1.0
    lin_uy: float = 0.0
    lin_dx: float = 0.0
    lin_dy: float = 0.0
    quality: float = 1.0


class Video(Flow):
    """Video flow."""

    __slots__ = ()
    container_s = VideoState


class ControlState(State):
    """
    Control state (left/right).

    Fields
    ------
    rpower: float
        right power
    lpower: float
        left power
    """

    rpower: float = 0.0
    lpower: float = 0.0


class Control(Flow):
    """Control flow."""

    __slots__ = ()
    container_s = ControlState


class SwitchState(State):
    """Power switch state (power on or off)."""

    power: bool = False


class Switch(Flow):
    """Power switch."""

    __slots__ = ()
    container_s = SwitchState


class CommsState(State):
    """
    Communications signal with override.

    Extends Pos and Control such that 'active' represents override activity and
    'on' represents whether data is active/passed.
    """

    pos: PosState = PosState()
    ctl: ControlState = ControlState()
    video: VideoState = VideoState()
    active: bool = False
    on: bool = False


class Comms(Flow):
    """External communications flow."""

    __slots__ = ()
    container_s = CommsState


class FaultStates(State):
    """Rover fault states (friction, transfer, and drift)."""

    transfer: float = 1.0
    friction: float = 0.0
    drift: float = 0.0


class FaultSig(Flow):
    """Rover fault signal."""

    __slots__ = ()
    container_s = FaultStates


# MODEL FUNCTIONS
class PlanPathMode(Mode):
    """
    Possible modes for rover path planning.

    Modes
    -------
    no_con : Fault
        Inability to send path planning commands
    crash : Fault
        Rover crashed or maximum deviation from the path breached
    drive : Mode
        Rover is in operation and sending position signals
    standby : Mode
        Rover is in standby
    em_off : Mode
        Rover is shut off due to an emergency
    finished : Mode
        Rover has competed the mission
    """

    fm_args = {"no_con": (1e-4, 200), "crash": (1e-4, 200)}
    opermodes = ("drive", "standby", "em_off", "finished")
    mode: str = "standby"


class PlanPathState(State):
    """
    States used in rover path planning.

    Fields
    ------
    u_self: tuple
        Position of the ownship.
    u_lin: tuple
        Position of the line.
    u_lin_dev: tuple
        Deviation of ownship from line.
    rdiff: float
        Projected turning around.
    vel_adj: float
        Velocity adjustment (for faults).
    """

    u_self: np.array = np.array([0.0, 0.0])
    u_lin: np.array = np.array([0.0, 0.0])
    u_lin_dev: np.array = np.array([0.0, 0.0])
    rdiff: float = 0.0
    vel_adj: float = 1.0

    def set_positions(self, pos_signal, video):
        """Get and group the ownship position, line position, and deviation."""
        self.u_self = pos_signal.s.get('ux', 'uy')
        self.u_lin = video.s.get('lin_ux', 'lin_uy')
        self.u_lin_dev = video.s.get('lin_dx', 'lin_dy')

    def set_turn(self):
        """Calculate turn rdiff from ownship position, line position, and deviation."""
        rdiff_track = rdiff_from_vects(self.u_self, self.u_lin)
        err_dist = np.linalg.norm(self.u_lin_dev)
        rdiff_err = rdiff_from_vects(self.u_self, self.u_lin_dev) * err_dist
        self.rdiff = rdiff_track + 0.15 * rdiff_err
        #if err_dist > 0.05:
        #    self.vel_adj = 0.2
        #else:
        #    self.vel_adj = 1.0


    def set_control(self, control):
        """Set rpower and lpower for control given intended turn and vel_adj."""
        control.s.put(rpower=self.vel_adj * (1 + (self.rdiff)),
                      lpower=self.vel_adj * (1 - (self.rdiff)))
        control.s.limit(rpower=(-1, 2), lpower=(-1, 2))


class PlanPath(Function):
    """Plan the next drive move based on the current state of the rover."""

    __slots__ = ("video", "pos_signal", "ground", "control", "fault_sig")
    container_m = PlanPathMode
    container_p = ResCorrection
    container_s = PlanPathState
    flow_video = Video
    flow_pos_signal = Pos
    flow_ground = Ground
    flow_control = Control
    flow_fault_sig = FaultSig
    flownames = {"auto_control": "control"}

    def dynamic_behavior(self, time):
        """
        Dynamic Behavior of Planning Module.

        Step 1: Assign Operational Mode.
        Step 2A: If Operational Mode is Drive determine if the rover has reached
        the end point or emergency shut off is required and assign those modes
        Step 2B: if none of the situations in step 2 applies, determines the
        control signal for left and right motor power, based on percieced line
        and current location.
        Step 4: if mode is "finished" or "em_off"" sets the left power and
        right power control signal to 0.
        """
        if not self.m.in_mode("no_con"):
            if time == 5:
                self.m.set_mode("drive")
            if time == 149:
                self.m.set_mode("standby")

        if self.m.in_mode("drive") and not self.m.in_mode("no_con"):
            if self.ground.at_end(self.pos_signal.s):
                self.m.set_mode("finished")
            elif self.video.s.quality == 0:
                self.m.set_mode("em_off")
            elif not self.faultstates_in_bounds():
                self.m.set_mode("em_off")
            else:
                self.drive_control()
        if self.m.in_mode("standby", "em_off", "finished"):
            self.control.s.put(rpower=0, lpower=0)

    def drive_control(self):
        """Control of rpower/lpower signal based on percieved line."""
        # comprehend: get u_self, u_line, u_dev
        self.s.set_positions(self.pos_signal, self.video)
        # project : calculate rdiffs and vel_adj from u_self, u_line, u_dev
        self.s.set_turn()
        # correct velocity
        self.s.vel_adj = self.correct_fault_vel(self.s.rdiff)
        # decide : rpower/lpower from rdiffs
        self.s.set_control(self.control)

    def correct_fault_vel(self, rdiff):
        """Slow down the rover when faults are detected."""
        # applying the correction factor if drift is present
        turn_fault_correction = self.p.cor_d * self.fault_sig.s.drift
        rdiff = rdiff + turn_fault_correction

        # applying a correction factor if friction or transfer is present
        # goal is to increase power if friction is present
        friction_vel_correction = ((1 - self.p.cor_f) * self.fault_sig.s.friction /
                                   (1 + self.fault_sig.s.friction))

        # increase power is transfer is below 1 and decrease power if it is more than 1
        transfer_vel_correction = ((1 + self.p.cor_t) * (self.fault_sig.s.transfer - 1)
                                   / (1 + self.fault_sig.s.transfer))
        vel_adj = 1 + friction_vel_correction - transfer_vel_correction
        return vel_adj

    def faultstates_in_bounds(self):
        """Check if fault-states are in-bounds."""
        return (self.p.lb_f <= self.fault_sig.s.friction <= self.p.ub_f
                and self.p.lb_d <= self.fault_sig.s.drift <= self.p.ub_d
                and self.p.lb_t <= self.fault_sig.s.transfer <= self.p.ub_t)



def rdiff_from_vects(u_self, u_lin):
    """Determine the needed correction to reach from point 1 to point 2."""
    d = np.dot(u_self, u_lin)
    dr = np.sign(np.cross(u_self, u_lin))
    rdiff = dr * np.arccos(d/(np.linalg.norm(u_self)*np.linalg.norm(u_lin)+0.00001))
    return rdiff


class DriveMode(Mode):
    """
    Instantiate Modes for the Drive Function.

    key_phases_by = 'plan_path' defines that the modes may be intantiated for
    certain phases of PlanPath. The phases are defined by opptvect.
    mode_args: determines if the how the modes in Drivemodes should be
    formulated (e.g., as a parameter, manually, as set of modes, etc. )
    """

    s: FaultStates = FaultStates()
    mode_args: tuple = tuple()
    deg_params: dict = dict
    fm_args = dict()

    def __init__(self, *args, mode_args=tuple(), deg_params=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        if "mode_args" in mode_args:
            self.mode_args = mode_args["mode_args"]
        else:
            self.mode_args = mode_args
        ph = {'drive': 1.0, 'start': 1.0}
        if self.mode_args == "degradation":
            self.add_degradation_modes(deg_params, ph)
        elif type(self.mode_args) is int:
            self.add_n_modes(ph)
        elif type(self.mode_args) is list:
            self.add_mode_list(ph)
        elif type(self.mode_args) is dict:
            self.init_faultstate_modes(manual_modes=self.mode_args, phases=ph)
        else:
            if "manual" in self.mode_args:
                self.add_manual_modes(ph)
            if "set" in self.mode_args:
                franges = {"friction": {1.5, 3.0, 10.0},
                           "transfer": {0.5, 0.0},
                           "drift": {-0.2, 0.2}}
                self.init_n_faultstates(franges, phases=ph)
            if "range" in self.mode_args:
                franges = {"friction": {*np.linspace(0.0, 20, 10)},
                           "transfer": {*np.linspace(1.0, 0.0, 10)},
                           "drift": {*np.linspace(-0.5, 0.5, 10)}}
                if "all" in self.mode_args:
                    self.init_n_faultstates(franges, phases=ph, n="all")
                else:
                    self.init_n_faultstates(franges, phases=ph, n=1)

    def add_manual_modes(self, ph, drift=0.0, friction=0.0):
        manual_modes = {"elec_open": {"transfer": 0.0},
                        "stuck": {"friction": 10.0+friction},
                        "stuck_right": {"friction": 3.0+friction, "drift": 0.2+drift},
                        "stuck_left": {"friction": 3.0+friction, "drift": -0.2+drift}}
        self.init_faultstate_modes(manual_modes, phases=ph)

    def add_degradation_modes(self, deg_params, ph):
        self.s.friction = deg_params.friction
        self.s.drift = deg_params.drift
        self.add_manual_modes(ph, drift=self.s.drift, friction=self.s.friction)

    def add_n_modes(self, ph):
        franges = {"friction": {*np.linspace(0.0, 20, 10)},
                   "transfer": {*np.linspace(1.0, 0.0, 10)},
                   "drift": {*np.linspace(-0.5, 0.5, 10)}}
        self.init_n_faultstates(franges, n=self.mode_args, phases=ph)

    def add_mode_list(self, ph):
        manual_modes = {"s_" + str(i):
                        {"friction": mode[0], "transfer": mode[1], "drift": mode[2]}
                        for i, mode in enumerate(self.mode_args)}
        self.init_faultstate_modes(manual_modes, phases=ph)


class Drive(Function):
    """The drive function determines the rover drive functionality."""

    __slots__ = ("ground", "motor_control", "ee_in", 'fault_sig', 'pos')
    container_m = DriveMode
    flow_fault_sig = FaultSig
    flow_ground = Ground
    flow_pos = Pos
    flow_motor_control = Control
    flow_ee_in = EE
    flownames = {"ee_15": "ee_in"}

    def dynamic_behavior(self, time):
        """Define the drive behavior for a given time step."""
        # calculate left and right motor power
        self.fault_sig.s.assign(self.m.s, "friction", "transfer", "drift")
        rpower = (self.m.s.transfer * self.ee_in.s.v * self.motor_control.s.rpower / 15
                  + self.m.s.drift)
        lpower = (self.m.s.transfer * self.ee_in.s.v * self.motor_control.s.lpower / 15
                  - self.m.s.drift)

        if self.m.has_fault("elec_open"):
            # Determine EE input based on elec_open if  fault is present
            self.ee_in.s.a = 0
        else:
            self.ee_in.s.a = (1.0 + self.m.s.friction) * (lpower + rpower) / 12

        if (lpower + rpower) > 100:
            # Set faulty behavior if motor power is beyond a threshold
            self.add_fault("elec_open")
        else:
            # determine new rover drive paramenters during nominal behaviors
            self.drive_nominal(rpower, lpower)
        self.ground.s.in_bound = self.ground.on_course(self.pos.s)

    def drive_nominal(self, rpower, lpower):
        """
        Calculate the new rover state based on left and right motor power.

        e.g.::
            >>> d = Drive()
            >>> d.pos.s
            PosState(x=0.0, y=0.0, vel=0.0, ux=1.0, uy=0.0)
            >>> d.drive_nominal(6, 6)
            >>> d.pos.s
            PosState(x=2.0, y=0.0, vel=2.0, ux=1.0, uy=0.0)

            >>> d = Drive()
            >>> d.pos.s.uy = 1
            >>> d.pos.s.ux = 0
            >>> d.pos.s
            PosState(x=0.0, y=0.0, vel=0.0, ux=0, uy=1)
            >>> d.drive_nominal(6, 6)
            >>> d.pos.s
            PosState(x=0.0, y=2.0, vel=2.0, ux=0.0, uy=1.0)

            >>> d = Drive()
            >>> d.pos.s.uy = -1
            >>> d.pos.s.ux = 0
            >>> d.pos.s
            PosState(x=0.0, y=0.0, vel=0.0, ux=0, uy=-1)
            >>> d.drive_nominal(6, 6)
            >>> d.pos.s
            PosState(x=0.0, y=-2.0, vel=2.0, ux=0.0, uy=-1.0)
        """
        # the division by 6 slows down the rover, so it can manage the course
        self.pos.s.vel = (rpower + lpower) / (5 + (1 + self.m.s.friction))
        ang_inc = np.arctan((rpower - lpower) / (rpower + lpower + 0.001))
        ux = np.cos(ang_inc) * self.pos.s.ux - np.sin(ang_inc) * self.pos.s.uy
        uy = np.sin(ang_inc) * self.pos.s.ux + np.cos(ang_inc) * self.pos.s.uy
        mag_u = np.linalg.norm([ux, uy])
        self.pos.s.put(ux=ux/mag_u, uy=uy/mag_u)
        self.pos.s.inc(x=self.pos.s.ux*self.pos.s.vel, y=self.pos.s.uy*self.pos.s.vel)

class PerceptionMode(Mode):
    """
    Possible modes for Perception function.

    Modes
    -------
    bad_feed : Fault
        poor video feed
    off: Mode
        video feed is off
    feed : Mode
        video feed is on

    Other Features
    -----------
    exlusive = True: Only one mode can be active at any given time
    """

    fm_args = ("bad_feed",)
    opermodes = ("off", "feed")
    mode: str = "off"
    exclusive = True


class Perception(Function):
    """Rover function that percieves the environment and creates the video feed."""

    __slots__ = ("ground", "ee", "video", 'pos', 'pos_signal')
    container_m = PerceptionMode
    flow_pos = Pos
    flow_pos_signal = Pos
    flow_ground = Ground
    flow_ee = EE
    flow_video = Video
    flownames = {"ee_12": "ee"}

    def dynamic_behavior(self, time):
        """Set the video feed based on the behavior mode at each timestep."""
        # Nominal Behavior
        if self.ee.s.v > 8:
            self.m.set_mode("feed")
        elif self.ee.s.v == 0:
            self.m.set_mode("off")

        if self.m.in_mode("off"):
            self.ee.s.a = 0
            self.video.s.put(lin_ux=0.0, lin_uy=0.0,
                             lin_dx=0.0, lin_dy=0.0, quality=0.0)
        elif self.m.in_mode("feed"):
            if self.ground.on_course(self.pos.s):
                self.get_line_ang()
            else:
                # Video quality drops off if rover is off course
                self.video.s.quality = 0.0
            self.pos_signal.s.assign(self.pos.s, "x", "y", "ux", "uy")

        # Faulty Behavior
        elif self.m.has_fault("bad_feed"):
            self.video.s.quality = 0.5

    def get_line_ang(self):
        """Determine the video feed states."""
        xy = self.pos.s.get('x', 'y')
        # calculate the unit vector to determine the direction of the line at
        # the nearest point in the line from the current location
        ux, uy = self.ground.ga.lines['line'].vect_at_shape(xy)
        # calculates the direction the rover needs to travel to reach the
        # nearest point in the line
        dx, dy = self.ground.ga.lines['line'].vect_to_shape(xy)
        self.video.s.put(lin_ux=ux[0], lin_uy=uy[0],
                         lin_dx=dx[0], lin_dy=dy[0], quality=1.0)


class PowerState(State):
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


class PowerMode(Mode):
    """
    Possible modes for Power function.

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

    fm_args = {"no_charge": (1e-5, 100, {"off": 1.0}),
               "short": (1e-5, 100, {"supply": 1.0})}
    opermodes = ("supply", "charge", "off")
    mode: str = "off"
    exclusive = True


class Power(Function):
    """Rover power supply."""

    __slots__ = ("ee_15", "ee_5", "ee_12", "switch")
    container_s = PowerState
    container_m = PowerMode
    flow_ee_15 = EE
    flow_ee_5 = EE
    flow_ee_12 = EE
    flow_switch = Switch

    def static_behavior(self, time):
        """Determine power use based on mode."""
        if self.m.in_mode("off"):
            self.off_power()
        elif self.m.in_mode("supply"):
            self.supply_power()
        elif self.m.in_mode("short"):
            self.short_power()
        elif self.m.in_mode("no_charge"):
            self.no_charge_power()

        if self.m.in_mode("charge"):
            self.charge_power_usage()
        else:
            self.power_usage()
            if self.m.in_mode("short"):
                self.short_power_usage()

    def dynamic_behavior(self, time):
        """Charge increment over time."""
        self.s.inc(charge=-self.s.power / 100)
        self.s.limit(charge=(0, 100))

    def short_power(self):
        """Power in case of a short has normal voltage."""
        self.ee_5.s.v = 5
        self.ee_12.s.v = 12
        self.ee_15.s.v = 15

    def no_charge_power(self):
        """Battery is out of charge."""
        self.ee_5.s.v = 0
        self.ee_12.s.v = 0
        self.ee_15.s.v = 0

    def off_power(self):
        """Power supply is shut off."""
        self.ee_5.s.put(v=0, a=0)
        self.ee_12.s.put(v=0, a=0)
        self.ee_15.s.put(v=0, a=0)
        if self.switch.s.power:
            self.m.set_mode("supply")

    def supply_power(self):
        """Power supply is in supply mode."""
        if self.s.charge > 0:
            self.ee_5.s.v = 5
            self.ee_12.s.v = 12
            self.ee_15.s.v = 15
        else:
            self.m.set_mode("no_charge")
        if not self.switch.s.power:
            self.m.set_mode("off")

    def power_usage(self):
        """Calculate the power usage in general."""
        self.s.power = (1.0 + self.ee_12.s.mul("v", "a") +
                        self.ee_5.s.mul("v", "a") + self.ee_15.s.mul("v", "a"))

    def charge_power_usage(self):
        """Calculate the power usage when the battery charges."""
        self.s.power = -1
        if self.s.charge == 100:
            self.m.set_mode("off")

    def short_power_usage(self):
        """Calculate power usage when there is a short (calculated as double)."""
        self.s.power = self.s.power * 2
        if self.s.charge == 0:
            self.m.set_mode("no_charge")
        if not self.switch.s.power:
            self.m.set_mode("off")


class OverrideMode(Mode):
    """
    Possible modes for Override function.

    Modes
    -------
    override: Mode
        override rover control
    standby: Mode
        override is in stand by
    off: Mode
        override is off
    """

    opermodes = ("off", "standby", "override")
    mode: str = "off"


class Override(Function):
    __slots__ = ("comms", "ee", "motor_control", "auto_control")
    container_m = OverrideMode
    flow_comms = Comms
    flow_ee = EE
    flow_motor_control = Control
    flow_auto_control = Control
    flownames = {"ee_5": "ee"}

    def dynamic_behavior(self, time):
        if self.ee.s.v > 4:
            if self.comms.s.active and self.comms.s.on:
                self.m.set_mode("override")
            else:
                self.m.set_mode("standby")
        else:
            self.m.set_mode("off")

        if self.m.in_mode("off"):
            self.ee.s.a = 0
        elif self.m.in_mode("standby"):
            self.ee.s.a = 1
            self.motor_control.s.assign(self.auto_control.s, "rpower", "lpower")
        elif self.m.in_mode("override"):
            self.ee.s.a = 1
            self.motor_control.s.assign(self.comms.s.ctl, "rpower", "lpower")


class Communications(Function):
    """Rover communications to the user."""

    __slots__ = ("ee_12", "comms", "pos_signal", "video")
    flow_ee_12 = EE
    flow_comms = Comms
    flow_pos_signal = Pos
    flow_video = Video

    def dynamic_behavior(self, time):
        """When active, convert position signals to communication signals."""
        if self.ee_12.s.v == 12:
            self.ee_12.s.a = 1
            self.comms.s.pos.assign(self.pos_signal.s, "x", "y", "vel", "ux", "uy")
            self.comms.s.video.assign(self.video.s)
            self.comms.s.on = True
        else:
            self.comms.s.pos.put(x=0, y=0, vel=0, ux=0, uy=0)
            self.comms.s.on = False


class Operator(Function):
    """Operator turns on or off the rover."""

    __slots__ = ("switch",)
    flow_switch = Switch

    def set_power(self, t):
        if t == 1:
            self.switch.s.power = True
        elif t == 200:
            self.switch.s.power = False

    def dynamic_behavior(self, t):
        self.set_power(t)


def gen_model_params(x, scen):
    """Generate model parameters for the scenario."""
    params = {"drive_modes": {"custom_fault": {"friction": x[scen][0][0],
                                               "drift": x[scen][0][1],
                                               "transfer": x[scen][0][2]}}}
    return params


class Rover(FunctionArchitecture):
    """
    Overall rover functional architecture.

    The functions override, communications, and operator are place holders
    in this model and do not affect the Rover Behavior. The remaining functions
    are setup to mimic a rover that percieved its environment through a video
    and follows a predefined path automatically.

    To study how the rover behaves when when faults occur, the simulation
    can be run with individual faults or combinations of faults that can be
    instantiated at different timesteps (or phases of operation).

    To study how the rover behaviors when environmental disturbances and
    external factors are affecting it, the model may be simulated with
    varying parameters. (e.g., friction can be increased if a friction on the
                         ground is expected to be high).
    """

    __slots__ = ()
    container_p = RoverParam
    default_sp = dict(end_time=150,
                      phases=(("start", 0, 30), ("end", 31, 150)),
                      end_condition="indicate_finished")

    def init_architecture(self, **kwargs):
        """Initialize the functional architecture."""
        self.add_flow("ground", Ground, p=self.p.ground)
        self.add_flow("pos_signal", Pos)
        self.add_flow('pos', Pos)
        self.add_flow("ee_12", EE)
        self.add_flow("ee_5", EE)
        self.add_flow("ee_15", EE)
        self.add_flow("video", Video)
        self.add_flow("auto_control", Control)
        self.add_flow("motor_control", Control)
        self.add_flow("switch", Switch)
        self.add_flow("comms", Comms)
        self.add_flow("fault_sig", FaultSig)

        self.add_fxn("power", Power, "ee_15", "ee_5", "ee_12", "switch")
        self.add_fxn("perception", Perception, "ground", 'pos', 'pos_signal',
                     "ee_12", "video")
        self.add_fxn("communications", Communications, "comms", "ee_12", "pos_signal",
                     "video")
        self.add_fxn("operator", Operator, "switch")
        self.add_fxn("plan_path", PlanPath, "video", "pos_signal", "ground",
                     "auto_control", "fault_sig", p=self.p.correction)
        self.add_fxn("override", Override, "comms", "ee_5", "motor_control",
                     "auto_control")
        drive_m = {"mode_args": self.p.drive_modes, 'deg_params': self.p.degradation}
        self.add_fxn("drive", Drive, "ground", 'pos', "ee_15", "motor_control",
                     "fault_sig", m=drive_m)

    def indicate_finished(self, time):
        """Determine if the rover has completed its mission (successful or not)."""
        if  (self.flows['ground'].at_end(self.flows['pos'].s)
             or (time > 5 and self.fxns["plan_path"].m.in_mode("standby"))
             or self.fxns["plan_path"].m.in_mode("em_off", "finished")):
            return True
        else:
            return False

    def find_classification(self, scen, mdlhist):
        """
        calculates metrics that need to be tracked for the simulation

        Returns
        ----------
        endclass : dict
            Dictionary with keys/values:
            "rate": float
                failure rate of the scenario
            "cost": float
                cost of the resulting failure
            "prob": float
                scenario probability
            "expected_cost": float
                expected cost of the failure
            "in_bound": bool
                is true if the rover is within the path bounds
            "at_finish": bool
                is true if the rover has reached the end point
            "line_dist": float
                line distance
            "num_modes": int
                numbers of faut modes present
            "end_dist": float
                minimum distance between rover and end point,
            "tot_deviation": float
                total deviation from the path through the mission
            "faults": list
                fault modes present
            "classification": str
                mission status (nominal, faulty, or incomplete)
            "endpt": list
                rovers last position (or end point),
        """
        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish = True

        # mission is incomplete if the rovr is not at the end point
        if not self.flows['ground'].at_end(self.flows['pos'].s):
            classification = "incomplete mission"
            at_finish = False

        # mission is fault if any fault modes are present
        if any(modes):
            classification = classification + " faulty"

        # missing is nominal in no fault modes are present
        if not classification:
            classification = "nominal mission"
        num_modes = len(modes)

        end_dist = self.flows['ground'].end_dist(self.flows['pos'].s)

        endpt = [self.flows["pos"].s.x, self.flows["pos"].s.y]

        in_bound = all(mdlhist.faulty.flows.ground.s.in_bound)
        line_dist = 1  # looks like this is not used. Is this needed? TODO: reimplement
        hist_xy = mdlhist.faulty.flows.pos.s.get('x', 'y')
        tot_deviation = self.flows['ground'].max_line_dist(hist_xy)
        return {"rate": scen.rate,
                "cost": 0,
                "prob": scen.prob,
                "expected_cost": 0,
                "in_bound": in_bound,
                "at_finish": at_finish,
                "line_dist": line_dist,
                "num_modes": num_modes,
                "end_dist": end_dist,
                "tot_deviation": tot_deviation,
                "faults": modes,
                "classification": classification,
                "end_x": endpt[0],
                "end_y": endpt[1],
                "endpt": endpt}


def gen_param_space():
    """Generate parameter space when a range of parameters needs to be simulated."""
    paramspace = []
    ranges = [x for x in itertools.product(np.arange(0, 10, 0.2), range(10, 50, 10))]
    for r in ranges:
        params = RoverParam(linetype="sine", amp=r[0], wavelength=r[1])
        paramspace.append(params)
    ranges = [x for x in itertools.product(range(5, 40, 5), range(0, 5, 20))]
    for r in ranges:
        params = RoverParam(linetype="turn", radius=r[0], start=r[1])
        paramspace.append(params)
    return paramspace


def plot_map(mdl, hists):
    fig, ax = hists.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y',
                                      time_groups=['nominal'])
    geoms = {'line': {'shapes': {'on': {}, 'shape': {}}}, 'start': {}, 'end': {}}
    fig, ax = mdl.flows['ground'].ga.show(geoms=geoms, fig=fig, ax=ax)
    return fig, ax


if __name__ == "__main__":
    import multiprocessing as mp
    from fmdtools.analyze import tabulate
    from fmdtools.sim.sample import SampleApproach, FaultDomain, FaultSample

    mdl = Rover()
    ec1, hist1 = prop.nominal(mdl)
    fig, ax = hist1.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y',
                                     time_groups=['nominal'])
    fig, ax = plot_map(mdl, hist1)
    ax.set_xlim(0,1.5)
    ax.set_ylim(0,1.5)
    fig.show()

    import doctest
    doctest.testmod(verbose=True)

    mdl.flows['ground'].ga.show(fig=fig, ax=ax)
    fig, ax = hist1.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y')

    mdl = Rover(p={'ground': GroundParam(linetype='turn')})
    ec, hist = prop.nominal(mdl)

    geoms = {'line': {'shapes': {'on': {}, 'shape': {}}}, 'start': {}, 'end': {}}
    fig, ax = mdl.flows['ground'].ga.show(geoms=geoms)

    fig, ax = hist.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y',
                                      time_groups=['nominal'], fig=fig, ax=ax)
    # ax.set_xlim(0, 5.0)
    # ax.set_ylim(-2.5, 2.5)
    hist.flows.pos.s.x
    fig, ax = hist.plot_trajectories('flows.pos.s.x', 'flows.pos.s.y')

    from fmdtools.sim.sample import ParameterSample, ParameterDomain
    pd_sine = ParameterDomain(RoverParam)
    pd_sine.add_constant("ground.linetype", "sine")
    pd_sine.add_variables("ground.amp", "ground.period",
                          lims={"ground.amp": (0, 8), "ground.period": (10, 50)})

    ps_sine = ParameterSample(pd_sine)
    comb_kwargs = {'resolutions': {'ground.amp': 2, "ground.period": 20}}
    ps_sine.add_variable_ranges(comb_kwargs=comb_kwargs)

    res, hist = prop.parameter_sample(mdl, ps_sine)
    comp = tabulate.Comparison(res, ps_sine,
                                metrics=['end_dist'],
                                factors=['p.ground.amp', 'p.ground.period'])
    res.plot_metric_dist('tot_deviation')
    # comp.sort_by_factor('p.ground.amp')
    comp.as_table()
    comp.as_plot("end_dist")

    comp1 = tabulate.Comparison(res, ps_sine,
                                metrics=['end_dist', "rate"],
                                factors=['p.ground.amp', 'p.ground.period'])
    # comp.sort_by_factor('p.ground.amp')
    comp1.as_table()
    # comp1.sort_by_factor('p.ground.period')
    # comp1.sort_by_factor('p.ground.amp')
    comp1.sort_by_factors()
    comp1.as_plot("end_dist")
    comp1.as_plot("end_dist", color_factor="p.ground.period")
    comp1.as_plot("end_dist", color_factor="p.ground.amp")
    comp1.as_plots("end_dist", "rate", color_factor="p.ground.amp")
    comp1.as_plots("end_dist", "rate", color_factor="p.ground.amp", title="hi", v_padding=0.4)
