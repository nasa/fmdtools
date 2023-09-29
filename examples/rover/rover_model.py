# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:22:05 2021

@author: dhulse

Functions:
    - Communications
    - Avionics
    - Camera/Guidance
    - Structures
    - Power
    - Thermal?

Flows:
    - Communications
    - ground
    - Force
    - EE
    - Camera
"""
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.mode import Mode
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow
from fmdtools.define.geom import PointParam, GeomPoint, LineParam, GeomLine, GeomArch
from fmdtools.define.environment import Environment
from fmdtools.sim.approach import SampleApproach
import fmdtools.analyze as an
import fmdtools.sim.propagate as prop
import itertools
import numpy as np
import matplotlib.pyplot as plt
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
        maximum x-value for line generation (sine or turn)
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

    def gen_ls_sine(self):
        """Generate coordinates in sine environment."""
        ls = tuple([[x, sin_func(x, self.amp, self.period)]
                    for x in np.arange(self.x_min, self.x_max, self.x_res)])
        return ls

    def gen_ls_turn(self):
        """Generate line coordinates in turn environment"""
        ls = [[x, turn_func(x, self.radius, self.x_start)]
              for x in np.arange(self.x_min, self.x_max, self.x_res)]
        if self.x_max >= self.x_start + self.radius:
            ls.append([self.x_max, self.radius + self.y_end])
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
        correction factor for drift (if out of bounds)
    cor_t: float
        correction factor for transfer (if out of bounds)
    cor_f: float
        correction for friction (if out of bounds)
    """

    ub_f: float = 10.0
    lb_f: float = -1.0
    ub_t: float = 10.0
    lb_t: float = -1.0
    ub_d: float = 2.0
    lb_d: float = -2.0
    cor_d: float = 1.0
    cor_t: float = 1.0
    cor_f: float = 1.0


class RoverParam(Parameter):
    """Parameters for rover."""

    ground: GroundParam = GroundParam()
    correction: ResCorrection = ResCorrection()
    degradation: DegParam = DegParam()
    drive_modes: dict = {"mode_args": "set"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, strict_immutability=False, **kwargs)


def sin_func(x, amp, period):
    """Sine line function generator"""
    return amp * np.sin(x * 2 * np.pi / period)


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


class DestState(State):
    """State defining whether rover is on or near point."""

    on: bool = False
    near: bool = False


class Dest(GeomPoint):
    """Start/end/point."""

    _init_p = DestParam
    _init_s = DestState


class PathParam(LineParam):
    """Parameter defining the path."""

    xys: tuple = tuple([[x, sin_func(x, 1, 1)] for x in np.arange(0, 100, 1)])
    buffer_on: float = 0.2
    buffer_poor: float = 0.3
    buffer_near: float = 0.4


class GroundGeomArch(GeomArch):
    """Geometry of rover environment--start, end, and line."""

    _init_p = GroundParam

    def init_geoms(self, **kwargs):
        """Initialize geometry with line and start/end points."""
        ls = self.p.gen_ls()
        self.add_geom('line', PathLine, p={'xys': ls})
        self.add_geom('start', Dest, p={'x': ls[0][0], 'y': ls[0][1]})
        self.add_geom('end', Dest, p={'x': ls[-1][0], 'y': ls[-1][1]})


class GroundState(State):
    in_bound: bool = True


class Ground(Environment):
    """Ground environment of the rover."""

    _init_ga = GroundGeomArch
    _init_p = GroundParam
    _init_s = GroundState

    def on_course(self, pos_state):
        return self.ga.geoms['line'].at(pos_state.get('x', 'y'), 'on')

    def at_end(self, pos_state):
        return self.ga.geoms['end'].at(pos_state.get("x", "y"), 'on')

    def end_dist(self, pos_state):
        return self.ga.geoms['end'].on.distance(Point(pos_state.get("x", "y")))

    def max_line_dist(self, pos_hist):
        xhist = pos_hist.x
        yhist = pos_hist.y
        max_dist = 0.0
        for i, x in enumerate(xhist):
            dist = self.ga.geoms['line'].shape.distance(Point(x, yhist[i]))
            if dist > max_dist:
                max_dist = dist
        return max_dist


class PathLine(GeomLine):
    """Rover Line."""

    _init_p = PathParam


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

    _init_s = PosState


class Pos_Signal(Flow):
    """Rover position signal flow."""

    _init_s = PosState


class EEState(State):
    """Electricity state (voltage v and amperage a)."""

    v: float = 0.0
    a: float = 0.0


class EE(Flow):
    """Electricity flow."""

    _init_s = EEState


class VideoState(State):
    """
    State of the rover video field.

    Fields
    ------
    lin_ux: float
        angle of the line in the x
    lin_uy: float
        angle of the line in the y
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

    _init_s = VideoState


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

    _init_s = ControlState


class SwitchState(State):
    """Power switch state (power on or off)."""

    power: float = 0.0


class Switch(Flow):
    """Power switch."""

    _init_s = SwitchState


class Comms(Flow):
    """External communications flow."""

    _init_s = PosState


class OverrideState(ControlState):
    """Override signal. 'active' represents activity."""

    active: float = 0.0


class OverrideComms(Flow):
    """Override communications flow."""

    _init_s = OverrideState


class FaultStates(State):
    """Rover fault states (friction, transfer, and drift)."""

    transfer: float = 1.0
    friction: float = 1.0
    drift: float = 0.0


class FaultSig(Flow):
    """Rover fault signal."""

    _init_s = FaultStates


# MODEL FUNCTIONS
class PlanPathMode(Mode):
    faultparams = {"no_con": (1e-4, 200), "crash": (1e-4, 200)}
    opermodes = ("drive", "standby", "em_off", "finished")
    mode: str = "standby"


class PlanPath(FxnBlock):
    __slots__ = ("video", "pos_signal", "ground", "control", "fault_sig", 'pos')
    _init_m = PlanPathMode
    _init_p = ResCorrection
    _init_video = Video
    _init_pos_signal = Pos_Signal
    _init_pos = Pos
    _init_ground = Ground
    _init_control = Control
    _init_fault_sig = FaultSig
    flownames = {"auto_control": "control"}

    def dynamic_behavior(self, time):
        if not self.m.in_mode("no_con"):
            if time == 5:
                self.m.set_mode("drive")
            if time == 149:
                self.m.set_mode("standby")

        if self.m.in_mode("drive"):
            self.pos_signal.s.assign(self.pos.s, "x", "y", "ux", "uy")

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
        u_self = self.pos_signal.s.get('ux', 'uy')
        u_lin = self.video.s.get('lin_ux', 'lin_uy')
        rdiff_track = rdiff_from_vects(u_self, u_lin)
        
        u_lin_dev = self.video.s.get('lin_dx', 'lin_dy')
        rdiff_err = rdiff_from_vects(u_self, u_lin_dev) * np.linalg.norm(u_lin_dev)
        
        rdiff = rdiff_track + 0.15 * rdiff_err
        
        turn_fault_correction = self.p.cor_d * self.fault_sig.s.drift
        vel_fault_correction = (self.p.cor_f * (self.fault_sig.s.friction)
                                + self.p.cor_t * (self.fault_sig.s.transfer - 1))
        vel_adj = max(0.2, 1 - 0.9 * abs(rdiff_err * 50)) * vel_fault_correction
        self.control.s.put(rpower=vel_adj * (1 + (rdiff)),
                           lpower=vel_adj * (1 - (rdiff)))
        self.control.s.limit(rpower=(-1, 2), lpower=(-1, 2))

    def faultstates_in_bounds(self):
        """Check if fault-states are in-bounds."""
        return (self.p.lb_f <= self.fault_sig.s.friction <= self.p.ub_f
                and self.p.lb_d <= self.fault_sig.s.drift <= self.p.ub_d
                and self.p.lb_t <= self.fault_sig.s.transfer <= self.p.ub_t)


def rdiff_from_vects(u_self, u_lin):
    d = np.dot(u_self, u_lin)
    dr = np.sign(np.cross(u_self, u_lin))
    rdiff = dr * np.arccos(d/(np.linalg.norm(u_self)*np.linalg.norm(u_lin)+0.00001))
    return rdiff
    


class DriveMode(Mode):
    s: FaultStates = FaultStates()
    mode_args: tuple = tuple()
    deg_params: dict = dict
    faultparams = dict()
    key_phases_by = "plan_path"

    def __init__(self, *args, mode_args=tuple(), deg_params=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        if "mode_args" in mode_args:
            self.mode_args = mode_args["mode_args"]
        else:
            self.mode_args = mode_args
        if self.mode_args == "degradation":
            self.s.friction = deg_params.friction
            self.s.drift = deg_params.drift
            self.assoc_faultstates(
                {
                    "friction": {
                        (self.s.friction + 0.5),
                        2 * (self.s.friction + 0.5),
                        5 * (self.s.friction + 0.5),
                    },
                    "transfer": {0.0},
                    "drift": {self.s.drift + 0.2, self.s.drift - 0.2},
                },
                "all",
                oppvect={"drive": 1.0},
            )
        elif type(self.mode_args) == int:
            self.assoc_faultstates(
                {
                    "friction": {*np.linspace(0.0, 20, 100)},
                    "transfer": {*np.linspace(1.0, 0.0, 100)},
                    "drift": {*np.linspace(-0.5, 0.5, 100)},
                },
                self.mode_args,
                oppvect={"drive": 1.0},
            )
        elif type(self.mode_args) == list:
            self.assoc_faultstates(
                manual_modes={
                    "s_"
                    + str(i): {
                        "friction": mode[0],
                        "transfer": mode[1],
                        "drift": mode[2],
                    }
                    for i, mode in enumerate(self.mode_args)
                },
                oppvect={"drive": 1.0},
            )
        elif type(self.mode_args) == dict:
            self.assoc_faultstates(manual_modes=self.mode_args, oppvect={"drive": 1.0})
        else:
            if "manual" in self.mode_args:
                self.assoc_faultstates(
                    manual_modes={
                        "elec_open": {"transfer": 0.0},
                        "stuck": {"friction": 10.0},
                        "stuck_right": {"friction": 3.0, "drift": 0.2},
                        "stuck_left": {"friction": 3.0, "drift": -0.2},
                    },
                    oppvect={"drive": 1.0},
                )
            if "set" in self.mode_args:
                self.assoc_faultstates(
                    {
                        "friction": {1.5, 3.0, 10.0},
                        "transfer": {0.5, 0.0},
                        "drift": {-0.2, 0.2},
                    },
                    "all",
                    oppvect={"drive": 1.0},
                )
            if "range" in self.mode_args:
                if "all" in kwargs["drive_modes"]:
                    self.assoc_faultstates(
                        {
                            "friction": np.linspace(0.0, 20, 10),
                            "transfer": np.linspace(1.0, 0.0, 10),
                            "drift": np.linspace(-0.5, 0.5, 10),
                        },
                        "all",
                        oppvect={"drive": 1.0},
                    )
                else:
                    self.assoc_faultstates(
                        {
                            "friction": np.linspace(0.0, 20, 100),
                            "transfer": np.linspace(1.0, 0.0, 100),
                            "drift": np.linspace(-0.5, 0.5, 100),
                        },
                        1000,
                        oppvect={"drive": 1.0},
                    )


class Drive(FxnBlock):
    __slots__ = ("ground", "motor_control", "ee_in", 'fault_sig', 'pos')
    _init_m = DriveMode
    _init_fault_sig = FaultSig
    _init_ground = Ground
    _init_pos = Pos
    _init_motor_control = Control
    _init_ee_in = EE
    flownames = {"ee_15": "ee_in"}

    def dynamic_behavior(self, time):
        self.fault_sig.s.assign(self.m.s, "friction", "transfer", "drift")
        rpower = (self.m.s.transfer * self.ee_in.s.v * self.motor_control.s.rpower / 15
                  + self.m.s.drift)
        lpower = (self.m.s.transfer * self.ee_in.s.v * self.motor_control.s.lpower / 15
                  - self.m.s.drift)

        if self.m.has_fault("elec_open"):
            self.ee_in.s.a = 0
        else:
            self.ee_in.s.a = (1.0 + self.m.s.friction) * (lpower + rpower) / 12
        if (lpower + rpower) > 100:
            self.add_fault("elec_open")
        else:
            self.drive_nominal(rpower, lpower)
        self.ground.s.in_bound = self.ground.on_course(self.pos.s)

    def drive_nominal(self, rpower, lpower):
        self.pos.s.vel = (rpower + lpower) / (1.0 + self.m.s.friction)
        ang_inc = np.arctan((rpower - lpower) / (rpower + lpower + 0.001))
        ux = np.cos(ang_inc) * self.pos.s.ux - np.sin(ang_inc) * self.pos.s.uy
        uy = np.sin(ang_inc) * self.pos.s.ux + np.cos(ang_inc) * self.pos.s.uy
        mag_u = np.linalg.norm([ux, uy])
        self.pos.s.put(ux=ux/mag_u, uy=uy/mag_u)
        self.pos.s.inc(x=self.pos.s.ux*self.pos.s.vel, y=self.pos.s.uy*self.pos.s.vel)


class PerceptionMode(Mode):
    faultparams = ("bad_feed",)
    opermodes = ("off", "feed")
    mode: str = "off"
    exclusive = True


class Perception(FxnBlock):
    __slots__ = ("ground", "ee", "video", 'pos')
    rad = 1
    _init_m = PerceptionMode
    _init_pos = Pos
    _init_ground = Ground
    _init_ee = EE
    _init_video = Video
    flownames = {"ee_12": "ee"}

    def dynamic_behavior(self, time):
        if self.m.in_mode("off"):
            self.ee.s.a = 0
            self.video.s.put(lin_ux=0.0, lin_uy=0.0,
                             lin_dx=0.0, lin_dy=0.0, quality=0.0)
            if self.ee.s.v == 12:
                self.m.set_mode("feed")
        elif self.m.in_mode("feed"):
            if self.ee.s.v > 8:
                if self.ground.on_course(self.pos.s):
                    self.get_line_ang()
                else:
                    self.video.s.quality = 0.0
            elif self.ee.s.v == 0:
                self.m.set_mode("off")
        elif self.m.has_fault("bad_feed"):
            self.video.quality = 0.5

    def get_line_ang(self):
        xy = self.pos.s.get('x', 'y')
        ux, uy = self.ground.ga.geoms['line'].vect_at_shape(xy)
        dx, dy = self.ground.ga.geoms['line'].vect_to_shape(xy)
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
    faultparams = {
        "no_charge": (1e-5, {"standby": 1.0}, 100),
        "short": (1e-5, {"supply": 1.0}, 100),
    }
    opermodes = ("supply", "charge", "standby", "off")
    mode: str = "off"
    exclusive = True
    key_phases_by = "self"


class Power(FxnBlock):
    """Rover power supply."""

    __slots__ = ("ee_15", "ee_5", "ee_12", "switch")
    _init_s = PowerState
    _init_m = PowerMode
    _init_ee_15 = EE
    _init_ee_5 = EE
    _init_ee_12 = EE
    _init_switch = Switch

    def static_behavior(self, time):
        """Determining power use based on mode."""
        if self.m.in_mode("off"):
            self.off_power()
        elif self.m.in_mode("supply"):
            self.supply_power()
        elif self.m.in_mode("short"):
            self.short_power()
        elif self.m.in_mode("no_charge"):
            self.no_charge_power()

        if self.m.in_mode("charge"):
            self.charge_power()
        else:
            self.on_power()

    def dynamic_behavior(self, time):
        """Charge increment over time."""
        self.s.inc(charge=-self.s.power / 100)
        self.s.limit(charge=(0, 100))

    def short_power(self):
        self.ee_5.s.v = 5
        self.ee_12.s.v = 12
        self.ee_15.s.v = 15

    def no_charge_power(self):
        self.ee_5.s.v = 0
        self.ee_12.s.v = 0
        self.ee_15.s.v = 0

    def off_power(self):
        self.ee_5.s.put(v=0, a=0)
        self.ee_12.s.put(v=0, a=0)
        self.ee_15.s.put(v=0, a=0)
        if self.switch.s.power == 1:
            self.m.set_mode("supply")

    def supply_power(self):
        if self.s.charge > 0:
            self.ee_5.s.v = 5
            self.ee_12.s.v = 12
            self.ee_15.s.v = 15
        else:
            self.m.set_mode("no_charge")
        if self.switch.s.power == 0:
            self.m.set_mode("off")
        
    def on_power(self):
        self.s.power = (1.0 + self.ee_12.s.mul("v", "a") +
                        self.ee_5.s.mul("v", "a") + self.ee_15.s.mul("v", "a"))

    def charge_power(self):
        self.s.power = -1
        if self.s.charge == 100:
            self.m.set_mode("off")
        



class OverrideMode(Mode):
    opermodes = ("off", "standby", "override")
    mode: str = "off"


class Override(FxnBlock):
    __slots__ = ("override_comms", "ee", "motor_control", "auto_control")
    _init_m = OverrideMode
    _init_override_comms = OverrideComms
    _init_ee = EE
    _init_motor_control = Control
    _init_auto_control = Control
    flownames = {"ee_5": "ee"}

    def dynamic_behavior(self, time):
        if self.m.in_mode("off"):
            self.ee.s.a = 0
            if self.ee.s.v == 5:
                self.m.set_mode("standby")
        elif self.m.in_mode("standby"):
            self.motor_control.s.assign(self.auto_control.s, "rpower", "lpower")
            if self.override_comms == "active" and self.EE.s.v > 4:
                self.m.set_mode("override")
        elif self.m.in_mode("override"):
            self.motor_control.s.assign(self.override_comms.s, "rpower", "lpower")


class Communications(FxnBlock):
    __slots__ = ("ee_12", "comms", "pos_signal")
    _init_ee_12 = EE
    _init_comms = Comms
    _init_pos_signal = Pos_Signal

    def dynamic_behavior(self, time):
        if self.ee_12.s.v == 12:
            self.ee_12.s.a = 1
            self.comms.s.assign(self.pos_signal.s, "x", "y", "vel", "ux", "uy")
        else:
            self.comms.s.put(x=0, y=0, vel=0, ux=0, uy=0)


class Operator(FxnBlock):
    __slots__ = ("switch",)
    _init_switch = Switch

    def dynamic_behavior(self, t):
        if t == 1:
            self.switch.s.power = 1
        elif t == 200:
            self.switch.s.power = 0


def gen_model_params(x, scen):
    params = {"drive_modes": {"custom_fault": {"friction": x[scen][0][0],
                                               "drift": x[scen][0][1],
                                               "transfer": x[scen][0][2]}}}
    return params


class Rover(Model):
    __slots__ = ()
    _init_p = RoverParam
    default_sp = dict(times=(0, 150),
                      phases=(("start", 0, 30), ("end", 31, 150)),
                      end_condition="indicate_finished")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_flow("ground", Ground, p=self.p.ground)
        self.add_flow("pos_signal", Pos_Signal)
        self.add_flow('pos', Pos)
        self.add_flow("ee_12", EE)
        self.add_flow("ee_5", EE)
        self.add_flow("ee_15", EE)
        self.add_flow("video", Video)
        self.add_flow("auto_control", Control)
        self.add_flow("motor_control", Control)
        self.add_flow("switch", Switch)
        self.add_flow("comms", Comms)
        self.add_flow("override_comms", OverrideComms)
        self.add_flow("fault_sig", FaultSig)
        # self.add_flow('Example_Disconnect')

        self.add_fxn("power", Power, "ee_15", "ee_5", "ee_12", "switch")
        self.add_fxn("operator", Operator, "switch")
        self.add_fxn("communications", Communications, "comms", "ee_12", "pos_signal")
        self.add_fxn("perception", Perception, "ground", 'pos', "ee_12", "video")
        self.add_fxn("plan_path", PlanPath, "video", "pos_signal", "ground", 'pos',
                     "auto_control", "fault_sig", p=self.p.correction)
        self.add_fxn("override", Override, "override_comms", "ee_5", "motor_control",
                     "auto_control")
        drive_m = {"mode_args": self.p.drive_modes, 'deg_params': self.p.degradation}
        self.add_fxn("drive", Drive, "ground", 'pos', "ee_15", "motor_control",
                     "fault_sig", m=drive_m)

        self.build()

    def indicate_finished(self, time):
        if  (self.flows['ground'].at_end(self.flows['pos'].s)
             or (time > 5 and self.fxns["plan_path"].m.in_mode("standby"))
             or self.fxns["plan_path"].m.in_mode("em_off", "finished")):
            return True
        else:
            return False

    def find_classification(self, scen, mdlhist):
        modes, modeproperties = self.return_faultmodes()
        classification = str()
        at_finish = True
        if not self.flows['ground'].at_end(self.flows['pos'].s):
            classification = "incomplete mission"
            at_finish = False
        if any(modes):
            classification = classification + " faulty"
        if not classification:
            classification = "nominal mission"
        num_modes = len(modes)

        end_dist = self.flows['ground'].end_dist(self.flows['pos'].s)

        endpt = [self.flows["pos"].s.x, self.flows["pos"].s.y]
        x_nom = mdlhist.nominal.flows.pos.s.x
        y_nom = mdlhist.nominal.flows.pos.s.x
        x_fault = mdlhist.faulty.flows.pos.s.x
        y_fault = mdlhist.faulty.flows.pos.s.x

        f_t = min(len(x_nom), len(y_nom))

        tot_deviation = np.sum(np.sqrt((x_nom[:f_t]- x_fault[:f_t])** 2 +
                                       (y_nom[:f_t]- y_fault[:f_t])** 2))

        in_bound = all(mdlhist.faulty.flows.ground.s.in_bound)
        
        line_dist = 1
        hist_xy = mdlhist.faulty.flows.pos.s.get('x', 'y')
        end_dist = self.flows['ground'].max_line_dist(hist_xy)
        return {
            "rate": scen.rate,
            "cost": 0,
            "prob": scen.prob,
            "expected cost": 0,
            "in_bound": in_bound,
            "at_finish": at_finish,
            "line_dist": line_dist,
            "num_modes": num_modes,
            "end_dist": end_dist,
            "tot_deviation": tot_deviation,
            "faults": modes,
            "classification": classification,
            "endpt": endpt,
        }


def gen_param_space():
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


if __name__ == "__main__":
    import multiprocessing as mp
    from fmdtools.analyze import show


    mdl = Rover()
    ec, hist = prop.nominal(mdl)
    fig, ax = show.trajectories(hist, 'flows.pos.s.x','flows.pos.s.y',
                                time_groups = ['nominal'])
    show.geomarch(mdl.flows['ground'].ga, fig=fig, ax=ax)
    fig, ax = show.trajectories(hist, 'flows.pos.s.x','flows.pos.s.y')
    
    mdl = Rover(p={'ground': GroundParam(linetype='turn')})
    ec, hist = prop.nominal(mdl)
    fig, ax = show.geomarch(mdl.flows['ground'].ga, 
                            geoms={'line': {'shapes': {'on': {} ,'shape': {}}},
                                   'start': {},
                                   'end': {}})
    fig, ax = show.trajectories(hist, 'flows.pos.s.x','flows.pos.s.y',
                                time_groups = ['nominal'], fig=fig, ax=ax)
    #ax.set_xlim(0, 5.0)
    #ax.set_ylim(-2.5, 2.5)
    hist.flows.pos.s.x
    fig, ax = show.trajectories(hist, 'flows.pos.s.x','flows.pos.s.y')
