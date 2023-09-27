# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.time import Time
from fmdtools.define.mode import Mode
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.sim.approach import SampleApproach
from fmdtools.define.environment import Environment
from fmdtools.define.coords import Coords, CoordsParam

import fmdtools.sim as fs

from examples.multirotor.drone_mdl_static import DistEE
from examples.multirotor.drone_mdl_static import HoldPayload as HoldPayloadStatic
from examples.multirotor.drone_mdl_static import StoreEE as StaticstoreEE
from examples.multirotor.drone_mdl_static import AffectDOF as AffectDOFStatic
from examples.multirotor.drone_mdl_static import CtlDOF
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj


class DroneEnvironmentGridParam(CoordsParam):
    """
    Define the grid parameters.

    By default the grid is a 16x16 grid of 10.0 m blocks. Defines the following

    Features/Collections
    -------
    viewed : State
        Whether the point as been viewed.
    target : State
        Whether the point is in the target area.
    start : Point
        Starting location
    start : Point
        Safe landing location (for emergencies)

    Other Fields
    -------
    loc : str
        Type of environment risk profile (urban or rural).
    """

    x_size: int = 16
    y_size: int = 16
    blocksize: float = 10.0
    state_viewed: tuple = (bool, False)
    feature_target: tuple = (bool, False)
    point_start: tuple = (0, 0)
    point_safe: tuple = (0, 50)
    loc: str = 'rural'


class SightGrid(Coords):
    """
    Define the Drone Grid environment.

    Used to calculate environmental risk and number of points viewed.
    """

    _init_p = DroneEnvironmentGridParam

    def init_properties(self, *args, **kwargs):
        """Set target true between 0 and 150 in the x and 10 and 160 in the y."""
        self.set_range("target", True, 0, 150, 10, 160)


class DroneEnvironment(Environment):
    """Drone environment flow (contains grid)."""

    _init_c = SightGrid
    _init_p = DroneEnvironmentGridParam


class StoreEE(StaticstoreEE):
    """Dynamic StoreEE function (adds energy usage)."""

    def condfaults(self, time):
        """When soc is 0, add 'nocharge' fault."""
        if self.s.soc < 1:
            self.s.soc = 0
            self.m.add_fault('nocharge')

    def behavior(self, time):
        """Energy storage/use behavior."""
        if self.m.has_fault('nocharge'):
            self.ee_out.s.effort = 0.0
        else:
            self.ee_out.s.effort = 1.0
        if time > self.t.time:
            self.s.inc(soc=-self.ee_out.s.mul('rate', 'effort')*(time-self.t.time)/2)


class HoldPayload(HoldPayloadStatic):
    """Holds payload (adapted with dynamic behaviors)."""

    def calc_force_gr(self):
        """Calculate ground force (dynamic adaptation).

        Makes it so that high velocity landings break the landing gear but low-velocity
        landings do not.
        """
        if self.at_ground():
            force_vel = - (abs(self.dofs.s.vertvel)
                           + abs(self.dofs.s.planvel)) / 60
            self.s.force_gr = min(-0.5, force_vel)
        else:
            self.s.force_gr = 0.0


class PlanPathMode(Mode):
    """
    Possible modes for drone path planning.

    Modes
    -------
    degloc : Fault
        Degraded location
    noloc : Fault
        Lost Location.
    taxi : Mode
        Drone on the ground.
    hover : Mode
        Drone hovers at a given point.
    move : Mode
        Drone moves from point to point.
    descend : Mode
        Drone descends to the ground.
    land : Mode
        Drone lands.
    """

    failrate = 1e-5
    faultparams = {'noloc': (0.2, 10000),
                   'degloc': (0.8, 10000)}
    opermodes = ('taxi', 'hover', 'move', 'descend', 'land')
    mode: int = 'taxi'


class PlanPathState(State):
    """
    Path planning states (extends dynamic model states).

    Fields
    -------
    pt : int
        current goal point (number) in sequence.
    goal : tuple
        current location of goal point.
    dist: float
        distance to goal point.
    """

    pt: int = 0
    goal: tuple = (0.0, 0.0, 50.0)
    dist: float = 0.0


class PlanPathParams(Parameter):
    """Defines flightplan."""

    goals = ((0.0,      0.0,    50.0),
             (100.0,    0.0,    50.0),
             (100.0,    100.0,  50.0),
             (150.0,    150.0,  50.0),
             (0.0,      0.0,    50.0),
             (0.0,      0.0,    0.0))


class PlanPathTime(Time):
    """Define pause timer for hovering at each point in the flightplan."""

    timernames = ('pause',)


class PlanPath(FxnBlock):
    """Path planning for the drone."""

    __slots__ = ('ee_ctl', 'dofs', 'des_traj', 'fs', 'dofs')
    _init_t = PlanPathTime
    _init_m = PlanPathMode
    _init_s = PlanPathState
    _init_p = PlanPathParams
    _init_ee_ctl = EE
    _init_dofs = DOFs
    _init_des_traj = DesTraj
    _init_fs = Force
    flownames = {'force_st': 'fs'}

    def condfaults(self, time):
        """Enter "noloc" fault if loses support."""
        if self.fs.s.support < 0.5:
            self.m.add_fault('noloc')

    def calc_dist_to_goal(self):
        """
        Calculate drone distance to goal.

        e.g.::
        >>> p = PlanPath()
        >>> p.s.dist
        0.0
        >>> p.calc_dist_to_goal()
        >>> p.s.dist
        50.0
        """
        loc = self.dofs.s.get('x', 'y', 'z')
        self.s.dist = finddist(loc, self.s.goal)

    def assign_vectdist_to_goal(self):
        """
        Assign the desired trajectory based on the goal.

        e.g.::
        >>> p = PlanPath()
        >>> p.des_traj.s
        DesTrajState(dx=1.0, dy=0.0, dz=0.0, power=1.0)
        >>> p.assign_vectdist_to_goal()
        >>> p.des_traj.s
        DesTrajState(dx=0.0, dy=0.0, dz=50.0, power=1.0)
        """
        loc = self.dofs.s.get('x', 'y', 'z')
        vd = vectdist(self.s.goal, loc)
        self.des_traj.s.assign(vd, "dx", "dy", "dz")

    def behavior(self, t):
        """
        Path planning behavior.

        Involves steps (1) calculating distance to goal (2) determining mode/next goal
        based on progress in flight plan and (3) asigning new trajectory.
        """
        self.s.goal = self.p.goals[self.s.pt]
        self.calc_dist_to_goal()


        if self.m.mode == 'taxi' and t > 5:
            self.m.mode = 'taxi'
        elif self.dofs.s.z < 1 and self.s.pt == 5:
            self.m.mode = 'taxi'
        elif self.s.dist < 5 and self.s.pt == 5:
            self.m.mode = 'land'
        elif self.s.pt == 5 and self.m.in_mode('move', 'hover'):
            self.m.mode = 'descend'
        elif self.s.dist > 5 and not (self.m.mode == 'descend'):
            self.m.mode = 'move'
        elif self.s.dist < 5 and self.m.in_mode('move', 'hover'):
            self.m.mode = 'hover'
            if t > self.t.time:
                self.t.pause.inc(1)
                if self.t.pause.t() > 2:
                    self.s.inc(pt=1)
                    self.s.limit(pt=(0, 5))
                    self.s.goal = self.p.goals[self.s.pt]
                    self.t.pause.reset()
        # nominal behaviors
        self.des_traj.s.power = 1.0
        if self.m.mode == 'taxi':
            self.des_traj.s.power = 0.0
        elif self.m.mode == 'hover':
            self.des_traj.s.assign([0, 0, 0], "dx", "dy", "dz")
        elif self.m.mode == 'move':
            self.assign_vectdist_to_goal()
        elif self.m.mode == 'descend':
            self.des_traj.s.assign([0, 0, -3*self.dofs.s.z/4], "dx", "dy", "dz")
        elif self.m.mode == 'land':
            self.des_traj.s.assign([0, 0, -2], "dx", "dy", "dz")
        # faulty behaviors
        if self.m.has_fault('noloc'):
            self.des_traj.s.assign([0, 0, 0], "dx", "dy", "dz")
        elif self.m.has_fault('degloc'):
            self.des_traj.s.assign([0, 0, -1], "dx", "dy", "dz")
        if self.ee_ctl.s.effort < 0.5:
            self.des_traj.s.assign([0, 0, 0, 0], "dx", "dy", "dz", "power")


class AffectDOF(AffectDOFStatic):
    """Dynamic extension of drone locomotion."""

    _init_des_traj = DesTraj

    def behavior(self, time):
        """Behavior in-time (fault effects on states and instantaneous power/force)."""
        self.calc_faults()
        self.calc_pwr()

    def dynamic_behavior(self, time):
        """Behavior at-time (calculating velociy and incrementing position)."""
        self.calc_vel()
        self.inc_pos()

    def calc_vel(self):
        """Calculate vertical/planar velocity based on power."""
        # calculate velocities from power
        self.dofs.s.put(vertvel=self.dofs.p.max_vel*(self.dofs.s.uppwr-1.0),
                        planvel=self.dofs.p.max_vel*self.dofs.s.planpwr)
        self.dofs.s.roundto(vertvel=0.001, planvel=0.001)
        # if at ground, takeoff, limit takeoff velocities.
        if self.dofs.s.z <= 0.0:
            self.dofs.s.put(planvel=0.0,
                            vertvel=max(0, self.dofs.s.vertvel))
        self.limit_falling_vel()

    def limit_falling_vel(self):
        """Limit falling distances based on xy-velocity and fall height."""
        min_fall_dist = self.get_fall_dist()
        # if falling, it can't reach the destination if it hits the ground first
        plan_dist = self.des_traj.s.dist2d()
        no_runway = (self.dofs.s.vertvel/self.t.dt < -min_fall_dist
                     and -self.dofs.s.vertvel > self.dofs.s.planvel)
        if no_runway:
            plan_dist = plan_dist*min_fall_dist/(-self.dofs.s.vertvel+0.001)
        self.dofs.s.limit(vertvel=(-min_fall_dist/self.t.dt, 300.0),
                          planvel=(0.0, plan_dist/self.t.dt),
                          z=(0, np.inf))

    def get_fall_dist(self):
        """Get the max distance possible to fall at the given point (dofs.s.z)."""
        return self.dofs.s.z

    def inc_pos(self):
        """
        Increments the drone position based on trajectory and calculated velocities.

        e.g.,::
        >>> a = AffectDOF()
        >>> a.des_traj.s.put(dx=1.0, dy=0.0, dz=0.0, power=1.0)
        >>> a.dofs.s.put(x=0.0, y=0.0, z=100.0, planvel=1.0, vertvel=0.0)
        >>> a.inc_pos()
        >>> a.dofs.s
        DOFstate(vertvel=0.0, planvel=1.0, planpwr=1.0, uppwr=1.0, x=1.0, y=0.0, z=100.0)

        or, for climbing::
        >>> a.dofs.s.put(vertvel=1.0, planvel=0.0)
        >>> a.inc_pos()
        >>> a.dofs.s
        DOFstate(vertvel=1.0, planvel=0.0, planpwr=1.0, uppwr=1.0, x=1.0, y=0.0, z=101.0)
        """
        # increment x,y,z
        vec_factor = self.des_traj.s.dist2d()
        norm_vel = self.dofs.s.planvel * self.t.dt / vec_factor
        self.dofs.s.inc(x=norm_vel*self.des_traj.s.dx,
                        y=norm_vel*self.des_traj.s.dy,
                        z=self.dofs.s.vertvel*self.t.dt)
        self.dofs.s.roundto(x=0.01, y=0.01, z=0.01)


class ViewEnvironment(FxnBlock):
    """Camera for the drone. Determines which aspects of the environment are viewed."""

    _init_dofs = DOFs
    _init_environment = DroneEnvironment

    def behavior(self, time):
        """Set points in grid as viewed if in range of view."""
        width = self.dofs.s.z
        height = self.dofs.s.z
        self.environment.c.set_range("viewed", True,
                                     self.dofs.s.x - width/2,
                                     self.dofs.s.x + width/2,
                                     self.dofs.s.y - height/2,
                                     self.dofs.s.y + height/2)


class Drone(Model):
    """Dynamic drone model."""

    __slots__ = ()
    default_sp = dict(phases=(('ascend', 0, 1),
                              ('forward', 2, 14),
                              ('descend', 15, 18),
                              ('taxi', 19, 20)),
                      times=(0, 20),
                      units='sec')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add flows to the model
        self.add_flow('force_st', Force)
        self.add_flow('force_lin', Force)
        self.add_flow('ee_1', EE)
        self.add_flow('ee_mot', EE)
        self.add_flow('ee_ctl', EE)
        self.add_flow('ctl', Control)
        self.add_flow('dofs', DOFs)
        self.add_flow('des_traj', DesTraj)
        self.add_flow('environment', DroneEnvironment)
        # add functions to the model
        self.add_fxn('store_ee', StoreEE, 'ee_1', 'force_st')
        self.add_fxn('dist_ee', DistEE, 'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof', AffectDOF, 'ee_mot', 'ctl', 'des_traj',
                     'dofs', 'force_lin')
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path', PlanPath, 'ee_ctl', 'des_traj', 'force_st', 'dofs')
        self.add_fxn('hold_payload', HoldPayload, 'force_lin', 'force_st', 'dofs')
        self.add_fxn('view_env', ViewEnvironment, 'dofs', 'environment')

        self.build()

    def find_classification(self, scen, mdlhists):
        """
        Calculate cost model based on scenario results.

        Cost model is based on repairs, getting lost, and crashing.

        Returns
        -------
        endclass : dict
            Rate, cost, and expected cost for scenario.
        """
        if -5 > mdlhists.faulty.flows.dofs.s.x[-1] or 5 < mdlhists.faulty.flows.dofs.s.x[-1]:
            lostcost = 50000
        elif -5 > mdlhists.faulty.flows.dofs.s.y[-1] or 5 < mdlhists.faulty.flows.dofs.s.y[-1]:
            lostcost = 50000
        elif mdlhists.faulty.flows.dofs.s.z[-1] > 5:
            lostcost = 50000
        else:
            lostcost = 0

        if any(mdlhists.faulty.fxns.hold_payload.m.faults['break']):
            crashcost = 100000
        else:
            crashcost = 0

        modes, modeprops = self.return_faultmodes()
        repcost = sum([c['rcost']
                       for f, m in modeprops.items()
                       for a, c in m.items()])

        totcost = repcost + crashcost + lostcost
        rate = scen.rate
        expcost = totcost*rate*1e5
        return {'rate': rate, 'cost': totcost, 'expected cost': expcost}


def finddist(p1, p2):
    """Find the 3d distance between two points."""
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def vectdist(p1, p2):
    """Find the 3d vector distance between two points."""
    return [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    
    from fmdtools import analyze as an
    mdl = Drone()
    ec, mdlhist = fs.propagate.nominal(mdl)
    fig, ax = an.show.trajectories(mdlhist, "dofs.s.x", "dofs.s.y", "dofs.s.z",
                                   time_groups=['nominal'], time_ticks=2.0)
    app = SampleApproach(mdl)

    mdl_quad_comp = Drone()
    quad_comp_app = SampleApproach(mdl_quad_comp,
                                   faults=[('affect_dof', 'mechbreak')],
                                   defaultsamp={'samp': 'evenspacing', 'numpts': 5})
    quad_comp_endclasses, quad_comp_mdlhists = fs.propagate.approach(mdl_quad_comp,
                                                                     quad_comp_app,
                                                                     staged=True)
    an.plot.hist(quad_comp_mdlhists.nominal, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')
    
    fig, ax = an.show.trajectories(quad_comp_mdlhists, "dofs.s.x", "dofs.s.y", "dofs.s.z",
                                   time_groups=['nominal'], indiv_kwargs={'faulty':{'alpha':0.15, 'color':'red'}})

    import fmdtools.analyze as an
    an.plot.samplemetric(quad_comp_app,
                         quad_comp_endclasses,
                         ('affect_dof', 'mechbreak'))

    quad_comp_endclasses_1, quad_comp_mdlhists_1 = fs.propagate.approach(mdl_quad_comp,
                                                                         quad_comp_app)

    cost_tests = [ec for ec in quad_comp_endclasses
                  if quad_comp_endclasses[ec] != quad_comp_endclasses_1[ec]]

    dist_tests = [ec for ec in quad_comp_mdlhists
                  if any(quad_comp_mdlhists.get(ec) != quad_comp_mdlhists_1.get(ec))]
