#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multirotor drone model (with simple dynamics).

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

from examples.multirotor.drone_mdl_static import DistEE
from examples.multirotor.drone_mdl_static import HoldPayload as HoldPayloadStatic
from examples.multirotor.drone_mdl_static import StoreEE as StaticstoreEE
from examples.multirotor.drone_mdl_static import AffectDOF as AffectDOFStatic
from examples.multirotor.drone_mdl_static import CtlDOF
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.time import Time
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.environment import Environment
from fmdtools.define.object.coords import Coords, CoordsParam
import fmdtools.sim as fs

import numpy as np


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

    x_size: np.int32 = 16
    y_size: np.int32 = 16
    blocksize: np.float64 = 10.0
    loc: str = 'rural'


class SightGrid(Coords):
    """
    Define the Drone Grid environment.

    Used to calculate environmental risk and number of points viewed.

    Example
    -------
    >>> mdl = Drone()
    >>> ec, hist = fs.propagate.nominal(mdl)
    >>> mdl.flows['environment'].c.assign_from(hist.flows.environment.c, 10)
    >>> fig, ax = mdl.flows['environment'].c.show({'viewed': {}})
    """

    state_viewed = (bool, False)
    feature_target = (bool, False)
    point_start = (0, 0)
    point_safe = (0, 50)
    container_p = DroneEnvironmentGridParam

    def init_properties(self, *args, **kwargs):
        """Set target true between 0 and 155 in the x and 10 and 155 in the y."""
        self.set_range("target", True, 0, 155, 10, 155)


class DroneEnvironment(Environment):
    """Drone environment flow (contains grid)."""

    coords_c = SightGrid
    container_p = DroneEnvironmentGridParam


class StoreEE(StaticstoreEE):
    """Dynamic StoreEE function (adds energy usage)."""

    def set_faults(self):
        """When soc is 0, add 'nocharge' fault."""
        if self.s.soc < 1:
            self.s.soc = 0
            self.m.add_fault('nocharge')

    def static_behavior(self):
        """Energy storage/use behavior."""
        self.set_faults()
        if self.m.has_fault('nocharge'):
            self.ee_out.s.effort = 0.0
        else:
            self.ee_out.s.effort = 1.0
        if not self.t.executed_static:
            self.s.inc(soc=-self.ee_out.s.mul('rate', 'effort')*self.t.dt/2)


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
    fault_noloc = (0.2, 10000)
    fault_degloc = (0.8, 10000)
    opermodes = ('taxi', 'hover', 'move', 'descend', 'land')
    mode: str = 'taxi'


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
    dist: np.float64 = 0.0


class PlanPathParams(Parameter):
    """Defines flightplan."""

    goals = ((0.0, 0.0, 50.0),
             (100.0, 0.0, 50.0),
             (100.0, 100.0, 50.0),
             (150.0, 150.0, 50.0),
             (0.0, 0.0, 50.0),
             (0.0, 0.0, 0.0))


class PlanPathTime(Time):
    """Define pause timer for hovering at each point in the flightplan."""

    timernames = ('pause',)


class PlanPath(Function):
    """Path planning for the drone."""

    container_t = PlanPathTime
    container_m = PlanPathMode
    container_s = PlanPathState
    container_p = PlanPathParams
    flow_ee_ctl = EE
    flow_dofs = DOFs
    flow_des_traj = DesTraj
    flow_fs = Force
    flownames = {'force_st': 'fs'}

    def set_faults(self):
        """Enter "noloc" fault if loses support."""
        if self.fs.s.support < 0.5:
            self.m.add_fault('noloc')

    def calc_dist_to_goal(self):
        """
        Calculate drone distance to goal.

        e.g.::
        >>> p = PlanPath()
        >>> p.s.dist
        np.float64(0.0)
        >>> p.calc_dist_to_goal()
        >>> p.s.dist
        np.float64(50.0)
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

    def static_behavior(self):
        """
        Path planning behavior.

        Involves steps (1) calculating distance to goal (2) determining mode/next goal
        based on progress in flight plan and (3) asigning new trajectory.
        """
        self.set_faults()
        self.s.goal = self.p.goals[self.s.pt]
        self.calc_dist_to_goal()

        if self.m.mode == 'taxi' and self.t.time > 5:
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
            if not self.t.executed_static:
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

    flow_des_traj = DesTraj

    def static_behavior(self):
        """Behavior in-time (fault effects on states and instantaneous power/force)."""
        self.calc_faults()
        self.calc_pwr()

    def dynamic_behavior(self):
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


class ViewEnvironment(Function):
    """Camera for the drone. Determines which aspects of the environment are viewed."""

    flow_dofs = DOFs
    flow_environment = DroneEnvironment

    def static_behavior(self):
        """Set points in grid as viewed if in range of view."""
        width = self.dofs.s.z
        height = self.dofs.s.z
        self.environment.c.set_range("viewed", True,
                                     self.dofs.s.x - width/2,
                                     self.dofs.s.x + width/2,
                                     self.dofs.s.y - height/2,
                                     self.dofs.s.y + height/2,
                                     outside_error=False)


class Drone(FunctionArchitecture):
    """Dynamic drone model."""

    default_sp = dict(phases=(('ascend', 0, 1),
                              ('forward', 2, 14),
                              ('descend', 15, 18),
                              ('taxi', 19, 20)),
                      end_time=20,
                      units='sec')

    def init_architecture(self, **kwargs):
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

    def classify(self, scen={}, mdlhists={}, **kwargs):
        """
        Calculate cost model based on scenario results.

        Cost model is based on repairs, getting lost, and crashing.

        Returns
        -------
        result : dict
            Rate, cost, and expected cost for scenario.
        """
        if -5 > self.h.flows.dofs.s.x[-1] or 5 < self.h.flows.dofs.s.x[-1]:
            lostcost = 50000
        elif -5 > self.h.flows.dofs.s.y[-1] or 5 < self.h.flows.dofs.s.y[-1]:
            lostcost = 50000
        elif self.h.flows.dofs.s.z[-1] > 5:
            lostcost = 50000
        else:
            lostcost = 0

        if any(self.h.fxns.hold_payload.m.faults['break']):
            crashcost = 100000
        else:
            crashcost = 0

        modeprops = self.return_faultmodes()
        repcost = sum([m['cost'] for f, m in modeprops.items()])

        totcost = repcost + crashcost + lostcost
        rate = scen.rate
        expcost = totcost*rate*1e5
        return {'rate': rate, 'cost': totcost, 'expected_cost': expcost,
                'lost_cost': lostcost, 'crashcost': crashcost}


def finddist(p1, p2):
    """Find the 3d distance between two points."""
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def vectdist(p1, p2):
    """Find the 3d vector distance between two points."""
    return [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]


def script_nominal_viewed(**kwargs):
    """Show viewed environment in nominal scenario."""
    mdl = Drone(**kwargs)
    ec, mdlhist = fs.propagate.nominal(mdl)
    mdl.flows['environment'].c.assign_from(mdlhist.flows.environment.c, 10)
    return mdl.flows['environment'].c.show({'viewed': {}})


def script_faulty_trajectories(**kwargs):
    """Show the faulty trajectories over a range of mechanical break scenarios."""
    mdl = Drone(**kwargs)
    ec, mdlhist = fs.propagate.nominal(mdl)

    fig, ax = mdlhist.plot_trajectories("dofs.s.x", "dofs.s.y", "dofs.s.z",
                                        time_groups=['nominal'], time_ticks=2.0)

    app_mechfaults = SampleApproach(mdl, phasemaps={'mdl': PhaseMap(mdl.sp.phases, dt=mdl.sp.dt)})
    app_mechfaults.add_faultdomain("mechfaults", "fault", "affect_dof", "mechbreak")
    app_mechfaults.add_faultsample("mechfault_scens", "fault_phases",
                                   "mechfaults", phasemap='mdl', args=(5,))

    quad_ec, quad_hist = fs.propagate.fault_sample(mdl, app_mechfaults, staged=True)
    quad_hist.nominal.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z',
                                'fxns.store_ee.s.soc')

    fig, ax = quad_hist.plot_trajectories("dofs.s.x", "dofs.s.y", "dofs.s.z",
                                          time_groups=['nominal'],
                                          indiv_kwargs={'faulty': {'alpha': 0.15,
                                                                   'color': 'red'}})

    an.phases.phaseplot(app_mechfaults.phasemaps)
    an.phases.samplemetric(app_mechfaults.faultsamples['mechfault_scens'], quad_ec)
    an.phases.samplemetrics(app_mechfaults, quad_ec)

    quad_ec_1, quad_hist_1 = fs.propagate.fault_sample(mdl, app_mechfaults)
    cost_tests = [ec for ec in quad_ec if quad_ec[ec] != quad_ec_1[ec]]

    fig, ax = quad_hist.plot_trajectories_from(10, ("dofs.s.x", "dofs.s.y", "dofs.s.z"),
                                               time_groups=['nominal'],
                                               indiv_kwargs={'faulty': {'alpha': 0.15,
                                                                        'color': 'red'}})
    quad_hist.animate('plot_trajectories_from',
                      plot_values=("dofs.s.x", "dofs.s.y", "dofs.s.z"))


def script_env_viewed(**kwargs):
    """Show the viewed properties of the environment in various configurations."""
    mdl = Drone(**kwargs)
    ec, mdlhist = fs.propagate.nominal(mdl)
    l_k = dict(new_labs={'viewed': 'viewed_true', 'target': 'target_true'})
    mdl.flows['environment'].c.assign_from(mdlhist.flows.environment.c, 10)
    mdl.flows['environment'].c.show({'viewed': {}})

    mdl.flows['environment'].c.show({'viewed': {}, 'target': {}})
    mdl.flows['environment'].c.show({'target': {},
                                     'viewed': {'alpha': 0.5}},
                                    legend_kwargs=l_k,
                                    collections={'start': {}, 'safe': {}})

    mdl.flows['environment'].c.show_from(10, mdlhist.flows.environment.c,
                                         {'viewed': {}}, title='hi',
                                         legend_kwargs=l_k)
    properties={'target': {'alpha': 0.6}, 'viewed': {'alpha': 0.5}}
    mdl.flows['environment'].c.show_over_time([0, 10, 15],
                                              history=mdlhist.flows.environment.c,
                                              properties=properties, title='hi')

    ani = mdl.flows['environment'].c.animate(mdlhist.flows.environment.c,
                                             properties=properties,
                                             collections={'start': {}, 'safe': {}})
    return ani


if __name__ == "__main__":
    from fmdtools.sim.sample import SampleApproach
    from fmdtools.analyze.phases import PhaseMap
    from fmdtools import analyze as an
    s = SightGrid()
    import doctest
    doctest.testmod(verbose=True)
    # script_nominal_viewed()
    script_faulty_trajectories()
    # ani = script_env_viewed()
