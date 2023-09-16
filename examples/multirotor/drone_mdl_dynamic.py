# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
from examples.multirotor.drone_mdl_static import CtlDOFMode
import numpy as np
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.time import Time
from fmdtools.define.mode import Mode
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.sim.approach import SampleApproach

import fmdtools.sim as fs

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from examples.multirotor.drone_mdl_static import DistEE, EngageLand, HoldPayload, AffectDOF
from examples.multirotor.drone_mdl_static import StoreEE as StaticstoreEE
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj


class StoreEE(StaticstoreEE):
    def condfaults(self, time):
        if self.s.soc < 1:
            self.s.soc = 0
            self.m.add_fault('nocharge')

    def behavior(self, time):
        if self.m.has_fault('nocharge'):
            self.ee_out.s.effort = 0.0
        else:
            self.ee_out.s.effort = 1.0
        if time > self.t.time:
            self.s.inc(soc=-self.ee_out.s.mul('rate', 'effort')*(time-self.t.time)/2)

class CtlDOFState(State):
    cs: float = 1.0
    vel: float = 0.0
    upthrottle: float = 0.0
    throttle: float = 0.0
    """
    Controller States. Has entries:
        cs: float
            Control signal transferrence (nominally 1.0)
        upthrottle: float
            Internal throttle signal (up) (0 is off, 1 is hover, 2 is max climb 5 m/s)
        throttle: float
            Internal throttle signal (forward)
        vel: float
            Percieved vertical velocity at the last timestep
    """

class CtlDOF(FxnBlock):
    __slots__ = ('ee_in', 'des_traj', 'ctl', 'dofs', 'fs')
    _init_s = CtlDOFState
    _init_m = CtlDOFMode
    _init_ee_in = EE
    _init_des_traj = DesTraj
    _init_ctl = Control
    _init_dofs = DOFs
    _init_fs = Force
    flownames = {'ee_ctl': 'ee_in', 'force_st': 'fs'}

    def condfaults(self, time):
        if self.fs.s.support < 0.5:
            self.m.add_fault('noctl')

    def behavior(self, time):
        if self.m.has_fault('noctl'):
            self.s.cs = 0.0
        elif self.m.has_fault('degctl'):
            self.s.cs = 0.5
        self.s.vel = self.dofs.s.vertvel
        self.s.upthrottle = 1.0
        if self.des_traj.s.z > 1:
            self.s.upthrottle = 2.0
        elif 0 < self.des_traj.s.z <= 1:
            self.s.upthrottle = self.des_traj.s.z + 1.0 
        elif self.des_traj.s.z == 0:
            damp = np.sign(self.s.vel)
            damp2 = damp*min(1.0, np.power(self.s.vel, 2))
            self.s.upthrottle = 1.0-0.2*damp2
        elif -1 < self.des_traj.s.z <= 0.0:
            damp = min(1.0, np.power(self.s.vel+0.5, 2))
            self.s.upthrottle = 0.5+0.5*damp
        elif self.des_traj.s.z <= -1.0:
            damp = min(0.75, np.power(self.s.vel+5.0, 2))
            self.s.upthrottle = 0.75+0.15*damp

        if self.des_traj.s.same([0.0, 0.0], 'x', 'y'):
            self.s.throttle = 0.0
        else:
            self.s.throttle = 1.0

        power = self.ee_in.s.effort*self.s.cs*self.des_traj.s.power
        self.ctl.s.put(forward=power*self.s.throttle,
                       upward=power*self.s.upthrottle)


class PlanPathMode(Mode):
    failrate = 1e-5
    faultparams = {'noloc': (0.2, 10000),
                   'degloc': (0.8, 10000)}
    opermodes = ('taxi', 'hover', 'move', 'descend', 'land')
    mode: int = 'taxi'


class PlanPathStates(State):
    """ """
    dx:     float = 0.0
    dy:     float = 0.0
    dz:     float = 0.0
    pt:     int = 0
    goal:   tuple = (0.0, 0.0, 50.0)


class PlanPathParams(Parameter):
    goals = ((0.0,      0.0,    50.0),
             (100.0,    0.0,    50.0),
             (100.0,    100.0,  50.0),
             (150.0,    150.0,  50.0),
             (0.0,      0.0,    50.0),
             (0.0,      0.0,    0.0))


class PlanPathTime(Time):
    timernames = ('pause',)


class PlanPath(FxnBlock):
    __slots__ = ('ee_in', 'dofs', 'des_traj', 'fs')
    _init_t = PlanPathTime
    _init_m = PlanPathMode
    _init_s = PlanPathStates
    _init_p = PlanPathParams
    _init_ee_in = EE
    _init_dofs = DOFs
    _init_des_traj = DesTraj
    _init_fs = Force
    flownames = {'ee_ctl': 'ee_in', 'force_st': 'fs'}

    def condfaults(self, time):
        if self.fs.s.support < 0.5:
            self.m.add_fault('noloc')

    def behavior(self, t):
        self.s.goal = self.p.goals[self.s.pt]
        loc = self.dofs.s.get('x', 'y', 'z')
        dist = finddist(loc, self.s.goal)
        self.s.assign(vectdist(self.s.goal, loc), 'dz', 'dy', 'dz')

        if self.m.mode == 'taxi' and t > 5:
            self.m.mode = 'taxi'
        elif self.dofs.s.z < 1 and self.s.pt == 5:
            self.m.mode = 'taxi'
        elif dist < 5 and self.s.pt == 5:
            self.m.mode = 'land'
        elif self.s.pt == 6 and self.m.in_mode('move', 'hover'):
            self.m.mode = 'descend'
        elif dist > 5 and not (self.m.mode == 'descend'):
            self.m.mode = 'move'
        elif dist < 5 and self.m.in_mode('move', 'hover'):
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
            self.des_traj.s.assign([0, 0, 0], "x", "y", "z")
        elif self.m.mode == 'move':
            self.des_traj.s.assign(vectdist(self.s.goal, loc), "x", "y", "z")
        elif self.m.mode == 'descend':
            self.des_traj.s.assign([0, 0, -0.5], "x", "y", "z")
        elif self.m.mode == 'land':
            self.des_traj.s.assign([0, 0, -0.1], "x", "y", "z")
        # faulty behaviors
        if self.m.has_fault('noloc'):
            self.des_traj.s.assign([0, 0, 0], "x", "y", "z")
        elif self.m.has_fault('degloc'):
            self.des_traj.s.assign([0, 0, -1], "x", "y", "z")
        if self.ee_in.s.effort < 0.5:
            self.des_traj.s.assign([0, 0, 0, 0], "x", "y", "z", "power")


class Trajectory(FxnBlock):
    __slots__ = ('dofs', 'des_traj', 'force_gr')
    _init_dofs = DOFs
    _init_des_traj = DesTraj
    _init_force_gr = Force

    def dynamic_behavior(self, time):
        if self.dofs.s.z <= 0.0:
            self.force_gr.s.support = min(-0.5,
                                          (self.dofs.s.vertvel - self.dofs.s.planvel)
                                          / 7.5)
            acc = 10*self.dofs.s.uppwr
        else:
            self.force_gr.s.support = 0.0
            acc = 10*(self.dofs.s.uppwr-1.0)

        sign = np.sign(self.dofs.s.vertvel)
        damp = (-0.02*sign*np.power(self.dofs.s.vertvel, 2)-0.1*self.dofs.s.vertvel)
        self.dofs.s.vertvel = self.dofs.s.vertvel+(acc+damp)
        self.dofs.s.planvel = 10.0*self.dofs.s.planpwr
        if self.dofs.s.z <= 0.0:
            self.dofs.s.vertvel = max(0, self.dofs.s.vertvel)
            self.dofs.s.planvel = 0.0

        self.dofs.s.inc(x=self.dofs.s.planvel*self.des_traj.s.x,
                        y=self.dofs.s.planvel*self.des_traj.s.y,
                        z=self.dofs.s.vertvel*self.des_traj.s.z)
        self.dofs.s.limit(z=(0.0, np.inf))


class ViewEnvironment(FxnBlock):
    _init_dofs = DOFs

    def __init__(self, name, flows, params={}, **kwargs):
        super().__init__(name, **kwargs)
        sq = square([0, 150], 160, 160)
        self.viewingarea = {(x, y): 'unviewed'
                            for x in range(int(sq[0][0]), int(sq[1][0])+10, 10)
                            for y in range(int(sq[0][1]), int(sq[2][1])+10, 10)}

    def dynamic_behavior(self, time):
        area = square((self.dofs.s.x, self.dofs.s.y), 10, 10)
        # TODO: This is *the* major bottleneck in the model.
        # Ideally we should try to speed it up
        for spot in self.viewingarea:
            if inrange(area, spot[0], spot[1]):
                self.viewingarea[spot] = 'viewed'


class Drone(Model):
    __slots__ = ()
    default_sp = dict(phases=(('ascend', 0, 4),
                              ('forward', 5, 94),
                              ('descend', 95, 100)),
                      times=(0, 135),
                      units='sec')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add flows to the model
        self.add_flow('force_st', Force)
        self.add_flow('force_lin', Force)
        self.add_flow('force_gr', Force)
        self.add_flow('force_lg', Force)
        self.add_flow('ee_1', EE)
        self.add_flow('ee_mot', EE)
        self.add_flow('ee_ctl', EE)
        self.add_flow('ctl', Control)
        self.add_flow('dofs', DOFs)
        self.add_flow('des_traj', DesTraj)
        # add functions to the model
        self.add_fxn('store_ee', StoreEE, 'ee_1', 'force_st')
        self.add_fxn('dist_ee', DistEE, 'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof', AffectDOF, 'ee_mot', 'ctl', 'dofs', 'force_lin')
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path', PlanPath, 'ee_ctl', 'des_traj', 'force_st')
        self.add_fxn('trajectory', Trajectory, 'dofs', 'des_traj', 'force_gr')
        self.add_fxn('engage_land', EngageLand, 'force_gr', 'force_lg')
        self.add_fxn('hold_payload', HoldPayload, 'force_lg', 'force_lin', 'force_st')
        self.add_fxn('view_env', ViewEnvironment, 'dofs')

        self.build()

    def find_classification(self, scen, mdlhists):
        if -5 > mdlhists.faulty.flows.dofs.s.x[-1] or 5 < mdlhists.faulty.flows.dofs.s.x[-1]:
            lostcost = 50000
        elif -5 > mdlhists.faulty.flows.dofs.s.y[-1] or 5 < mdlhists.faulty.flows.dofs.s.y[-1]:
            lostcost = 50000
        elif mdlhists.faulty.flows.dofs.s.z[-1] > 5:
            lostcost = 50000
        else:
            lostcost = 0

        if any(abs(mdlhists.faulty.flows.force_gr.s.support) > 2.0):
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


def square(center, xw, yw):
    square = [[center[0]-xw/2, center[1]-yw/2],
              [center[0]+xw/2, center[1]-yw/2],
              [center[0]+xw/2, center[1]+yw/2],
              [center[0]-xw/2, center[1]+yw/2]]
    return square

# checks to see if a point with x-y coordinates is in the area a


def inrange(area, x, y):
    point = Point(x, y)
    polygon = Polygon(area)
    return polygon.contains(point)


def finddist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def calcdist(p1, p2):
    return np.sqrt((p1[0]-p2.x)**2+(p1[1]-p2.y)**2+(p1[2]-p2.z)**2)


def vectdist(p1, p2):
    return [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]


def vectdir(p1, p2):
    return vectdist(p1, p2)/(finddist(p1, p2)+0.00001)


if __name__ == "__main__":
    from fmdtools import analyze as an
    mdl = Drone()
    app = SampleApproach(mdl)

    mdl_quad_comp = Drone()
    quad_comp_app = SampleApproach(mdl_quad_comp,
                                   faults=[('affect_dof', 'mechbreak')],
                                   defaultsamp={'samp': 'evenspacing', 'numpts': 5})
    quad_comp_endclasses, quad_comp_mdlhists = fs.propagate.approach(mdl_quad_comp,
                                                                     quad_comp_app,
                                                                     staged=True)
    an.plot.hist(quad_comp_mdlhists.nominal, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')

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
