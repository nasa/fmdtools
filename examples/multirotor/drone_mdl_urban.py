# -*- coding: utf-8 -*-
"""
Variant of drone model for modelling computer vision in urban settings.
"""
from examples.multirotor.drone_mdl_static import EE, Force, Control
from examples.multirotor.drone_mdl_static import DistEE
from drone_mdl_opt import DesTraj, DOFs, HSig, RSig
from drone_mdl_opt import ManageHealth, StoreEE, AffectDOF, CtlDOF, HoldPayload
from drone_mdl_opt import PlanPath as PlanPathOpt
from drone_mdl_opt import DroneParam as DroneParamOpt
from drone_mdl_opt import HoldPayload as HoldPayloadOpt
from drone_mdl_opt import AffectDOF as AffectDOFOpt
from drone_mdl_opt import rect, inrange

from fmdtools.define.block import CompArch, Component
from fmdtools.define.mode import Mode
from fmdtools.define.flow import Flow
from fmdtools.define.state import State
from fmdtools.define.parameter import Parameter
from fmdtools.define.model import Model


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D, art3d
from recordclass import asdict


class EnvironmentState(State):
    safe: bool = True
    allowed: bool = True
    landed: bool = True
    occupied: bool = False
    """
    States relating the drone with its environmnet:
        -safe: bool
            whether the drone is above a safe grid point (where no people are)
        -allowed: bool
            whether the drone is above an allowed grid location
        -landed: bool
            whether the drone has landed
        -occupied: bool
            whether the space is occupied
    """


class EnvironmentParameter(Parameter):
    blocksize: int = 100
    roadwidth: int = 15
    maxheight: int = 100
    num_allowed: int = 4
    num_unsafe: int = 3
    num_occupied: int = 20
    x_size: int = 1000
    y_size: int = 1000
    seed: int = 100


class Environment(Flow):
    _init_s = EnvironmentState
    _init_p = EnvironmentParameter
    """
    The environment the drone flies through. 

    And attributes:
        - grid:     grid of points corresponding to the map size and parameters in env_params
        - pts:      list of grid points
        - map:      dict mapping each grid point to safe, allowed, and height attributes
        - start:    start point
        - end:      end point
    As well as custom methods for interacting/visualizing the underlying map.
    """

    def __init__(self, name, s={}, p={}):
        super().__init__(name, s=s, p=p)
        self.build_map()

    def build_map(self):
        """ Builds the map based on class parameters defined in env_params"""
        rng = np.random.default_rng(self.p.seed)
        self.grid = np.array([[(i, j) for i in range(0, self.p.x_size, self.p.blocksize)]
                              for j in range(0, self.p.y_size, self.p.blocksize)])
        self.pts = self.grid.reshape(int(self.grid.size/2), 2)
        self.map = {tuple(pt):{'safe': True,
                               'allowed': False,
                               'height': rng.integers(0, self.p.maxheight),
                               'occupied': False} for pt in self.pts}
        self.start = self.pts[0]
        self.end = self.pts[-1]
        rand_pts = rng.choice(self.pts[1:-1],
                              self.p.num_allowed + self.p.num_unsafe,
                              replace=False)
        self.allow = np.array([*rand_pts[:self.p.num_allowed], self.start, self.end])
        self.unsafe = rand_pts[self.p.num_allowed:]
        self.occ = rng.choice(self.pts[1:-1], self.p.num_occupied, replace=False)
        self.open = np.array([pt for pt in self.pts
                              if not any([all(pt == pto) for pto in self.occ])])

        for pt in self.allow:
            self.set_pt(pt, allowed=True)
        for pt in self.unsafe:
            self.set_pt(pt, safe=False)
        for pt in self.occ:
            self.set_pt(pt, occupied=True)
        self.set_pts([self.start, self.end], height=0.0)

    def to_gridpoint(self, *args):
        """Finds the grid point closest to the given x/y values"""
        return tuple(round(arg/self.p.blocksize)*self.p.blocksize for arg in args)

    def set_pts(self, pts, **kwargs):
        """Sets a given list of points to have the safe/allowed/height attributes given
        in kwargs"""
        for pt in pts:
            self.set_pt(pt, **kwargs)

    def set_pt(self, pt, **kwargs):
        """Sets a given point to have the safe/allowed/height attributes given in
        kwargs"""
        pt_round = self.to_gridpoint(*pt)
        self.map[pt_round].update(kwargs)

    def ground_height(self, x, y, z):
        """Gets the distance of the height z above the ground at point x,y"""
        env_height = self.get_below(x, y)['height']
        return z-env_height

    def set_states(self, x, y, z):
        """Set the landing states safe_land and allowed_land since the drone has
        landed"""
        props = self.get_below(x, y)
        self.s.safe = props['safe']
        self.s.allowed = props['allowed']
        self.s.occupied = props['occupied']
        if self.ground_height(x, y, z) <= 0.1:
            self.s.landed = True
        else:
            self.s.landed = False

    def in_area(self, x, y, where="pts"):
        """Check if the drone is in a given area (pts, allowed, start, end, etc)"""
        pts = getattr(self, where)
        pt = self.to_gridpoint(x, y)
        if pt in pts:
            return True

    def get_closest(self, x, y, where="pts", include_pt=True):
        """Get the closest point in a given area (pts, allowed, start, end, etc)"""
        pts = getattr(self, where)
        pt = self.to_gridpoint(x, y)
        if pt in pts and include_pt:
            xy = pt
        else:
            if not include_pt:
                pts = np.array([p for p in pts if all(p != pt)])
            dists = np.sqrt(np.sum((np.array([x, y])-pts)**2, 1))
            closest_ind = np.argmin(dists)
            xy = pts[closest_ind]
        z = self.get_below(*xy)['height']
        return np.array([xy[0], xy[1], z])

    def get_below(self, x, y):
        """Gets all map properties of the gridpoint below"""
        pt = self.to_gridpoint(x, y)
        block = rect(pt,
                     self.p.blocksize-self.p.roadwidth,
                     self.p.blocksize-self.p.roadwidth)
        if inrange(block, x, y):
            properties = self.map.get(pt, {"safe": False,
                                           "allowed": False,
                                           "height": 0.0,
                                           "occupied": True})
        else:
            properties = {"safe": False,
                          "allowed": False,
                          "height": 0.0,
                          "occupied": True}
        return properties

    def show_grid(self):
        """Shows the environment gridworld. Returns matplotlib figure and axis."""
        fig, ax = plt.subplots()
        for pt in self.pts:
            rect = self._create_rect(pt)
            if rect.get_label() != 'disallowed':
                plt.text(pt[0], pt[1], rect.get_label(),
                         horizontalalignment="center", verticalalignment="center")
            ax.add_patch(rect)
        ax.set_xlim(-self.p.blocksize, self.p.x_size)
        ax.set_ylim(-self.p.blocksize, self.p.y_size)
        # ax.legend()
        return fig, ax

    def _create_rect(self, pt):
        if all(pt == self.start):
            color = "green"
            label = "start"
        elif all(pt == self.end):
            color = "green"
            label = "end"
        elif self.map[tuple(pt)]['allowed']:
            color = "blue"
            label = "allowed"
        elif not self.map[tuple(pt)]['safe']:
            color = "red"
            label = "unsafe"
        else:
            color = "gray"
            label = "disallowed"
        if self.map[tuple(pt)]['occupied']:
            hatch = "//"
        else:
            hatch = ""
        lw = int(5*self.map[tuple(pt)]['height']/self.p.maxheight)
        rect = Rectangle(pt-[self.p.blocksize/2, self.p.blocksize/2],
                         self.p.blocksize-self.p.roadwidth,
                         self.p.blocksize-self.p.roadwidth,
                         facecolor=color, linewidth=lw, edgecolor="black",
                         label=label, hatch=hatch, alpha=0.5)
        return rect

    def show_3d(self):
        """Shows a 3d version of the environment. Returns matplotlib figure and axis."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pt in self.pts:
            rect = self._create_rect(pt)
            ax.add_patch(rect)
            art3d.patch_2d_to_3d(rect, z=self.map[tuple(pt)]['height'])
        ax.set_xlim3d(0, self.p.x_size)
        ax.set_ylim3d(0, self.p.y_size)
        ax.set_zlim3d(0, self.p.maxheight)
        return fig, ax

class AffectDOF(AffectDOFOpt):
    _init_environment = Environment

    def inc_takeoff(self):
        self.environment.set_states(self.dofs.s.x, self.dofs.s.y, self.dofs.s.z)
        # can only take off at ground
        if self.environment.s.landed:
            self.dofs.s.put(planvel=0.0, vertvel=max(0, self.dofs.s.vertvel))

    def dynamic_behavior(self, time):
        self.calc_vel()
        self.inc_takeoff()
        g_h = self.environment.ground_height(self.dofs.s.x,
                                             self.dofs.s.y,
                                             self.dofs.s.z)
        self.inc_falling(min_fall_dist=g_h)
        self.inc_pos()
        self.environment.set_states(self.dofs.s.x, self.dofs.s.y, self.dofs.s.z)

class ComputerVisionMode(Mode):
    faultparams = {'undesired_detection': (0.5, {'move': 1.0}, 0),
                   'lack_of_detection': (0.5, {'move': 1.0}, 0)}


class ComputerVision(Component):
    _init_m = ComputerVisionMode
    """
    Component for percieving if a landing location is occupied.
    Has modes:
        -undesired_detection: detects an occupied space (even if it isn't')
        -lack_of_detection: doesn't detect occupied spaces
    """

    def check_if_occupied(self, environment, dofs):
        if self.m.has_fault("undesired_detection"):
            occ = True
        elif self.m.has_fault("lack_of_detection"):
            occ = False
        else:
            occ = environment.get_below(dofs.x, dofs.y)['occupied']
        self.m.remove_fault("undesired_detection")
        return occ

    def find_nearest_open(self, environment, dofs, include_pt=False):
        if self.m.has_fault("undesired_detection"):
            pt = environment.open[round(len(environment.open)/3)]
            return np.array([*pt, environment.get_below(*pt)["height"]])
        elif self.m.has_fault("lack_of_detection"):
            return environment.get_closest(dofs.x, dofs.y,
                                           where="pts", include_pt=include_pt)
        else:
            return environment.get_closest(dofs.x, dofs.y, where="open",
                                           include_pt=include_pt)


class VisionArch(CompArch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.make_components(ComputerVision, 'vision')


class PlanPathParam(Parameter):
    height: float = 200.0


class PlanPath(PlanPathOpt):
    _init_environment = Environment
    _init_c = VisionArch
    _init_p = PlanPathParam

    def init_goals(self):
        self.make_goals([*self.environment.start, 0], [*self.environment.end, 0])

    def make_goals(self, start, end):
        self.goals = {0: start,
                      1: [start[0], start[1], self.p.height],
                      2: [end[0], end[1], self.p.height],
                      3: end}

    def reconfigure_plan(self, new_landing):
        self.make_goals([self.dofs.x, self.dofs.y, self.dofs.z], new_landing)
        self.set_mode("move_em")
        self.pt = 0

    def reconfigure_path(self):
        vis = self.c.components['vision']
        # reconfigure path based on mode
        if (self.m.in_mode('emland')
                and vis.check_if_occupied(self.environment, self.dofs)):
                self.reconfigure_plan(vis.find_nearest_open(self.environment, self.dofs))
        elif (self.m.in_mode('land')
                and vis.check_if_occupied(self.environment, self.dofs)):
            self.reconfigure_plan(self.find_nearest())
        elif self.m.in_mode('to_nearest'):
            self.reconfigure_plan(self.find_nearest())
        elif self.m.in_mode('to_home'):
            self.reconfigure_plan([*self.environment.start, 0.0])

    def behavior(self, t):
        self.s.ground_height = self.environment.ground_height(self.dofs.s.x,
                                                              self.dofs.s.y,
                                                              self.dofs.s.z)
        self.update_mode(t)
        self.reconfigure_path()
        self.update_dist()
        self.update_traj()

    def dynamic_behavior(self):
        self.increment_point()


class DroneParam(DroneParamOpt):
    plan_param: PlanPathParam = PlanPathParam()
    env_param: EnvironmentParameter = EnvironmentParameter()

class Drone(Model):
    _init_p = DroneParam
    default_sp = dict(phases=(('ascend', 0, 0),
                              ('forward', 1, 11),
                              ('taxi', 12, 20)),
                      times=(0, 30),
                      units='min',
                      dt=0.1)

    def __init__(self, name='drone', **kwargs):
        super().__init__(name=name, **kwargs)

        self.add_flow('force_st', Force)
        self.add_flow('force_lin', Force)
        self.add_flow('hsig_dofs', HSig)
        self.add_flow('hsig_bat', HSig)
        self.add_flow('rsig_traj', RSig)
        self.add_flow('ee_1', EE)
        self.add_flow('ee_mot', EE)
        self.add_flow('ee_ctl', EE)
        self.add_flow('ctl', Control)
        self.add_flow('dofs', DOFs)
        self.add_flow('des_traj', DesTraj)
        self.add_flow('environment', Environment, p=self.p.env_param)

        # add functions to the model
        flows = ['ee_ctl', 'force_st', 'hsig_dofs', 'hsig_bat', 'rsig_traj']
        self.add_fxn('manage_health', ManageHealth, *flows, p=asdict(self.p.respolicy))

        store_ee_p = {'archtype': self.p.bat, 'weight': (
            self.p.batweight+self.p.archweight)/2.2, 'drag': self.p.archdrag}
        self.add_fxn('store_ee',    StoreEE, 'ee_1',
                     'force_st', 'hsig_bat', c=store_ee_p)
        self.add_fxn('dist_ee',     DistEE,   'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof',  AffectDOF, 'ee_mot', 'ctl', 'dofs', 'des_traj',
                     'force_lin', 'hsig_dofs',  c={'archtype': self.p.linearch})
        self.add_fxn('ctl_dof',     CtlDOF,   'ee_ctl',
                     'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path',   PlanPath, 'ee_ctl', 'dofs',
                     'des_traj', 'force_st', 'rsig_traj', p=self.p.plan_param)
        self.add_fxn('hold_payload', HoldPayload, 'dofs', 'force_lin', 'force_st')

        self.build()


if __name__ == "__main__":
    p = PlanPath("test", {})
    
    e = Environment("env")
    e.show_grid()
    e.show_3d()



