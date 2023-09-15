# -*- coding: utf-8 -*-
"""
Variant of drone model for modelling computer vision in urban settings.
"""
from examples.multirotor.drone_mdl_static import EE, Force, Control
from examples.multirotor.drone_mdl_static import DistEE
from drone_mdl_rural import DesTraj, DOFs, HSig, RSig
from drone_mdl_rural import ManageHealth, StoreEE, AffectDOF, CtlDOF
from drone_mdl_rural import PlanPath as PlanPathOpt
from drone_mdl_rural import DronePhysicalParameters, ResPolicy
from drone_mdl_rural import HoldPayload as HoldPayloadOpt
from drone_mdl_rural import AffectDOF as AffectDOFOpt
from drone_mdl_rural import Drone as DroneOpt

from fmdtools.define.block import CompArch, Component
from fmdtools.define.mode import Mode
from fmdtools.define.flow import Flow
from fmdtools.define.state import State
from fmdtools.define.parameter import Parameter
from fmdtools.define.model import Model
from fmdtools.define.environment import Grid, GridParam, Environment

from fmdtools.analyze.result import History
from fmdtools.analyze import show

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D, art3d
from recordclass import asdict


class EnvironmentState(State):
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
    safe: bool = True
    allowed: bool = True
    landed: bool = True
    occupied: bool = False


class UrbanGridParam(GridParam):
    """
    Defines the grid parameters, including resolution as well as number of allowed,
    unsafe, and occupied spaces, max height of the buildings, and road width.
    """
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 100.0
    num_allowed: int = 10
    num_unsafe: int = 10
    num_occupied: int = 10
    max_height: float = 100.0
    roadwidth: int = 15
    loc: str = 'urban'


class StreetGrid(Grid):
    """
    Defines the urban environment (buildings, streets, etc) the drone flies through.

    Note the aspects of the Grid:
        - safe: feature
            places where the drone is safe to land
        - allowed: feature
            places the drone is allowed to land
        - occupied: feature
            places where people are (landing would be dangerous)
        - height: feature
            height of the buildings to fly over
        - start: point
            where the drone starts
        - end: point
            where the drone ends
        - all_occupied: collection
            all points that are occupied
        - all_safe: collection
            all points that are safe to land at
    """
    _init_p = UrbanGridParam

    _feature_safe = (bool, True)
    _feature_allowed = (bool, False)
    _feature_occupied = (bool, False)
    _feature_height = (float, 0.0)

    _point_start = (0, 0)
    _point_end = (900, 900)
    _collect_all_occupied = ("occupied", True)
    _collect_all_safe = ("safe", True)

    def init_properties(self, *args, **kwargs):
        """Randomly allocates the allowed/occupied points, as well as the building 
        heights."""
        rand_pts = self.r.rng.choice(self.pts[1:-1],
                                     self.p.num_allowed + self.p.num_unsafe,
                                     replace=False)
        for i, pt in enumerate(rand_pts):
            if i < self.p.num_allowed:
                self.set(*pt, 'allowed', True)
            else:
                self.set(*pt, 'safe', False)
        self.set_rand_pts('occupied', True, 10, pts=self.pts[1:-1])
        self.set_prop_dist("height", "uniform", low=0.0, high=self.p.max_height)
        self.set(0, 0, "height", 0.0)
        self.set(900, 900, "height", 0.0)


class UrbanDroneEnvironment(Environment):
    _init_p = UrbanGridParam
    _init_g = StreetGrid
    _init_s = EnvironmentState

    def ground_height(self, dofs):
        """Gets the distance of the height z above the ground at point x,y"""
        env_height = self.g.get(dofs.s.x, dofs.s.y, 'height', dofs.s.z)
        return dofs.s.z-env_height

    def set_states(self, dofs):
        """Set the landing states safe_land and allowed_land since the drone has
        landed"""
        if self.g.in_range(dofs.s.x, dofs.s.y):
            props = self.g.get_properties(dofs.s.x, dofs.s.y)
            self.s.safe = props['safe']
            self.s.allowed = props['allowed']
            self.s.occupied = props['occupied']
    
            if self.ground_height(dofs) <= 0.1:
                self.s.landed = True
            else:
                self.s.landed = False

class AffectDOF(AffectDOFOpt):
    _init_environment = UrbanDroneEnvironment

    def inc_takeoff(self):
        self.environment.set_states(self.dofs)
        # can only take off at ground
        if self.environment.s.landed:
            self.dofs.s.put(planvel=0.0, vertvel=max(0, self.dofs.s.vertvel))

    def dynamic_behavior(self, time):
        self.calc_vel()
        self.inc_takeoff()
        g_h = self.environment.ground_height(self.dofs)
        self.inc_falling(min_fall_dist=g_h)
        self.inc_pos()
        self.environment.set_states(self.dofs)

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
            occ = environment.g.get(dofs.s.x, dofs.s.y, 'occupied', True)
        self.m.remove_fault("undesired_detection")
        return occ

    def find_nearest_open(self, environment, dofs, include_pt=False):
        if self.m.has_fault("undesired_detection"):
            pt = environment.open[round(len(environment.open)/3)]
            return np.array([*pt, environment.g.get(*pt, "height", 0.0)])
        elif self.m.has_fault("lack_of_detection"):
            return environment.g.find_closest(dofs.s.x, dofs.s.y, 'pts',
                                              include_pt=include_pt)
        else:
            return environment.g.find_closest(dofs.s.x, dofs.s.y, "all_safe",
                                              include_pt=include_pt)


class VisionArch(CompArch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.make_components(ComputerVision, 'vision')


class PlanPathParam(Parameter):
    height: float = 200.0


class PlanPath(PlanPathOpt):
    _init_environment = UrbanDroneEnvironment
    _init_c = VisionArch
    _init_p = PlanPathParam

    def init_goals(self):
        self.make_goals([*self.environment.g.start, 0], [*self.environment.g.end, 0])

    def make_goals(self, start, end):
        self.goals = {0: start,
                      1: [start[0], start[1], self.p.height],
                      2: [end[0], end[1], self.p.height],
                      3: end}

    def reconfigure_plan(self, new_landing, newmode="move_em"):
        self.make_goals([self.dofs.s.x, self.dofs.s.y, self.dofs.s.z], new_landing)
        self.m.set_mode("move_em")
        self.s.pt = 0

    def reconfigure_path(self):
        vis = self.c.components['vision']
        land_occupied = vis.check_if_occupied(self.environment, self.dofs)
        # reconfigure path based on mode
        if (self.m.in_mode('emland') and land_occupied):
            self.reconfigure_plan(vis.find_nearest_open(self.environment, self.dofs))
        elif (self.m.in_mode('land') and land_occupied):
            self.reconfigure_plan(self.find_nearest())
        elif self.m.in_mode('to_nearest'):
            self.reconfigure_plan(self.find_nearest())
        elif self.m.in_mode('to_home'):
            self.reconfigure_plan([*self.environment.g.start, 0.0])

    def find_nearest(self):
        return self.environment.g.find_closest(self.dofs.s.x, self.dofs.s.y,
                                               "all_safe", include_pt=False)

    def behavior(self, t):
        self.s.ground_height = self.environment.ground_height(self.dofs)
        self.update_mode(t)
        self.reconfigure_path()
        self.update_dist()
        self.update_traj()

    def dynamic_behavior(self, time):
        self.increment_point()


class HoldPayload(HoldPayloadOpt):
    _init_environment = UrbanDroneEnvironment

    def at_ground(self):
        ground_height = self.environment.ground_height(self.dofs)
        return ground_height <= 0.0


class DroneParam(Parameter):
    plan_param: PlanPathParam = PlanPathParam()
    env_param: UrbanGridParam = UrbanGridParam()
    phys_param: DronePhysicalParameters = DronePhysicalParameters()
    respolicy: ResPolicy = ResPolicy()


class Drone(DroneOpt):
    _init_p = DroneParam
    default_sp = dict(phases=(('ascend', 0, 0),
                              ('forward', 1, 11),
                              ('taxi', 12, 20)),
                      times=(0, 30),
                      units='min',
                      dt=0.1)

    def __init__(self, name='drone', **kwargs):
        Model.__init__(self, name=name, **kwargs)

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
        self.add_flow('environment', UrbanDroneEnvironment, g=dict(p=self.p.env_param))

        # add functions to the model
        flows = ['ee_ctl', 'force_st', 'hsig_dofs', 'hsig_bat', 'rsig_traj']
        self.add_fxn('manage_health', ManageHealth, *flows, p=asdict(self.p.respolicy))

        store_ee_p = {'archtype': self.p.phys_param.bat,
                      'weight': self.p.phys_param.batweight+self.p.phys_param.archweight,
                      'drag': self.p.phys_param.archdrag}
        self.add_fxn('store_ee', StoreEE, 'ee_1', 'force_st', 'hsig_bat', c=store_ee_p)
        self.add_fxn('dist_ee', DistEE, 'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof', AffectDOF, 'ee_mot', 'ctl', 'dofs', 'des_traj',
                     'force_lin', 'hsig_dofs', 'environment',
                     c={'archtype': self.p.phys_param.linearch})
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path',   PlanPath, 'ee_ctl', 'dofs', 'des_traj', 'force_st',
                     'rsig_traj', 'environment', p=self.p.plan_param)
        self.add_fxn('hold_payload', HoldPayload, 'dofs', 'force_lin', 'force_st', 
                     'environment')

        self.build()

    def indicate_landed(self, time):
        """
        Custom indicator for ending the simulation. Returns true if the
        drone has entered the "landed" state.
        """
        return time > 1 and self.fxns['plan_path'].m.mode == 'taxi'

    def at_start(self, dofs):
        return self.flows['environment'].g.in_area(dofs.s.x, dofs.s.y, "start")
    
    def at_safe(self, dofs):
        return self.flows['environment'].g.in_area(dofs.s.x, dofs.s.y, "all_safe")
    
    def at_dangerous(self, dofs):
        return self.flows['environment'].g.in_area(dofs.s.x, dofs.s.y, "all_occupied")

    def find_classification(self, scen, mdlhist):
        faulttime = self.h.get_fault_time(metric='total')

        land_metrics = self.calc_land_metrics(scen, mdlhist, faulttime)

        # repair costs
        repcost = self.calc_repaircost(max_cost=1500)

        totcost = (land_metrics['landcost']
                   + land_metrics['safecost']
                   + repcost)

        metrics = {'rate': scen.rate,
                   'cost': totcost,
                   'expected cost': totcost * scen.rate * 1e5,
                   'repcost': repcost,
                   'unsafe_flight_time': faulttime,
                   **land_metrics}
        return metrics

def plot_env_with_traj(mdlhists, mdl, legend=True, title="trajectory"):
    """
    Plots given 2d Drone trajectories over the gridword.

    Parameters
    ----------
    mdlhist : dict
        Dict of model histories.
    mdl : Drone
        Drone model object.
    title : str
        Title for the plot. The default is 'Trajectory'.
    legend : bool, optional
        Whether to include a legend. The default is False.

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axis
    """
    fig, ax = show.grid(mdl.flows['environment'].g, "height",
                        collections={"all_occupied": {"color": "red"},
                                     "start": {"color": "blue"},
                                     "end": {"color": "blue"}})
    fig, ax = show.trajectories(mdlhists, "dofs.s.x", "dofs.s.y",
                                fig=fig, ax=ax, legend=legend, title=title)
    return fig, ax

def plot_env_with_traj3d(mdlhists, mdl, legend=True, title="trajectory"):
    """
    Plots given 3d Drone trajectories over the gridword.

    Parameters
    ----------
    mdlhist : dict
        Dict of model histories.
    mdl : Drone
        Drone model object.
    title : str
        Title for the plot. The default is 'Trajectory'.
    legend : bool, optional
        Whether to include a legend. The default is False.

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axis
    """
    collections = {"all_safe": {"color": "blue", "label": False},
                   "all_occupied": {"color": "red", "label": False},
                   "start": {"color": "yellow", "label": True, "text_z_offset": 30},
                   "end": {"color": "yellow", "label": True, "text_z_offset": 30}}
    
    fig, ax = show.grid3d(mdl.flows['environment'].g, "height",
                        collections=collections)
    fig, ax = show.trajectories(mdlhists, "dofs.s.x", "dofs.s.y", "dofs.s.z",
                                fig=fig, ax=ax, legend=legend, title=title)
    ax.set_zlim3d(0, mdl.p.plan_param.height)
    for goal, loc in mdl.fxns['plan_path'].goals.items():
        ax.text(loc[0], loc[1], loc[2], str(goal), fontweight='bold', fontsize=12)
        ax.plot([loc[0]], [loc[1]], [loc[2]],
                marker='o', markersize=10, color='red', alpha=0.5)
    return fig, ax


def make_move_quad(mdlhist, move_phase, weights = [0.003, 0.5, 0.46]):
    """
    Creates a quadrature for SampleApproach over a provided phase that has the drone
    over an unsafe area, over a safe area, and over the landing area.
    """
    unsafe_times = [t for i, t in enumerate(mdlhist['time'])
                    if not mdlhist.flows.environment.s.safe[i]
                    and mdlhist.fxns.plan_path.m.mode[i] == 'move']
    safe_times = [t for i, t in enumerate(mdlhist['time'])
                  if mdlhist.flows.environment.s.safe[i]
                  and not mdlhist.flows.environment.s.allowed[i]
                  and mdlhist.fxns.plan_path.m.mode[i] == 'move']
    land_times = [t for i, t in enumerate(mdlhist['time'])
                  if mdlhist.flows.environment.s.safe[i]
                  and mdlhist.flows.environment.s.allowed[i]
                  and mdlhist.fxns.plan_path.m.mode[i] == 'move']

    nodes = []
    ws = []
    for i, times in enumerate([unsafe_times, safe_times, land_times]):
        if times:
            time = np.percentile(times, 50, method='nearest')
            node = 2*(time-move_phase[0])/(move_phase[1]-move_phase[0])-1.0
            nodes.append(node)
            ws.append(2*weights[i])
    return {'samp': 'quadrature', 'quad': {'nodes': nodes, 'weights': weights}}

if __name__ == "__main__":
    from fmdtools.sim import propagate
    from fmdtools import analyze as an
    from fmdtools.sim.approach import SampleApproach
    
    #UrbanDroneEnvironment("a")
    
    #PlanPath._init_environment("a")

    #p = PlanPath("test", {})
    
    e = UrbanDroneEnvironment("env")
    show.grid(e.g, "height")
    show.grid3d(e.g, "height")
    
    mdl = Drone()
    # ec, mdlhist_fault = propagate.one_fault(mdl, "plan_path", "vision_lack_of_detection", time=4.5)

    ec, mdlhist = propagate.nominal(mdl)
    
    phases, modephases = mdlhist.get_modephases()
    an.plot.phases(phases, modephases)
    
    an.plot.hist(mdlhist, "flows.dofs.s.planvel","flows.dofs.s.vertvel", "fxns.store_ee.s.soc")
    plot_env_with_traj(mdlhist, mdl)
    plot_env_with_traj3d(mdlhist, mdl)
    
    
    
    move_quad=make_move_quad(mdlhist, phases['plan_path']['move'])
    
    """
    ec, mdlhist = propagate.one_fault(mdl, 'store_ee', 'nocharge', 4.5)
    app = SampleApproach(mdl, phases=phases, modephases=modephases,
                         sampparams = {('PlanPath','move'): move_quad})
    app
    endresults, hists = propagate.approach(mdl, app, staged=False)
    statsfmea = an.tabulate.fmea(endresults, app, group_by='fxnfault',
                                 weight_metrics=['rate'],
                                 avg_metrics=['unsafe_flight_time', 'cost', 'repcost',
                                              'landcost', 'body_strikes',
                                              'head_strikes', 'property_restrictions'],
                                 sort_by='cost')
    plot_traj(hists, mdl, title="", legend=True)
    """
    #move_quad = make_move_quad(mdlhist, phases['PlanPath']['move'])
