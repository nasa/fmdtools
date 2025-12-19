# -*- coding: utf-8 -*-
"""
Contingency management model aircraft.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from examples.airspacelib.base.arch.flows import Trajectories, Force, Electricity
from examples.airspacelib.base.arch.aviate import Aviate
from examples.airspacelib.base.arch.controlflight import ControlFlight, ControlState
from examples.airspacelib.base.arch.storeee import StoreAndSupplyElectricity
from examples.airspacelib.base.arch.perceiveenvironment import PerceiveEnvironment
from examples.airspacelib.base.arch.holdpayload import HoldPayload

from examples.airspacelib.contingencymanagement.environment import ContingencyEnvironment, properties, collections
from examples.airspacelib.contingencymanagement.environment import ContingencyConditions
from examples.airspacelib.contingencymanagement.flightplanner import DroneFlightGrid, DroneFlightGridParam

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.architecture.function import FunctionArchitectureGraph
from fmdtools.analyze.common import consolidate_legend, add_title_xylabs, setup_plot
from fmdtools.analyze.common import prep_animation_title, clear_prev_figure
from fmdtools.define.container.parameter import Parameter
from fmdtools.analyze.history import History
from fmdtools.analyze.phases import from_hist
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample

import numpy as np


class ContingencyControlState(ControlState):
    """
    ControlState defining the current state of the planning over the grid.

    Fields
    ------
    closest_dist: float
        Distance to the closest drone.
    flightgrid: ContingencyFlightGrid
        FlightGrid for current aviation plan; subject to update with faults/
        environmental updates
    endpt : tuple
        End-point of current flight plan
    planned: bool
        Whether or not the flight-plan has been generated to the current goal.
    reconfigured_proxthreat : bool
        Whether or not proxthreat has triggered flight replanning (replans if not)
    reconfigured_charge : bool
        Whether or not low charge has triggered flight replanning (replans if not)
    """

    closest_dist: float = 100.0
    flightgrid: DroneFlightGrid = DroneFlightGrid(env = ContingencyEnvironment())
    endpt: tuple = (100.0, 100.0)
    planned: bool = False
    reconfigured_proxthreat: bool = False
    reconfigured_charge: bool = False

    
class ContingencyControlParameter(Parameter):
    """
    Parameter defining the control of the drone.

    Parameters
    ----------
    with_proxthreat : bool
        Whether or not proximity to threat functionality is enabled
    fuel_rate : float
        The fuel usage rate of the drone. Default is 20.0
    disallowed_cost : float
        Cost for flying over disallowed areas. Default is 10.0
    occupied_cost : float
        Cost for flying over occupied areas. Default is 20.0
    restricted_cost : float
        Cost for flying in restricted areas. Default is 1000000.
    max_distance : int
        Maximum distance for grid? Default is 5.
    blocksize : float
        Block size of grid. Default is 2.5
    """

    with_proxthreat: bool = True
    fuel_rate: float = 20.0
    disallowed_cost: float = 10.0
    occupied_cost: float = 20.0
    restricted_cost: float = 1000000.0
    max_distance: int = 5
    blocksize: float = 2.5
    
class ContingencyControlFlight(ControlFlight):
    """Function that controls the drone."""

    flow_environment = ContingencyEnvironment
    container_s = ContingencyControlState
    container_p = ContingencyControlParameter
    default_track = {'s': ["closest_dist", 'planned', 'pt', 'flightplan'], 'm': ['mode']}
    
    def set_faultmode(self):
        """Set contingency actions for the drone in proxthreat/low charge cases."""
        super().set_faultmode()

        if 0.0 < self.electricity.s.charge <= 25.0:
            if self.s.pt > 0 and not self.s.reconfigured_charge:
                self.set_depletion_land_loc()
                self.replan_mission()
                self.s.pt = 1
                self.s.reconfigured_charge = True
            elif self.electricity.s.charge <= 15.0:
                if not self.m.any_faults():
                    self.m.set_mode('descend')

        if self.p.with_proxthreat:
            dists = self.environment.ga.calc_dist_to_threats()
            if dists:
                self.s.closest_dist = min([*dists.values()])
            if self.s.closest_dist <= 0.0 and not self.m.any_faults():
                self.m.set_mode('pause')
            elif self.s.closest_dist <= 10.0 and not self.m.any_faults():
                if not self.s.reconfigured_proxthreat:
                    self.replan_mission()
                    self.s.pt = 1
                    self.s.reconfigured_proxthreat = True
            else:
                if self.m.in_mode('pause'):
                    self.m.set_mode('flight')
                self.s.reconfigured_proxthreat = False
                
    def static_behavior(self):
        """Simulate static behavior for planning."""
        if not self.s.planned:
            self.replan_mission()
        super().static_behavior()

    def set_depletion_land_loc(self):
        """Set new landing location in depletion scenario."""
        ts = self.trajectories.s.copy()
        ts.assign(self.environment.c.start, "goal_x", "goal_y")
        dist_to_start = ts.calc_dist_to_travel(1000)
        ts.assign(self.environment.c.end, "goal_x", "goal_y")
        dist_to_end = ts.calc_dist_to_travel(1000)
        if dist_to_end < self.electricity.s.charge:
            self.s.endpt = self.environment.c.end
        elif dist_to_start < self.electricity.s.charge:
            self.s.endpt = self.environment.c.start
        else:
            pt = self.trajectories.s.gett('x', 'y')
            self.s.endpt = self.environment.c.find_closest(*pt, 'suitable')

    def gen_flight_grid(self):
        """Generate flight grid."""
        # max cost set such that re-planning doesn't occur in DroneFlightGrid
        dfgp = DroneFlightGridParam(blocksize=self.p.blocksize,
                                    x_size=120/self.p.blocksize,
                                    y_size=120/self.p.blocksize,
                                    max_cost=np.inf)
        grid = DroneFlightGrid(env = self.environment, p = dfgp)
        return grid
        
    def replan_mission(self):
        """Re-evaluate flight path based on flight circumstance."""
        grid = self.gen_flight_grid()
        # if UAV within distance 10, require A* consideration for UAV safety
        obstacle = self.s.closest_dist <= 10
        # if battery low, bump fuel cost proportionally to current battery level
        if 0.0 < self.electricity.s.charge <= 25.0:
            fuel_rate = self.p.fuel_rate * 25000.0 / self.electricity.s.charge
        else:
            fuel_rate = self.p.fuel_rate
        curr_pt = self.trajectories.perc_traj.s.gett('x', 'y')
        new_path = grid.a_star_worldcoords(
            start_xy = curr_pt, 
            goal_xy = self.s.endpt, 
            max_distance = self.p.max_distance,
            disallowed_cost = self.p.disallowed_cost,
            occupied_cost = self.p.occupied_cost,
            restricted_cost = self.p.restricted_cost,
            fuel_rate = fuel_rate, 
            obstacle = obstacle
            )

        self.s.flightplan = new_path
        self.s.pt = 0
        self.s.planned = True
        

class ContingencyAviate(Aviate):
    """Movement of the drone. Updates encironment point."""

    def dynamic_behavior(self):
        super().dynamic_behavior()
        self.environment.ga.points['self'].s.assign(self.trajectories.s, 'x', 'y', 'z')



class ContingencyAircraftArchParameter(Parameter):
    """Overall Parameter Defining the AircraftArchitecture."""

    startpt: tuple = (10.0, 10.0)
    endpt: tuple = (100.0, 100.0)
    flightplan: tuple = ((10.0, 10.0), (50.0, 10.0), (50.0, 100.0), (100.0, 100.0)) 
    height: float = 25.0
    depletion: float = 25.0
    with_proxthreat: bool = True
    intruders: str = "across"
    gridcase: str = "mix"
    fuel_rate: float = 20.0
    disallowed_cost: float = 10.0
    occupied_cost: float = 20.0
    restricted_cost: float = 1000000.0
    max_distance: int = 5

class ContingencyAircraftArchitecture(FunctionArchitecture):
    """
    Overall drone architecture.

    Involves flows:
        - force: the flow of force between physical components
        - electricity: the flow of electrical energy from the power supply to functions
        - trajectories: the 3d position and velocity of the aircraft
        - environment: the environment the aircraft inhabits and interacts with
    And functions:
        - control_flight : flight planning and control
        - aviate : the function that moves the drone in the x/y/z
        - store_and_supply_ee : the aircraft power supply/battery
        - perceive_environment : drone perception and localization
        - hold_payload : the structure of the drone
    """

    container_p = ContingencyAircraftArchParameter
    # default_sp = {'end_condition': 'indicate_landed'}
    default_sp = {'end_time': 100}

    def init_architecture(self, **kwargs):
        """Initialize the architecture of the aircraft."""
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories,
                      s={'x': self.p.startpt[0], 'y': self.p.startpt[1]})
        self.add_flow('environment', ContingencyEnvironment,
                      p={'intruders': self.p.intruders, 'gridcase': self.p.gridcase})

        self.add_fxn('conditions', ContingencyConditions, 'environment')
        self.add_fxn('control_flight', ContingencyControlFlight,
                     'trajectories', 'force', 'electricity', 'environment',
                     s={'flightplan': self.p.flightplan, 'height': self.p.height, 'endpt': self.p.endpt},
                     p={'with_proxthreat': self.p.with_proxthreat,
                        'fuel_rate': self.p.fuel_rate,
                        'disallowed_cost': self.p.disallowed_cost,
                        'occupied_cost':   self.p.occupied_cost,
                        'restricted_cost': self.p.restricted_cost,
                        'max_distance':    self.p.max_distance})
        self.add_fxn('aviate', ContingencyAviate,
                     'trajectories', 'force', 'electricity', 'environment')
        m = {'fault_depletion':
             {'disturbances': (('electricity.s.charge', self.p.depletion), )}}
        self.add_fxn('store_and_supply_ee', StoreAndSupplyElectricity,
                     'force', 'electricity', m=m)
        self.add_fxn('perceive_environment', PerceiveEnvironment,
                     'environment', 'force', 'electricity', 'trajectories')
        self.add_fxn('hold_payload', HoldPayload, 'trajectories', 'force')

    def indicate_unsuitable_landing(self):
        """Indicate if landed in an unsuitable area."""
        coords = [*self.flows['trajectories'].s.get('x', 'y')]
        if self.flows['trajectories'].s.z > 0.0:
            return False
        else:
            return coords not in [[*i] for i in [*self.flows['environment'].c.suitable]]

    def indicate_landed(self):
        """Indicate if landed."""
        return self.flows['trajectories'].s.z == 0.0 and self.t.time > 5.0

    def classify(self, scen, **kwargs):
        """Classify the simulation results."""
        endloc = self.flows['trajectories'].s.get('x', 'y')
        coords = self.flows['environment'].c
        xs = self.h.flows.trajectories.s.x
        ys = self.h.flows.trajectories.s.y
        any_rest = any([coords.get(x, ys[i], 'restricted') for i, x in enumerate(xs)
                        if coords.in_range(x, ys[i])])
        mission_complete = all(endloc == self.p.flightplan[-1])
        landing_damage = self.fxns['hold_payload'].m.any_faults()
        crash = self.fxns['aviate'].m.has_fault('crash')
        return {'faultmodes': {*self.return_faultmodes()},
                'unsuitable_landing': self.indicate_unsuitable_landing(),
                'disallowed_landing': coords.get(*endloc, 'disallowed', outside=True),
                'occupied_landing': coords.get(*endloc, 'occupied', outside=True),
                'restricted_landing': coords.get(*endloc, 'restricted', outside=True),
                'restricted_flight': any_rest,
                'mission_complete': mission_complete,
                'landing_damage': landing_damage,
                'crash': crash}


def plot_environment(mdl, properties=properties, collections=collections,
                     fig={}, ax={}, legend_kwargs={}, **kwargs):
    """
    Plot the environment of the drone.

    Parameters
    ----------
    mdl : ContingencyAircraftArchitecture
        Aircraft architecture simualted.
    properties : dict, optional
        Properties of the Coords to show. The default is properties.
    collections : dict, optional
        Collections of the Coords to show. The default is collections.
    fig : mpl.Figure, optional
        Figure to show on. The default is {}.
    ax : mpl.axis, optional
        Axis to add to. The default is {}.
    legend_kwargs : dict, optional
        Keyword arguments to legend. The default is {}.
    **kwargs : kwargs
        Other kwargs (unused)

    Returns
    -------
    fig : mpl.Figure
        Figure with environment of drone shown.
    ax : mpl.axis
        Corresponding figure axis.
    """
    fig, ax = mdl.flows['environment'].c.show(properties=properties,
                                              collections=collections,
                                              fig=fig, ax=ax,
                                              legend_kwargs=legend_kwargs)
    start = mdl.p.flightplan[0]
    end   = mdl.p.flightplan[-1]
    ax.scatter([start[0]], [start[1]], label="start", color="green", s=100, marker="X")
    ax.scatter([end[0]],   [end[1]],   label="end", color="red", s=100, marker="X")
    return fig, ax


def collect_plans(history):
    """Get all flight plans from a given history and their indices (helper)."""
    plan_hist = history.fxns.control_flight.s.flightplan
    plans = [plan_hist[0]]
    inds = [0]
    for i, plan in enumerate(plan_hist):
        if plan != plans[-1]:
            plans.append(plan)
            inds.append(i)
    return plans, inds


def plot_plan(ax, plan, inds, i, history, plan_colors=['gray', 'purple', 'orange', 'yellow']):
    """Plot a flight plan on an axis at point i in a history (helper)."""
    xs, ys = zip(*plan)
    ind = inds[i]
    ax.plot(xs, ys, '--', label='plan t='+str(ind), color=plan_colors[i], linewidth=1)  # Make line red
    ax.scatter(xs, ys, marker='o', label='waypoints', color=plan_colors[i], s=10)  # Make dots red, smaller
    if i > 0:
        ax.scatter(history.flows.trajectories.s.x[ind],
                   history.flows.trajectories.s.y[ind],
                   marker="*", s=20, color=plan_colors[i],
                   label="replan pt="+str(ind))


def plot_locations(ax, hists, text="", label="location", color="blue"):
    """Plot the locations of the drone on an axis with given text (helper)."""
    for scen, hist in hists.nest(1).items():
        x = hist.flows.trajectories.s.x[-1]
        y = hist.flows.trajectories.s.y[-1]
        ax.scatter(x, y, marker="^", color=color, label=label, s=50)
        if text:
            if text == "split":
                scen = scen.split("_")[-1]
            ax.text(x, y, scen)


def plot_flightpath(mdl={}, history={}, fig={}, ax={}, with_plans=True, with_locations=True,
                    boundaries_at=False, ft_kwar={}, text="", title="",
                    legend_kwargs={}, **kwargs):
    """
    Plot the fligthpath of a drone over its environment, along with other information.

    Parameters
    ----------
    mdl : ContingencyAircraftArchitecture
        Aircraft architecture simualted.
    history : History, optional
        History (may be nested) of model states from simulation. The default is {}.
    fig : mpl.Figure, optional
        Figure to show on. The default is {}.
    ax : mpl.axis, optional
        Axis to add to. The default is {}.
    with_plans : bool, optional
        Whether to overlay flight plans on the plot. The default is True.
    with_locations : bool, optional
        Whether to overlay locations on the plot. The default is True.
    boundaries_at : bool, optional
        Whether or not to show threat/intruder shapes on the plot. The default is False.
    ft_kwar : dict, optional
        Keyword arguments to faulty trajectories. The default is {}.
    text :str, optional
        Text to show for points ("split" is just last portion or "" is none).
        The default is "".
    title : str, optional
        Title for the plot. The default is "".
    legend_kwargs : dict, optional
        Keyword arguments to legend. The default is {}.
    **kwargs : kwargs
        kwargs to add_title_xylabs

    Returns
    -------
    fig : mpl.Figure
        Figure with environment of drone shown.
    ax : mpl.axis
        Corresponding figure axis.
    """
    fig, ax = plot_environment(mdl, fig=fig, ax=ax, legend_kwargs=legend_kwargs,
                               **kwargs)


    if 'nominal' in history.nest(1) and not ft_kwar:
        ft_kwar={'faulty': dict(linewidth=1, alpha=0.5, color='red'),
                 'nominal': dict(linewidth=3, color='blue', linestyle="--")}
        kw = {}
    else:
        kw = dict(linewidth=3, color='blue', linestyle="--")
    kw['legend'] = legend_kwargs is not False
    if 'time_ticks' in kwargs:
        kw['time_ticks'] = kwargs.pop("time_ticks")
    if 'time_groups' in kwargs:
        kw['time_groups'] = kwargs.pop("time_groups")
    fig, ax = history.plot_trajectories('trajectories.s.x',
                                        'trajectories.s.y',
                                        fig=fig, ax=ax, indiv_kwargs=ft_kwar, **kw)

    if with_plans:
        if 'nominal' in history.nest(1):
            phist = history.nominal
        else:
            phist = history
        plans, inds = collect_plans(phist)
        for i, plan in enumerate(plans):
            plot_plan(ax, plan, inds, i, phist)
    if with_locations:
        if 'nominal' not in history.nest(1):
            hists = History(nominal=history)
        else:
            hists = history
        plot_locations(ax, hists, text=text)
    if boundaries_at is not False:
        if 'nominal' in history.nest(1):
            bhist = history.nominal
        else:
            bhist = history
        geoms = dict()
        for pt in mdl.flows['environment'].ga.points:
            if pt == 'self':
                geoms[pt] = {'shapes': {'envelope': {'color': 'blue'}}}
            else:
                geoms[pt] = {'shapes': {'safety': {'color': 'orange'}, 'envelope': {'color': 'gray'}}}
        fig, ax = mdl.flows['environment'].ga.show_from(bhist.flows.environment.ga,
                                                        boundaries_at,
                                                        fig=fig, ax=ax, geoms=geoms,
                                                        legend = legend_kwargs is not False)
        

    add_title_xylabs(ax, title=title, aspect='equal', **kwargs)
    if legend_kwargs is not False:
        consolidate_legend(ax, **legend_kwargs)
    return fig, ax


def plot_flightpath_from(time, history={}, with_boundaries=True, **kwargs):
    """
    Plot combined environment and flightpath at time.

    Parameters
    ----------
    time : int
        Simulation time.
    history : History, optional
        History to plot from. The default is {}.
    with_boundaries : bool, optional
        Whether to plot boundaries of threats. The default is True.
    **kwargs : kwargs
        kwargs for plot_flightpath.

    Returns
    -------
    fig : mpl.Figure
        Figure with environment of drone shown.
    ax : mpl.axis
        Corresponding figure axis.

    """
    hist = history.cut(time, newcopy=True)
    kwargs = prep_animation_title(time, **kwargs)
    kwargs = clear_prev_figure(**kwargs)
    if with_boundaries:
        with_boundaries=time
    return plot_flightpath(history=hist, boundaries_at=with_boundaries, **kwargs)


def create_legend_plot(ax, **kwargs):
    """Plot the legend for an axis ax on its own."""
    fig, ax2 = setup_plot(**kwargs)
    ax2.axis("off")
    consolidate_legend(ax2, loc='center', bbox_to_anchor=(0.5,0.5), add_handles=[ax],
                       old_legend=ax.get_legend())
    return fig, ax2

if __name__ == "__main__":

    haa = ContingencyAircraftArchitecture()
    haa()
    haa2 = haa.copy()

    hcs = ContingencyControlState()
    hcs.create_hist([0.0, 1.0])


    h = ContingencyAviate()
    hc = ContingencyConditions()
    hcf = ContingencyControlFlight()

    ha = ContingencyAircraftArchitecture(p={'intruders': ''})

    hcs = ContingencyControlState()
    hcs2 = hcs.copy()


    cf = ha.fxns['control_flight']
    cf.static_behavior()
    fg = FunctionArchitectureGraph(ha)
    fg.draw()
    res, hist = propagate.nominal(ha)
    hist.plot_line_from(15, plot_values=('flows.trajectories.s.x', 'flows.trajectories.s.y','flows.trajectories.s.z', 'fxns.control_flight.m.mode', 'flows.electricity.s.charge'), t_line=True)
    fig, ax = plot_flightpath(ha, hist, legend_kwargs=False, boundaries_at=10)
    hist.plot_line('flows.trajectories.s.x', 'flows.trajectories.s.y','flows.trajectories.s.z', 'fxns.control_flight.m.mode', 'flows.electricity.s.charge')
    hist.fxns.control_flight.s.flightplan


    ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 10)
    pms = from_hist(hist)
    pm = pms['store_and_supply_ee']

    fd = FaultDomain(ha)
    # fd.add_fault('store_and_supply_ee', 'depletion', '1', disturbances=(('electricity.s.charge', 1.0), ))
    # fd.add_fault('store_and_supply_ee', 'depletion', '16', disturbances=(('electricity.s.charge', 16.0), ))
    fd.add_fault('store_and_supply_ee', 'depletion', '25', disturbances=(('electricity.s.charge', 25.0), ))

    fs = FaultSample(fd, phasemap=pm)
    fs.add_fault_phases("in_use", args=(10, ))
    # fs.add_fault_phases('in_use', method='all')
    fs

    ress, hists = propagate.fault_sample(ha, fs)


    fig, ax = ha.flows['environment'].c.show_z("disallowed", z="", collections=collections)
    hists.plot_trajectories('trajectories.s.x', 'trajectories.s.y', 'trajectories.s.z', fig=fig, ax=ax)
    fig, ax = plot_flightpath(ha, hists, with_locations=True, text="split")

    import doctest
    doctest.testmod(verbose=True)

    from fmdtools.analyze.phases import from_hist
    fig, ax = plot_flightpath(ha, hist)
    ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 11, fig=fig, ax=ax)
    #ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 10, fig=fig, ax=ax)
    
    ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 18, fig=fig, ax=ax)
    ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 20, fig=fig, ax=ax)


    ani = hist.animate(plot_flightpath_from, mdl=ha)

    # haa = ContingencyAircraftArchitecture(p={'depletion': 40.0})

    # res, hist = prop.nominal(haa)
    # pm = from_hist(hist)

    # # res, hist = prop.one_fault(haa, 'store_and_supply_ee', 'break', 8, desired_result=['endclass', 'graph'])
    # res, hist = prop.one_fault(haa, 'store_and_supply_ee', 'depletion', 18, desired_result=['endclass', 'graph'])
    # res, hist = prop.one_fault(haa, 'control_flight', 'loss', 19, desired_result=['endclass', 'graph'])
    # res.graph.draw()

    # fig, ax = haa.flows['environment'].c.show(properties=properties,
    #                                           collections=collections)

    # hist.plot_trajectories('trajectories.s.x', 'trajectories.s.y', fig=fig, ax=ax)


    # fig, ax = haa.flows['environment'].c.show_collection('suitable', z=0,
    #                                                      **collections['suitable'])

    # fig, ax = hist.plot_trajectories('trajectories.s.x',
    #                                  'trajectories.s.y',
    #                                  'trajectories.s.z',
    #                                  time_groups='nominal', time_ticks=2.0, fig=fig, ax=ax)

    # fig, ax = hist.plot_trajectories('environment.ga.points.uav.s.x',
    #                                  'environment.ga.points.uav.s.y',
    #                                  'environment.ga.points.uav.s.z',
    #                                  time_groups='nominal', time_ticks=2.0, fig=fig, ax=ax)


    # hist.plot_line('flows.electricity.s.charge',
    #                'fxns.control_flight.m.mode',
    #                'fxns.aviate.m.mode')
    # plot_flightpath(haa, hist)



