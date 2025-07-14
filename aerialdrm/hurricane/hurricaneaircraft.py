# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:10:02 2025

@author: dhulse
"""
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.architecture.function import FunctionArchitectureGraph
from fmdtools.analyze.common import consolidate_legend
from fmdtools.define.container.parameter import Parameter
import fmdtools.sim.propagate as prop


from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity
from aerialdrm.base.aircraft.arch.aviate import Aviate
from aerialdrm.base.aircraft.arch.controlflight import ControlFlight, ControlState
from aerialdrm.base.aircraft.arch.storeee import StoreAndSupplyElectricity
from aerialdrm.base.aircraft.arch.perceiveenvironment import PerceiveEnvironment
from aerialdrm.base.aircraft.arch.holdpayload import HoldPayload


from aerialdrm.hurricane.hurricaneenvironment import HurricaneEnvironment, properties, collections
from aerialdrm.hurricane.hurricaneenvironment import HurricaneConditions
from aerialdrm.hurricane.hurricaneflightpath import DroneFlightGrid, DroneFlightGridParam
from aerialdrm.base.aircraft.state import AircraftPosition
import math

class HurricaneControlState(ControlState):
    """ControlState defining the current state of the aircraft's FlightGrid and 
    correspondent planning.
    
    Fields
    ------
    closest_dist: float [outdated?]
    flightgrid: HurricaneFlightGrid
        FlightGrid for current aviation plan; subject to update with faults/
        environmental updates
    planned: bool
        Whether or not flightgrid is up to date
    """
    closest_dist: float = 100.0
    flightgrid: object = None
    planned: bool = False

    
class HurricaneControlParameter(Parameter):
    with_proxthreat: bool = True
    fuel_rate: float = 20.0
    disallowed_cost: float = 10.0
    occupied_cost: float = 20.0
    restricted_cost: float = 1000000.0
    max_distance: int = 5
    blocksize: float = 2.5
    
class HurricaneControlFlight(ControlFlight):
    __slots__ = ()
    flow_environment = HurricaneEnvironment
    container_s = HurricaneControlState
    container_p = HurricaneControlParameter
    
    def set_faultmode(self):
        super().set_faultmode()

        if 0.0 < self.electricity.s.charge <= 25.0:
            if self.s.pt > 0:
                self.replan_mission()
            elif (not self.m.any_faults() and len(self.s.flightplan) > 1) or self.electricity.s.charge <= 15.0:
                if not self.m.any_faults():
                    self.m.set_mode('descend')

        dists = self.environment.ga.calc_dist_to_threats()
        self.s.closest_dist = min([*dists.values()])
        if self.p.with_proxthreat:
            if self.s.closest_dist <= 0.0 and not self.m.any_faults():
                self.m.set_mode('pause')
            elif self.m.in_mode('pause'):
                self.m.set_mode('flight')
                
    def static_behavior(self, time):
        if not self.s.planned:
            self.replan_mission()
            self.s.planned = True
            
    def gen_flight_grid(self):
        dfgp = DroneFlightGridParam(
            fuel_rate = self.p.fuel_rate,
            disallowed_cost = self.p.disallowed_cost,
            occupied_cost = self.p.occupied_cost,
            restricted_cost = self.p.restricted_cost)
        grid = DroneFlightGrid(p = dfgp)
        grid.env_coords = self.environment.c
        return grid
        
    def replan_mission(self):
        """
        re-evaluates flight path based on flight circumstance. 
        
        If flight conditions nominal, use A* as normal.
        
        If battery low,  bump fuel cost significantly based on present fuel
        and plan as if nominal.
        
        If obstacle appears in the path, evaluate between options:
            1. Hold still until it disappears
            2. Temporary descent (necessitates safe landing zone existence, 
            dynamic cost consideration)
            3. Replan around the obstacle 
        """
        
        # print(f"[DEBUG] replan from {curr} to {goal}")
        ap = AircraftPosition() # just a calculator! 
        ap.assign(self.trajectories.perc_traj.s) # initialize this state as current perceived state.
        start = self.s.flightplan[0]
        ap.assign(start, 'goal_x', 'goal_y')
        start_dist = ap.calc_dist()
        end = self.s.flightplan[-1]
        ap.assign(end, 'goal_x', 'goal_y')
        end_dist = ap.calc_dist()
        if 0.0 < self.electricity.s.charge <= 25.0:
            # DO WE NEED A PERCEPTION /= REALITY FAULT? YES
            # 20 below arbitrary
            if start_dist > 20 and end_dist > 20:
                curr_pt = self.trajectories.perc_traj.s.get('x', 'y')
                land_pt = self.environment.c.find_closest(*curr_pt, 'suitable')
            elif start_dist < end_dist:
                land_pt = start
            else:
                land_pt = end
                # closest suitable spot DNE MOST optimal suitable spot. perhaps should prioritize suitable spot closest to the goal?
                # in this case, must consider: 1. feasibility 2. distance to goal
            self.s.flightplan = (tuple(land_pt),)
            self.s.pt = 0
            return
        if self.new_obstacle_present():
            self.s.planned = False # ?
            return
        else:
            curr_x, curr_y = self.trajectories.perc_traj.s.get('x', 'y')
            goal_x, goal_y = self.s.flightplan[-1]
            grid = self.gen_flight_grid()
            new_path = grid.a_star_worldcoords(
                start_xy = (curr_x, curr_y), 
                goal_xy = (goal_x, goal_y), 
                max_distance = self.p.max_distance,
                disallowed_cost = self.p.disallowed_cost,
                occupied_cost = self.p.occupied_cost,
                restricted_cost = self.p.restricted_cost,
                fuel_rate = self.p.fuel_rate
            )
            if new_path and new_path[0] == (curr_x, curr_y):
                new_path = new_path[1:]
            self.s.flightplan = new_path
            self.s.pt = 0
            
    def new_obstacle_present(self):
        return False
        
class HurricaneAviate(Aviate):
    __slots__ = ()

    def dynamic_behavior(self, time):
        super().dynamic_behavior(time)
        self.environment.ga.points['self'].s.assign(self.trajectories.s, 'x', 'y', 'z')



class HurricaneAircraftArchParameter(Parameter):
    """Overall Parameter Defining the AircraftArchitecture."""

    startpt: tuple = (10.0, 10.0)
    flightplan: tuple = ((10.0, 10.0), (50.0, 10.0), (50.0, 100.0), (100.0, 100.0)) 
    height: float = 25.0
    depletion: float = 25.0
    with_proxthreat: bool = True
    # below: new
    fuel_rate: float       = 20.0
    disallowed_cost: float = 10.0
    occupied_cost: float   = 20.0
    restricted_cost: float = 1000000.0
    max_distance: int      = 5

class HurricaneAircraftArchitecture(FunctionArchitecture):
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

    __slots__ = ()
    container_p = HurricaneAircraftArchParameter
    # default_sp = {'end_condition': 'indicate_landed'}
    default_sp = {'end_time': 35}

    def init_architecture(self, **kwargs):
        """Initialize the architecture of the aircraft."""
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories,
                      s={'x': self.p.startpt[0], 'y': self.p.startpt[1]})
        self.add_flow('environment', HurricaneEnvironment)

        self.add_fxn('conditions', HurricaneConditions, 'environment')
        self.add_fxn('control_flight', HurricaneControlFlight,
                     'trajectories', 'force', 'electricity', 'environment',
                     s={'flightplan': self.p.flightplan, 'height': self.p.height},
                     p={'with_proxthreat': self.p.with_proxthreat,
                        'fuel_rate': self.p.fuel_rate,
                        'disallowed_cost': self.p.disallowed_cost,
                        'occupied_cost':   self.p.occupied_cost,
                        'restricted_cost': self.p.restricted_cost,
                        'max_distance':    self.p.max_distance})
        self.add_fxn('aviate', HurricaneAviate,
                     'trajectories', 'force', 'electricity', 'environment')
        m = {'fault_depletion':
             {'disturbances': (('electricity.s.charge', self.p.depletion), )}}
        self.add_fxn('store_and_supply_ee', StoreAndSupplyElectricity,
                     'force', 'electricity', m=m)
        self.add_fxn('perceive_environment', PerceiveEnvironment,
                     'environment', 'force', 'electricity', 'trajectories')
        self.add_fxn('hold_payload', HoldPayload, 'trajectories', 'force')

    def indicate_unsuitable_landing(self):
        coords = [*self.flows['trajectories'].s.get('x', 'y')]
        if self.flows['trajectories'].s.z > 0.0:
            return False
        else:
            return coords not in [[*i] for i in [*self.flows['environment'].c.suitable]]

    def indicate_landed(self, time):
        return self.flows['trajectories'].s.z == 0.0 and time > 5.0

    def find_classification(self, scen, mdlhists):
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


def plot_flightpath(mdl, hist, **kwargs):
    fig, ax = mdl.flows['environment'].c.show(properties=properties,
                                              collections=collections)
    start = mdl.p.flightplan[0]
    end   = mdl.p.flightplan[-1]
    ax.scatter([start[0]], [start[1]], label="start", color="green")
    ax.scatter([end[0]],   [end[1]],   label="end",   color="red")
    fig, ax = hist.plot_trajectories('trajectories.s.x',
                                     'trajectories.s.y',
                                     fig=fig, ax=ax, **kwargs)
    cf   = mdl.fxns['control_flight']
    plan = cf.s.flightplan
    if plan:
        xs, ys = zip(*plan)
        ax.plot(xs, ys, '--', label='planned path', color='red')  # Make line red
        ax.scatter(xs, ys, marker='o', label='waypoints', color='red', s=10)  # Make dots red, smaller
    consolidate_legend(ax)
    return fig, ax



if __name__ == "__main__":

    from fmdtools.analyze.phases import from_hist
    from fmdtools.sim import propagate
    from fmdtools.sim.sample import FaultDomain, FaultSample

    ha = HurricaneAircraftArchitecture()
    
    
    
    cf = ha.fxns['control_flight']
    cf.static_behavior(time=0.0)
    fg = FunctionArchitectureGraph(ha)
    fg.draw()
    res, hist = propagate.nominal(ha)


    ha.flows['environment'].ga.show_from(hist.flows.environment.ga, 10)
    pms = from_hist(hist)
    pm = pms['store_and_supply_ee']

    fd = FaultDomain(ha)
    # fd.add_fault('store_and_supply_ee', 'depletion', '1', disturbances=(('electricity.s.charge', 1.0), ))
    # fd.add_fault('store_and_supply_ee', 'depletion', '16', disturbances=(('electricity.s.charge', 16.0), ))
    fd.add_fault('store_and_supply_ee', 'depletion', '25', disturbances=(('electricity.s.charge', 25.0), ))

    fs = FaultSample(fd, phasemap=pm)
    fs.add_fault_phases('in_use', method='all')
    fs

    ress, hists = propagate.fault_sample(ha, fs)

    hists.plot_trajectories('trajectories.s.x', 'trajectories.s.y', 'trajectories.s.z')

    import doctest
    doctest.testmod(verbose=True)

    from fmdtools.analyze.phases import from_hist
    plot_flightpath(ha, hist)
    # haa = HurricaneAircraftArchitecture(p={'depletion': 40.0})

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



