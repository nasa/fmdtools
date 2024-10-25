#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:33:54 2024

@author: smbaye
"""

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import GenericFxn
from fmdtools.define.block.function import Function
# from fmdtools.define.container.mode import Mode
# from fmdtools.define.flow.base import Flow
# from fmdtools.define.architecture.function import FunctionArchitecture
# from fmdtools.define.architecture.base import check_model_pickleability
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
# from fmdtools.define.container.time import Time
# import fmdtools.analyze as an
# import fmdtools.sim.propagate as propagate
import numpy as np
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment


class FireMapParam(CoordsParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 1000.0
    strike_prob: float = 0.1
    state_to_burn: tuple = (bool, False)
    state_burned: tuple = (bool, False)
    feature_strike: tuple = (bool, False)
    feature_tree: tuple = (bool, True)
    feature_water: tuple = (bool, False)
    feature_grass: tuple = (bool, False)
    point_base: tuple = (0.0, 0.0)


class FireMap(Coords):
    container_p = FireMapParam

    def init_properties(self, *args, **kwargs):
        self.set_prop_dist('strike', 'binomial', 1, self.p.strike_prob)
        self.set_range('tree', False, ymin=5000, ymax=10000, xmin=0, xmax=10000)
        self.set_range('water', True, xmin=0, xmax=4000, ymin=5000, ymax=10000)
        self.set_range('grass', True, xmin= 5000, xmax=10000, ymin=5000, ymax=10000)

    def get_neighbors(self, x, y):
        ind = self.to_index(x, y)
        neighbor_list = [(ind[0]+1, ind[1]),
                         (ind[0]-1, ind[1]),
                         (ind[0], ind[1]+1),
                         (ind[0], ind[1]-1)]
        neighbors = []
        for i, n_point in enumerate(neighbor_list):
            if not (n_point[0] < 0 or n_point[1] < 0 or n_point[0]>=self.p.x_size or n_point[1]>=self.p.y_size):
                neighbors.append(self.grid[n_point[0], n_point[1]])
        return neighbors


class FireEnvironment(Environment):
    __slots__ = ()
    coords_c = FireMap

    def prop_time(self):
        for pt in self.c.pts:
            if self.c.get(*pt, 'to_burn'):
                self.c.set(*pt, 'burned', True)

        for pt in self.c.pts:
            # light the fire where lightning has struck
            if self.c.get(*pt, 'strike'):
                self.c.set(*pt, 'burned', True)
            # light the fire next to burning points
            if self.c.get(*pt, 'burned'):
                possible = self.c.get_neighbors(*pt)
                for pt in possible:
                    self.c.set(*pt, 'to_burn', True)


class AircraftStates(State):
    retardant_status: float = 100 # starting retardant at 100%
    fuel_status: float = 100 # starting fuel at 100%
    goal_x: float = 100.0
    goal_y: float = 100.0
    location_x: float = 0.0
    location_y: float = 0.0
    direction: np.array = np.array([0, 0])
    
    def find_direction(self):
        vector_dist = np.array([self.goal_x-self.location_x, self.goal_y-self.location_y])
        goal_dist = np.sqrt(vector_dist[0]**2+vector_dist[1]**2)
        return vector_dist/goal_dist
    
class AircraftParam(Parameter, readonly=True):
    number: int = 1
    max_range: float = 1287.0 # in km
    max_speed: float = 6.6  # in km/min
    base: int = 1
    
class Aircraft(Function):
    __slots__=('fireenvironment')
    container_s = AircraftStates
    container_p = AircraftParam
    flow_fireenvironment = FireEnvironment

    def indicate_at_goal(self):
        return self.s.same([self.s.goal_x, self.s.goal_y],
                           'location_x', 'location_y')

    def indicate_in_range(self):
        return (abs(self.s.goal_x - self.s.location_x) <= 10.0 and
                abs(self.s.goal_y - self.s.location_y) <= 10.0)
    
    def dynamic_behavior(self, time):
        self.s.fuel_status = 100-time
        if self.indicate_at_goal():
            a = 1
        elif self.indicate_in_range():
            self.s.location_x = self.s.goal_x
            self.s.location_y = self.s.goal_y
        else:
            direction = self.s.find_direction()
            self.s.inc(location_x=self.p.max_speed*direction[0],
                       location_y=self.p.max_speed*direction[1])


class FirePropagation(Function):
    __slots__ = ('fireenvironment')
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self, time):
        self.fireenvironment.prop_time()


class WildfireSim(FunctionArchitecture):
    __slots__=()
    default_sp={}
    """
    flows: environment, supplies
    functions: fire_propagation, aircraft, bases
    """
    def init_architecture(self, **kwargs):

        # self.add_flow("supplies")
        self.add_flow("fireenvironment", FireEnvironment)
        
        # self.add_fxn("fire_propagation", GenericFxn, "environment", "supplies")
        # self.add_fxn("aircraft", GenericFxn, "environment", "supplies")
        # self.add_fxn("bases", GenericFxn, "supplies", "environment")
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")
        self.add_fxn("aircraft", Aircraft, "fireenvironment")
        




if __name__ == "__main__":
    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    import fmdtools.sim.propagate as prop
    mdl = WildfireSim()

    mdl_graph = FunctionArchitectureGraph(mdl)
    mdl_graph.draw()

    s = AircraftStates()
    p = AircraftParam()

    a = Aircraft()

    # res, hist = prop.nominal(a)
    # hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    # hist.plot_trajectory('s.location_x', 's.location_y')

    a1 = Aircraft(s={'goal_x': 390, 'goal_y': 510})
    res, hist = prop.nominal(a1, protect=False)
    hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    hist.plot_trajectory('s.location_x', 's.location_y')

    from matplotlib import colormaps as cmaps
    fm = FireMap()
    fm.show_property('tree')
    fm = FireMap(p=dict(x_size=10, y_size=10, strike_prob = 0.1))
    fm.show_property('tree')
    fm.show_property('strike', color="yellow")
    fm.show_property('water', color="blue")
    fm.show_property('grass', color="green")

    fe = FireEnvironment()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")

    fp_mdl = FirePropagation()
    res, hist = prop.nominal(mdl, protect=False)
    hist.flows.fireenvironment.c.burned
    """
    Next to do: Set up an environment, to model certain aspects of the fire  
    Get it to have a grid where the aircraft can fly around, as a base to build
    some behaviors around that  
    """