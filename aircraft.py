# -*- coding: utf-8 -*-
"""
Module for assets (e.g. aircraft, etc.)
"""

import numpy as np
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.block.function import Function

from fireenvironment import FireEnvironment

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


if __name__ == "__main__":
    import fmdtools.sim.propagate as prop
    a = Aircraft()

    # res, hist = prop.nominal(a)
    # hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    # hist.plot_trajectory('s.location_x', 's.location_y')
    
    a1 = Aircraft(s={'goal_x': 390, 'goal_y': 510})
    res, hist = prop.nominal(a1, protect=False)
    hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    hist.plot_trajectory('s.location_x', 's.location_y')