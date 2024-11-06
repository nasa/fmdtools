# -*- coding: utf-8 -*-
"""
Module for assets (e.g. aircraft, etc.)
"""

import numpy as np
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode

from fireenvironment import FireEnvironment

class AircraftStates(State):
    retardant_status: float = 100 # starting retardant at 100%
    fuel_status: float = 100 # starting fuel at 100%
    goal_x: float = 10.0
    goal_y: float = 10.0
    location_x: float = 0.0
    location_y: float = 0.0
    direction: np.array = np.array([0, 0])

    def find_direction(self):
        return self.calc_vector_dist()/self.calc_dist()

    def calc_vector_dist(self):
        return np.array([self.goal_x-self.location_x, self.goal_y-self.location_y])

    def calc_dist(self):
        vector_dist = self.calc_vector_dist()
        return np.sqrt(vector_dist[0]**2+vector_dist[1]**2)

    def at_goal(self):
        return self.same(self.gett("goal_x", "goal_y"), "location_x", "location_y")

    def in_range(self):
        return (abs(self.goal_x - self.location_x) <= 10.0 and
                abs(self.goal_y - self.location_y) <= 10.0)

class  AircraftModes(Mode):
    opermodes = ("resupply", "fly_to_fire", "mitigate_fire", "fly_to_base")
    mode: str = "resupply"

class AircraftParam(Parameter, readonly=True):
    number: int = 1
    max_range: float = 1287.0  # in km
    max_speed: float = 6.6  # in km/min
    base: int = 0
    resupply_time: float = 10.0  # 10 minute resupply time


from fmdtools.define.container.time import Time

class AircraftTime(Time):
    timernames = ('resupply', )


class Aircraft(Function):
    __slots__=('fireenvironment')
    container_s = AircraftStates
    container_p = AircraftParam
    container_m = AircraftModes
    container_t = AircraftTime
    flow_fireenvironment = FireEnvironment

    def init_block(self, **kwargs):
        self.s.location_x = self.fireenvironment.c.p.base_locations[self.p.base][0]
        self.s.location_y = self.fireenvironment.c.p.base_locations[self.p.base][1]

    def indicate_at_goal(self):
        return self.s.at_goal()

    def indicate_in_range(self):
        return self.s.in_range()

    def set_fire_goal(self):
        if [*self.fireenvironment.c.find_all_prop("burning")]:
            self.m.set_mode("fly_to_fire")
            closest = self.fireenvironment.c.find_closest(*self.s.get("location_x", "location_y"), "burning")
            self.s.assign(closest, "goal_x", "goal_y")

    def fly_to_goal(self):
        if not self.indicate_at_goal():
            dist = self.s.calc_dist()
            if self.s.in_range():
                # if in range, clip to goal location
                self.s.location_x = self.s.goal_x
                self.s.location_y = self.s.goal_y
                self.s.inc(fuel_status=-dist/self.p.max_range)
            else:
                # otherwise, move in a straight line
                direction = self.s.find_direction()
                self.s.inc(location_x=self.p.max_speed*direction[0],
                           location_y=self.p.max_speed*direction[1],
                           fuel_status=-self.p.max_speed/self.p.max_range)

    def dynamic_behavior(self, time):
        print(self)
        if self.m.in_mode("resupply"):
            if self.t.timers['resupply'].indicate_complete() or self.t.timers['resupply'].indicate_standby():
                self.s.retardant_status = 100
                self.s.fuel_status = 100
                self.t.timers['resupply'].set_timer(self.p.resupply_time)
                self.m.set_mode('fly_to_fire')
                self.set_fire_goal()
            else:
                self.t.timers['resupply'].inc()
        elif self.m.in_mode("fly_to_fire"):
            self.set_fire_goal()
            self.fly_to_goal()
            if self.indicate_at_goal():
                self.m.set_mode("mitigate_fire")
        elif self.m.in_mode("mitigate_fire"):
            self.s.retardant_status = 0
            self.fireenvironment.c.set(*self.s.gett("location_x", "location_y"),
                                       "extinguished", True)
            self.fireenvironment.c.set(*self.s.gett("location_x", "location_y"),
                                       "burning", False)
            self.m.set_mode("fly_to_base")
            self.s.goal_x = self.fireenvironment.c.p.base_locations[self.p.base][0]
            self.s.goal_y = self.fireenvironment.c.p.base_locations[self.p.base][1]
        elif self.m.in_mode("fly_to_base"):
            self.fly_to_goal()
            if self.indicate_at_goal():
                self.m.set_mode("resupply")


if __name__ == "__main__":
    import fmdtools.sim.propagate as prop
    a = Aircraft()
    fe = FireEnvironment(c={"p": {"base_locations": ((40.0, 20.0),), "num_strikes": 3}})
    fe.prop_time()
    # res, hist = prop.nominal(a)
    # hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    # hist.plot_trajectory('s.location_x', 's.location_y')

    a1 = Aircraft(s={'goal_x': 30, 'goal_y': 40}, fireenvironment=fe, track="all")

    res, hist = prop.nominal(a1, protect=False)
    hist.plot_line('s.fuel_status', 's.location_x', 's.location_y', 'm.mode')

    fig, ax = a1.fireenvironment.c.show_from(30, hist.fireenvironment.c,
                                             properties={'burning': {"color": "red", "as_bool": True}, "base": {"color": "grey"}, "extinguished": {"color": "blue", "alpha": 0.5}})
    hist.plot_trajectory('s.location_x', 's.location_y', fig=fig, ax=ax)