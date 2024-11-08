#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location for Environmental Flows (e.g., Ground, AirSpace, etc)
"""

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
import numpy as np


class FireMapParam(CoordsParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 5.0  # 5 kilometers
    state_to_burn: tuple = (float, np.NaN)
    state_burning: tuple = (bool, False)
    state_to_extinguish: tuple = (float, np.NaN)
    state_extinguished: tuple = (bool, False)
    feature_strike: tuple = (bool, False)
    feature_grass: tuple = (bool, False)
    feature_scrub: tuple = (bool, False)
    feature_forest: tuple = (bool, False)
    feature_water: tuple = (bool, False)
    feature_base: tuple = (bool, False)
    base_locations: tuple = ((0.0, 0.0),)
    num_strikes: int = 1
    map_type: str = "uniform-grass"
    grass_ig_time: float = 50.0  # 5 km every 50 timesteps (~3mph)
    grass_ex_time: float = 90.0
    scrub_ig_time: float = 75.0
    scrub_ex_time: float = 200.0
    forest_ig_time: float = 100.0
    forest_ex_time: float = 400.0


class FireMap(Coords):
    container_p = FireMapParam

    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        # self.set_prop_dist('strike', 'binomial', 1, self.p.strike_prob)
        strike_pts = self.r.rng.choice(self.pts, self.p.num_strikes, replace=False)
        self.set_pts(strike_pts, "strike", True)

        mapchars = self.p.map_type.split("-")
        half_x0 = (self.p.x_size/2-1)*self.p.blocksize
        half_x1 = (self.p.x_size/2)*self.p.blocksize
        half_y0 = (self.p.y_size/2-1)*self.p.blocksize
        half_y1 = (self.p.y_size/2)*self.p.blocksize
        if mapchars[0] == "uniform":
            self.set_range(mapchars[1], True)
        elif mapchars[0] == "split":
            self.set_range(mapchars[1], True, xmax=half_x0)
            self.set_range(mapchars[2], True, xmin=half_x1)
        elif len(mapchars) == 3:
            self.set_range(mapchars[0], True, xmax=half_x0, ymax=half_y0)
            self.set_range(mapchars[1], True, xmax=half_x0, ymin=half_y1)
            self.set_range(mapchars[2], True, xmin=half_x1)
        self.set_strike_burn()

    def get_ignition_time(self, *pt):
        if self.get(*pt, "grass"):
            return self.p.grass_ig_time
        elif self.get(*pt, "forest"):
            return self.p.forest_ig_time
        elif self.get(*pt, "scrub"):
            return self.p.scrub_ig_time
        else:
            return np.inf

    def get_extinguish_time(self, *pt):
        if self.get(*pt, "grass"):
            return self.p.grass_ex_time
        elif self.get(*pt, "forest"):
            return self.p.forest_ex_time
        elif self.get(*pt, "scrub"):
            return self.p.scrub_ex_time
        else:
            return np.inf

    def set_to_burn(self, tstep=1.0):
        for pt in self.find_all_prop("burning"):
            # light the fire next to burning points
            possible = self.get_neighbors(*pt)
            for ppt in possible:
                if not self.get(*ppt, "extinguished") and not self.get(*ppt, 'burning'):
                    to_burn = self.get(*ppt, "to_burn")
                    if np.isnan(to_burn):
                        self.set(*ppt, "to_burn", self.get_ignition_time(*ppt))
                    else:
                        self.set(*ppt, "to_burn", to_burn-tstep)

    def set_burning(self):
        for pt in self.find_all_prop("to_burn", value=0.0, comparator=np.less_equal):
            self.set(*pt, 'burning', True)
            self.set(*pt, 'to_burn', np.NaN)
            self.set(*pt, 'to_extinguish', self.get_extinguish_time(*pt))

    def set_extinguished(self, tstep=1.0):
        for pt in self.find_all_prop("burning"):
            to_extinguish = self.get(*pt, 'to_extinguish')
            if to_extinguish <= 0.0:
                self.set(*pt, 'extinguished', True)
                self.set(*pt, 'burning', False)
                self.set(*pt, 'to_extinguish', np.NaN)
            else:
                self.set(*pt, 'to_extinguish', to_extinguish-tstep)

    def set_strike_burn(self):
        for pt in self.find_all_prop("strike"):
            # light the fire where lightning has struck
            if not self.get(*pt, 'burning'):
                self.set(*pt, 'burning', True)
                self.set(*pt, 'to_extinguish', self.get_extinguish_time(*pt))

    def prop_fire(self, tstep=1.0):
        self.set_to_burn(tstep=tstep)
        self.set_burning()
        self.set_extinguished(tstep=tstep)

    def calc_area_burned(self):
        return self.p.blocksize**2 * (len(self.find_all_prop("burning"))+len(self.find_all_prop("extinguished")))

    def calc_perc_burned(self):
        return self.calc_area_burned()/(self.p.blocksize**2 * self.p.x_size*self.p.y_size)


class FireEnvironment(Environment):
    __slots__ = ()
    coords_c = FireMap

    def prop_time(self, tstep=1.0):
        self.c.prop_fire(tstep=tstep)


class FirePropagationState(State):

    perc_burned: float = 0.0

class FirePropagation(Function):
    __slots__ = ('fireenvironment')
    container_s = FirePropagationState
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self, time):
        self.s.perc_burned = self.fireenvironment.c.calc_perc_burned()
        if time > 0:
            self.fireenvironment.prop_time(self.t.dt)


if __name__ == "__main__":

    fm = FireMap(p={'map_type': "forest-grass-scrub"})
    fm.show({'forest': {'color': 'darkgreen'},
             'grass': {'color': 'lightgreen'},
             'scrub': {'color': 'gold'}})
    # fm.show_property('tree')
    fm = FireMap(p=dict(x_size=10, y_size=10, num_strikes=3,
                        base_locations=((0.0, 40.0), (30.0, 30.0)),
                        map_type="forest-grass-scrub"))
    # fm.show_property('tree')
    fm.show_property('strike', color="yellow")
    # fm.show_property('water', color="blue")
    # fm.show_property('grass', color="green")
    fm.show_property('base', color="grey")

    fe = FireEnvironment(c={'p': {'map_type': "forest-grass-scrub"}})
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.c.show_property('extinguished', color="blue")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=20.0)
    fe.c.show_property('burning', color="red")

    fp_mdl = FirePropagation()
