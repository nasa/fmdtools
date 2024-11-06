#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location for Environmental Flows (e.g., Ground, AirSpace, etc)
"""

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
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
    feature_tree: tuple = (bool, True)
    feature_water: tuple = (bool, False)
    feature_grass: tuple = (bool, False)
    feature_base: tuple = (bool, False)
    base_locations: tuple = ((0.0, 0.0),)
    num_strikes: int = 1


class FireMap(Coords):
    container_p = FireMapParam

    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        # self.set_prop_dist('strike', 'binomial', 1, self.p.strike_prob)
        strike_pts = self.r.rng.choice(self.pts, self.p.num_strikes, replace=False)
        self.set_pts(strike_pts, "strike", True)
        self.set_strike_burn()
        # self.set_range('tree', False, ymin=5000, ymax=10000, xmin=0, xmax=10000)
        # self.set_range('water', True, xmin=0, xmax=4000, ymin=5000, ymax=10000)
        # self.set_range('grass', True, xmin= 5000, xmax=10000, ymin=5000, ymax=10000)

    def set_to_burn(self, tstep=1.0):
        for pt in self.find_all_prop("burning"):
            # light the fire next to burning points
            possible = self.get_neighbors(*pt)
            for pt in possible:
                if not self.get(*pt, "extinguished") and not self.get(*pt, 'burning'):
                    to_burn = self.get(*pt, "to_burn")
                    if np.isnan(to_burn):
                        self.set(*pt, "to_burn", 50.0)
                    else:
                        self.set(*pt, "to_burn", to_burn-tstep)

    def set_burning(self):
        for pt in self.find_all_prop("to_burn", value=0.0, comparator=np.less_equal):
            self.set(*pt, 'burning', True)
            self.set(*pt, 'to_burn', np.NaN)
            self.set(*pt, 'to_extinguish', 100.0)

    def set_extinguished(self, tstep=1.0):
        for pt in self.find_all_prop("to_extinguish", value=-np.inf, comparator=np.greater_equal):
            to_extinguish = self.get(*pt, 'to_extinguish')
            if to_extinguish <= 0.0:
                self.set(*pt, 'extinguished', True)
                self.set(*pt, 'burning', True)
                self.set(*pt, 'to_extinguish', np.NaN)
            else:
                self.set(*pt, 'to_extinguish', to_extinguish-tstep)

    def set_strike_burn(self):
        for pt in self.find_all_prop("strike"):
            # light the fire where lightning has struck
            if not self.get(*pt, 'burning'):
                self.set(*pt, 'burning', True)
                self.set(*pt, 'to_extinguish', 100.0)

    def prop_fire(self, tstep=1.0):
        self.set_to_burn(tstep=tstep)
        self.set_burning()
        self.set_extinguished(tstep=tstep)


class FireEnvironment(Environment):
    __slots__ = ()
    coords_c = FireMap

    def prop_time(self, tstep=1.0):
        self.c.prop_fire(tstep=tstep)

from fmdtools.define.container.time import Time

# class FirePropagationTime(Time):
#     """
#     Propagation time depends on assumptions, see:
#         * ~15 mph or 0.5 km/min == prop 5 km every 10 timesteps for directional fires
#         * ~3mph or 0.1 km/min == prop 5 km every 50 timesteps for no-wind fires
#     """
#     local_dt = 50.0  # ~15 mph or 0.5 km/min == prop 5 km every 10 timesteps

class FirePropagation(Function):
    __slots__ = ('fireenvironment')
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self, time):
        if time > 0:
            self.fireenvironment.prop_time(self.t.dt)


if __name__ == "__main__":

    fm = FireMap()
    # fm.show_property('tree')
    fm = FireMap(p=dict(x_size=10, y_size=10, num_strikes=3, base_locations=((0.0, 40.0), (30.0, 30.0))))
    # fm.show_property('tree')
    fm.show_property('strike', color="yellow")
    # fm.show_property('water', color="blue")
    # fm.show_property('grass', color="green")
    fm.show_property('base', color="grey")

    fe = FireEnvironment()
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=50.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=50.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=50.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=50.0)
    fe.c.show_property('burning', color="red")
    fe.prop_time(tstep=50.0)
    fe.c.show_property('burning', color="red")
    fe.c.show_property('extinguished', color="blue")

    fp_mdl = FirePropagation()
