#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location for Environmental Flows (e.g., Ground, AirSpace, etc)
"""

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.analyze.common import setup_plot, consolidate_legend
import numpy as np


class FireMapParam(CoordsParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 5.0  # 5 kilometers
    state_to_burn: tuple = (float, np.nan)
    state_burning: tuple = (bool, False)
    state_to_extinguish: tuple = (float, np.nan)
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

    def get_leading_edge(self, direction='direct'):
        # get all points
        burn_pts = [*self.find_all_prop("burning", True, np.equal)]
        leading_edge = []
        for i, pt2 in enumerate(burn_pts):
            neighbors = self.get_neighbors(*pt2, direction='direct')
            any_to_burn = any([not (self.get(*p3, "burning")
                                    or self.get(*p3, "extinguished"))
                               for p3 in neighbors])
            if any_to_burn:
                leading_edge.append(pt2)
        return leading_edge

    def find_closest_edge(self, *pt):
        burn_pts = self.get_leading_edge()
        if burn_pts:
            dists = np.sqrt(np.sum((np.array([*pt])-burn_pts)**2, 1))
            closest_ind = np.argmin(dists)
            return burn_pts[closest_ind]
        else:
            return []

    def set_to_burn(self, tstep=1.0):
        for pt in self.find_all_prop("burning"):
            # light the fire next to burning points
            possible = self.get_neighbors(*pt, direction="direct")
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
            self.set(*pt, 'to_burn', np.nan)
            self.set(*pt, 'to_extinguish', self.get_extinguish_time(*pt))

    def set_extinguished(self, tstep=1.0):
        for pt in self.find_all_prop("burning"):
            to_extinguish = self.get(*pt, 'to_extinguish')
            if to_extinguish <= 0.0:
                self.set(*pt, 'extinguished', True)
                self.set(*pt, 'burning', False)
                self.set(*pt, 'to_extinguish', np.nan)
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

    def calc_area_burning(self):
        return self.p.blocksize**2 * len(self.find_all_prop("burning"))

    def calc_perc_burning(self):
        return self.calc_area_burning()/(self.p.blocksize**2 * self.p.x_size*self.p.y_size)

    def get_all_burned(self):
        return np.logical_or(self.burning, self.extinguished)

    def calc_area_burned(self):
        return self.p.blocksize**2 * np.sum(self.get_all_burned())

    def calc_perc_burned(self):
        return self.calc_area_burned()/(self.p.blocksize**2 * self.p.x_size*self.p.y_size)

    def indicate_contained(self):
        """Contained when nowhere left to spread"""
        return len(self.get_leading_edge()) <= 0

    def show_base_placement(self, fig=None, ax=None, figsize=(6.0, 4.0), color="blue",
                            linewidths=3.0, **leg_kwargs):
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        xs = [p[0] for p in self.p.base_locations]
        ys = [p[1] for p in self.p.base_locations]
        ax.scatter(xs, ys, marker="*", label="bases", color=color,
                   linewidths=linewidths)
        consolidate_legend(ax, **leg_kwargs)
        return fig, ax


class FireEnvironment(Environment):
    """Map of fire propagation properties and air bases."""

    __slots__ = ()
    coords_c = FireMap

    def prop_time(self, tstep=1.0):
        self.c.prop_fire(tstep=tstep)


class FirePropagationState(State):

    perc_burned: float = 0.0
    leading_edge_length: int = 0

class FirePropagation(Function):
    """Propagates fire behavior over map."""

    __slots__ = ('fireenvironment')
    container_s = FirePropagationState
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self, time):
        self.s.perc_burned = self.fireenvironment.c.calc_perc_burned()
        self.s.leading_edge_length = len(self.fireenvironment.c.get_leading_edge())
        if time > 0:
            self.fireenvironment.prop_time(self.t.dt)




double_size_p = dict(x_size=20, y_size=20, blocksize=2.5,
                     map_type="forest-grass-scrub", num_strikes=3,
                     grass_ig_time=25.0, scrub_ig_time=37.0, forest_ig_time=50.0,
                     grass_ex_time=45.0, scrub_ex_time=100.0, firest_ex_time=200.0,
                     base_locations=((0.0, 40.0), (30.0, 30.0)))


show_properties = {'forest': {'color': 'darkgreen'},
                   'grass': {'color': 'lightgreen'},
                   'scrub': {'color': 'gold'}}

sim_properties={'grass': {'color': 'lightgreen'},
                'forest': {'color': 'darkgreen'},
                'scrub': {'color': 'gold'},
                'burning': {'color': "red", "as_bool": True, 'alpha': 0.5},
                "base": {"color": "black"},
                "to_burn": {"color": "yellow", "as_bool": True, "alpha": 0.5},
                "extinguished": {"color": "grey"}}

if __name__ == "__main__":

    fm = FireMap(p={'map_type': "forest-grass-scrub"})
    fm.show(properties=show_properties)
    # fm.show_property('tree')
    fm = FireMap(p=double_size_p)
    # fm.show_property('tree')
    fm.show_property('strike', color="yellow")
    # fm.show_property('water', color="blue")
    # fm.show_property('grass', color="green")
    fig, ax = fm.show_property('base', color="grey")
    fig, ax = fm.show_base_placement(fig=fig, ax=ax)

    fe = FireEnvironment(c={'p': double_size_p})
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

    fe.c.get_leading_edge()
    fp_mdl = FirePropagation()
