#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Flows (e.g., Ground, AirSpace, etc) used in wildfire response model.

Includes classes for defining the fire map (fuels, bases, etc) and its behavior.

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

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.analyze.common import setup_plot, consolidate_legend

import numpy as np


class FireMapParam(CoordsParam):
    """
    Parameter defining the fire map.

    Parameters
    ----------
    x_size: int
        Number of grid cells in the x. Default is 10.
    y_size: int
        Number of grid cells in the y. Default is 10.
    blocksize: float
        Size of grid cels. Default is 5.0, or 5 kilometers.
    base_locations: tuple
        Locations in the grid to put a base. Default is  ((0.0, 0.0),).
    num_strikes: int
        Number of strikes to initiate in the grid. Default is 1.
    map_type: str
        Type of map, specified as "uniform-xx","split-xx-yy", or "xx-yy-zz", where
        xx, yy, and zz are "grass", "forest", or "scrub" fuels
    grass_ig_time : float
        Grass cell ignition time. Default is 50.0 minutes
    grass_ex_time : float
        Grass cell extinguish time. Default is 90.0 minutes.
    scrub_ig_time : float
        Scrub cell ignition time. Default is 75.0 minutes
    scrub_ex_time : float
        Scrub cell extinguish time. Default is 200.0 minutes.
    forest_ig_time : float
        Forest cell ignition time. Default is 100.0 minutes
    grass_ex_time : float
        Forest cell extinguish time. Default is 400.0 minutes.
    """

    x_size: int = 10
    y_size: int = 10
    blocksize: float = 5.0  # 5 kilometers
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
    """
    Grid of points defining fire properties.

    The following define the grid arrays:

    States
    ------
    to_burn: float
        Timesteps to ignition for the given cell.
    burning: bool
        Whether or not a cell is burning.
    to_extinguish: float
        Timesteps to extinguishing a given cell.
    extinguished: bool
        Whether or not a cell has been extinguished.

    Features
    --------
    strike : bool
        Whether or not the cell is a lighting strike location.
    grass : bool
        Whether or not a cell is in the grassland.
    scrub : bool
        Whether or not a cell is in the scrubland.
    forest : bool
        Whether or not the cell is in the forest.
    base : bool
        Whether or not the cell is an air base.
    """

    container_p = FireMapParam
    state_to_burn: tuple = (float, np.nan)
    state_burning: tuple = (bool, False)
    state_to_extinguish: tuple = (float, np.nan)
    state_extinguished: tuple = (bool, False)
    feature_strike: tuple = (bool, False)
    feature_grass: tuple = (bool, False)
    feature_scrub: tuple = (bool, False)
    feature_forest: tuple = (bool, False)
    feature_base: tuple = (bool, False)

    def init_properties(self, *args, **kwargs):
        """
        Initialize properties of the grid.

        Sets fuel distributions based on map_type and sets strike locations on fire.
        """
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
        """Get ignition time for a given point x,y."""
        if self.get(*pt, "grass"):
            return self.p.grass_ig_time
        elif self.get(*pt, "forest"):
            return self.p.forest_ig_time
        elif self.get(*pt, "scrub"):
            return self.p.scrub_ig_time
        else:
            return np.inf

    def get_extinguish_time(self, *pt):
        """Get extinguish time for a given point x,y."""
        if self.get(*pt, "grass"):
            return self.p.grass_ex_time
        elif self.get(*pt, "forest"):
            return self.p.forest_ex_time
        elif self.get(*pt, "scrub"):
            return self.p.scrub_ex_time
        else:
            return np.inf

    def get_leading_edge(self, direction='direct'):
        """Get the leading edge where the fire is spreading from."""
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
        """Find the closest leading edge of the fire to the point x,y."""
        burn_pts = self.get_leading_edge()
        if burn_pts:
            dists = np.sqrt(np.sum((np.array([*pt])-burn_pts)**2, 1))
            closest_ind = np.argmin(dists)
            return burn_pts[closest_ind]
        else:
            return []

    def set_to_burn(self, tstep=1.0):
        """Set the to_burn property for the fire over the timestep tstep."""
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
        """Set the burning property of the fire for cells where to_burn<=0.0."""
        for pt in self.find_all_prop("to_burn", value=0.0, comparator=np.less_equal):
            self.set(*pt, 'burning', True)
            self.set(*pt, 'to_burn', np.nan)
            self.set(*pt, 'to_extinguish', self.get_extinguish_time(*pt))

    def set_extinguished(self, tstep=1.0):
        """Set the extinguished property for cells where to_extinguish<=0.0."""
        for pt in self.find_all_prop("burning"):
            to_extinguish = self.get(*pt, 'to_extinguish')
            if to_extinguish <= 0.0:
                self.set(*pt, 'extinguished', True)
                self.set(*pt, 'burning', False)
                self.set(*pt, 'to_extinguish', np.nan)
            else:
                self.set(*pt, 'to_extinguish', to_extinguish-tstep)

    def set_strike_burn(self):
        """Set a strike location as burning."""
        for pt in self.find_all_prop("strike"):
            # light the fire where lightning has struck
            if not self.get(*pt, 'burning'):
                self.set(*pt, 'burning', True)
                self.set(*pt, 'to_extinguish', self.get_extinguish_time(*pt))

    def prop_fire(self, tstep=1.0):
        """Propagate fire behavior through the grid over tstep."""
        self.set_to_burn(tstep=tstep)
        self.set_burning()
        self.set_extinguished(tstep=tstep)

    def calc_area_burning(self):
        """Calculate the total area burning (in sq km)."""
        return self.p.blocksize**2 * len(self.find_all_prop("burning"))

    def calc_perc_burning(self):
        """Calculate the percentage of the area burning."""
        return self.calc_area_burning()/(self.p.blocksize**2 * self.p.x_size*self.p.y_size)

    def get_all_burned(self):
        """Get all points currently burning."""
        return np.logical_or(self.burning, self.extinguished)

    def calc_area_burned(self):
        """Calculate the total area burned (in sq km)."""
        return self.p.blocksize**2 * np.sum(self.get_all_burned())

    def calc_perc_burned(self):
        """Calculate the percentage of the area burned."""
        return self.calc_area_burned()/(self.p.blocksize**2 * self.p.x_size*self.p.y_size)

    def indicate_contained(self):
        """Determine when the fire is contained - nowhere left to spread."""
        return len(self.get_leading_edge()) <= 0

    def show_base_placement(self, fig=None, ax=None, figsize=(6.0, 4.0), color="blue",
                            linewidths=3.0, **leg_kwargs):
        """Show the placement of bases on a plot."""
        fig, ax = setup_plot(fig=fig, ax=ax, figsize=figsize)
        xs = [p[0] for p in self.p.base_locations]
        ys = [p[1] for p in self.p.base_locations]
        ax.scatter(xs, ys, marker="*", label="bases", color=color,
                   linewidths=linewidths)
        consolidate_legend(ax, **leg_kwargs)
        return fig, ax


class FireEnvironment(Environment):
    """Environment flow containing the FireMap for use in a larger model."""

    coords_c = FireMap

    def prop_time(self, tstep=1.0):
        """Propagate a timestep of fire behavior."""
        self.c.prop_fire(tstep=tstep)


class FirePropagationState(State):
    """
    Fire propagation State.

    Fields
    ------
    perc_burned : float
        Percentage of the fire burned.
    leading_edge_length : int
        Length of the leading edge of the fire.
    """

    perc_burned: float = 0.0
    leading_edge_length: int = 0

class FirePropagation(Function):
    """Function propagating fire behavior over map."""

    container_s = FirePropagationState
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self):
        """Propagate fire behavior and update propagation states."""
        self.s.perc_burned = self.fireenvironment.c.calc_perc_burned()
        self.s.leading_edge_length = len(self.fireenvironment.c.get_leading_edge())
        if self.t.time > 0:
            self.fireenvironment.prop_time(self.t.dt)


"""Scaled parameters for more detailed grid."""
double_size_p = dict(x_size=20, y_size=20, blocksize=2.5,
                     map_type="forest-grass-scrub", num_strikes=3,
                     grass_ig_time=25.0, scrub_ig_time=37.0, forest_ig_time=50.0,
                     grass_ex_time=45.0, scrub_ex_time=100.0, firest_ex_time=200.0,
                     base_locations=((0.0, 40.0), (30.0, 30.0)))


"""Default simulation viz arguments to show()."""
sim_properties={'grass': {'color': 'lightgreen'},
                'forest': {'color': 'darkgreen'},
                'scrub': {'color': 'gold'},
                'burning': {'color': "red", "as_bool": True, 'alpha': 0.5},
                "base": {"color": "black"},
                "to_burn": {"color": "yellow", "as_bool": True, "alpha": 0.5},
                "extinguished": {"color": "grey"}}

if __name__ == "__main__":

    fm = FireMap(p={'map_type': "forest-grass-scrub"})
# %%
    fm.show(properties=sim_properties)

    # fm.show_property('tree')
    fm = FireMap(p=double_size_p)
    # fm.show_property('tree')
    fm.show_property('strike', color="yellow")
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
