#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location for Environmental Flows (e.g., Ground, AirSpace, etc)
"""

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function

class FireMapParam(CoordsParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 5.0 # 5 kilometers
    strike_prob: float = 0.1
    state_to_burn: tuple = (bool, False)
    state_burned: tuple = (bool, False)
    feature_strike: tuple = (bool, False)
    feature_tree: tuple = (bool, True)
    feature_water: tuple = (bool, False)
    feature_grass: tuple = (bool, False)
    feature_base: tuple = (bool, False)
    base_locations: tuple = ((0.0, 0.0),)


class FireMap(Coords):
    container_p = FireMapParam

    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        self.set_prop_dist('strike', 'binomial', 1, self.p.strike_prob)
        # self.set_range('tree', False, ymin=5000, ymax=10000, xmin=0, xmax=10000)
        # self.set_range('water', True, xmin=0, xmax=4000, ymin=5000, ymax=10000)
        # self.set_range('grass', True, xmin= 5000, xmax=10000, ymin=5000, ymax=10000)

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

    def prop_fire(self):
        for pt in self.pts:
            if self.get(*pt, 'to_burn'):
                self.set(*pt, 'burned', True)

        for pt in self.pts:
            # light the fire where lightning has struck
            if self.get(*pt, 'strike'):
                self.set(*pt, 'burned', True)
            # light the fire next to burning points
            if self.get(*pt, 'burned'):
                possible = self.get_neighbors(*pt)
                for pt in possible:
                    self.set(*pt, 'to_burn', True)


class FireEnvironment(Environment):
    __slots__ = ()
    coords_c = FireMap

    def prop_time(self):
        self.c.prop_fire()


class FirePropagation(Function):
    __slots__ = ('fireenvironment')
    flow_fireenvironment = FireEnvironment

    def dynamic_behavior(self, time):
        self.fireenvironment.prop_time()


if __name__ == "__main__":

    fm = FireMap()
    # fm.show_property('tree')
    fm = FireMap(p=dict(x_size=10, y_size=10, strike_prob=0.1, base_locations=((0.0, 40.0), (30.0, 30.0))))
    # fm.show_property('tree')
    fm.show_property('strike', color="yellow")
    # fm.show_property('water', color="blue")
    # fm.show_property('grass', color="green")
    fm.show_property('base', color="grey")

    fe = FireEnvironment()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")
    fe.prop_time()
    fe.c.show_property('burned', color="red")

    fp_mdl = FirePropagation()
