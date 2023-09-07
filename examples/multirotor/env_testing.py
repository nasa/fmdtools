# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:03:39 2023

@author: dhulse
"""


from fmdtools.define.environment import Grid, GridParam

# Grid - collection of points




# need to enable both params and gridparam


class SpecialGridParam(GridParam):
    x_size: int = 20
    y_size: int = 20
    blocksize: float = 100.0
    num_allowed: int = 10
    num_unsafe: int = 10
    num_occupied: int = 10


class SpecialGrid(Grid):
    _init_p = SpecialGridParam

    _feature_safe = (bool, True)
    _feature_allowed = (bool, False)
    _feature_occupied = (bool, False)
    _feature_height = (float, 0.0)

    _state_wind = (float, 1.0)

    _point_start = (0, 0)
    _point_end = (90, 90)
    _collect_all_occupied = ("occupied", True)
    _collect_all_safe = ("safe", True)

    def init_properties(self, *args, **kwargs):
        rand_pts = self.r.rng.choice(self.pts[1:-1],
                                     self.p.num_allowed + self.p.num_unsafe,
                                     replace=False)
        for i, pt in enumerate(rand_pts):
            if i < self.p.num_allowed:
                self.set(*pt, 'allowed', True)
            else:
                self.set(*pt, 'safe', False)
        self.set_rand_pts('occupied', True, 10, pts=self.pts[1:-1])


# Map - collection of lines



b = Grid()
c = SpecialGrid()
c.find_all("safe")