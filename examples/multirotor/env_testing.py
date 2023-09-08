# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:03:39 2023

@author: dhulse
"""


from fmdtools.define.environment import Grid, GridParam

# Grid - collection of points




# need to enable both params and gridparam


class SpecialGridParam(GridParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 100.0
    num_allowed: int = 10
    num_unsafe: int = 10
    num_occupied: int = 10
    max_height: float = 100.0


class SpecialGrid(Grid):
    _init_p = SpecialGridParam

    _feature_safe = (bool, True)
    _feature_allowed = (bool, False)
    _feature_occupied = (bool, False)
    _feature_height = (float, 0.0)

    _state_wind = (float, 1.0)

    _point_start = (0, 0)
    _point_end = (900, 900)
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
        self.set_prop_dist("height", "uniform", low=0.0, high=self.p.max_height)

# Map - collection of lines



b = Grid()
c = SpecialGrid()
c.find_all("safe")
c.show("safe")
c.show("height")
c.show3d("allowed", z="height", legend_args=dict(bbox_to_anchor=(0.9, 1)),
         collections={"all_safe": {"color": "blue", "label": False},
                      "all_occupied": {"color": "red", "label": False},
                      "start": {"color": "yellow", "label": True, "text_z_offset": 30},
                      "end": {"color": "yellow", "label": True, "text_z_offset": 30}})
c.show_collection("end")

h = c.create_hist([0,1,2,3], "all")