# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:55:43 2025

@author: dhulse
"""

from aerialdrm.base.aircraft.arch.flows import AircraftEnvironment
from fmdtools.define.object.coords import Coords, CoordsParam
import numpy as np


class HurricaneCoordsParam(CoordsParam):
    x_size: int = 12
    y_size: int = 12
    blocksize: float = 10.0
    feature_occupied: tuple = (bool, False)
    feature_disallowed: tuple = (bool, False)
    feature_restricted: tuple = (bool, False)
    collect_suitable: tuple = (("occupied", False, np.equal),
                               "and", ("disallowed", False, np.equal),
                               "and", ("restricted", False, np.equal))
    point_start: tuple = (10.0, 10.0)
    point_end: tuple = (100.0, 100.0)


class HurricaneCoords(Coords):
    container_p = HurricaneCoordsParam

    def init_properties(self, **kwargs):
        self.set_rand_pts('occupied', True, 50)
        self.set_range('disallowed', True, xmin=30, xmax=60, ymin=70)
        self.set_range('disallowed', True, xmin=20, xmax=60, ymax=30)
        # self.set_rand_pts('disallowed', True, 30)
        self.set_range('restricted', True, xmax=20, ymin=30)
        self.set_range('restricted', True, xmin=60, ymax=70)
        self.set_range('restricted', True, ymin=110)
        self.set_pts([self.start, self.end], 'occupied', False)
        self.set_pts([self.start, self.end], 'disallowed', False)


properties = {'disallowed': {'color': 'blue', 'proplab': 'disallowed', 'alpha': 0.5},
              'occupied': {'color': 'red', 'proplab': 'occupied', 'alpha': 0.5},
              'restricted': {'color': 'grey', 'proplab': 'restricted', 'alpha': 0.75}}

collections = {'suitable': {"label": False, 'color': 'lightgreen'}}


class HurricaneEnvironment(AircraftEnvironment):

    coords_c = HurricaneCoords


if __name__ == "__main__":
    hc = HurricaneCoords()
    # hc.show(properties=properties, collections=collections)
    # hc.show(collections={'suitable': {}})
    # hc.show_collection("suitable", **collections['suitable'])

    hc.show(properties={'restricted': {'color': 'red', 'proplab': 'restricted'}},
            collections={'start': {'color': 'lightblue'},
                         'end': {'color': 'lightgreen'}})

    from fmdtools.analyze.common import setup_plot, add_title_xylabs
    from matplotlib.colors import to_rgba, ListedColormap, TABLEAU_COLORS
    # fig, ax = setup_plot(fig=None, ax=None)
    # pallette=[*TABLEAU_COLORS.keys()]
    # hc._show_properties({}, fig, ax, pallette)
    # hc._show_collections(collections, fig, ax, pallette, c_offset=0)
    # add_title_xylabs(ax, title='li', xlabel='x', ylabel='y')

    fig, ax = hc.show(collections={'start': {'color': 'lightblue'},
                                   'end': {'color': 'lightgreen'}},
                      coll_overlay=False, border_offset=0.0)

    # he = HurricaneEnvironment()