# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:55:43 2025

@author: dhulse
"""

from dronelib.base.arch.flows import AircraftEnvironment
from dronelib.base.state import AircraftPosition3
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.object.geom import GeomPoint, PointParam
from fmdtools.define.block.function import Function

from shapely import distance
import numpy as np

class ContingencyCoordsParam(CoordsParam):
    x_size: int = 12
    y_size: int = 12
    blocksize: float = 10.0

class ContingencyCoords(Coords):
    container_p = ContingencyCoordsParam
    feature_occupied = (bool, False)
    feature_disallowed = (bool, False)
    feature_restricted = (bool, False)
    collection_suitable = (("occupied", False, np.equal),
                           "and", ("disallowed", False, np.equal),
                           "and", ("restricted", False, np.equal))

    point_start = (10.0, 10.0)
    point_end = (100.0, 100.0)

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


class ThreatState(AircraftPosition3):

    buffer_speed: float = 10.0

    def update_speed(self):
        self.buffer_speed = self.get_vel()


class ThreatParam(PointParam):

    buffer_envelope: float = 1.0
    buffer_safety: float = 25.0


class Threat(GeomPoint):
    container_p = ThreatParam
    container_s = ThreatState

    def update_position(self):
        self.s.update_position(self.s.buffer_speed)


class ContingencyThreats(GeomArchitecture):

    def init_architecture(self, **kwargs):
        self.add_point('self', Threat)
        s = {'buffer_speed': 5.0, 'x': 100, 'y': 0.0, 'z': 25.0,
             'goal_x': 0.0, 'goal_y': 100.0, 'goal_z': 25.0}
        self.add_point("uav", Threat, s=s)

    def update_positions(self):
        for threatname, threat in self.points.items():
            if threatname != 'self':
                threat.update_position()

    def calc_dist_to_threats(self, self_shape='envelope', threat_shape='safety'):
        dists = {}
        self_envelope = self.points['self'].get_shape(self_shape)
        for threatname, threat in self.points.items():
            if threatname != 'self':
                threat_envelope = threat.get_shape(threat_shape)
                dists[threatname] = distance(self_envelope, threat_envelope)
        return dists

class ContingencyEnvironment(AircraftEnvironment):

    coords_c = ContingencyCoords
    arch_ga = ContingencyThreats 
    def show(self, *args, **kwargs):
        fig, ax = self.c.show(properties=properties, collections=collections,
                              coll_overlay=False)
        self.ga.show(fig=fig, ax=ax)
        return fig, ax
    """override environment init method. Seems AircraftEnvironment flow is TBD?"""
    
class ContingencyConditions(Function):
    __slots__ = ('environment', )
    flow_environment = ContingencyEnvironment

    def dynamic_behavior(self):
        self.environment.ga.update_positions()


if __name__ == "__main__":
    hc = ContingencyCoords()
    # hc.show(properties=properties, collections=collections)
    # hc.show(collections={'suitable': {}})
    # hc.show_collection("suitable", **collections['suitable'])
    props = {'restricted': {'color': 'red', 'proplab': 'restricted'}}
    colls = {'start': {'color': 'lightblue'}, 'end': {'color': 'lightgreen'}}

    hc.show(properties=props, collections=colls)

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

    # he = ContingencyEnvironment()
    ht = ContingencyThreats()
    ht.show()

    he = ContingencyEnvironment()
    he.show()
    he.ga.update_positions()
    he.show()
    he.ga.update_positions()
    he.show()

    hc = ContingencyConditions(track=['environment'])
    from fmdtools.sim import propagate
    res, hist = propagate.nominal(hc)

    hist.plot_trajectories('environment.ga.points.uav.s.x',
                           'environment.ga.points.uav.s.y')