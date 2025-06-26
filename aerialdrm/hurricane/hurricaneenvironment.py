# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:55:43 2025

@author: dhulse
"""

from aerialdrm.base.aircraft.arch.flows import AircraftEnvironment
from aerialdrm.base.aircraft.state import AircraftPosition3
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.object.geom import GeomPoint, PointParam
from fmdtools.define.block.function import Function

from shapely import distance
import numpy as np

from hurricaneflightpath import DroneFlightGrid
"""import new file!"""

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


class HurricaneThreats(GeomArchitecture):

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

class HurricaneEnvironment(AircraftEnvironment):

    coords_c = HurricaneCoords
    arch_ga = HurricaneThreats
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 
        # 
        # TASKS FOR 6/26: 
        # FIGURE THIS __init__ METHOD OUT, INTEGRATE FLIGHTPATH INTO THIS CLASS
        # get_grid_cost N^2 -> N by only initializing needed grid costs
        # ask about replan_mission, how other aircraft are represented, testhurricaneflightpath.py -> modular testing style?
        # Additionally: Gaussian blur (taking in threat.p.buffer_envelope and threat.p.buffer_safety? more params in hurricaneflightpath?)
        # 
        # 
    def show(self, *args, **kwargs):
        fig, ax = self.c.show(properties=properties, collections=collections,
                              coll_overlay=False)
        self.ga.show(fig=fig, ax=ax)
        return fig, ax
    """override environment init method. Seems AircraftEnvironment flow is TBD?"""
    
class HurricaneConditions(Function):
    __slots__ = ('environment', )
    flow_environment = HurricaneEnvironment

    def dynamic_behavior(self, time):
        self.environment.ga.update_positions()


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
    ht = HurricaneThreats()
    ht.show()

    he = HurricaneEnvironment()
    he.show()
    he.ga.update_positions()
    he.show()
    he.ga.update_positions()
    he.show()

    hc = HurricaneConditions(track=['environment'])
    from fmdtools.sim import propagate
    res, hist = propagate.nominal(hc)

    hist.plot_trajectories('environment.ga.points.uav.s.x',
                           'environment.ga.points.uav.s.y')