# -*- coding: utf-8 -*-
"""
Environment for contingency management model.

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

from examples.airspacelib.base.arch.flows import AircraftEnvironment
from examples.airspacelib.base.state import AircraftPosition3

from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.object.geom import GeomPoint, PointParam
from fmdtools.define.block.function import Function

from shapely import distance
import numpy as np

class ContingencyEnvironmentParam(CoordsParam):
    """
    Parameter defining the Environment flow.

    Parameters
    ----------
    x_size: int
        Number of grid cells in the x. Default is 12.
    y_size: int
        Number of grid cells in the y. Default is 12.
    blocksize: float
        Size of grid cells. Default is 10.0, or 10 meters.
    gridcase : str
        Type of grid with options (Default is "mix"):
            "mix" - assortment of properties through allowed corridor
            "all_disallowed" - all landing areas disallowed.
    intruders : str
        Whether and how there are intruders (Default is "accross"):
            -"accross" has an intruder fly across the path
            "middle" has an intruder waiting in the middle of the path
            "down" has an intruder fly down through part of the path
            "down-over" has an intruder fly down over the path
    """

    x_size: int = 12
    y_size: int = 12
    blocksize: float = 10.0
    gridcase: str = "mix"
    intruders: str = 'across'

class ContingencyCoords(Coords):
    """
    Coordinate grid defining environment the drone flies over.

    Features
    --------
    occupied : bool
        Whether or not a ground point is occupied (e.g., by a person or vehicle).
    disallowed : bool
        Whether or not the drone is disallowed from landing on a cell.
    restricted : bool
        Whether or not a given area has flight restrictions barring entry.

    Collections
    -----------
    suitable : bool
        Whether a cell is suitable to land in.

    Points
    -----
    start : tuple
        Starting location of the drone.
    end : tuple
        Ending location of hte drone.
    """

    container_p = ContingencyEnvironmentParam
    feature_occupied = (bool, False)
    feature_disallowed = (bool, False)
    feature_restricted = (bool, False)
    collection_suitable = (("occupied", False, np.equal),
                           "and", ("disallowed", False, np.equal),
                           "and", ("restricted", False, np.equal))

    point_start = (10.0, 10.0)
    point_end = (100.0, 100.0)

    def init_properties(self, **kwargs):
        """Set properties of the grid based on the grid case."""
        if self.p.gridcase == 'mix':
            self.set_rand_pts('occupied', True, 50)
            self.set_range('disallowed', True, xmin=30, xmax=60, ymin=70)
            self.set_range('disallowed', True, xmin=20, xmax=60, ymax=30)
            # self.set_rand_pts('disallowed', True, 30)
            self.set_range('restricted', True, xmax=20, ymin=30)
            self.set_range('restricted', True, xmin=60, ymax=70)
            self.set_range('restricted', True, ymin=110)
        elif self.p.gridcase == 'all_disallowed':
            self.set_range('disallowed', True, xmin=0, xmax=110, ymin=0, ymax=110)
            self.set_range('restricted', True, xmax=20, ymin=30)
            self.set_range('restricted', True, xmin=60, ymax=70)
            self.set_range('restricted', True, ymin=110)
        else:
            raise Exception("Invalid grid case option: "+self.p.gridcase)
        self.set_pts([self.start, self.end], 'occupied', False)
        self.set_pts([self.start, self.end], 'disallowed', False)


"""Default properties to show (and their attributes) in vizualizations."""
properties = {'disallowed': {'color': 'blue', 'proplab': 'disallowed', 'alpha': 0.5},
              'occupied': {'color': 'red', 'proplab': 'occupied', 'alpha': 0.5},
              'restricted': {'color': 'grey', 'proplab': 'restricted', 'alpha': 0.75}}
collections = {'suitable': {"label": "suitable", 'color': 'lightgreen'}}


class ThreatState(AircraftPosition3):
    """
    State of an external threat (e.g., intruder).

    Extends AircraftPosition3 with the following:

    Parameters
    ----------
    buffer_speed: float
        The speed of the threat.
    """

    buffer_speed: float = 10.0

    def update_speed(self):
        """Update speed buffer based on its velocity."""
        self.buffer_speed = self.get_vel()


class ThreatParam(PointParam):
    """
    Parameter defining the exteral threat shape.

    Parameters
    ----------
    buffer_envelope : float
        The physical envelope of the threat. Default is 1.0 meters.
    buffer_safety : float
        The distance from the threat needed for safety. Default is 25.0 meters.
    """

    buffer_envelope: float = 1.0
    buffer_safety: float = 25.0


class Threat(GeomPoint):
    """Point geometry (and buffers) defining drone and intruders."""

    container_p = ThreatParam
    container_s = ThreatState

    def update_position(self):
        """Update position given known speed."""
        self.s.update_position(self.s.buffer_speed)


class ContingencyThreats(GeomArchitecture):
    """Overall environment-defining geometries of the drone and external threats."""

    container_p = ContingencyEnvironmentParam

    def init_architecture(self, **kwargs):
        """Initialize drone and intruders given 'intruders' options."""
        self.add_point('self', Threat)
        if self.p.intruders == "across":
            s = {'buffer_speed': 3.5, 'x': 100, 'y': 0.0, 'z': 25.0,
                 'goal_x': 0.0, 'goal_y': 100.0, 'goal_z': 25.0}
            self.add_point("uav", Threat, s=s)
        elif self.p.intruders == "middle":
            s = {'buffer_speed': 0.0, 'x': 60, 'y': 60, 'z': 25.0,
                 'goal_x': 60.0, 'goal_y': 60.0, 'goal_z': 25.0}
            self.add_point("uav", Threat, s=s)
        elif self.p.intruders == "down":
            s = {'buffer_speed': 2.5,
                 'x': 60, 'y': 120, 'z': 25.0,
                 'goal_x': 60.0, 'goal_y': 0.0, 'goal_z': 25.0}
            self.add_point("uav", Threat, s=s)
        elif self.p.intruders == "down-over":
            s = {'buffer_speed': 2.5,
                 'x': 40, 'y': 120, 'z': 25.0,
                 'goal_x': 40.0, 'goal_y': 0.0, 'goal_z': 25.0}
            self.add_point("uav", Threat, s=s)
        elif self.p.intruders:
            raise Exception("Invalid option for intruders: "+self.p.intruders)
            

    def update_positions(self):
        """Update positions of the threats."""
        for threatname, threat in self.points.items():
            if threatname != 'self':
                threat.update_position()

    def calc_dist_to_threats(self, self_shape='envelope', threat_shape='safety'):
        """Calculate distancses b/t self_shape for self and threat_shape for threats."""
        dists = {}
        self_envelope = self.points['self'].get_shape(self_shape)
        for threatname, threat in self.points.items():
            if threatname != 'self':
                threat_envelope = threat.get_shape(threat_shape)
                dists[threatname] = distance(self_envelope, threat_envelope)
        return dists


class ContingencyEnvironment(AircraftEnvironment):
    """Overall environment of drone with threats and grid."""

    container_p = ContingencyEnvironmentParam
    coords_c = ContingencyCoords
    arch_ga = ContingencyThreats 

    def show(self, *args, **kwargs):
        """Show combined view of threats and coords."""
        fig, ax = self.c.show(properties=properties, collections=collections,
                              coll_overlay=False)
        self.ga.show(fig=fig, ax=ax)
        return fig, ax


class ContingencyConditions(Function):
    """Function to update positions of external intruders."""

    flow_environment = ContingencyEnvironment

    def dynamic_behavior(self):
        """Update intruder positions."""
        self.environment.ga.update_positions()


if __name__ == "__main__":
    hc = ContingencyCoords(p={'gridcase': 'all_disallowed'})
    # hc.show(properties=properties, collections=collections)
    # hc.show(collections={'suitable': {}})
    # hc.show_collection("suitable", **collections['suitable'])
    props = {'restricted': {'color': 'red', 'proplab': 'restricted'}}
    colls = {'start': {'color': 'lightblue'}, 'end': {'color': 'lightgreen'}}

    hc.show(properties=props, collections=colls)


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