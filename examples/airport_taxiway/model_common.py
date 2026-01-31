#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common methods and Flows for the Taxiway model

Contains the following model aspects:
    - The Environment and related parameters
    - Asset Allocation, parameter generation
    - Default flow dictionaries
    - Plotting

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.analyze.common import consolidate_legend

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from shapely.geometry import LineString
from shapely.geometry import Point


"""
:param:`default_aircoords`:   Defines the segments that define the map and
their coordinates.`

Segments must follow these rules:

1) The segments names should always start with the following convention
and should have a number attached if there are more than 1 similar segments:
     gates --> "gateX" --> e.g., gate1
     helipad --> "heli...X" --> e.g. helipad1
     take off area --> "take...X" --> e.g.,takeOff
     landing area --> "land..X" --> e.g.,landing1
     Ground Vehicles --> "Ground...X" --> e.g., GroundV1
     Taxiways -->"seg_<taxiwayname>X" -- e.g., seg_A1

2) Intersections should always be at an end or begining point of a segment
    (e.g., seg_R1 in the default map is not allowed)

3) A single segment may contain multiple turns as long as there are no intersections
in the middle of it (e.g., seg_A1 in the default map)

4) There must be no more than one turns/junctions within the  maximum travel distance
    in a timestep (max speed * timestep duration) segment, boundaries exluded
"""
default_aircoords = {
    "seg_A1": [(20, 0), (0, 5), (0, 10), (10, 15)],
    "seg_A2": [(10, 15), (45, 15)],
    "takeoff1": [(10, 15), (10, 20)],
    "seg_A3": [(45, 15), (55, 15)],
    "landing1": [(55, 15), (55, 20)],
    "seg_A4": [(55, 15), (65, 10)],
    "helipad1": [(65, 10), (65, 15)],
    "seg_A5": [(65, 10), (65, 5), (45, 0)],
    "seg_R1": [(20, 0), (25, 0)],
    "seg_R2": [(25, 0), (30, 0)],
    "seg_R3": [(30, 0), (32.5, 0)],
    "seg_R4": [(32.5, 0), (35, 0)],
    "seg_R5": [(35, 0), (40, 0)],
    "seg_R6": [(40, 0), (45, 0)],
    "seg_B1": [(32.5, 0), (32.5, 7.5), (10, 7.5), (10, 15)],
    "seg_C1": [(45, 15), (45, 10), (65, 10)],
    "gate1": [(20, 0), (20, -5)],
    "gate2": [(25, 0), (25, -5)],
    "gate3": [(30, 0), (30, -5)],
    "gate4": [(35, 0), (35, -5)],
    "gate5": [(40, 0), (40, -5)],
    "gate6": [(45, 0), (45, -5)],
}
"""
:param:`default_speeds`:   Defines the Speed to be followed at each segment.

NOTE: if speeds are too fast (>5), the AC will jump outside its vision cone and crash
"""
default_speeds = {
    "seg_A1": 5.0,
    "seg_A2": 5.0,
    "takeoff1": 3.0,
    "seg_A3": 5.0,
    "landing1": 5.0,
    "seg_A4": 5.0,
    "helipad1": 5.0,
    "seg_A5": 5.0,
    "seg_R1": 5.0,
    "seg_R2": 5.0,
    "seg_R3": 2.0,
    "seg_R4": 2.0,
    "seg_R5": 2.0,
    "seg_R6": 5.0,
    "seg_B1": 5.0,
    "seg_C1": 5.0,
    "gate1": 3.0,
    "gate2": 3.0,
    "gate3": 3.0,
    "gate4": 3.0,
    "gate5": 3.0,
    "gate6": 3.0,
}
"""
:param:`default_routes`: dict
    Defines the routes which can be followed between segments

Rules for creating taxi/ground vehicle routes:
1) Taxi routes must be specified for each gate for both landing and take off.
2) Ground vehicle routes are optional for gates. If ground vehicle gate taxi conflicts
    are present. in the airport map and that needs to be modeled then this must be
    specificed to capture how ground vehicle to to and from taxiways to and from ground
    vehicle stations.
3) if Ground vehicles service helipads the routes to the helipad1 and back to the ground
    vehicle stations must be specied.
4) One may speicify more than 1 route for an asset to reach and return from specific
    areas. For example, an airport may have 2 taxi routes to a runway takeoff points.
5) The taxi routes must include routes to/from all runways from/t0 all gates. These
    routes may be shutoff during the simulation as needed.
6) The following naming convention should be followed when naming routes.
- For taxiways in should start with the gate name, followed by T for 'takeoff;
or L for'landing' followed by runway number. The name should with the a route number.
- For example gate1_t1_2 would mean that this route is the second taxi route for
take off from gate 1 to runway 1.
- For Ground vehicle routes it states with the ground station name followed by end
location name followed by In or Out to indicate if the route is to the end location or
from the end location.
- For example GroundStation1_helipad1_In indicates that the route is from
Groundstation1 to helipad1.
7) The routes must be specified through a list of segment names where the segments are
    ordered from the start location to the end location in the order they are travelled.
"""
default_routes = {
    "gate1_t1_1": ["gate1", "seg_A1", "takeoff1"],
    "gate2_t1_1": ["gate2", "seg_R1", "seg_A1", "takeoff1"],
    "gate3_t1_1": ["gate3", "seg_R2", "seg_R1", "seg_A1", "takeoff1"],
    "gate4_t1_1": ["gate4", "seg_R4", "seg_B1", "takeoff1"],
    "gate5_t1_1": ["gate5", "seg_R5", "seg_R4", "seg_B1", "takeoff1"],
    "gate6_t1_1": ["gate6", "seg_R6", "seg_R5", "seg_R4", "seg_B1", "takeoff1"],
    "gate1_l1_1": ["landing1", "seg_A4", "seg_A5", "seg_R6", "seg_R5", "seg_R4",
                   "seg_R3", "seg_R2", "seg_R1", "gate1"],
    "gate2_l1_1": ["landing1", "seg_A4", "seg_A5", "seg_R6", "seg_R5", "seg_R4",
                   "seg_R3", "seg_R2", "gate2"],
    "gate3_l1_1": ["landing1", "seg_A4", "seg_A5", "seg_R6", "seg_R5", "seg_R4",
                   "seg_R3", "gate3"],
    "gate4_l1_1": ["landing1", "seg_A4", "seg_A5", "seg_R6", "seg_R5", "gate4"],
    "gate5_l1_1": ["landing1", "seg_A4", "seg_A5", "seg_R6", "gate5"],
    "gate6_l1_1": ["landing1", "seg_A4", "seg_A5", "gate6"],
}
"""
:param:`default_air_loc`: Coordinates of the air location (for visualizing flight).
"""
default_air_loc = (30.0, 25.0)


class AssetParams(Parameter, readonly=True):
    """
    `param`:AssetParams: dict defining performance parameters for the assets
    """

    uas: tuple = ()  # names for uaVs (filled on initialization)
    mas: tuple = ()  # names for piloted aircraft
    hs: tuple = ()  # names for helicopters
    gatetime: np.float64 = 45.0  # time spent at gate parking
    gatetime_lim = (0.0, 120.0)
    landtime: np.float64 = 3.0
    takeofftime: np.float64 = 5.0
    num_ua: np.int32 = 3  # number of uaS
    num_ua_lim = (0, 8)
    num_ma: np.int32 = 3  # number of mas
    num_ma_lim = (0, 8)
    num_h: np.int32 = 2  # number of helicopters
    num_h_lim = (0, 4)
    ground_ua: np.int32 = 1  # number of uas initialized on the ground
    ground_ua_lim = (0, 4)
    ground_ma: np.int32 = 1  # number of mas initialized on the ground
    ground_ma_lim = (0, 4)
    ground_h: np.int32 = 1  # number of Hs initialized on the ground
    ground_h_lim = (0, 1)

    def __init__(self, *args, **kwargs):
        names = self.set_names(*args, **kwargs)
        if args:
            args = tuple(*names, args[3:])
        else:
            args = names
        super().__init__(*args, **kwargs)

    def set_names(self, *args, **kwargs):
        """Initialize the Hs, mas, uas names given certain numua/numma/numH fields."""
        names = []
        for i in ["ua", "ma", "h"]:
            num_id = "num_" + i
            ind = self.__fields__.index(num_id)
            if num_id in kwargs:
                num = kwargs[num_id]
            else:
                num = self.__default_vals__[ind]
            names.append(tuple(self.create_namelist(i, num)))
        return names

    def create_namelist(self, name, num):
        """
        Create a list of names.

        Follows the convention name+ind for a name (string) and number of names (int).
        """
        return [str(name) + str(Nnum + 1) for Nnum in range(0, int(num))]

    def assetnames(self):
        return (*self.hs, *self.uas, *self.mas)


class TaxiwayParams(Parameter, readonly=True):
    """Aircoords, numbers of assets, and gatetime/recoverytime."""

    assetparams: AssetParams = AssetParams()
    seed: np.int32 = 42  # random seed for map generation
    aircoords = default_aircoords
    speeds = default_speeds
    routes = default_routes
    air_loc = default_air_loc

    def assetnames(self):
        return self.assetparams.assetnames()


default_params = TaxiwayParams()


class TaxiwayStates(State):
    """
    Taxiway States.

    Fields
    ------
    - area_allocation: dict
        Where each asset is allowed to be (e.g., the route)
        Structure: {area: set(asset1, asset2...)}
    - asset_area: dict
        Where each asset is currently {asset1:area}
    - asset_assignment: dict
        Where each asset is intended to be (e.g., end of the route) {asset1: area}
    """

    area_allocation: dict = {}
    asset_area: dict = {}
    asset_assignment: dict = {}


class Environment(MultiFlow):
    """
    Flow combining the map that the assets navigate on with their assignment/allocation.

    Has attributes:
        - p : TaxiwayParams
            TaxiwayParams
        - s : TaxiwayStates
            Taxiway States.
        - :param:`places`:
            Coordinates of the segments in the airfield {place:[x,y]}
        - :param:`airfield`:
            Line segments corresponding to each segment in the airfield {place:lineseg}
    """

    container_p = TaxiwayParams
    container_s = TaxiwayStates

    def __init__(self, name='', glob=[], **kwargs):
        """
        Creates map of environment with given segments

        Parameters
        ----------
        params :  tuple
            (TaxiwayParams, seed)
        name : str
            Name for the flow
        ftype : str
            Flow type indicator
        """
        super().__init__(name=name, glob=glob, **kwargs)
        self.build_map()
        self.init_places()
        self.allocate_assets()

    def build_map(self):
        """
        Builds airfield property as a dict of linestrings corresponding to aircoords.
        """
        self.airfield = {}
        for segname, coords in self.p.aircoords.items():
            self.airfield[segname] = LineString(coords)

    def init_places(self):
        """
        Builds places property as a dict of coordinates using the airfield property
        """
        places = dict()
        for key, lin in self.airfield.items():
            if key.startswith("gate"):
                places[key] = (lin.bounds[0], lin.bounds[1])
            if key.startswith("heli"):
                places[key] = (lin.bounds[2], lin.bounds[3])
            if key.startswith("ground"):
                places[key] = (lin.bounds[0], lin.bounds[1])
            if key.startswith("take"):
                places[key] = (lin.bounds[2], lin.bounds[3])
            if key.startswith("land"):
                places[key] = (lin.bounds[2], lin.bounds[3])
        places["air_loc"] = self.p.air_loc
        self.places = places

    def allocate_assets(self):
        """
        Determine the initial allocation of assets within the airfield.

        e.g. to (gate, helipad, runway, air_loc, etc).
        Returns area_allocation, asset_area, and asset_assignment,
        setting area_assignment=asset_area.
        """
        rng = np.random.default_rng(self.p.seed)
        area_allocation = {p: set() for p in self.places}
        asset_area = {asset: "air_loc" for asset in self.p.assetnames()}
        asset_assignment = {asset: "air_loc" for asset in self.p.assetnames()}

        heli_areas = [p for p in self.places if "helipad" in p]
        heli_choice = [
            *rng.choice(heli_areas, self.p.assetparams.ground_h, replace=False)
        ]

        ma_areas = [p for p in self.places if "helipad" not in p]
        ma_choice = [*rng.choice(ma_areas, self.p.assetparams.ground_ma, replace=False)]

        ua_areas = [p for p in ma_areas if p not in ma_choice]
        ua_choice = [*rng.choice(ua_areas, self.p.assetparams.ground_ua, replace=False)]

        for area in area_allocation:
            if area in heli_choice:
                allocation = self.p.assetparams.hs[len(heli_choice) - 1]
                heli_choice.pop()
            elif area in ma_choice:
                allocation = self.p.assetparams.mas[len(ma_choice) - 1]
                ma_choice.pop()
            elif area in ua_choice:
                allocation = self.p.assetparams.uas[len(ua_choice) - 1]
                ua_choice.pop()
            else:
                allocation = False
            if allocation:
                area_allocation[area].add(allocation)
                asset_area[allocation] = area
                asset_assignment[allocation] = area
        self.s.area_allocation.update(area_allocation)
        self.s.asset_area.update(asset_area)
        self.s.asset_assignment.update(asset_assignment)

    def area_to_standby(self, asset, new_area=[]):
        """
        Clear the area allocation for standby (sets location/assignment to given area).

        Useful for parking at gate and takeoff, where the asset no longer moves through
        the airfield.

        Parameters
        ----------
        asset : str
            Name of asset
        new_area : str, optional
            Are the asset is entering a standby (or done) mode. The default is [].
        """
        if not new_area:
            new_area = self.s.asset_area[asset]
        for area in self.s.area_allocation:
            self.s.area_allocation[area].discard(asset)
        self.s.asset_area[asset] = new_area
        self.s.asset_assignment[asset] = new_area
        if new_area in self.s.area_allocation:
            self.s.area_allocation[new_area].add(asset)

    def show_map(self, legend=False, figsize=(6.5, 3.5), show_area_allocation=False,
                 areas_to_label=["gate", "landing", "helipad", "takeoff"]):
        """
        Show the map as a plot.

        Parameters
        ----------
        legend : bool
            whether to include a legend
        figsize : tuple
            Figure size

        Returns
        -------
        fig: matplotlib figure
        ax: matplotlib axis
        """
        fig, ax = plt.subplots(figsize=figsize)
        airspace = plt.Rectangle(
            (self.p.air_loc[0] - 5, self.p.air_loc[1] - 5),
            10,
            10,
            fc="white",
            ec="blue",
        )
        ax.add_patch(airspace)
        plt.text(
            self.p.air_loc[0],
            self.p.air_loc[1] + 5,
            "In Air",
            horizontalalignment="center",
        )
        for segname, lineobj in self.airfield.items():
            if any([segname.startswith(prefix) for prefix in areas_to_label]):
                label = "Destination"
                color = "blue"
                if show_area_allocation == "size":
                    entry = len(self.area_allocation[segname])
                elif type(show_area_allocation) == dict:
                    entry = show_area_allocation[segname]
                elif show_area_allocation:
                    entry = self.area_allocation[segname]
                else:
                    entry = False
                if not entry:
                    area_text = ""
                elif type(entry) == int:
                    area_text = ": (" + str(entry) + ")"
                else:
                    area_text = "\n" + str(entry)
                text = segname + area_text
                plt.text(
                    lineobj.centroid.xy[0][0],
                    lineobj.centroid.xy[1][0],
                    text,
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="top",
                    rotation_mode="anchor",
                )
            else:
                label = "segment"
                color = "gray"
            ax.plot(*lineobj.xy, label=label, color=color)

        if legend:
            consolidate_legend(ax)
        plt.axis('off')
        return fig, ax

    def show_route(self, routename, color="red", legend=True, **kwargs):
        """
        Highlight a given route on the map.

        Parameters
        ----------
        routename : str
            Route name (in routes)
        color : str, optional
            Color to highlight. The default is "red".
        legend : bool, optional
            Whether to include a legend. The default is True.
        **kwargs : kwargs
            arguments to show_map

        Returns
        -------
        fig: matplotlib figure
        ax: matplotlib axis
        """
        fig, ax = self.show_map(legend=False, **kwargs)
        for segname in self.p.routes[routename]:
            lineobj = self.airfield[segname]
            ax.plot(*lineobj.xy, color=color, label=routename)
        if legend:
            consolidate_legend(ax)

    def show_all_routes(self, color="red", legend=True, **kwargs):
        """
        Iterate through routes to show all routes on different plots.

        Parameters
        ----------
        color : str, optional
            Color to highlight. The default is "red".
        legend : bool, optional
            Whether to include a legend. The default is True.
        **kwargs : kwargs
            arguments to show_map
            DESCRIPTION.

        Returns
        -------
        figs : dict
            dict of tuples with structure {route:(fig, ax)}
        """
        figs = {}
        for route in self.p.routes:
            figs[route] = self.show_route(route, color=color, legend=legend, **kwargs)
        return figs

    def indicate_overbooked(self):
        """Return true if too many assets are assigned in area_allocation."""
        overbooked = False
        for k in self.s.area_allocation:
            if k not in ('air_loc', 'takeoff1') and len(self.s.area_allocation[k]) > 1:
                overbooked = True
        return overbooked

    def indicate_incorrect_perception(self):
        """Return true for the atc if it doesn't match the true allocation."""
        if self.name == 'atc':
            if self.s.area_allocation != self.glob.s.area_allocation:
                return True
            else:
                return False
        else:
            return False


class LocationState(State):
    """State defining the default values of the location flow for assets."""

    x: np.float64 = default_air_loc[0]
    y: np.float64 = default_air_loc[1]
    xd: np.float64 = 0.0
    yd: np.float64 = 0.0
    speed: np.float64 = 0.0
    stage: str = "flight"  # phase of operation (flight, land, taxi, park, takeoff)
    mode: str = "standby"  # completion of the phase (standby, continue, hold, done)


class Location(MultiFlow):
    container_s = LocationState

    def dist(self, other):
        """
        Calculate the x-y distance between two locations.

        Parameters
        ----------
        other : Location
            Other location to compare against

        Returns
        -------
        distance : Float
            Distance between points
        """
        p_self = Point(self.s.get('x', 'y'))
        p_other = Point(other.s.get('x', 'y'))
        return p_self.distance(p_other)

    def get_closest_loc(self, *locs):
        """
        Get the closest distance to the given location.

        Parameters
        ----------
        *locs : str
            If provided, only finds the distance to the given names

        Returns
        -------
        closest : str
            name of the closest location
        dist : float
            distance to the closest location
        """
        dist = 1000.0
        if len(locs) == 0:
            locs = self.glob.locals
        closest = ""
        for i in locs:
            if i != self.name:
                dist_i = self.dist(getattr(self.glob, i))
                if dist_i < dist:
                    dist = dist_i
                    closest = i
        return closest, dist

    def indicate_unsafe(self):
        """If the real location is too close to the closest location, set unsafe."""
        if self.name not in ['percieved', 'closest']:
            closest, dist = self.get_closest_loc()
            if (self.s.stage in ['land', 'taxi', 'park', 'takeoff'] and
                    dist < 1 and getattr(self.glob, closest).s.stage != 'flight'):
                return True
            else:
                return False

        else:
            return False

    def indicate_nosight(self):
        """Indicate as 'incorrect' if percieved doesn't match the closest location."""
        if "percieved" in self.name:
            if self.glob.s != self.s:
                return True
            else:
                return False
        elif "closest" in self.name and not self.glob.name.startswith("h"):
            closest, dist = self.glob.get_closest_loc(*[i for i in self.glob.glob.locals
                                                        if not i.startswith('h')])
            if closest:
                closest_loc = getattr(self.glob.glob, closest)
                if (self.glob.s.stage in ['land', 'taxi', 'park', 'takeoff']
                    and closest_loc.s.stage in ['land', 'taxi', 'park', 'takeoff']
                        and self.s != closest_loc.s):
                    perc_dist = self.glob.dist(self)
                    # somewhat hacky patch for when movement changes the closest
                    if dist < 3 and perc_dist - dist > 1:
                        return True
                    elif dist < 5 and perc_dist - dist > 4.5:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False


class RequestState(State):
    """
    State  defining communications to/from ATCs and assets.

    Fields
    ------
    atc_com : str
        what ATC wants asset to do, e.g. (takeoff, land, taxi, hold)
    asset_req : str
        what the asset wants to do,
        e.g. (taxi_to_gate, taxi_to_runway, takeoff, landing)
    route : str
        Name of the route assigned (from map)
    """

    atc_com: str = "None"
    asset_req: str = "None"
    route: str = "           "


class Requests(CommsFlow):
    container_s = RequestState

    def indicate_incorrect(self):
        """
        Indicate whether the request is based on accurate information.

        Including: (1) that atc_com matches what atc_com is actually sending for that
        asset, and (2) that asset_req matches what the asset is sending as asset_req.
        """
        if self == self.glob:
            condition = False
        elif self.glob.name == 'requests' and self.name != 'atc' and "_out" not in self.name:
            condition = self.s.atc_com != getattr(self.glob.atc, self.name).s.atc_com
        elif self.glob.name == 'requests' and '_out' in self.name and self.name != 'atc_out':
            name = self.name[:-4]
            true_req = self.glob.fxns[name]['internal']
            condition = self.s.asset_req != true_req.s.asset_req
        elif self.glob.name == 'atc':
            condition = getattr(self.glob.glob, self.name).s.atc_com != self.s.atc_com
        else:
            condition = False
        if condition:
            pass
        return condition

    def indicate_duplicate_land(self):
        """Indicate true if same landing command has been given as another flow."""
        if self.glob.name == 'atc' and self.s.atc_com == 'land':
            other_requests = self.glob.locals
            condition = False
            for req in other_requests:
                if (getattr(self.glob, req).s.atc_com == 'land' and
                        req != self.name):
                    condition = True
            return condition
        else:
            return False


def plot_one_path(mdlhist, asset, color="red", showtimes=False):
    """Plot the path of one asset on the course."""
    x = mdlhist.flows.location[asset + ".s.x"]
    y = mdlhist.flows.location[asset + ".s.y"]
    plt.plot(x, y, color=color)
    if showtimes:
        showtimes = int(showtimes)
        for i, t in enumerate(mdlhist["time"]):
            if not (i % showtimes):
                plt.text(x[i], y[i], str(t))


def plot_course(mdl, mdlhist, asset, showtimes=False, color="red", title=""):
    """
    Highlight the path of one asset on the map.

    Parameters
    ----------
    mdl : model
        model with ground and Location
    mdlhist : dict
        Model history
    asset : str
        Name of the asset
    showtimes : bool/int, optional
        Whether to show the time the asset was at each location (or index of times).
        The default is False.
    color : str, optional
        Color to highlightt route with. The default is "red".

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis
    """
    fig, ax = mdl.flows["ground"].show_map()
    plot_one_path(mdlhist, asset, showtimes=showtimes, color=color)
    if title:
        plt.title(title)
    return fig, ax


def att_to_text(att):
    if type(att) in [float, np.float64]:
        att = np.round(att, 3)
    return str(att)


def plot_tstep(mdl, mdlhist, t, fxnattr="", locattr="", markersize=10,
               show_area_allocation=True, asset_assignment=False, title="",
               assets_to_label="all", **kwargs):
    """
    Plot the location of the assets at a given timestep t.

    Parameters
    ----------
    mdl : model
        model with ground and Location
    mdlhist : dict
        Model history
    t : int
        time-step to plot
    fxnattr : str
        function attribute to display on the plot (e.g., visioncov, mode, segment)
    locattr : str
        location attribute to display on the plot (e.g., speed, x, y)
    markersize : int
        size for the asset markers
    show_area_allocation : bool/"size"
        Show where the assets have been allocated on the map
    asset_assignment : bool
        Show asset assignment/area next to the assets
    title : str
        Title for the plot
    kwargs: kwargs
        kwargs for ground.show_map

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """
    if show_area_allocation == "size":
        show_area_allocation = {
            pl: len(plhist[t])
            for pl, plhist in mdlhist.flows.ground.s.area_allocation.items()
        }
    elif show_area_allocation:
        show_area_allocation = {
            pl: plhist[t]
            for pl, plhist in mdlhist.flows.ground.s.area_allocation.items()
        }
    fig, ax = mdl.flows["ground"].show_map(
        **kwargs, show_area_allocation=show_area_allocation
    )
    texts = []
    for f in mdl.flows["location"].locals:
        loc = mdlhist.flows.location.get(f)
        x, y = loc.s.x[t], loc.s.y[t]
        xd, yd = loc.s.xd[t], loc.s.yd[t]
        mode, stage = loc.s.mode[t], loc.s.stage[t]
        angle = np.arctan2(yd, xd)
        if f.startswith("h"):
            marker = "X"
        else:
            marker = (3, 0, angle * 90 + 30)
        if stage in ["park", "flight"]:
            color = "grey"
        elif "taxi" in stage:
            color = "blue"
        elif stage in ["takeoff", "land"]:
            color = "green"
        elif stage in ["hold"]:
            color = "red"
        else:
            raise Exception("stage: " + stage)
        if mode == "standby":
            markeredgecolor = "purple"
        else:
            markeredgecolor = "grey"
        plt.plot(x, y, marker=marker, markersize=markersize,
                 markeredgecolor=markeredgecolor, color=color)
        text = f
        if assets_to_label == "all" or f in assets_to_label:
            if fxnattr:
                try:
                    atthist = mdlhist.fxns.get(f + "." + fxnattr)
                except Exception or AttributeError:
                    atthist = None
                if (not atthist is None) and len(atthist) > 0:
                    if fxnattr == "s.visioncov" and atthist[t]:
                        plt.plot(*atthist[t].exterior.xy)
                    if fxnattr in ["mode", "segment"]:
                        text_add = mdlhist.fxns.get(f + "." + fxnattr)[t]
                        text = text + ": " + text_add
                    if fxnattr == "faults":
                        faults = {
                            fault
                            for fault, hist in mdlhist.fxns["functions"][f][
                                "faults"
                            ].items()
                            if hist[t]
                        }
                        if faults:
                            text = text + ": " + str(faults)
            if locattr:
                text_add = att_to_text(
                    mdlhist.flows.location.get(f + ".s." + locattr)[t]
                )
                text = text + ": " + text_add
            if asset_assignment:
                assignment = str(mdlhist.fxns["ground"]["asset_assignment"][f][t])
                area = str(mdlhist.fxns["ground"]["asset_area"][f][t])
                text = text + ": (" + area + "->" + assignment + ")"
            textax = plt.text(x, y, text)
            texts.append(textax)
    if texts:
        adjust_text(texts)
    plt.title(title + "(t=" + str(t) + ")")
    plt.axis('off')
    return fig, ax
