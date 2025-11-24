#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model of taxiway assets.

Assumptions:
    1) all assets will slow down to 0.3 km/min when approaching turns or intersections
about 500 m away
    2) Distance between grid points is 100 m: 10 grid points equals 1 km.
    3) If an asset detects another asset approaching and it does not have priority it is
stop 100 m before the intersection,
    to allow for the approaching assets to pass

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

from common import AssetParams, Location, Requests, Environment

from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.container.time import Time
from fmdtools.define.block.function import Function

from shapely.geometry import Point, Polygon
import numpy as np


class AssetTime(Time):
    timernames = ('takeoff', 'land', 'park')


class Asset(Function):
    """Superclass for Helicopters and Aircraft (uas and mas)."""

    __slots__ = ('perc_requests', 'perc_location', 'closest_location')
    container_p = AssetParams
    container_t = AssetTime
    flow_location = Location
    flow_requests = Requests
    flow_ground = Environment

    def init_block(self, **kwargs):
        self.perc_requests = self.requests.create_comms(self.name)
        area = self.ground.s.asset_area[self.name]
        self.location = self.location.create_local(self.name)
        self.location.s.assign([*self.ground.places[area]], "x", "y")

        atc_requests = getattr(self.requests.atc, self.name)
        if "gate" in area:
            self.location.s.put(stage="park")
            self.perc_requests.s.put(asset_req='taxi')
        elif "takeoff" in area:
            self.location.s.put(stage="takeoff")
            self.perc_requests.s.put(asset_req='takeoff')
        elif "helipad" in area or "landing" in area:
            self.location.s.put(stage="land")
            self.perc_requests.s.put(asset_req='taxi')
        elif "air_loc" in area:
            self.location.s.put(stage="flight", mode="standby")
            self.perc_requests.s.put(asset_req="land")
        atc_requests.s.assign(self.perc_requests.s, as_copy=True)

        self.perc_requests.update("asset_req", to_update="atc", to_get="local",)

        self.perc_location = self.location.create_local("percieved", s=self.location.s)

        self.closest_location = self.location.create_local("closest")
        closest, dist = self.location.get_closest_loc()
        if closest:
            self.closest_location.s.assign(getattr(self.location.glob, closest).s)

        self.m.set_mode(self.perc_location.s.stage)
        self.send_perception()

    def dynamic_behavior(self):
        self.receive_perception()
        self.check_crash()
        self.m.set_mode(self.perc_location.s.stage)

        if not self.m.has_fault("crash", "immobile"):
            if self.m.in_mode('park'):
                self.park_cycle()
            if self.m.in_mode('takeoff'):
                self.takeoff_cycle()
            if self.m.in_mode('land'):
                self.land_cycle()
            if self.m.in_mode('taxi'):
                self.move_cycle()
            if self.m.in_mode('flight'):
                self.flight_cycle()

        self.check_crash()
        self.send_perception()

    def check_crash(self):
        if self.m.in_mode("park", "takeoff", "land", "taxi", "hold") and not all(self.location.s.get("x", "y") == self.ground.p.air_loc):
            # triggers a crash if too close
            closestasset, distance_to_asset = self.find_closest_asset(in_vision=False)
            if distance_to_asset < 1:
                self.m.add_fault("crash")

    def receive_perception(self):
        # update perception
        self.perc_requests.receive()
        self.perc_requests.clear_inbox()

        if self.perc_requests.s.atc_com == "land":  # add condition for in_air
            self.location.s.put(stage="land", mode="continue")
        if 'taxi' in self.perc_requests.s.atc_com:
            if self.perc_requests.s.route.startswith('gate'):
                self.location.s.mode = "continue"
            else:
                self.location.s.mode = "standby"
        if 'hold' in self.perc_requests.s.atc_com:
            self.location.s.mode = "hold"

        # update percieved location
        self.perc_location.s.assign(self.location.s)

    def takeoff_cycle(self):
        # can add code to abort take off and return to the landing location if needed.
        # As a resilience measure from the pilot or atc
        if self.t.takeoff.indicate_complete():
            self.t.takeoff.reset()
            self.update_takeoff()
        elif self.t.takeoff.indicate_standby():
            self.t.takeoff.set_timer(self.p.takeofftime)
            self.location.s.put(mode='continue')
        else:
            self.t.takeoff.inc()
            self.location.s.put(mode='continue')

    def land_cycle(self):
        # can add code to abort landing
        if self.t.land.indicate_complete():
            self.t.land.reset()
            self.update_landing()
        elif self.t.land.indicate_standby():
            self.t.land.set_timer(self.p.landtime)
            self.location.s.put(mode='continue')
        else:
            self.t.land.inc()
            self.location.s.put(mode='continue')

    def check_landing(self, landplace):
        aircraft_at_landing = [a for a, l in self.ground.s.asset_area.items()
                               if (l == landplace and a != self.name)]
        return aircraft_at_landing

    def update_landing(self):
        landplace = self.ground.s.asset_assignment[self.name]
        if self.m.has_fault("lost_sight") or not (self.check_landing(landplace)):
            self.location.s.assign([*self.ground.places[landplace]], "x", "y")
            self.ground.area_to_standby(self.name, landplace)
        else:
            pass
            #print(self.name+" correction worked at t="+str(self.t.time))

    def park_cycle(self):
        self.location.s.speed = 0.0
        if self.t.park.indicate_complete():
            self.t.park.reset()
            self.update_park()
        elif self.t.park.indicate_standby():
            if self.perc_requests.s.atc_com != 'hold':
                self.t.park.set_timer(self.p.gatetime)
            else:
                self.perc_requests.s.put(asset_req='none')
        else:
            self.t.park.inc()

    def flight_cycle(self):
        if not self.s.cycled:
            self.perc_requests.s.put(asset_req="land")
        else:
            self.perc_requests.s.put(asset_req="none")
    def send_perception(self):
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.send('atc', 'asset_req')
    def determine_visioncov(self, xd, yd):
        """
        Create the polygon corresponding to the asset's vision cone.

        Parameters
        ----------
        xd, yd : float
            x/y direction corresponding to the way the asset is looking (along the segment)

        Returns
        -------
        visioncov : Polygon
            Shapely polygon corresponding to the asset's vision
        """
        x, y = self.perc_location.s.get("x", "y")
        junction = (x, y) in [*self.ground.airfield[self.s.segment].coords]
        if self.name.startswith('ua'):
            covcenter = Point(x, y)
            visioncov = covcenter.buffer(8)
        else:
            if junction:
                # assumes a 60 degree coverage zone with 1 km visibility,
                # This can be parametrized by passing in front vision and peripheral
                # vision coverage angles
                visioncoords = calc_vision_coords(x, y, xd, yd, 10, np.pi/6)
                visioncoords.insert(0, (x, y))
            else:
                visioncoords = calc_vision_coords(x, y, xd, yd, 5, np.pi/6)
                # 120 degrees mid peripheral vision
                temp = calc_vision_coords(x, y, xd, yd, 5, np.pi/3)
                visioncoords.insert(0, temp[0])
                visioncoords.append(temp[1])
                # 280 (60+80)degrees for neck turn
                temp = calc_vision_coords(x, y, xd, yd, 5, 14 * np.pi / 9)
                visioncoords.insert(0, temp[0])
                visioncoords.append(temp[1])
                visioncoords.append((x, y))
            visioncov = Polygon(visioncoords)
        return visioncov

    def find_closest_asset(self, xd=0.0, yd=0.0, in_vision=True):
        """
        Find asset closest to the current. If in_vision, update closest location.

        Parameters
        ----------
        xd : float
            direction of vision cone in the x
        yd : float
            direction of vision cone in the y
        in_vision : bool, optional
            Whether the check is percieved (using vision cone) or actual
            (using true location). The default is True.

        Returns
        -------
        assetloc : location
            Location corresponding to the closest asset
        distance : float
            Distance of current location to the closest asset
        """
        colassets = {}
        assetloc = {}
        distance = 1000
        if in_vision:
            self.s.visioncov = self.determine_visioncov(xd, yd)

        local_pos = Point(self.location.s.get('x', 'y'))
        for i in self.p.uas + self.p.mas + self.p.hs:
            if i != self.name:
                assetpos = Point(getattr(self.location.glob, i).s.get('x', 'y'))
                if not(in_vision) or self.s.visioncov.covers(assetpos):
                    colassets[i] = local_pos.distance(assetpos)
        if len(colassets) > 0:
            asset = min(colassets, key=colassets.get)
            assetloc = getattr(self.location.glob, asset)
            distance = colassets[asset]
        if in_vision and assetloc:
            self.closest_location.s.assign(assetloc.s)
        elif in_vision:
            self.closest_location.reset()
        return assetloc, distance


def calc_vision_coords(x, y, xd,yd, visibility, angle):
    """
    Determine the coordinates of the vision coverage area.

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate
    xd : float
        x-direction
    yd : float
        y-direction.
    visibility : float
        Length of the cone from the perciever
    angle : float
        Angle of the cone (in radians)

    Returns
    -------
    visioncoords : list
        List of two points [(x2, y2), (x3, y3)] corresponding to the edge of 
        perception to the reciever.
    """
    mag = np.sqrt(xd**2+yd**2)+0.000001
    sidelen = abs(visibility / np.cos(angle))
    ptx2 = x + (((xd/mag)*np.cos(angle) + (yd/mag)*np.sin(angle)) * sidelen)
    pty2 = y + (((yd/mag)*np.cos(angle) - (xd/mag)*np.sin(angle)) * sidelen)
    ptx3 = x + ((np.sin((np.pi/2)-angle)*(xd/mag) -
                np.cos((np.pi/2)-angle)*(yd/mag)) * sidelen)
    pty3 = y + ((np.cos((np.pi/2)-angle)*(xd/mag) +
                np.sin((np.pi/2)-angle)*(yd/mag)) * sidelen)
    visioncoords = [(ptx2, pty2), (ptx3, pty3)]
    return visioncoords


def get_next_pt(lineseg, endpt, subseg, num_coords):
    """Get the point on the line segment at the subseg index (w- reversal)."""
    if endpt == Point(lineseg.coords[0]):
        return Point(lineseg.coords[num_coords-(subseg+1)])
    elif endpt == Point(lineseg.coords[-1]):
        return Point(lineseg.coords[subseg])


def get_unit_vect(pt1, pt2):
    """Return unit vector xd, yd for the distance between two points w- x and y attr."""
    xd = (pt2.x-pt1.x)/np.sqrt((pt2.x-pt1.x)**2 + (pt2.y-pt1.y)**2+0.00001)
    yd = (pt2.y-pt1.y)/np.sqrt((pt2.x-pt1.x)**2 + (pt2.y-pt1.y)**2+0.00001)
    return xd, yd


class AircraftState(State):
    visioncov: Polygon = Polygon()
    cycled: bool = False
    segment: str = "          "
    segments: list = []
    subseg: int = 0
    seg_ind: int = 0


class AircraftMode(Mode):
    """
    Aircraft modes.

    Modes
    -----
    crash : Fault
        Mainly a triggered fault that happens when aircraft collide.
    immobile : Fault
        Plane stopped on runway and can't move.
    lost_sight : Fault
        Plane cannot see other aircraft.
    land : Mode
        Aircraft landing
    takeoff : Mode
        Aircraft taking off
    taxi : Mode
        Aircraft taxiing on taxiway
    park : Mode
        Aircraft parked at gate
    """

    fault_crash = (1e-7, 1e7, (("land", 1.0), ("takeoff", 1.0), ("taxi", 1.0)))
    fault_immobile = (1e-5, 1e5, (("taxi", 1.0)))
    fault_lost_sight = (1e-5, 1e5, (("land", 1.0), ("takeoff", 1.0), ("taxi", 1.0)))
    opermodes = ('land', 'takeoff', 'taxi', 'park')
    mode: str = 'park'


class Aircraft(Asset):

    container_s = AircraftState
    container_m = AircraftMode

    def update_landing(self):
        """
        Land the aircraft.

        Put the aircraft at the assigned location, sets stage to taxi, and requests taxi
        from atc.
        """
        super().update_landing()
        
        self.location.s.put(stage="taxi", mode="standby")
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.asset_req ='taxi_to_gate'

    def update_takeoff(self):
        self.ground.area_to_standby(self.name, "air_loc")
        self.location.s.put(x=self.ground.p.air_loc[0], y=self.ground.p.air_loc[1],
                            xd=0.0, yd=0.0, speed=0.0, stage='flight', mode='done')
        self.s.put(segments=[], segment="", seg_ind=0, visioncov=Polygon())
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.put(asset_req='none', route="")
        self.s.cycled = True

    def update_park(self):
        self.ground.area_to_standby(self.name)
        self.s.put(segments=[], segment="", seg_ind=0)
        self.location.s.put(xd=0.0, yd=0.0, speed=0.0, stage="taxi", mode="standby")
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.asset_req = 'taxi_to_runway'

    def move_cycle(self):
        # speed in km per minute and timestep in mins
        # if no segment, get segment from from map/route
        if (not self.s.segment or (" " in self.s.segment)) and self.perc_requests.s.route.startswith('gate'):
            self.s.segments = self.ground.p.routes[self.perc_requests.s.route]
            self.s.segment = self.s.segments[0]
            self.s.seg_ind = 0
        # if in continue mode (ends if gets to gate/atc tells it to hold)
        if self.perc_location.s.mode=='continue':
            # get current segment
            self.s.segment = self.s.segments[self.s.seg_ind]
            # pilot goes speed set by speed limit (in ground)
            self.perc_location.s.speed = self.ground.p.speeds[self.s.segment]
            self.location.s.speed = self.perc_location.s.speed
            # determine proposed travel trajectory, etc
            endpt, is_last_subseg = self.find_segment_endpt()
            currpt = Point(self.location.s.get("x", "y"))
            seg_dist = endpt.distance(currpt)
            self.perc_location.s.assign([*get_unit_vect(currpt, endpt)], 'xd', 'yd')
            # collision avoidance - need to check that xd doesn't conflict with another
            # aircraft coming the other way
            if not self.m.has_fault("lost_sight"):
                self.avoid_collision()
            dist_to_travel = self.location.s.speed * self.t.dt
            # increment location
            if dist_to_travel >= seg_dist:
                if is_last_subseg:
                    self.s.seg_ind += 1
                    self.s.subseg = 0
                self.location.s.put(x=endpt.x, y=endpt.y)
                self.location.s.assign(self.perc_location.s, 'xd', 'yd')
            elif dist_to_travel < seg_dist:
                self.location.s.assign(self.perc_location.s, 'xd', 'yd')
                self.location.s.inc(x=dist_to_travel*self.location.s.xd,
                                    y=dist_to_travel*self.location.s.yd)
            # if at the end of the route, switch modes and standby
            if self.s.seg_ind >= len(self.s.segments):
                self.s.segments = []
                self.s.seg_ind = 0
                self.s.segment = ""
                # now need to send requests/change mode for takeoff or parking
                if 't1' in self.perc_requests.s.route:
                    self.location.s.put(stage='takeoff', mode="standby")
                    self.perc_requests.s.asset_req = 'takeoff'
                elif 'l1' in self.perc_requests.s.route:
                    self.location.s.put(stage='park', mode="standby")
                    self.perc_requests.s.asset_req = "none"

            self.perc_location.s.assign(self.location.s)
        self.update_route_area_allocation()

    def update_route_area_allocation(self):
        if self.s.segment:
            self.ground.s.asset_area[self.name] = self.s.segment
        for i, seg in enumerate(self.s.segments):
            if seg in self.ground.s.area_allocation:
                if i < self.s.seg_ind:
                    self.ground.s.area_allocation[seg].discard(self.name)
                else:
                    self.ground.s.area_allocation[seg].add(self.name)

    def find_segment_endpt(self):
        """
        Find the end point of the current line segment to go towards.

        If not the last segment, just goes to the point that's concurrent with the nex
        segment. Otherwise, picks the point farthest from the last segment.
        """
        lineseg = self.ground.airfield[self.s.segment]
        # get the end of the segment if at the end
        # (assumes end segment is a single subsegment)
        if self.s.seg_ind >= len(self.s.segments)-1:
            prev_seg = self.s.segments[self.s.seg_ind-1]
            prev_line = self.ground.airfield[prev_seg]
            pt_0 = Point(lineseg.coords[0])
            pt_1 = Point(lineseg.coords[1])
            if prev_line.distance(pt_0) < prev_line.distance(pt_1):
                endpt = pt_1
            else:
                endpt = pt_0
        else:
            next_seg = self.s.segments[self.s.seg_ind+1]
            next_line = self.ground.airfield[next_seg]
            endpt = next_line.intersection(lineseg)
            if not [*endpt.coords]:
                raise Exception("Unable to find point for segment: "+self.s.segment
                                + " index:"+str(self.s.seg_ind))
        # get the closest subsegment point between here and the end of the segment
        num_coords = len(lineseg.coords)
        if num_coords > 2:
            nextpt = get_next_pt(lineseg, endpt, self.s.subseg, num_coords)
            currpt = Point(self.location.s.get("x", "y"))
            if currpt == nextpt and not(currpt == endpt):
                self.s.subseg += 1
                nextpt = get_next_pt(lineseg, endpt, self.s.subseg, num_coords)
            is_last = (endpt == nextpt)
            endpt = nextpt
        else:
            is_last = True
        return endpt, is_last

    def avoid_collision(self):
        """Slow down/stop if the asset is moving in xd, yd direction."""
        # get closest asset (for collision avoidance)
        asset, dist = self.find_closest_asset(*self.perc_location.s.get('xd', 'yd'))
        # check if both assets are on the same subsegment - change to distance-based
        if asset:
            self.set_collision_avoid_speed(asset, dist)

    def set_collision_avoid_speed(self, closestasset, distance_to_asset):
        """
        Set the collision avoidance speed required to avoid the closest asset.

        Parameters
        ---------- 
        closestasset : location flow
            The location corresponding to the closest asset
        distance_to_asset : float
            Distance to the closest asset
        """
        currpt = self.perc_location.s.get('x', 'y')
        nextpt = proj_next_pt(self.perc_location.s, self.t.dt)

        currpt_closest = closestasset.s.get('x', 'y')
        vect_to_next = nextpt - currpt
        vect_to_closest = currpt_closest - currpt
        dist_to_closest = np.linalg.norm(vect_to_closest)
        vect_next_to_closest = currpt_closest - nextpt
        dist_next_to_closest = np.linalg.norm(vect_next_to_closest)
        # if it's not going to go past the other point
        if np.dot(vect_to_closest, vect_next_to_closest) > 0.0:
            if (dist_next_to_closest > dist_to_closest and
                    np.dot(vect_to_closest, vect_next_to_closest) > 0.0):
                # if will be further away (and not ahead)
                speed = self.perc_location.s.speed
            else:
                # if it will be closer, it can follow slowly to a point
                slowspeed = 1.0
                speed = min(max((dist_to_closest-2.0)/self.t.dt, 0.0), slowspeed)
        else:
            # it's going to go past the other point, stop
            speed = 0.0

        self.perc_location.s.speed = speed
        self.location.s.speed = speed


def proj_next_pt(loc_state, dt):
    """Project the next point from going the current direction."""
    return loc_state.speed*dt*loc_state.get('xd', 'yd')+loc_state.get('x', 'y')


class HelicopterState(State):
    cycled: bool = False


class HelicopterMode(Mode):
    """ Helicopter Mode--subset of AircraftMode for Helicopters."""
    fault_crash = (1e-7, 1e7, (("land", 1.0), ("takeoff", 1.0), ("taxi", 1.0)))
    fault_immobile = (1e-5, 1e5, (("park", 1.0)))
    fault_lost_sight = (1e-5, 1e5, (("land", 1.0), ("takeoff", 1.0), ("taxi", 1.0)))
    opermodes = ('land', 'takeoff', 'park')
    mode: str = 'park'


class Helicopter(Asset):

    container_s = HelicopterState
    container_m = HelicopterMode

    def update_landing(self):
        super().update_landing()

        self.location.s.put(stage='park', mode="standby")
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.put(asset_req="park")

    def update_takeoff(self):
        self.ground.area_to_standby(self.name, "air_loc")
        self.location.s.assign([*self.ground.p.air_loc], "x", "y")
        self.location.s.put(stage="flight", mode="done")
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.put(asset_req='none')
        self.s.cycled = True

    def update_park(self):
        self.location.s.put(stage="takeoff", mode="standby")
        self.perc_location.s.assign(self.location.s)
        self.perc_requests.s.asset_req = 'takeoff'
