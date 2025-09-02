#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

from common import Location, Requests, TaxiwayParams, Environment

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode


class ATCModes(Mode):

    fault_lost_ground_perception = ()
    fault_wrong_land_command = ()
    fault_single_wrong_command = ()


class ATC(Function):

    __slots__ = ('perc_requests', 'perc_ground')
    container_p = TaxiwayParams
    container_m = ATCModes
    flow_location = Location
    flow_requests = Requests
    flow_ground = Environment

    def init_block(self, **kwargs):
        self.perc_requests = self.requests.create_comms(self.name,
                                                        ports=self.p.assetnames())
        self.perc_ground = self.ground.create_local(self.name, "area_allocation",
                                                    p=self.p)

    def dynamic_behavior(self):
        # get communications, look at aircraft
        self.perc_requests.receive()
        if not self.m.has_fault("lost_ground_perception"):
            self.perc_ground.update(to_update="local", to_get="global")

        # go through requests and approve/assign routes and locations
        for asset in self.perc_requests.received().keys():
            perc_asset_requests = getattr(self.perc_requests, asset)
            if perc_asset_requests.s.asset_req == "land":
                self.clear_landing(perc_asset_requests)
            if perc_asset_requests.s.asset_req == "taxi_to_gate":
                self.clear_taxi_to_gate(perc_asset_requests)
            if perc_asset_requests.s.asset_req == "park":
                self.clear_park(perc_asset_requests)
            if perc_asset_requests.s.asset_req == "taxi_to_runway":
                self.clear_taxi_to_runway(perc_asset_requests)
            if perc_asset_requests.s.asset_req == "takeoff":
                self.clear_takeoff(perc_asset_requests)

        # enable further comms
        self.perc_requests.clear_inbox()
        self.perc_requests.send("all")
        self.perc_ground.update(to_update="global", to_get="local")

    def clear_landing(self, asset_req):
        """Send command to land and allocates runway if there is a place to land."""
        if asset_req.name[0] == "h":
            landtype = "helipad"
        elif asset_req.name[0] in ["m", "u"]:
            landtype = "landing"

        if not self.perc_ground.s.asset_assignment[asset_req.name].startswith(landtype):
            if self.m.has_fault("wrong_land_command"):
                free_runways = self.assign_runways_wrong(landtype)
            elif self.m.has_fault("single_wrong_command"):
                free_runways = self.assign_runways_single_wrong(asset_req, landtype)
                self.m.remove_fault("single_wrong_command")
            else:
                free_runways = self.assign_runways(asset_req, landtype)

            if free_runways:
                runway = free_runways[0]
                self.perc_ground.s.area_allocation[runway].add(asset_req.name)
                self.perc_ground.s.asset_assignment[asset_req.name] = runway
                asset_req.s.put(atc_com="land", route="none")

    def assign_runways(self, asset_req, landtype):
        free_runways = [pl for pl, alloc in self.perc_ground.s.area_allocation.items()
                        if len(alloc.difference({asset_req.name})) == 0
                        and pl.startswith(landtype)]
        return free_runways

    def assign_runways_single_wrong(self, asset_req, landtype):
        free_runways = [pl for pl, alloc in self.perc_ground.s.area_allocation.items()
                        if len(alloc.difference({asset_req.name})) > 0
                        and pl.startswith(landtype)]
        return free_runways

    def assign_runways_wrong(self, landtype):
        free_runways = [pl for pl, alloc in self.perc_ground.s.area_allocation.items()
                        if pl.startswith(landtype)]
        return free_runways

    def clear_taxi_to_gate(self, asset_req):
        """Send route and allocate gate provided there is an empty gate."""
        if (
            not (self.perc_ground.s.asset_assignment[asset_req.name].startswith("gate"))
            and ("l1" not in asset_req.s.route)
            and ("landing" in self.perc_ground.s.asset_area[asset_req.name])
        ):
            free_gates = [
                pl
                for pl, alloc in self.perc_ground.s.area_allocation.items()
                if len(alloc.difference({asset_req.name})) == 0
                and pl.startswith("gate")
            ]
            if free_gates:
                gate = free_gates[0]
                runway = self.perc_ground.s.asset_area[asset_req.name]
                self.perc_ground.s.area_allocation[gate].add(asset_req.name)
                self.perc_ground.s.asset_assignment[asset_req.name] = gate
                route = gate + "_" + runway[0] + runway[-1] + "_1"
                asset_req.s.put(atc_com="taxi_to_gate", route=route)

    def clear_park(self, asset_req):
        """Clear parking. De-allocates any (non-parking) space given to the asset."""
        self.perc_ground.area_to_standby(asset_req.name)
        asset_req.s.put(atc_com="park", route="none")

    def clear_taxi_to_runway(self, asset_req):
        """
        Send route and allocates runway.

        Runways don't need to be empty (Aircraft should form a line).
        """
        if (
            not (
                self.perc_ground.s.asset_assignment[asset_req.name].startswith(
                    "takeoff"
                )
            )
            and ("t1" not in asset_req.s.route)
            and ("gate" in self.perc_ground.s.asset_area[asset_req.name])
        ):
            runway_use = [
                (k, len(v))
                for k, v in self.perc_ground.s.area_allocation.items()
                if "takeoff" in k
            ]
            runway = sorted(runway_use, key=lambda runw: runw[1])[0][0]
            gate = self.perc_ground.s.asset_area[asset_req.name]
            self.perc_ground.s.area_allocation[runway].add(asset_req.name)
            self.perc_ground.s.asset_assignment[asset_req.name] = runway
            route = gate + "_" + runway[0] + runway[-1] + "_1"
            asset_req.s.put(atc_com="taxi_to_runway", route=route)

    def clear_takeoff(self, asset_req):
        """Clear the aircraft/helicopter for takeoff."""
        self.perc_ground.area_to_standby(asset_req.name)
        asset_req.s.put(atc_com="taxi_to_gate")

    def get_taxi_route(self, gate, runway):
        """
        Get the taxiroute segments.

        determines the segments given a routename thats in the format of gate1_T1 where
        T stands for take off and the following number represents a runway number.
        L may be used for landing.
        """
        routename = gate + "_" + runway[0] + runway[-1]
        taxiroute = False
        for route in self.Ground.s.routes.keys():
            if (
                routename in route
            ):  # will need to re-add way of selecting between multiple routes based on usage
                taxiroute = route
        if not taxiroute:
            raise Exception("No route between gate: " + gate + " and runway: " + runway)
        return taxiroute
