#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for taxiway model assets.

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

from common import (
    TaxiwayParams,
    AssetParams,
    Environment,
    Location,
    Requests,
    plot_course,
    plot_tstep,
)
from asset import Aircraft, Helicopter

from fmdtools.define.block.function import Function
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.sim import propagate as prop
from fmdtools.analyze.common import suite_for_plots

import unittest


# test_setup
class LoadingForAircraft(Function):

    __slots__ = ('perc_requests',)
    flow_requests = Requests
    flow_ground = Environment
    container_p = AssetParams

    def init_block(self, **kwargs):
        self.perc_requests = self.requests.create_comms("atc", ports=self.p.mas)
        self.ground.area_to_standby("ma1", "air_loc")

    def dynamic_behavior(self):
        if self.t.time == 1.0:
            self.perc_requests.ma1.s.put(atc_com="land")
            self.ground.s.asset_assignment["ma1"] = "landing1"
        # if t>7.0:
        #    self.perc_requests.receive()
        self.perc_requests.receive()
        if self.perc_requests.ma1.s.asset_req == "taxi_to_gate":
            self.perc_requests.ma1.s.put(atc_com="taxi_to_gate", route="gate2_l1_1")
            self.ground.s.asset_assignment["ma1"] = "gate2"
            # update area allocation for route??

        if self.perc_requests.ma1.s.asset_req == "taxi_to_runway":
            self.perc_requests.ma1.s.put(atc_com="taxi_to_runway", route="gate2_t1_1")
            self.ground.s.asset_assignment["ma1"] = "takeoff1"

        self.perc_requests.clear_inbox()

        self.perc_requests.send("ma1", "atc_com", "route")


class test_aircraft_model(FunctionArchitecture):
    container_p = TaxiwayParams
    default_sp = dict(end_time=120)

    def init_architecture(self, **kwargs):
        self.add_flow("ground", Environment, p=self.p)
        self.add_flow("location", Location)
        self.add_flow("requests", Requests)

        self.add_fxn("loading", LoadingForAircraft, "ground", "requests",
                     p=self.p.assetparams)
        self.add_fxn("ma1", Aircraft, "ground", "location", "requests",
                     p=self.p.assetparams)
        # second aircraft is collision avoidance test
        if self.p.assetparams.num_ma >= 2:
            self.add_fxn(
                "ma2", Aircraft, "ground", "location", "requests", p=self.p.assetparams
            )
            self.flows["location"].ma2.s.put(
                x=float(self.flows["ground"].p.aircoords["seg_A1"][0][0]),
                y=float(self.flows["ground"].p.aircoords["seg_A1"][0][1]),
                mode="standby",
                stage="hold",
            )


class LoadingForHeli(Function):
    __slots__ = ('perc_requests', )
    flow_requests = Requests
    flow_ground = Environment

    def init_block(self, **kwargs):
        self.perc_requests = self.requests.create_comms("atc", ports=["h1"])

    def dynamic_behavior(self):
        if self.t.time == 1.0:
            self.perc_requests.h1.s.put(atc_com="land")
            self.ground.s.asset_assignment["h1"] = "helipad1"
        self.perc_requests.receive()
        if self.perc_requests.h1.s.asset_req == "park":
            self.perc_requests.h1.s.put(atc_com="park")
        if self.perc_requests.h1.s.asset_req == "takeoff":
            self.perc_requests.h1.s.put(atc_com="takeoff")
            self.ground.s.asset_assignment["h1"] = "air_loc"
            # update area allocation for route??
        self.perc_requests.clear_inbox()
        self.perc_requests.send("h1", "atc_com")


class test_heli_model(FunctionArchitecture):
    container_p = TaxiwayParams
    default_sp = dict(end_time=120)

    def init_architecture(self, **kwargs):
        self.add_flow("ground", Environment, p=self.p)
        self.add_flow("location", Location)
        self.add_flow("requests", Requests)

        self.add_fxn("loading", LoadingForHeli, "ground", "requests")
        self.add_fxn("h1", Helicopter, "ground", "location", "requests",
                     p=self.p.assetparams)


class AssetTests(unittest.TestCase):
    def setUp(self):
        single_ac_params = AssetParams(
            num_ma=1, ground_ma=0, num_ua=0, ground_ua=0, num_h=0, ground_h=0
        )
        self.single_ac_model = test_aircraft_model(
            p=TaxiwayParams(assetparams=single_ac_params)
        )
        two_ac_params = AssetParams(
            num_ma=2, ground_ma=1, num_ua=0, ground_ua=0, num_h=0, ground_h=0
        )
        self.two_ac_model = test_aircraft_model(
            p=TaxiwayParams(assetparams=two_ac_params)
        )
        heli_params = AssetParams(
            num_ma=0, ground_ma=0, num_ua=0, ground_ua=0, num_h=1, ground_h=0
        )
        self.heli_model = test_heli_model(p=TaxiwayParams(assetparams=heli_params))

    def test_one_cycle(self):
        """Test that a single airborne asset will land, park at a gate, and take off."""
        for t in range(100):
            self.single_ac_model(t)
            ma_xy = self.single_ac_model.flows["location"].ma1.s.get("x", "y")
            if t == 5:  # test - does it make it to the landing spot?
                self.assertTrue(
                    all(
                        ma_xy == self.single_ac_model.flows["ground"].places["landing1"]
                    )
                )
            if t == 27:  # test - does it make it to the gate?
                self.assertTrue(
                    all(ma_xy == self.single_ac_model.flows["ground"].places["gate2"])
                )
            if t == 95:  # test - does it fly again?
                self.assertTrue(
                    all(ma_xy == self.single_ac_model.flows["ground"].places["air_loc"])
                )
                self.assertTrue(self.single_ac_model.fxns["ma1"].s.cycled)

    def test_one_cycle_plot(self):
        """Plot verification of test_one_cycle"""
        endresults, mdlhist = prop.nominal(self.single_ac_model)
        plot_course(self.single_ac_model, mdlhist, "ma1",
                    title="ma1 should cycle all the way around the taxiway")

    def test_avoid(self):
        """Test that a single airborne asset will land, park at a gate, and take off as
        desired"""
        for t in range(100):
            self.two_ac_model(t)
            if t == 77:
                self.assertAlmostEqual(
                    self.two_ac_model.flows["location"].ma1.s.speed, 1.0, places=3
                )
            if t >= 79:
                self.assertAlmostEqual(
                    self.two_ac_model.flows["location"].ma1.s.speed, 0.0, places=3
                )

    def test_avoid_plot(self):
        endresults, mdlhist = prop.nominal(self.two_ac_model)
        plot_tstep(
            self.two_ac_model,
            mdlhist,
            76,
            fxnattr="s.visioncov",
            locattr="speed",
            assets_to_label=["ma1"],
            title="Visioncov means ma1 slows down",
        )
        plot_tstep(
            self.two_ac_model,
            mdlhist,
            77,
            fxnattr="s.visioncov",
            locattr="speed",
            assets_to_label=["ma1"],
            title="Visioncov means ma1 stops",
        )
        plot_tstep(
            self.two_ac_model,
            mdlhist,
            78,
            fxnattr="s.visioncov",
            locattr="speed",
            assets_to_label=["ma1"],
            title="Visioncov means ma1 stops",
        )

    def test_lost_sight(self):
        endresults, mdlhist = prop.one_fault(self.two_ac_model, "ma1", "lost_sight")
        fhist = mdlhist.faulty
        # should end with crash attribute
        self.assertTrue(fhist.fxns.ma1.m.faults.crash[-1])
        self.assertTrue(fhist.fxns.ma2.m.faults.crash[-1])
        # should end
        x1, y1 = fhist.flows.location.ma1.s.x[-1], fhist.flows.location.ma1.s.y[-1]
        x2, y2 = fhist.flows.location.ma2.s.x[-1], fhist.flows.location.ma2.s.y[-1]
        self.assertEqual(x1, x2)
        self.assertEqual(y1, y2)

    def test_lost_sight_plot(self):
        endresults, mdlhist = prop.one_fault(self.two_ac_model, "ma1", "lost_sight")
        fhist = mdlhist.faulty
        plot_tstep(
            self.two_ac_model,
            fhist,
            76,
            fxnattr="s.visioncov",
            assets_to_label=["ma1"],
            title="ma1 approaches ma2 with no vision cone",
        )
        plot_tstep(
            self.two_ac_model,
            fhist,
            77,
            fxnattr="m.faults",
            assets_to_label=["ma1"],
            title="ma1 crashes into ma2 (should show faults)",
        )

    def test_heli_cycle(self):
        """Tests that a single helicopter will land and take off as desired"""
        for t in range(100):
            self.heli_model(t)
            heli_xy = self.heli_model.flows["location"].h1.s.get("x", "y")
            if t == 7:  # lands
                self.assertTrue(
                    all(heli_xy == self.heli_model.flows["ground"].places["helipad1"])
                )
            if t == 20:
                self.assertEqual(self.heli_model.fxns["h1"].m.mode, "park")
            if t >= 59:  # takes off
                self.assertTrue(
                    all(heli_xy == self.heli_model.flows["ground"].places["air_loc"])
                )
                self.assertTrue(self.heli_model.fxns["h1"].s.cycled)

    def test_heli_cycle_plot(self):
        endresults, mdlhist = prop.nominal(self.heli_model)
        plot_tstep(self.heli_model, mdlhist, 1, title="Empty landing spot")
        plot_tstep(self.heli_model, mdlhist, 7, title="h1 at landing spot")
        plot_tstep(self.heli_model, mdlhist, 59, title="h1 back in air")
        mdlhist.fxns.h1
        # mdlhist['flows']['Perc_location']['h1']

        # mdlhist['flows']['Perc_location']['h1']['stage']
        # mdlhist['flows']['Requests']['h1']
        # mdlhist['flows']['Requests']['h1']['atc_com']
        # mdlhist['h1']


if __name__ == "__main__":
    single_ac_params = AssetParams(num_ma=1, ground_ma=0, num_ua=0, ground_ua=0,
                                   num_h=0, ground_h=0)
    single_ac_model = test_aircraft_model(p=TaxiwayParams(assetparams=single_ac_params))
    single_ac_model.copy()

    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(AssetTests, plottests=False))
    runner.run(suite_for_plots(AssetTests, plottests=True))

    # test_mdl = test_heli_model(params=gen_params(num_ma=0, ground_ma=0, num_ua=0, ground_ua=0, num_h=1, ground_h=0))
    # test_mdl = test_aircraft_model(params=gen_params(num_ma=2, ground_ma=1, num_ua=0, ground_ua=0, num_h=0, ground_h=0))
    # endresults, mdlhist = prop.nominal(test_mdl)

    # plot_course(test_mdl, mdlhist, "h1")

    # plot_tstep(test_mdl, mdlhist, 12, fxnattr="mode", locattr="mode", perc_locattr="stage")
    # plot_tstep(test_mdl, mdlhist, 12, fxnattr="mode", locattr="mode", perc_locattr="stage", show_area_allocation=True)
    # plot_tstep(test_mdl, mdlhist, 12, asset_assignment=True)

    # plot_course(test_mdl, mdlhist, "ma1", showtimes=5)
    # plot_course(test_mdl, mdlhist, "ma1")
