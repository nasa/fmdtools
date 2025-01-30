#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated taxiway model.

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

from ATC import ATC
from asset import Aircraft, Helicopter
from common import (
    TaxiwayParams,
    Environment,
    Location,
    Requests,
    plot_course,
    plot_one_path,
    plot_tstep,
)

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.sim import propagate as prop
from fmdtools.analyze.history import History

import numpy as np


class taxiway_model(FunctionArchitecture):
    container_p = TaxiwayParams
    default_sp = dict(end_time=120)
    default_track = "all"

    def init_architecture(self, **kwargs):
        self.add_flow("ground", Environment, p=self.p)
        self.add_flow("location", Location)
        self.add_flow("requests", Requests)

        self.add_fxn("atc", ATC, "ground", "location", "requests", p=self.p)

        for Aname in [*self.p.assetparams.mas, *self.p.assetparams.uas]:
            self.add_fxn(
                Aname, Aircraft, "ground", "location", "requests", p=self.p.assetparams
            )

        for Hname in self.p.assetparams.hs:
            self.add_fxn(
                Hname,
                Helicopter,
                "ground",
                "location",
                "requests",
                p=self.p.assetparams,
            )

    def find_classification(self, scen, mdlhists):
        num_cycled = len(
            [f for f, fxn in self.fxns.items() if f != "atc" and fxn.s.cycled]
        )
        perc_cycled = num_cycled / (len(self.fxns) - 1)
        num_crashed = len(
            [
                f
                for f, fxn in self.fxns.items()
                if f != "atc" and fxn.m.has_fault("crash")
            ]
        )
        return {
            "num_cycled": num_cycled,
            "perc_cycled": perc_cycled,
            "num_crashed": num_crashed,
        }

def create_fault_scen_metrics(mdlhist):
    nomhist = History({'incorrect_fields':
             mdlhist.nominal.get_metric("i.incorrect", method=np.sum, axis=0),
             'assets_without_sight':
             mdlhist.nominal.get_metric("i.nosight", method=np.sum, axis=0),
             "unsafe_distances":
            mdlhist.nominal.get_metric("i.unsafe", method=np.sum, axis=0),
            "overbooked_locations":
            mdlhist.nominal.get_metric("i.overbooked", method=np.sum, axis=0),
            "incorrect_perception":
            mdlhist.nominal.get_metric("i.incorrect_perception", method=np.sum, axis=0),
            "duplicate_land_commands":
            mdlhist.nominal.get_metric("i.duplicate_land", method=np.sum, axis=0),
            "cycled_assets":
            mdlhist.nominal.get_metric('cycled', methodc=np.sum, axis=0),
            "degraded_fields":
            0*mdlhist.nominal.time,
            "faulty_functions":
            0*mdlhist.nominal.time,
            "time": mdlhist.nominal.time})
        
    faulthist = History({"incorrect_fields":
             mdlhist.faulty.get_metric("i.incorrect", method=np.sum, axis=0),
             'assets_without_sight':
             mdlhist.faulty.get_metric("i.nosight", method=np.sum, axis=0),
             "unsafe_distances":
            mdlhist.faulty.get_metric("i.unsafe", method=np.sum, axis=0),
            "overbooked_locations":
            mdlhist.faulty.get_metric("i.overbooked", method=np.sum, axis=0),
            "incorrect_perception":
            mdlhist.faulty.get_metric("i.incorrect_perception", method=np.sum, axis=0),
            "duplicate_land_commands":
            mdlhist.faulty.get_metric("i.duplicate_land", method=np.sum, axis=0),
            "cycled_assets": 
            mdlhist.faulty.get_metric('cycled', method=np.sum, axis=0),
            "degraded_fields":
            100*mdlhist.get_degraded_hist(*mdlhist.nominal.flows.keys())['total']/len(mdlhist.nominal.flows.keys()),
            "faulty_functions":
            mdlhist.get_faulty_hist(*mdlhist.nominal.fxns.nest().keys())['total']/len(mdlhist.nominal.fxns.nest().keys()),
            "time": mdlhist.faulty.time})
    
    ind_hist = History({"nominal": nomhist, "faulty": faulthist})
    return ind_hist
    

if __name__ == "__main__":
    import networkx as nx
    from fmdtools.define.flow.multiflow import MultiFlowGraph
    from fmdtools.define.flow.commsflow import CommsFlowGraph

    mdl = taxiway_model()

    mg = mdl.as_modelgraph()
    mg.draw()

    ground_args = {'include_glob':True, "include_containers": ['s'],
              'send_connections':{"asset_area":"asset_area", 
                                  "area_allocation":"area_allocation",
                                  "asset_assignment":"asset_assignment"}}
    req_args = {'include_glob':False, "ports_only":True}
    endresults, mdlhist = prop.sequence(mdl, faultseq={7:{"atc": ["wrong_land_command"]}, 9: {"ua2":["lost_sight"]}}, 
                                         desired_result={10:{"graph.flows.requests":(CommsFlowGraph, req_args)},
                                                         11:{"graph.flows.requests":(CommsFlowGraph, req_args),
                                                            "graph.flows.ground":(MultiFlowGraph, ground_args)},
                                                         19:{"graph.flows.requests":{'include_glob':False, "ports_only":True}},
                                                         20:["graph"], 120:"endclass"})
    ind_hist = create_fault_scen_metrics(mdlhist)
    fig, ax = ind_hist.plot_line("degraded_fields",
                                 "cycled_assets",
                                 "unsafe_distances",
                                 "assets_without_sight",
                                 "faulty_functions",
                                 "duplicate_land_commands", time_slice=[8, 10, 19])

    endresults, mdlhist = prop.one_fault(
        mdl,
        "ma3",
        "lost_sight",
        desired_result={
            93: {"graph.flows.location": {"include_glob": False}},
            110: {"graph.flows.location": {"include_glob": False}},
            20: ["graph"],
            120: ["graph", "endclass"],
        },
    )

    ind_hist = create_fault_scen_metrics(mdlhist)
    fig, ax = ind_hist.plot_line("incorrect_fields",
                                 "unsafe_distances",
                                 "overbooked_locations",
                                 "incorrect_perception", time_slice=0)

    mdlhist.nominal.get_metric("i.unsafe", metric=np.sum, axis=1)

    mdlhist.get_metric("i.overbooked", metric=np.sum, axis=0)

    mdl.flows["ground"].show_map()

    endresults, mdlhist = prop.nominal(mdl)
    a = mdlhist.fxns.get("ua3.s.visioncov")

    plot_course(mdl, mdlhist, "ua1")
    plot_course(mdl, mdlhist, "ua2")
    plot_course(mdl, mdlhist, "ua3")

    plot_tstep(mdl, mdlhist, 50, fxnattr="s.visioncov")
    plot_tstep(mdl, mdlhist, 75, fxnattr="s.visioncov")
    plot_tstep(mdl, mdlhist, 100, fxnattr="s.visioncov")
    plot_tstep(mdl, mdlhist, 110, fxnattr="s.visioncov")
    plot_tstep(mdl, mdlhist, 115, fxnattr="s.visioncov")

    vals = {"flows": {"location": {"ua2": {"s": {"x", "y", "speed", "mode", "stage"}}}}}
    mdlhist.plot_line(vals)

    uncycled = [
        f for f, fxn in mdlhist.items() if ("cycled" in fxn) and not fxn["cycled"][-1]
    ]



    endresults, mdlhist = prop.sequence(
        mdl,
        {8: {"atc": ["single_wrong_command"]}, 10: {"ua2": ["lost_sight"]}},
        {},
        desired_result={
            11: {"graph.flows.requests": {}},
            20: ["graph"],
            120: "endclass",
        },
    )
    plot_tstep(mdl, mdlhist.faulty, 110, title="Aircraft crashed")
    plot_tstep(mdl, mdlhist.faulty, 25, fxnattr="visioncov", areas_to_label=[])

    ind_hist = create_fault_scen_metrics(mdlhist)
    fig, ax = ind_hist.plot_line("degraded_fields",
                                 "cycled_assets",
                                 "unsafe_distances",
                                 "assets_without_sight",
                                 "faulty_functions", time_slice=[8, 10])

    endresults.get("t120p0")

    degraded = mdlhist.get_degraded_hist(*mdl.fxns, *mdl.flows)
    faulty = mdlhist.get_faulty_hist(*mdl.fxns)
    # pos = nx.spring_layout(mdl.bipartite)
    # rd.graph.result_from(mdl, reshists['faulty'], 10, pos=pos)
    # rd.graph.result_from(mdl, reshists['faulty'], 25, pos=pos)

    endresults.t11p0.graph.flows.requests.draw()
    endresults.t20p0.graph.draw()


