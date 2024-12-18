#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiFlow demo model (for testing and demos).

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

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.define.container.state import State


class LocationState(State):
    x: float = 0.0
    y: float = 0.0


class Communications(CommsFlow):
    container_s = LocationState


class Location(MultiFlow):
    container_s = LocationState


class MoveParam(Parameter):
    x_up: float = 0.0
    y_up: float = 0.0


class Mover(Function):

    __slots__ = ('communications', 'location', 'internal_info', 'loc')
    container_p = MoveParam
    flow_communications = Communications
    flow_location = Location

    def init_block(self, **kwargs):
        self.internal_info = self.communications.create_comms(self.name)
        self.loc = self.location.create_local(self.name)

    def dynamic_behavior(self, time):
        # move
        self.loc.s.inc(x=self.p.x_up, y=self.p.y_up)
        # the inbox should be cleared each timestep to allow new messages
        self.internal_info.clear_inbox()

    def static_behavior(self, time):
        # recieve messages
        self.internal_info.receive()
        # communicate
        if self.p.x_up != 0.0:
            self.internal_info.s.x = self.loc.s.x
            self.internal_info.send("all", "x")
        elif self.p.y_up != 0.0:
            self.internal_info.s.y = self.loc.s.y
            self.internal_info.send("all", "y")

    def find_classification(self, scen, fxnhist):
        return {"last_x": self.loc.s.x, "min_x": fxnhist.faulty.location.get(self.name).x}


class Coordinator(Function):

    __slots__ = ('communications', 'coord_view')
    flow_communications = Communications

    def init_block(self, **kwargs):
        self.coord_view = self.communications.create_comms(self.name,
                                                           ports=["mover_1", "mover_2"])

    def dynamic_behavior(self, time):
        self.coord_view.clear_inbox()

    def static_behavior(self, time):
        self.coord_view.receive()
        self.coord_view.update("y", to_update="local", to_get="mover_1")
        self.coord_view.update("x", to_update="local", to_get="mover_2")


class ExModel(FunctionArchitecture):
    __slots__ = ()
    default_sp = dict(end_time=10)

    def init_architecture(self, **kwargs):

        self.add_flow("communications", Communications)
        self.add_flow("location", Location)
        self.add_fxn("mover_1", Mover, "communications", "location", p={"x_up": 1.0})
        self.add_fxn("mover_2", Mover, "communications", "location", p={"y_up": 1.0})

        self.add_fxn("coordinator", Coordinator, "communications")


if __name__ == '__main__':
    mdl = ExModel()
    mdl.flows["communications"].mover_1.s.x = 25
    mdl.flows["communications"].mover_1.send("mover_2")
    from fmdtools.sim import propagate
    from fmdtools.analyze.graph.base import Graph

    g = mdl.flows['communications'].create_graph(role_nodes=['local'],
                                                 roles_to_connect=["locals"],
                                                 recursive=True,
                                                 with_root=False)

    g2 = Graph(g)
    g2.draw()

    g = mdl.flows['communications'].create_graph(role_nodes=['local'],
                                                 recursive=True)

    g2 = Graph(g)
    g2.draw()

    from fmdtools.define.flow.multiflow import MultiFlowGraph
    from fmdtools.define.flow.commsflow import CommsFlowGraph
    MultiFlowGraph(mdl.flows['communications']).draw()
    CommsFlowGraph(mdl.flows['communications']).draw()

    # endres, mdlhist = propagate.nominal(mdl, desired_result='graph.flows.communications')

    # endres.graph.flows.communications.set_node_labels(title='id', subtext='s')
    # endres.graph.flows.communications.draw()
    # endres.graph.flows.communications.draw_graphviz()
