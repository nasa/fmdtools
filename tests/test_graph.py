#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing some different graph plotting methods.

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

from examples.pump.ex_pump import Pump
from examples.rover.rover_model import Rover
from fmdtools.define.architecture.function import FunctionArchitectureGraph, FunctionArchitectureFxnGraph
from fmdtools.define.architecture.function import FunctionArchitectureFlowGraph, FunctionArchitectureTypeGraph
from fmdtools.analyze.common import suite_for_plots
from fmdtools.sim import propagate

import unittest


class ModelGraphTests(unittest.TestCase):
    def setUp(self):
        self.mdl = Pump()
        self.rvr = Rover()

    def test_modelgraph_plot(self):
        a = FunctionArchitectureGraph(self.mdl)
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()

        b = FunctionArchitectureGraph(self.rvr)
        b.set_exec_order(self.rvr, next_edges={"nx_edge_color": "red"})
        b.draw(title="Should show Order, timestep, and dynamic properties of"
               + " FunctionArchitectureGraph with red arrows for next")

    def test_fxngraph_plot(self):
        a = FunctionArchitectureFxnGraph(self.mdl)
        # a.set_edge_labels(title='label', subtext='flows')
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()

        b = FunctionArchitectureFxnGraph(self.rvr)
        b.set_exec_order(self.rvr)
        b.draw()

    def test_flowgraph_plot(self):
        a = FunctionArchitectureFlowGraph(self.mdl)
        # a.set_edge_labels(title='label', subtext='functions')
        a.draw()
        a.set_exec_order(self.mdl)
        a.draw()

        b = FunctionArchitectureFlowGraph(self.rvr)
        b.set_exec_order(self.rvr)
        b.draw(title="Should show Order, timestep, and dynamic properties of FlowGraph")

    def test_typegraph_plot(self):
        a = FunctionArchitectureTypeGraph(self.mdl)
        a.draw(title="Should show the Pump model Containing functions, which in turn"
               + " contain Signal, Water, Electricity Flows")

    def test_fault_plot(self):
        er, mh = propagate.one_fault(self.mdl, 'move_water', 'short', time=10,
                                     desired_result=['graph', 'endclass', 'endfaults'])
        er.graph.set_node_styles(degraded={}, faulty={})
        er.graph.draw(title="Should show Faults (color edges) as well as"
                              + "degradations (orange color)")
        degraded = {'node_color': 'green'}
        faulty = {'node_size': 1500, 'edgecolors': 'purple'}
        er.graph.set_node_styles(degraded=degraded, faulty=faulty)
        er.graph.draw(title="Should be identical but faulty nodes are large"
                              + " and have purple edges while degradations are green")

    def test_result_from_plot(self):
        des_res = ['graph', 'endclass', 'endfaults']
        er, hist = propagate.one_fault(self.mdl, 'move_water', 'short',
                                       time=10, track='all', desired_result=des_res)
        mg = FunctionArchitectureGraph(self.mdl)
        mg.draw_from(11, hist)
        mg.draw_graphviz_from(11, hist, disp=False)

# def test_move_nodes(self):
#    p = endresults.graph.move_nodes()


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(ModelGraphTests, plottests=False))
    runner.run(suite_for_plots(ModelGraphTests, plottests=True))

    mdl = Pump()
    des_res = ['graph', 'endclass', 'endfaults']
    endresults, mdlhist = propagate.one_fault(mdl, 'move_water', 'short', time=10,
                                              desired_result=des_res, track='all')

    # p = endresults.graph.move_nodes()
    # endresults.graph.set_node_styles(degraded={}, faulty={})
    # endresults.graph.set_node_labels(title='id', subtext='faults')
    # endresults.graph.draw()

    a = FunctionArchitectureTypeGraph(mdl)
    # a.draw_pyvis()
    # a.draw_from(10, mdlhist)
    # a.draw_from(50, mdlhist)

    # an = a.animate(mdlhist)
    # from IPython.display import HTML
    # HTML(an.to_jshtml())

    # p = endresults.graph.move_nodes()
