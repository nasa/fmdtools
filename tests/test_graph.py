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
import networkx as nx

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
                                     to_return=['graph', 'classify', 'faults'])
        graph = er.faulty.tend.graph
        graph.set_node_styles(degraded={}, faulty={})
        graph.draw(title="Should show Faults (color edges) as well as"
                   + "degradations (orange color)")
        degraded = {'node_color': 'green'}
        faulty = {'node_size': 1500, 'edgecolors': 'purple'}
        graph.set_node_styles(degraded=degraded, faulty=faulty)
        graph.draw(title="Should be identical but faulty nodes are large"
                   + " and have purple edges while degradations are green")

    def test_result_from_plot(self):
        des_res = ['graph', 'classify', 'faults']
        er, hist = propagate.one_fault(self.mdl, 'move_water', 'short',
                                       time=10, track='all', to_return=des_res)
        mg = FunctionArchitectureGraph(self.mdl)
        mg.draw_from(11, hist)
        mg.draw_graphviz_from(11, hist, disp=False)

    def test_plot_centrality(self):
        """Test centrality visualization with predefined and custom metrics."""
        # Test predefined metrics
        for metric in ['betweenness', 'closeness', 'degree']:
            mg = FunctionArchitectureGraph(self.mdl)
            mg.set_pos(auto='spring')
            fig = mg.plot_node_percentiles(metric=metric)
            self.assertIsNotNone(fig)
            # Verify node groups were created
            self.assertGreater(len(mg.node_groups), 0)

        # Test custom metric function
        mg = FunctionArchitectureGraph(self.mdl)
        mg.set_pos(auto='spring')
        custom_metric = lambda g: nx.pagerank(g)
        fig = mg.plot_node_percentiles(metric=custom_metric, title='PageRank')
        self.assertIsNotNone(fig)

        # Test custom quartiles
        mg = FunctionArchitectureGraph(self.mdl)
        mg.set_pos(auto='spring')
        fig = mg.plot_node_percentiles(metric='degree', quartiles=[0, 30, 70, 100])
        self.assertIsNotNone(fig)

    def test_summary(self):
        """Test graph summary statistics."""
        mg = FunctionArchitectureGraph(self.mdl)
        summary = mg.summary()

        # Check all expected keys are present
        expected_keys = ['num_nodes', 'num_edges', 'density', 'is_connected',
                        'num_components', 'avg_degree', 'aspl', 'modularity']
        for key in expected_keys:
            self.assertIn(key, summary)

        # Check types and value constraints
        self.assertIsInstance(summary['num_nodes'], int)
        self.assertGreater(summary['num_nodes'], 0)

        self.assertIsInstance(summary['num_edges'], int)
        self.assertGreaterEqual(summary['num_edges'], 0)

        self.assertIsInstance(summary['density'], float)
        self.assertGreaterEqual(summary['density'], 0.0)
        self.assertLessEqual(summary['density'], 1.0)

        self.assertIsInstance(summary['is_connected'], bool)

        self.assertIsInstance(summary['num_components'], int)
        self.assertGreater(summary['num_components'], 0)

        self.assertIsInstance(summary['avg_degree'], float)
        self.assertGreaterEqual(summary['avg_degree'], 0.0)

        # ASPL should be float if connected, None if not
        if summary['is_connected']:
            self.assertIsInstance(summary['aspl'], float)
            self.assertGreater(summary['aspl'], 0.0)
        else:
            self.assertIsNone(summary['aspl'])

        self.assertIsInstance(summary['modularity'], float)

    def test_compare_with(self):
        """Test graph comparison between model variants."""
        pump_graph = FunctionArchitectureGraph(self.mdl)
        rover_graph = FunctionArchitectureGraph(self.rvr)

        comparison = pump_graph.compare_with(rover_graph)

        # Check all expected keys are present
        expected_keys = ['nodes_added', 'nodes_removed', 'edges_added', 'edges_removed',
                        'structure_similarity', 'summary_this', 'summary_other']
        for key in expected_keys:
            self.assertIn(key, comparison)

        # Check types
        self.assertIsInstance(comparison['nodes_added'], set)
        self.assertIsInstance(comparison['nodes_removed'], set)
        self.assertIsInstance(comparison['edges_added'], set)
        self.assertIsInstance(comparison['edges_removed'], set)
        self.assertIsInstance(comparison['structure_similarity'], float)
        self.assertIsInstance(comparison['summary_this'], dict)
        self.assertIsInstance(comparison['summary_other'], dict)

        # Check value constraints
        self.assertGreaterEqual(comparison['structure_similarity'], 0.0)
        self.assertLessEqual(comparison['structure_similarity'], 1.0)

        # Pump and Rover should have different structures
        self.assertLess(comparison['structure_similarity'], 1.0)
        # Should have some differences (nodes or edges)
        has_differences = (len(comparison['nodes_added']) > 0 or
                          len(comparison['nodes_removed']) > 0 or
                          len(comparison['edges_added']) > 0 or
                          len(comparison['edges_removed']) > 0)
        self.assertTrue(has_differences)

# def test_move_nodes(self):
#    p = endresults.graph.move_nodes()


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(ModelGraphTests, plottests=False))
    runner.run(suite_for_plots(ModelGraphTests, plottests=True))

    mdl = Pump()
    des_res = ['graph', 'classify', 'endfaults']
    endresults, mdlhist = propagate.one_fault(mdl, 'move_water', 'short', time=10,
                                              to_return=des_res, track='all')

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
