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

    def test_centrality_methods(self):
        """Test new centrality calculation methods."""
        mg = FunctionArchitectureGraph(self.mdl)
        
        # Test betweenness centrality
        bc = mg.calc_betweenness_centrality()
        self.assertIsInstance(bc, dict)
        self.assertGreater(len(bc), 0)
        
        # Test closeness centrality
        cc = mg.calc_closeness_centrality()
        self.assertIsInstance(cc, dict)
        self.assertGreater(len(cc), 0)
        
        # Test eigenvector centrality
        ec = mg.calc_eigenvector_centrality()
        self.assertIsInstance(ec, dict)
        self.assertGreater(len(ec), 0)

    def test_plot_centrality(self):
        """Test centrality visualization method."""
        mg = FunctionArchitectureGraph(self.mdl)
        mg.set_pos(auto='spring')
        
        # Test different centrality metrics
        for metric in ['betweenness', 'closeness', 'degree']:
            fig = mg.plot_centrality(metric=metric)
            self.assertIsNotNone(fig)

    def test_summary_method(self):
        """Test graph summary method."""
        mg = FunctionArchitectureGraph(self.mdl)
        summary = mg.summary()
        
        # Check all expected keys are present
        expected_keys = ['num_nodes', 'num_edges', 'density', 'is_connected',
                        'num_components', 'avg_degree', 'modularity']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check types
        self.assertIsInstance(summary['num_nodes'], int)
        self.assertIsInstance(summary['num_edges'], int)
        self.assertIsInstance(summary['density'], float)
        self.assertIsInstance(summary['is_connected'], bool)

    def test_compare_with_method(self):
        """Test graph comparison method."""
        mg1 = FunctionArchitectureGraph(self.mdl)
        mg2 = FunctionArchitectureGraph(self.mdl)

        # Compare identical graphs
        comparison = mg1.compare_with(mg2)
        self.assertEqual(comparison['structure_similarity'], 1.0)
        self.assertEqual(len(comparison['nodes_added']), 0)
        self.assertEqual(len(comparison['nodes_removed']), 0)

        # Test with faulty graph
        er, hist = propagate.one_fault(self.mdl, 'move_water', 'short',
                                       time=10, to_return=['graph'])
        mg_faulty = er.faulty.tend.graph
        comparison = mg1.compare_with(mg_faulty)
        self.assertIsInstance(comparison['structure_similarity'], float)
        self.assertGreaterEqual(comparison['structure_similarity'], 0.0)
        self.assertLessEqual(comparison['structure_similarity'], 1.0)

    def test_path_analysis_methods(self):
        """Test path finding methods."""
        mg = FunctionArchitectureGraph(self.rvr)
        nodes = list(mg.g.nodes())

        if len(nodes) >= 2:
            source, target = nodes[0], nodes[-1]

            # Test find_critical_paths
            paths = mg.find_critical_paths(source, target, k=3)
            self.assertIsInstance(paths, list)

            # Test find_all_simple_paths
            all_paths = mg.find_all_simple_paths(source, target, cutoff=5)
            self.assertIsInstance(all_paths, list)

            # Test with invalid nodes
            with self.assertRaises(ValueError):
                mg.find_critical_paths('nonexistent_node', target)

    def test_subgraph_extraction(self):
        """Test subgraph extraction method."""
        from fmdtools.analyze.graph.base import Graph
        mg = FunctionArchitectureGraph(self.rvr)
        nodes = list(mg.g.nodes())

        if len(nodes) > 0:
            # Test ego graph extraction
            center = nodes[0]
            subgraph = mg.extract_subgraph(center_node=center, radius=1)
            self.assertIsInstance(subgraph, Graph)
            self.assertGreater(subgraph.g.number_of_nodes(), 0)

            # Test with node list
            if len(nodes) >= 3:
                subgraph2 = mg.extract_subgraph(nodes=nodes[:3])
                self.assertEqual(subgraph2.g.number_of_nodes(), 3)

            # Test error handling
            with self.assertRaises(ValueError):
                mg.extract_subgraph(center_node='nonexistent_node')

    def test_resilience_score(self):
        """Test resilience scoring method."""
        mg = FunctionArchitectureGraph(self.mdl)

        # Test combined metric
        score = mg.calc_resilience_score(metric='combined')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

        # Test other metrics
        for metric in ['connectivity', 'redundancy', 'modularity']:
            score = mg.calc_resilience_score(metric=metric)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

        # Test invalid metric
        with self.assertRaises(ValueError):
            mg.calc_resilience_score(metric='invalid')

    def test_node_removal_impact(self):
        """Test node removal impact assessment."""
        mg = FunctionArchitectureGraph(self.mdl)
        nodes = list(mg.g.nodes())

        if len(nodes) > 0:
            node = nodes[0]
            impact = mg.assess_node_removal_impact(node)

            # Check all expected keys present
            expected_keys = ['disconnected_nodes', 'new_components',
                           'original_components', 'aspl_change', 'connectivity_loss']
            for key in expected_keys:
                self.assertIn(key, impact)

            # Check types
            self.assertIsInstance(impact['disconnected_nodes'], list)
            self.assertIsInstance(impact['new_components'], int)
            self.assertIsInstance(impact['original_components'], int)
            self.assertIsInstance(impact['connectivity_loss'], float)

            # Test error handling
            with self.assertRaises(ValueError):
                mg.assess_node_removal_impact('nonexistent_node')

    def test_vulnerable_nodes(self):
        """Test vulnerable node identification."""
        mg = FunctionArchitectureGraph(self.rvr)

        # Test combined metric
        vulnerable = mg.find_vulnerable_nodes(metric='combined', top_k=3)
        self.assertIsInstance(vulnerable, list)
        self.assertLessEqual(len(vulnerable), 3)

        # Check structure
        if len(vulnerable) > 0:
            self.assertIsInstance(vulnerable[0], tuple)
            self.assertEqual(len(vulnerable[0]), 2)

        # Test other metrics
        for metric in ['betweenness', 'degree', 'bridge']:
            vulnerable = mg.find_vulnerable_nodes(metric=metric, top_k=5)
            self.assertIsInstance(vulnerable, list)
            self.assertLessEqual(len(vulnerable), 5)

        # Test invalid metric
        with self.assertRaises(ValueError):
            mg.find_vulnerable_nodes(metric='invalid')

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
