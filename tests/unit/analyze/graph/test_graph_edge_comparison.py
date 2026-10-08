#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for orientation-independent undirected graph comparison.

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

import unittest
from unittest.mock import patch

import networkx as nx

from fmdtools.analyze.graph.base import Graph
from fmdtools.define.architecture.function import FunctionArchitectureFxnGraph
from fmdtools_examples.water_pump.model_main import Pump


class TestGraphEdgeComparison(unittest.TestCase):
    """Compare topology without depending on undirected endpoint order."""

    def test_reordered_undirected_graphs_have_identical_structure(self):
        for graph_type in (nx.Graph, nx.MultiGraph):
            with self.subTest(graph_type=graph_type.__name__):
                left = graph_type()
                left.add_edges_from(
                    [("a", "b"), ("b", "c"), ("c", "c")], label="connection"
                )
                left.nodes["a"]["state"] = "nominal"
                if left.is_multigraph():
                    left.add_edge("a", "b", label="connection")
                right = graph_type()
                right.add_nodes_from(reversed(list(left.nodes(data=True))))
                right.add_edges_from(
                    (v, u, data)
                    for u, v, data in reversed(list(left.edges(data=True)))
                )
                self.assertTrue(nx.utils.graphs_equal(left, right))
                before_left, before_right = left.copy(), right.copy()
                result = Graph(left, check_info=False).compare_with(
                    Graph(right, check_info=False)
                )
                for key in (
                    "nodes_added", "nodes_removed", "edges_added", "edges_removed"
                ):
                    self.assertEqual(result[key], set())
                self.assertEqual(result["structure_similarity"], 1.0)
                self.assertEqual(result["summary_this"], result["summary_other"])
                self.assertTrue(nx.utils.graphs_equal(left, before_left))
                self.assertTrue(nx.utils.graphs_equal(right, before_right))

    def test_edge_matching_does_not_require_orderable_node_labels(self):
        left = nx.Graph([(1, "middle"), ("middle", ("sink", 2))])
        right = nx.Graph([(("sink", 2), "middle"), ("middle", 1)])
        # Greedy modularity independently requires comparable node labels.
        with patch.object(Graph, "calc_modularity", return_value=0.0):
            result = Graph(left, check_info=False).compare_with(
                Graph(right, check_info=False)
            )
        self.assertEqual(result["edges_added"], set())
        self.assertEqual(result["edges_removed"], set())
        self.assertEqual(result["structure_similarity"], 1.0)

    def test_real_changes_keep_source_edge_tuples_and_correct_similarity(self):
        left = nx.Graph()
        left.add_nodes_from(["a", "b", "c", "removed"])
        left.add_edges_from([("a", "b"), ("b", "c")])
        right = nx.Graph()
        right.add_nodes_from(["added", "c", "b", "a"])
        right.add_edges_from([("b", "a"), ("added", "c")])
        before = Graph(left, check_info=False)
        after = Graph(right, check_info=False)
        result = before.compare_with(after)
        self.assertEqual(result["nodes_added"], {"added"})
        self.assertEqual(result["nodes_removed"], {"removed"})
        self.assertEqual(result["edges_added"], {("added", "c")})
        self.assertEqual(result["edges_removed"], {("b", "c")})
        # Three of five nodes and one of three edges are shared.
        self.assertAlmostEqual(result["structure_similarity"], 7 / 15)
        reverse = after.compare_with(before)
        self.assertEqual(reverse["edges_added"], result["edges_removed"])
        self.assertEqual(reverse["edges_removed"], result["edges_added"])
        self.assertEqual(
            reverse["structure_similarity"], result["structure_similarity"]
        )

    def test_directed_edges_keep_their_orientation(self):
        for graph_type in (nx.DiGraph, nx.MultiDiGraph):
            with self.subTest(graph_type=graph_type.__name__):
                left = graph_type([("a", "b"), ("b", "c")])
                right = graph_type([("c", "b"), ("a", "b")])
                result = Graph(left, check_info=False).compare_with(
                    Graph(right, check_info=False)
                )
                self.assertEqual(result["edges_added"], {("c", "b")})
                self.assertEqual(result["edges_removed"], {("b", "c")})
                # All nodes and one of three directed edges are shared.
                self.assertAlmostEqual(result["structure_similarity"], 2 / 3)

    def test_reordered_pump_function_graph_has_no_structural_changes(self):
        model_graph = FunctionArchitectureFxnGraph(Pump())
        reordered = nx.Graph()
        reordered.add_nodes_from(reversed(list(model_graph.g.nodes(data=True))))
        reordered.add_edges_from(
            (v, u, data)
            for u, v, data in reversed(list(model_graph.g.edges(data=True)))
        )
        self.assertTrue(nx.utils.graphs_equal(model_graph.g, reordered))
        result = model_graph.compare_with(Graph(reordered))
        self.assertEqual(result["nodes_added"], set())
        self.assertEqual(result["nodes_removed"], set())
        self.assertEqual(result["edges_added"], set())
        self.assertEqual(result["edges_removed"], set())
        self.assertEqual(result["structure_similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
