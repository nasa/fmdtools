#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for model calculation correctness.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “Fault Model Design tools - fmdtools version 2” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import copy
import unittest

import networkx as nx
import numpy as np

from fmdtools.analyze.graph.base import Graph
from fmdtools.define.architecture.function import ExFxnArch, FunctionArchitectureGraph


def largest_surviving_component(graph, nodes):
    """Find the largest surviving component independently by breadth-first search."""
    unseen = set(nodes)
    largest = 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            reached = unseen.intersection(graph.neighbors(node))
            unseen.difference_update(reached)
            stack.extend(reached)
        largest = max(largest, size)
    return largest


def reference_coefficient(graph, trials, seed):
    graph = graph.to_undirected()
    nodes = list(graph)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(trials):
        order = rng.choice(len(nodes), len(nodes), replace=False)
        sizes = [
            largest_surviving_component(graph, [nodes[i] for i in order[k:]])
            for k in range(len(nodes))
        ]
        values.append((200 * sum(sizes) - 100 * sizes[0]) / len(nodes) ** 2)
    return sum(values) / len(values)


class TestDisconnectedGraphRobustness(unittest.TestCase):
    def test_edgeless_graphs_include_the_entire_removal_curve(self):
        for size in (1, 2, 4, 8):
            for trials in (1, 7):
                with self.subTest(size=size, trials=trials):
                    graph = nx.empty_graph(size)
                    expected = 100 * (2 * size - 1) / size**2
                    self.assertAlmostEqual(
                        Graph(graph, check_info=False).calc_robustness_coefficient(
                            trials=trials, seed=5
                        ),
                        expected,
                    )

    def test_disconnected_graphs_match_full_order_surviving_subgraphs(self):
        graphs = [
            nx.disjoint_union(nx.complete_graph(3), nx.empty_graph(2)),
            nx.disjoint_union(nx.path_graph(3), nx.path_graph(4)),
            nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(2)),
        ]
        for graph in graphs:
            for seed in (1, 5, 19):
                for trials in (1, 9):
                    with self.subTest(
                        nodes=len(graph),
                        edges=len(graph.edges),
                        seed=seed,
                        trials=trials,
                    ):
                        actual = Graph(
                            graph, check_info=False
                        ).calc_robustness_coefficient(trials, seed)
                        self.assertAlmostEqual(
                            actual,
                            reference_coefficient(graph, trials, seed),
                            places=12,
                        )

    def test_node_order_labels_and_directed_inputs_preserve_original_graph(self):
        base = nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2))
        labels = {0: "a", 1: ("b", 1), 2: 17, 3: ("isolated",), 4: "other"}
        base = nx.relabel_nodes(base, labels)
        for order in (list(base), list(base)[::-1]):
            for directed in (False, True):
                graph = nx.DiGraph() if directed else nx.Graph()
                graph.add_nodes_from((node, {"label": str(node)}) for node in order)
                graph.add_edges_from(base.edges)
                before = copy.deepcopy(graph)
                with self.subTest(order=order, directed=directed):
                    result = Graph(graph, check_info=False).calc_robustness_coefficient(
                        11, 13
                    )
                    self.assertAlmostEqual(
                        result, reference_coefficient(graph, 11, 13), places=12
                    )
                    self.assertTrue(nx.utils.graphs_equal(graph, before))
                    self.assertEqual(list(graph), list(before))

    def test_connected_graph_results_and_seeded_repeatability_are_unchanged(self):
        for graph in (
            nx.path_graph(5),
            nx.star_graph(4),
            nx.cycle_graph(5),
            nx.complete_graph(5),
        ):
            with self.subTest(edges=list(graph.edges)):
                wrapped = Graph(graph, check_info=False)
                value = wrapped.calc_robustness_coefficient(8, 42)
                self.assertEqual(value, wrapped.calc_robustness_coefficient(8, 42))
                self.assertAlmostEqual(
                    value, reference_coefficient(graph, 8, 42), places=12
                )
        self.assertEqual(
            Graph(nx.complete_graph(5), check_info=False).calc_robustness_coefficient(
                8, 42
            ),
            100.0,
        )

    def test_model_graph_uses_all_nodes_after_connections_are_lost(self):
        model = ExFxnArch()
        graph = FunctionArchitectureGraph(model)
        graph.g.remove_edges_from(list(graph.g.edges))
        size = len(graph.g)
        self.assertGreater(size, 1)
        expected = 100 * (2 * size - 1) / size**2
        self.assertAlmostEqual(graph.calc_robustness_coefficient(3, 7), expected)
        self.assertEqual(len(graph.g), size)
        self.assertEqual(len(graph.g.edges), 0)


if __name__ == "__main__":
    unittest.main()
