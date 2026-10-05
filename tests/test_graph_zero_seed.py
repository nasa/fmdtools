#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault summaries and model calculations.

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
from unittest.mock import patch

import networkx as nx
import numpy as np

from fmdtools.analyze.graph.base import Graph
from fmdtools.define.architecture.function import ExFxnArch, FunctionArchitectureGraph


def reference_robustness(graph, trials, seed):
    graph = graph.to_undirected()
    nodes = list(graph)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        order = rng.choice(len(nodes), len(nodes), replace=False)
        sizes = []
        for start in range(len(nodes)):
            unseen = {nodes[i] for i in order[start:]}
            largest = 0
            while unseen:
                stack, count = [unseen.pop()], 0
                while stack:
                    node = stack.pop()
                    count += 1
                    reached = unseen.intersection(graph.neighbors(node))
                    unseen.difference_update(reached)
                    stack.extend(reached)
                largest = max(largest, count)
            sizes.append(largest)
        scores.append((200 * sum(sizes) - 100 * sizes[0]) / len(nodes) ** 2)
    return sum(scores) / len(scores)


class TestGraphZeroSeed(unittest.TestCase):
    def test_python_and_numpy_zero_seeds_reach_the_random_generator(self):
        for seed in (0, np.int64(0), np.uint32(0), 7):
            with self.subTest(seed_type=type(seed).__name__, seed=seed):
                graph = nx.path_graph(7)
                expected = reference_robustness(graph, 7, seed)
                with patch(
                    "fmdtools.analyze.graph.base.np.random.default_rng",
                    wraps=np.random.default_rng,
                ) as factory:
                    actual = Graph(graph, check_info=False).calc_robustness_coefficient(
                        7, seed
                    )
                    factory.assert_called_once_with(seed=seed)
                self.assertAlmostEqual(actual, expected, places=12)

    def test_zero_seed_curves_match_reference_and_repeat_for_graph_families(self):
        graphs = (
            nx.path_graph(7),
            nx.star_graph(6),
            nx.cycle_graph(7),
            nx.disjoint_union(nx.path_graph(3), nx.path_graph(4)),
        )
        for graph in graphs:
            for trials in (1, 9):
                with self.subTest(edges=list(graph.edges), trials=trials):
                    wrapped = Graph(graph, check_info=False)
                    expected = reference_robustness(graph, trials, 0)
                    first = wrapped.calc_robustness_coefficient(trials, 0)
                    second = wrapped.calc_robustness_coefficient(trials, 0)
                    self.assertEqual(first, second)
                    self.assertAlmostEqual(first, expected, places=12)

    def test_default_false_and_none_keep_unseeded_initialization(self):
        for supplied in ({}, {"seed": False}, {"seed": None}):
            with self.subTest(supplied=supplied):
                generator = np.random.default_rng(31)
                with patch(
                    "fmdtools.analyze.graph.base.np.random.default_rng",
                    return_value=generator,
                ) as factory:
                    result = Graph(
                        nx.complete_graph(4), check_info=False
                    ).calc_robustness_coefficient(trials=2, **supplied)
                    factory.assert_called_once_with()
                self.assertEqual(result, 100.0)

    def test_directed_model_graphs_are_reproducible_and_not_modified(self):
        graph = FunctionArchitectureGraph(ExFxnArch())
        before = copy.deepcopy(graph.g)
        expected = reference_robustness(graph.g, 11, 0)
        actual = graph.calc_robustness_coefficient(trials=11, seed=0)
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(
            actual, graph.calc_robustness_coefficient(trials=11, seed=np.int64(0))
        )
        self.assertTrue(nx.utils.graphs_equal(graph.g, before))
        self.assertEqual(list(graph.g), list(before))


if __name__ == "__main__":
    unittest.main()
