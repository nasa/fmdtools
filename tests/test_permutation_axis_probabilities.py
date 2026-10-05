#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for permutations.

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

import itertools
import unittest
from collections import Counter

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    Rand,
    calc_prob_for_permuted,
    calc_prob_for_shuffle_permutation,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation

SOURCE = np.array([[0, 1], [0, 1], [2, 3]])


class LayoutState(State):
    layout: np.array = SOURCE.copy()
    layout_update = ("permutation", (SOURCE,))


class LayoutRand(Rand):
    s: LayoutState = LayoutState()


class ScoreState(State):
    score: np.float64 = 0.0


class LayoutFunction(Function):
    container_r = LayoutRand
    container_s = ScoreState

    def dynamic_behavior(self):
        self.s.score += (self.r.s.layout * np.arange(6).reshape(3, 2)).sum()

    def classify(self, **kwargs):
        return {"score": self.s.score}


class TestPermutationAxisProbabilities(unittest.TestCase):
    def test_probabilities_match_exhaustive_shared_axis_permutations(self):
        sources = [
            np.array([0, 1, 2]),
            np.array([0, 0, 1]),
            SOURCE,
            np.array([[0, 0, 1], [2, 2, 3]]),
            np.ones((3, 2)),
            np.arange(12).reshape(2, 3, 2),
        ]
        for source in sources:
            for axis in range(-source.ndim, source.ndim):
                with self.subTest(shape=source.shape, axis=axis):
                    counts = Counter(
                        tuple(np.take(source, order, axis=axis).ravel())
                        for order in itertools.permutations(range(source.shape[axis]))
                    )
                    total = sum(counts.values())
                    masses = []
                    for key, count in counts.items():
                        value = np.array(key).reshape(source.shape)
                        expected = count / total
                        for method in ("permutation", "shuffle"):
                            self.assertAlmostEqual(
                                get_prob_for_rand(value, method, source, axis),
                                expected,
                                13,
                            )
                        self.assertAlmostEqual(
                            calc_prob_for_shuffle_permutation(value, source, axis),
                            expected,
                            13,
                        )
                        masses.append(
                            get_prob_for_rand(value, "permutation", source, axis)
                        )
                    self.assertAlmostEqual(sum(masses), 1.0, 13)

    def test_invalid_outputs_have_zero_mass(self):
        source = np.array([[1, 2], [3, 4], [5, 6]])
        for value in (
            source.ravel(),
            source[:-1],
            np.tile(source[:1], (3, 1)),
            np.array([[1, 4], [3, 2], [5, 6]]),
        ):
            with self.subTest(value=value.tolist()):
                self.assertEqual(
                    calc_prob_for_shuffle_permutation(value, source, 0), 0.0
                )
                self.assertEqual(get_prob_for_rand(value, "permutation", source), 0.0)
        self.assertEqual(calc_prob_for_shuffle_permutation([1, 1, 1], [1, 2, 3]), 0.0)

    def test_empty_integer_and_variadic_inputs_preserve_values(self):
        for shape in ((0,), (2, 0), (0, 3), (1,), (1, 1)):
            source = np.zeros(shape, dtype=int)
            with self.subTest(shape=shape):
                self.assertEqual(get_prob_for_rand(source, "permutation", source), 1.0)
        self.assertAlmostEqual(get_prob_for_rand([2, 0, 1], "permutation", 3), 1 / 6)
        self.assertAlmostEqual(get_pfunc_for_dist("permutation", 3)(2, 0, 1), 1 / 6)
        source = SOURCE[:, ::-1].copy()
        source.setflags(write=False)
        values = source[::-1]
        self.assertAlmostEqual(get_prob_for_rand(values, "permutation", source), 1 / 3)
        np.testing.assert_array_equal(source, SOURCE[:, ::-1])
        self.assertAlmostEqual(
            calc_prob_for_shuffle_permutation([99], 3, check_valid=False), 1 / 6
        )

    def test_repeated_strings_nan_rows_and_legacy_flattening(self):
        for source in (
            np.array(["a", "a", "b"]),
            np.array([[np.nan, 1.0], [np.nan, 1.0], [3.0, 4.0]]),
        ):
            with self.subTest(source=source.tolist()):
                self.assertAlmostEqual(
                    get_prob_for_rand(source[::-1], "permutation", source), 1 / 3
                )
        self.assertAlmostEqual(
            calc_prob_for_permuted(np.arange(4).reshape(2, 2)), 1 / 24
        )
        self.assertAlmostEqual(
            calc_prob_for_permuted(np.arange(6).reshape(3, 2), 0), 1 / 6
        )

    def test_tracked_draws_preserve_generator_copy_and_reset(self):
        rand = LayoutRand(seed=17, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(17)
        for _ in range(3):
            rand.set_rand_state("layout", "permutation", SOURCE)
            np.testing.assert_array_equal(rand.s.layout, reference.permutation(SOURCE))
            self.assertAlmostEqual(rand.probs[-1], 1 / 3)
            self.assertEqual(rand.gen_state(), reference.bit_generator.state)
        self.assertAlmostEqual(rand.return_probdens(), 1 / 27)
        clone = rand.copy()
        for state in (rand, clone):
            state.set_rand_state("layout", "permutation", SOURCE)
        np.testing.assert_array_equal(rand.s.layout, clone.s.layout)
        rand.reset()
        rand.set_rand_state("layout", "permutation", SOURCE)
        np.testing.assert_array_equal(
            rand.s.layout, np.random.default_rng(17).permutation(SOURCE)
        )
        self.assertAlmostEqual(rand.return_probdens(), 1 / 3)

    def test_simulation_records_nonzero_layout_probabilities(self):
        model = LayoutFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        layouts = np.array([rng.permutation(SOURCE) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.layout"], layouts)
        np.testing.assert_allclose(
            history["r.probdens"], np.full(len(history.time), 1 / 3)
        )
        increments = (layouts[1:] * np.arange(6).reshape(3, 2)).sum(axis=(1, 2))
        expected = np.r_[0.0, np.cumsum(increments)]
        np.testing.assert_allclose(history["s.score"], expected)
        self.assertEqual(result["tend.classify.score"], expected[-1])


if __name__ == "__main__":
    unittest.main()
