#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independently permuted array slices.

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

from collections import Counter
import itertools
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import (
    Rand,
    calc_prob_for_permuted,
    get_pfunc_for_dist,
    get_prob_for_rand,
)
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


SOURCE = np.array([[0, 0, 1], [2, 3, 4]])


def enumerated_outputs(source, axis):
    """Count indexed shuffles independently of the probability implementation."""
    source = np.asarray(source)
    moved = source.ravel()[None, :] if axis is None else np.moveaxis(source, axis, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    outputs = Counter()
    orders = [list(itertools.permutations(range(len(row)))) for row in rows]
    for indices in itertools.product(*orders):
        shuffled = np.array([row[list(order)] for row, order in zip(rows, indices)])
        shuffled = shuffled.reshape(moved.shape)
        result = (
            shuffled.reshape(source.shape)
            if axis is None
            else np.moveaxis(shuffled, -1, axis)
        )
        outputs[tuple(result.ravel())] += 1
    return outputs


class LayoutState(State):
    layout: np.array = SOURCE.copy()
    layout_update = ("permuted", (SOURCE,))


class LayoutRand(Rand):
    s: LayoutState = LayoutState()


class ScoreState(State):
    score: float = 0.0


class LayoutFunction(Function):
    container_r = LayoutRand
    container_s = ScoreState

    def dynamic_behavior(self):
        self.s.score += (self.r.s.layout * np.arange(6).reshape(2, 3)).sum()

    def classify(self, **kwargs):
        return {"score": self.s.score}


class TestPermutedSliceProbabilities(unittest.TestCase):
    def test_small_outcomes_match_exhaustive_index_permutations(self):
        for source in (np.arange(6).reshape(2, 3), SOURCE, np.ones((2, 2), dtype=int)):
            for axis in (None, 0, 1, -1):
                with self.subTest(source=source.tolist(), axis=axis):
                    counts = enumerated_outputs(source, axis)
                    count_all = sum(counts.values())
                    pfunc = get_pfunc_for_dist("permuted", source, axis)
                    mass_sum = 0.0
                    for key, count in counts.items():
                        values = np.array(key).reshape(source.shape)
                        expected = count / count_all
                        self.assertAlmostEqual(pfunc(values), expected, 13)
                        self.assertAlmostEqual(
                            calc_prob_for_permuted(values, axis), expected, 13
                        )
                        mass_sum += pfunc(values)
                    self.assertAlmostEqual(mass_sum, 1.0, 12)

    def test_wrong_multisets_and_wrong_slice_membership_have_zero_mass(self):
        source = np.arange(6).reshape(2, 3)
        different = source.copy()
        different[0, 0] = 99
        cross_slice = source.copy()
        cross_slice[0, 0], cross_slice[1, 0] = source[1, 0], source[0, 0]
        for axis in (None, 0, 1):
            with self.subTest(axis=axis):
                pfunc = get_pfunc_for_dist("permuted", source, axis)
                self.assertEqual(pfunc(different), 0.0)
                self.assertEqual(pfunc(source.ravel()), 0.0)
        self.assertEqual(get_prob_for_rand(cross_slice, "permuted", source, 1), 0.0)
        self.assertAlmostEqual(
            get_prob_for_rand(cross_slice, "permuted", source), 1 / 720, 13
        )

    def test_empty_singleton_variadic_and_output_buffer_arguments_preserve_inputs(self):
        for shape in ((0,), (2, 0), (0, 3), (1,), (1, 1, 1)):
            for axis in (None, 0, -1):
                with self.subTest(shape=shape, axis=axis):
                    source = np.zeros(shape, dtype=int)
                    self.assertEqual(calc_prob_for_permuted(source, axis), 1.0)
                    self.assertEqual(
                        get_prob_for_rand(source, "permuted", source, axis), 1.0
                    )
        original = SOURCE.copy()
        values = np.random.default_rng(11).permuted(original, axis=1)
        buffer = np.full_like(original, -99)
        pfunc = get_pfunc_for_dist("permuted", original, 1, buffer)
        self.assertAlmostEqual(pfunc(values), 1 / 18, 13)
        self.assertAlmostEqual(pfunc(*values), 1 / 18, 13)
        np.testing.assert_array_equal(original, SOURCE)
        np.testing.assert_array_equal(buffer, np.full_like(original, -99))
        self.assertEqual(get_prob_for_rand(1, "permuted"), 1.0)
        with self.assertRaises(IndexError):
            calc_prob_for_permuted(original, 2)

    def test_tracked_default_draws_keep_generator_state_copies_and_reset(self):
        rand = LayoutRand(seed=19, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(19)
        counts = enumerated_outputs(SOURCE, None)
        total = sum(counts.values())
        for _ in range(2):
            expected = reference.permuted(SOURCE)
            rand.set_rand_state("layout", "permuted", SOURCE)
            np.testing.assert_array_equal(rand.s.layout, expected)
            self.assertEqual(
                rand.rng.bit_generator.state, reference.bit_generator.state
            )
            self.assertAlmostEqual(
                rand.probs[-1], counts[tuple(expected.ravel())] / total, 13
            )
        clone = rand.copy()
        for state in (rand, clone):
            state.set_rand_state("layout", "permuted", SOURCE)
        np.testing.assert_array_equal(rand.s.layout, clone.s.layout)
        rand.reset()
        self.assertEqual(rand.probs, [])
        rand.set_rand_state("layout", "permuted", SOURCE)
        np.testing.assert_array_equal(
            rand.s.layout, np.random.default_rng(19).permuted(SOURCE)
        )
        np.testing.assert_array_equal(SOURCE, [[0, 0, 1], [2, 3, 4]])

    def test_real_simulation_records_permutations_and_complete_joint_masses(self):
        model = LayoutFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        values = np.array([reference.permuted(SOURCE) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.layout"], values)
        np.testing.assert_allclose(history["r.probdens"], np.full(4, 1 / 360))
        increments = (values[1:] * np.arange(6).reshape(2, 3)).sum(axis=(1, 2))
        totals = np.r_[0.0, np.cumsum(increments)]
        np.testing.assert_array_equal(history["s.score"], totals)
        self.assertEqual(result["tend.classify.score"], totals[-1])
        np.testing.assert_array_equal(model.r.s.layout, SOURCE)


if __name__ == "__main__":
    unittest.main()
