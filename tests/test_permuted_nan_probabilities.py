#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for permutations of arrays containing NaN values.

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

import itertools
import unittest
from collections import Counter

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation

SOURCE = np.array([[1.0, np.nan, np.nan], [2.0, 2.0, 3.0]])


def canonical(values):
    return tuple("nan" if np.isnan(value) else value for value in values.ravel())


def enumerated(source, axis):
    moved = source.ravel()[None, :] if axis is None else np.moveaxis(source, axis, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    counts = Counter()
    examples = {}
    for orders in itertools.product(
        *[list(itertools.permutations(range(len(row)))) for row in rows]
    ):
        shuffled = np.array(
            [row[list(order)] for row, order in zip(rows, orders)]
        ).reshape(moved.shape)
        values = (
            shuffled.reshape(source.shape)
            if axis is None
            else np.moveaxis(shuffled, -1, axis)
        )
        key = canonical(values)
        counts[key] += 1
        examples[key] = values
    return counts, examples


class LayoutState(State):
    layout: np.array = SOURCE.copy()
    layout_update = ("permuted", (SOURCE,))


class LayoutRand(Rand):
    s: LayoutState = LayoutState()


class SumState(State):
    score: np.float64 = 0.0


class PermutedFunction(Function):
    container_r = LayoutRand
    container_s = SumState

    def dynamic_behavior(self):
        self.s.score += np.nansum(self.r.s.layout * np.arange(6).reshape(2, 3))

    def classify(self, **kwargs):
        return {"score": self.s.score}


class TestPermutedNaNProbabilities(unittest.TestCase):
    def test_valid_outcomes_match_exhaustive_index_permutations(self):
        for source in (
            np.array([1.0, np.nan, np.nan]),
            SOURCE,
            np.full((2, 2), np.nan),
        ):
            for dtype in (np.float32, np.float64):
                original = source.astype(dtype)
                for axis in (None, *range(original.ndim), -1):
                    with self.subTest(shape=original.shape, dtype=dtype, axis=axis):
                        counts, examples = enumerated(original, axis)
                        total = sum(counts.values())
                        pfunc = get_pfunc_for_dist("permuted", original, axis)
                        for key, count in counts.items():
                            self.assertAlmostEqual(
                                pfunc(examples[key]), count / total, 13
                            )
                        self.assertAlmostEqual(
                            sum(pfunc(value) for value in examples.values()), 1.0, 13
                        )

    def test_changed_nan_counts_values_and_slice_membership_are_rejected(self):
        missing = SOURCE.copy()
        missing[0, 1] = 1.0
        extra = SOURCE.copy()
        extra[1, 0] = np.nan
        changed = SOURCE.copy()
        changed[1, 0] = 99.0
        moved = SOURCE.copy()
        moved[0, 1], moved[1, 0] = SOURCE[1, 0], SOURCE[0, 1]
        for values in (missing, extra, changed, SOURCE.ravel()):
            for axis in (None, 0, 1):
                with self.subTest(values=values, axis=axis):
                    self.assertEqual(
                        get_prob_for_rand(values, "permuted", SOURCE, axis), 0.0
                    )
        self.assertEqual(get_prob_for_rand(moved, "permuted", SOURCE, 1), 0.0)
        self.assertAlmostEqual(get_prob_for_rand(moved, "permuted", SOURCE), 1 / 180)

    def test_readonly_noncontiguous_buffer_and_finite_controls(self):
        original = SOURCE[:, ::-1].copy()
        original.setflags(write=False)
        values = np.random.default_rng(7).permuted(original, axis=-1)
        buffer = np.full_like(original, -99.0)
        pfunc = get_pfunc_for_dist("permuted", original, -1, buffer)
        self.assertAlmostEqual(pfunc(values[:, ::-1]), 1 / 9)
        self.assertAlmostEqual(pfunc(*values), 1 / 9)
        np.testing.assert_array_equal(buffer, np.full_like(original, -99.0))
        np.testing.assert_array_equal(original, SOURCE[:, ::-1])
        for source in (
            np.array([1, 1, 2]),
            np.array(["a", "a", "b"]),
            np.array([1.0, np.inf, -np.inf]),
            np.array([1.0 + 0.0j, complex(np.nan, 0), complex(np.nan, 0)]),
        ):
            with self.subTest(source=source):
                expected = (
                    1 / 6
                    if np.iscomplexobj(source) is False and source.dtype.kind == "f"
                    else 1 / 3
                )
                self.assertAlmostEqual(
                    get_prob_for_rand(source[::-1], "permuted", source), expected
                )
        for shape in ((0,), (2, 0), (0, 3)):
            empty = np.empty(shape)
            self.assertEqual(get_prob_for_rand(empty, "permuted", empty), 1.0)

    def test_tracking_preserves_draws_rng_copy_and_reset(self):
        state = LayoutRand(seed=19, run_stochastic=True, track_pdf=True)
        reference = np.random.default_rng(19)
        for _ in range(3):
            state.set_rand_state("layout", "permuted", SOURCE)
            np.testing.assert_array_equal(state.s.layout, reference.permuted(SOURCE))
            self.assertAlmostEqual(state.probs[-1], 1 / 180)
            self.assertEqual(state.gen_state(), reference.bit_generator.state)
        self.assertAlmostEqual(state.return_probdens(), (1 / 180) ** 3)
        clone = state.copy()
        for current in (state, clone):
            current.set_rand_state("layout", "permuted", SOURCE)
        np.testing.assert_array_equal(state.s.layout, clone.s.layout)
        state.reset()
        state.set_rand_state("layout", "permuted", SOURCE)
        np.testing.assert_array_equal(
            state.s.layout, np.random.default_rng(19).permuted(SOURCE)
        )
        self.assertAlmostEqual(state.return_probdens(), 1 / 180)
        np.testing.assert_array_equal(SOURCE, [[1.0, np.nan, np.nan], [2.0, 2.0, 3.0]])

    def test_simulation_records_valid_nonzero_probability_history(self):
        model = PermutedFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 31},
        )
        result, history = Simulation(mdl=model)()
        reference = np.random.default_rng(31)
        draws = np.array([reference.permuted(SOURCE) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.layout"], draws)
        np.testing.assert_allclose(history["r.probdens"], np.full(4, 1 / 180))
        increments = np.nansum(draws[1:] * np.arange(6).reshape(2, 3), axis=(1, 2))
        np.testing.assert_array_equal(
            history["s.score"], np.r_[0.0, np.cumsum(increments)]
        )
        self.assertEqual(result["tend.classify.score"], increments.sum())
        np.testing.assert_array_equal(model.r.s.layout, SOURCE)


if __name__ == "__main__":
    unittest.main()
