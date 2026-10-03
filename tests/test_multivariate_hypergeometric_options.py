#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for multivariate hypergeometric sampling options.

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
from itertools import product
import math
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand, get_pfunc_for_dist, get_prob_for_rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


def exact_mass(values, colors, count):
    """Product of finite-urn probabilities, independent of scipy.stats."""
    denominator = math.comb(sum(colors), count)
    rows = np.asarray(values).reshape(-1, len(colors))
    return math.prod(
        math.prod(math.comb(int(c), int(x)) for c, x in zip(colors, row)) / denominator
        for row in rows
    )


class UrnState(State):
    counts: np.array = np.zeros((2, 3), dtype=int)
    counts_update = ("multivariate_hypergeometric", ([5, 3, 2], 4, 2, "marginals"))


class UrnRand(Rand):
    s: UrnState = UrnState()


class TotalState(State):
    total: np.float64 = 0.0


class UrnFunction(Function):
    container_r = UrnRand
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += (self.r.s.counts @ np.array([1, 2, 4])).sum()

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestMultivariateHypergeometricOptions(unittest.TestCase):
    def test_vector_and_batch_probabilities_match_combinatorial_formula(self):
        for colors, count in (([5, 3, 2], 4), ([0, 4, 1], 0), ([3], 3)):
            for size in (None, (), 1, (2, 3), 0, (2, 0)):
                for method in ("marginals", "count"):
                    with self.subTest(
                        colors=colors, count=count, size=size, method=method
                    ):
                        draws = np.random.default_rng(13).multivariate_hypergeometric(
                            colors, count, size, method
                        )
                        before = draws.copy()
                        expected = exact_mass(draws, colors, count)
                        for args in (
                            (colors, count),
                            (colors, count, size),
                            (colors, count, size, method),
                        ):
                            actual = get_prob_for_rand(
                                draws, "multivariate_hypergeometric", *args
                            )
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-12, atol=0
                            )
                            self.assertEqual(np.ndim(actual), 0)
                        np.testing.assert_array_equal(draws, before)

    def test_all_small_outcomes_normalize_and_impossible_draws_keep_zero_mass(self):
        colors, count = [2, 1, 2], 3
        probability = get_pfunc_for_dist(
            "multivariate_hypergeometric", colors, count, None, "count"
        )
        total = 0.0
        for row in product(*(range(c + 1) for c in colors)):
            if sum(row) == count:
                expected = exact_mass(row, colors, count)
                self.assertAlmostEqual(probability(*row), expected, 13)
                total += probability(row)
        self.assertAlmostEqual(total, 1.0, 13)
        for row in ((3, 0, 0), (-1, 2, 2), (0, 0, 0)):
            with self.subTest(row=row):
                self.assertEqual(probability(row), 0.0)
        with self.assertRaises(TypeError):
            get_pfunc_for_dist(
                "multivariate_hypergeometric", colors, count, None, "count", "extra"
            )

    def test_tracked_updates_copies_reset_and_disabled_controls_preserve_rng(self):
        colors = np.array([5, 3, 2])
        for method in ("marginals", "count"):
            for size in (None, (2, 3), 0):
                with self.subTest(method=method, size=size):
                    random = UrnRand(seed=19, run_stochastic=True, track_pdf=True)
                    reference = np.random.default_rng(19)
                    masses = []
                    for _ in range(2):
                        expected = reference.multivariate_hypergeometric(
                            colors, 4, size, method
                        )
                        random.set_rand_state(
                            "counts",
                            "multivariate_hypergeometric",
                            colors,
                            4,
                            size,
                            method,
                        )
                        np.testing.assert_array_equal(random.s.counts, expected)
                        self.assertEqual(
                            random.rng.bit_generator.state,
                            reference.bit_generator.state,
                        )
                        masses.append(exact_mass(expected, colors.tolist(), 4))
                    np.testing.assert_allclose(random.probs, masses, rtol=1e-12)
                    np.testing.assert_allclose(
                        random.return_probdens(), np.prod(masses), rtol=1e-12
                    )
                    cloned = random.copy()
                    for instance in (random, cloned):
                        instance.set_rand_state(
                            "counts",
                            "multivariate_hypergeometric",
                            colors,
                            4,
                            size,
                            method,
                        )
                    np.testing.assert_array_equal(random.s.counts, cloned.s.counts)
                    self.assertEqual(
                        random.rng.bit_generator.state, cloned.rng.bit_generator.state
                    )
                    random.reset()
                    self.assertEqual(random.probs, [])
                    random.set_rand_state(
                        "counts", "multivariate_hypergeometric", colors, 4, size, method
                    )
                    expected = np.random.default_rng(19).multivariate_hypergeometric(
                        colors, 4, size, method
                    )
                    np.testing.assert_array_equal(random.s.counts, expected)
        for enabled, tracked in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, tracked=tracked):
                random = UrnRand(seed=7, run_stochastic=enabled, track_pdf=tracked)
                before = copy.deepcopy(random.rng.bit_generator.state)
                random.set_rand_state(
                    "counts", "multivariate_hypergeometric", colors, 4, 2
                )
                self.assertEqual(random.probs, [])
                self.assertEqual(random.rng.bit_generator.state == before, not enabled)
        np.testing.assert_array_equal(colors, [5, 3, 2])

    def test_real_stochastic_simulation_records_counts_probabilities_and_totals(self):
        model = UrnFunction(
            sp={"end_time": 3.0, "run_stochastic": True, "track_pdf": True},
            r={"seed": 29},
        )
        result, history = Simulation(mdl=model)()
        generator = np.random.default_rng(29)
        counts = np.array(
            [
                generator.multivariate_hypergeometric([5, 3, 2], 4, 2, "marginals")
                for _ in history.time
            ]
        )
        np.testing.assert_array_equal(history["r.s.counts"], counts)
        expected = [exact_mass(row, [5, 3, 2], 4) for row in counts]
        np.testing.assert_allclose(history["r.probdens"], expected, rtol=1e-12)
        totals = np.r_[0.0, np.cumsum((counts[1:] @ np.array([1, 2, 4])).sum(axis=1))]
        np.testing.assert_array_equal(history["s.total"], totals)
        self.assertEqual(result["tend.classify.total"], totals[-1])
        np.testing.assert_array_equal(model.r.s.counts, np.zeros((2, 3), dtype=int))


if __name__ == "__main__":
    unittest.main()
