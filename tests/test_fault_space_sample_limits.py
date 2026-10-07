#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for honor zero and numpy integer fault-space limits.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
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

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


def domain(limit, seed=7):
    result = FaultDomain(ExampleFunction(sp={"end_time": 3.0}))
    result.add_fault_space(
        "examplefunction",
        "low",
        {"s.x": {1.0, 3.0, 5.0}, "s.y": {2.0, 4.0}},
        n=limit,
        seed=seed,
    )
    return result


def definitions(value):
    return {key: fault.asdict() for key, fault in value.faults.items()}


class TestFaultSpaceSampleLimits(unittest.TestCase):
    def test_numpy_limits_match_builtin_integers_and_never_exceed_them(self):
        for seed in (0, 7, 19):
            for count in (1, 2, 5, 12, 20):
                expected = domain(count, seed)
                for dtype in (np.int32, np.int64, np.uint64):
                    with self.subTest(seed=seed, count=count, dtype=dtype):
                        actual = domain(dtype(count), seed)
                        self.assertEqual(definitions(actual), definitions(expected))
                        self.assertLessEqual(len(actual.faults), count)

    def test_zero_is_a_noop_for_existing_domains_and_supplied_ranges(self):
        for zero in (0, np.int64(0), np.uint64(0)):
            with self.subTest(zero=zero):
                target = domain("all")
                before = definitions(target)
                ranges = {"s.x": {1.0, 2.0}, "s.y": (0.0, 2.0, 3)}
                original = copy.deepcopy(ranges)
                target.add_fault_space("examplefunction", "low", ranges, n=zero)
                self.assertEqual(definitions(target), before)
                self.assertEqual(ranges, original)
                self.assertEqual(domain(zero).faults, {})

    def test_invalid_limits_fail_before_changing_the_domain_or_ranges(self):
        for limit in (
            -1,
            np.int64(-1),
            1.5,
            2.0,
            True,
            np.bool_(True),
            None,
            "2",
            "ALL",
            [2],
            np.array([2]),
        ):
            with self.subTest(limit=repr(limit)):
                target = domain(2)
                before = definitions(target)
                ranges = {"s.x": {1.0, 2.0}}
                original = copy.deepcopy(ranges)
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    target.add_fault_space("examplefunction", "low", ranges, n=limit)
                self.assertEqual(definitions(target), before)
                self.assertEqual(ranges, original)

    def test_all_and_large_limits_keep_full_space_and_explicit_probabilities(self):
        self.assertEqual(definitions(domain("all")), definitions(domain(100)))
        target = FaultDomain(ExampleFunction())
        target.add_fault_space(
            "examplefunction", "low", {"s.x": {1.0, 2.0}}, n=np.int64(10), prob=0.25
        )
        self.assertEqual(len(target.faults), 2)
        self.assertTrue(all(fault.prob == 0.25 for fault in target.faults.values()))

    def test_real_fault_samples_match_the_equivalent_builtin_limit(self):
        outputs = []
        for count in (3, np.int64(3)):
            selected = domain(count)
            sample = FaultSample(selected, def_mdl_phasemap=False)
            sample.add_fault_times([1.0])
            result, history = propagate.fault_sample(
                selected.mdl, sample, showprogress=False
            )
            outputs.append((sample, result, history))
        left, right = outputs
        self.assertEqual(
            [s.asdict() for s in left[0].scenarios()],
            [s.asdict() for s in right[0].scenarios()],
        )
        self.assertEqual(set(left[1]), set(right[1]))
        self.assertEqual(set(left[2]), set(right[2]))
        for index in (1, 2):
            for key in left[index]:
                np.testing.assert_array_equal(left[index][key], right[index][key])


if __name__ == "__main__":
    unittest.main()
