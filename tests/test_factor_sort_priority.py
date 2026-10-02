#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for table factor sort priority.

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
import itertools
import unittest

import numpy as np
from pandas.testing import assert_frame_equal

from fmdtools.analyze.tabulate import BaseTab, FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestFactorSortPriority(unittest.TestCase):
    def make_table(self, names=("plant", "mode", "time")):
        keys = list(itertools.product(("b", "a"), (2, 1), (3, 1, 2)))
        table = BaseTab(
            {
                "cost": {key: i + 0.5 for i, key in enumerate(keys)},
                "count": {key: 2 * i for i, key in enumerate(reversed(keys))},
            }
        )
        table.factors = names
        return table

    def assert_priority(self, table, selectors):
        original = copy.deepcopy(table.data)
        factors = list(table.factors)
        indices = [
            factors.index(value)
            if isinstance(value, str)
            else list(range(len(factors)))[value]
            for value in selectors
        ]
        indices += [i for i in range(len(factors)) if i not in indices]
        expected = sorted(
            next(iter(original.values())),
            key=lambda key: tuple(key[i] for i in indices),
        )
        self.assertIsNone(table.sort_by_factors(*selectors))
        for metric, values in table.items():
            self.assertEqual(list(values), expected)
            self.assertEqual(values, original[metric])
        self.assertEqual(list(table.factors), factors)
        return expected

    def test_default_order_follows_declared_factors(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                keys = list(itertools.product(*([("b", "a")] * count)))
                table = BaseTab({"cost": {key: i for i, key in enumerate(keys)}})
                table.factors = ["factor_" + str(i) for i in range(count)]
                self.assertEqual(self.assert_priority(table, ()), sorted(keys))

    def test_every_partial_priority_preserves_remaining_factor_order(self):
        for count in range(4):
            for indices in itertools.permutations(range(3), count):
                for named in (False, True):
                    with self.subTest(indices=indices, named=named):
                        table = self.make_table()
                        selectors = (
                            tuple(table.factors[i] for i in indices)
                            if named
                            else indices
                        )
                        self.assert_priority(table, selectors)

    def test_mixed_and_negative_selectors_match_their_named_priority(self):
        for selectors in (
            (np.int64(0),),
            (-1,),
            ("time", -3),
            (np.int64(1), "plant"),
            (-2, "time"),
            ("plant", 0, "time"),
        ):
            with self.subTest(selectors=selectors):
                self.assert_priority(self.make_table(), selectors)

    def test_invalid_selectors_leave_existing_row_order_untouched(self):
        for selectors, error in (
            (("missing",), ValueError),
            ((3,), IndexError),
            ((-4,), IndexError),
            (("plant", 0.5), TypeError),
        ):
            with self.subTest(selectors=selectors):
                table = self.make_table()
                before = {
                    metric: list(values.items()) for metric, values in table.items()
                }
                with self.assertRaises(error):
                    table.sort_by_factors(*selectors)
                self.assertEqual(
                    {metric: list(values.items()) for metric, values in table.items()},
                    before,
                )

    def test_actual_fmea_keeps_default_priority_and_metric_alignment(self):
        model = ExampleFunction(sp={"end_time": 4.0})
        domain = FaultDomain(model)
        domain.add_faults(("examplefunction", "low"), ("examplefunction", "no_charge"))
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([3.0, 1.0, 2.0])
        result, _ = propagate.fault_sample(model, sample, showprogress=False)
        for selectors in ((), ("obj",), (0,), ("time",)):
            with self.subTest(selectors=selectors):
                table = FMEA(
                    result,
                    sample,
                    group_by=("obj", "fault", "time"),
                    sum_metric=["xy"],
                    average_metric=["scenario_rate"],
                )
                before = table.as_table(sort=False)
                expected = self.assert_priority(table, selectors)
                after = table.as_table(sort=False)
                self.assertEqual(list(after.index), expected)
                assert_frame_equal(after, before.reindex(after.index))


if __name__ == "__main__":
    unittest.main()
