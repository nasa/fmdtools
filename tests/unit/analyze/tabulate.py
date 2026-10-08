#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for sorting tuple-valued analysis table factors.

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

import numpy as np
from pandas.testing import assert_frame_equal

from fmdtools.analyze.tabulate import BaseTab, FMEA
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestTupleTableFactors(unittest.TestCase):
    """Keep a grouped factor as a whole sorting value."""

    def make_table(self, factors):
        keys = [(factor, index) for index, factor in enumerate(factors)]
        table = BaseTab(
            {
                "cost": {key: float(10 + i) for i, key in enumerate(keys)},
                "count": {key: i for i, key in enumerate(keys)},
            }
        )
        table.factors = ("modes", "time")
        return table

    def assert_sort(self, table, selector, reverse):
        before = copy.deepcopy(table.data)
        index = table.factors.index(selector) if isinstance(selector, str) else selector
        expected = sorted(before["cost"], key=lambda key: key[index])
        if reverse:
            expected.reverse()
        self.assertIsNone(table.sort_by_factor(selector, reverse=reverse))
        for metric in table:
            self.assertEqual(list(table[metric]), expected)
            self.assertEqual(table[metric], before[metric])
        return expected

    def test_uniform_ragged_empty_and_nested_tuple_factors_are_sorted(self):
        cases = [
            [(2, 1), (1, 9), (1, 2), (1, 9)],
            [("low",), ("short", "low"), (), ("low", "short")],
            [(("b", 1),), (("a", 2), ("b", 0)), (("a", 1),)],
            [(), (), ()],
        ]
        for factors in cases:
            for selector in ("modes", 0, -2, np.int64(0)):
                for reverse in (False, True):
                    with self.subTest(
                        factors=factors, selector=selector, reverse=reverse
                    ):
                        self.assert_sort(self.make_table(factors), selector, reverse)

    def test_tuple_indices_do_not_require_optional_factor_names(self):
        table = self.make_table([(2, 9), (1, 2), (2, 3)])
        expected = sorted(table["cost"])
        del table.factors
        table.sort_by_factor(0)
        self.assertEqual(list(table["cost"]), expected)
        self.assertEqual(list(table["count"]), expected)

    def test_multifactor_sorting_and_dataframe_rows_keep_values_aligned(self):
        table = self.make_table([("b", "c"), ("a",), ("b", "a"), ("a",)])
        expected = self.make_table([("b", "c"), ("a",), ("b", "a"), ("a",)])
        keys = sorted(expected["cost"], key=lambda key: (key[0], key[1]))
        for metric in expected:
            expected[metric] = {key: expected[metric][key] for key in keys}
        table.sort_by_factors("modes", "time")
        assert_frame_equal(table.as_table(sort=False), expected.as_table(sort=False))

    def test_scalar_sorting_keeps_numpy_order_and_invalid_selectors_are_atomic(self):
        for factors in ([2, 1, 2, 0], ["b", "a", "c", "a"], [2.0, np.nan, 1.0, 2.0]):
            for reverse in (False, True):
                with self.subTest(factors=factors, reverse=reverse):
                    table = self.make_table(factors)
                    keys = list(table["cost"])
                    order = np.argsort(factors, kind="stable")
                    if reverse:
                        order = order[::-1]
                    expected = [keys[i] for i in order]
                    table.sort_by_factor(0, reverse=reverse)
                    self.assertEqual(list(table["cost"]), expected)
        for selector, error in (("missing", ValueError), (2, IndexError)):
            table = self.make_table([("b",), ("a",)])
            before = copy.deepcopy(table.data)
            with self.assertRaises(error):
                table.sort_by_factor(selector)
            self.assertEqual(table.data, before)
            self.assertEqual(list(table["cost"]), list(before["cost"]))
        table = self.make_table([])
        table.sort_by_factor("modes")
        self.assertEqual(table.data, {"cost": {}, "count": {}})

    def test_joint_fault_simulation_fmea_can_sort_mode_tuples(self):
        model = ExFxnArch(sp={"end_time": 3.0})
        domain = FaultDomain(model)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        for time in (2.0, 1.0):
            sample.add_joint_fault_scenario(
                (("ex_fxn", "low"), ("ex_fxn2", "short")), time
            )
            sample.add_joint_fault_scenario(
                (("ex_fxn", "short"), ("ex_fxn2", "short")), time
            )
        results, _ = propagate.fault_sample(model, sample, showprogress=False)
        table = FMEA(
            results,
            sample,
            group_by=("modes", "time"),
            sum_metric=["flowval"],
            average_metric=["scenario_rate"],
        )
        before = copy.deepcopy(table.data)
        keys = sorted(before["sum_flowval"], key=lambda key: key[0])
        table.sort_by_factor("modes")
        self.assertEqual(len(keys), 4)
        for metric in table:
            self.assertEqual(list(table[metric]), keys)
            self.assertEqual(table[metric], before[metric])
        self.assertEqual(list(table.as_table(sort=False).index), keys)


if __name__ == "__main__":
    unittest.main()
