#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for numerical factor indices in analysis tables.

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
from matplotlib import pyplot as plt
from pandas.testing import assert_frame_equal

from fmdtools.analyze.tabulate import BaseTab, FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestTableFactorIndices(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def make_table(self, with_names=True):
        keys = [("b", 2), ("a", 3), ("b", 1), ("a", 2), ("b", 3), ("a", 1)]
        table = BaseTab(
            {
                "cost": {key: float(10 * i) for i, key in enumerate(keys)},
                "count": {key: i for i, key in enumerate(keys)},
            }
        )
        if with_names:
            table.factors = ("mode", "time")
        return table

    def assert_reordered(self, table, before, expected):
        self.assertEqual(set(table), set(before))
        for metric, values in table.items():
            self.assertEqual(list(values), expected)
            self.assertEqual(values, before[metric])

    def test_integer_indices_match_named_factor_sorting(self):
        for index in (0, 1, -1, -2, np.int64(0), np.int64(1)):
            for reverse in (False, True):
                with self.subTest(index=index, reverse=reverse):
                    table = self.make_table()
                    reference = self.make_table()
                    before = copy.deepcopy(table.data)
                    reference.sort_by_factor(table.factors[index], reverse=reverse)
                    result = table.sort_by_factor(index, reverse=reverse)
                    self.assertIsNone(result)
                    self.assert_reordered(table, before, list(reference["cost"]))
                    for metric in table:
                        self.assertEqual(
                            list(table[metric].items()), list(reference[metric].items())
                        )

    def test_indices_work_without_optional_factor_name_metadata(self):
        for index in (0, 1, -1):
            with self.subTest(index=index):
                table = self.make_table(with_names=False)
                before = copy.deepcopy(table.data)
                expected = sorted(before["cost"], key=lambda key: key[index])
                table.sort_by_factor(index)
                self.assert_reordered(table, before, expected)

    def test_multifactor_indices_and_mixed_selectors_match_name_priority(self):
        for selectors, names in (
            ((0, 1), ("mode", "time")),
            ((1, 0), ("time", "mode")),
            ((0, "time"), ("mode", "time")),
            (("time", 0), ("time", "mode")),
        ):
            with self.subTest(selectors=selectors):
                table = self.make_table()
                reference = self.make_table()
                before = copy.deepcopy(table.data)
                reference.sort_by_factors(*names)
                table.sort_by_factors(*selectors)
                self.assert_reordered(table, before, list(reference["cost"]))

    def test_invalid_indices_and_names_leave_all_metrics_unchanged(self):
        for selector, error in (
            (2, IndexError),
            (-3, IndexError),
            ("missing", ValueError),
        ):
            with self.subTest(selector=selector):
                table = self.make_table()
                before = copy.deepcopy(table.data)
                with self.assertRaises(error):
                    table.sort_by_factor(selector)
                self.assertEqual(table.data, before)
                for metric in table:
                    self.assertEqual(list(table[metric]), list(before[metric]))

    def test_dataframe_and_bar_plot_preserve_sorted_rows_and_values(self):
        table = self.make_table()
        reference = self.make_table()
        reference.sort_by_factor("time")
        table.sort_by_factor(1)
        assert_frame_equal(table.as_table(sort=False), reference.as_table(sort=False))
        frame = table.as_table(sort=False)
        self.assertEqual(list(frame.index), list(table["cost"]))
        fig, ax = table.as_plot("cost")
        self.assertEqual(
            [bar.get_height() for bar in ax.patches], list(table["cost"].values())
        )
        fig.canvas.draw()

    def test_fmea_from_actual_fault_simulations_accepts_factor_indices(self):
        model = ExampleFunction(sp={"end_time": 4.0})
        domain = FaultDomain(model)
        domain.add_faults(("examplefunction", "low"), ("examplefunction", "no_charge"))
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([3.0, 1.0, 2.0])
        result, _ = propagate.fault_sample(model, sample, showprogress=False)
        table = FMEA(result, sample, group_by=("time", "fault"), sum_metric=["xy"])
        reference = FMEA(result, sample, group_by=("time", "fault"), sum_metric=["xy"])
        before = copy.deepcopy(table.data)
        reference.sort_by_factor("time")
        table.sort_by_factor(0)
        self.assert_reordered(table, before, list(reference["sum_xy"]))
        assert_frame_equal(table.as_table(sort=False), reference.as_table(sort=False))
        self.assertEqual(len(table["sum_xy"]), 6)

    def test_named_factor_and_metric_sorting_controls_are_unchanged(self):
        table = self.make_table()
        before = copy.deepcopy(table.data)
        table.sort_by_factor("mode")
        self.assert_reordered(
            table, before, sorted(before["cost"], key=lambda key: key[0])
        )
        table.sort_by_metric("cost", reverse=True)
        self.assert_reordered(
            table, before, sorted(before["cost"], key=before["cost"].get, reverse=True)
        )


if __name__ == "__main__":
    unittest.main()
