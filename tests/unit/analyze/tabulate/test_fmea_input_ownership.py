#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for input ownership while constructing FMEA tables.

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

from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestFmeaInputOwnership(unittest.TestCase):
    def setUp(self):
        self.model = ExampleFunction(sp={"end_time": 3.0})
        domain = FaultDomain(self.model)
        domain.add_fault("examplefunction", "low")
        self.sample = FaultSample(domain, def_mdl_phasemap=False)
        self.sample.add_fault_times([1.0, 2.0])
        self.result, self.history = propagate.fault_sample(
            self.model, self.sample, showprogress=False
        )
        self.scenarios = self.sample.scenarios()

    def assert_mapping_unchanged(self, original, snapshot):
        self.assertEqual(list(original), list(snapshot))
        flat = Result(original).flatten()
        expected = Result(snapshot).flatten()
        self.assertEqual(list(flat), list(expected))
        for key in expected:
            np.testing.assert_equal(flat[key], expected[key])

    def table(self, result, added, **kwargs):
        return FMEA(
            result,
            self.sample,
            add_res=added,
            group_by=("time",),
            sum_metric="xy",
            **kwargs,
        )

    def test_additions_and_overrides_leave_both_inputs_unchanged(self):
        for nested in (False, True):
            for added_as_dict in (False, True):
                with self.subTest(nested=nested, added_as_dict=added_as_dict):
                    source = copy.deepcopy(self.result)
                    second_key = self.scenarios[1].name + ".tend.classify.xy"
                    added = Result(
                        {
                            self.scenarios[0].name + ".tend.classify.xy": 50.0,
                            second_key: source[second_key],
                            self.scenarios[1].name + ".tend.classify.extra": 8.0,
                        }
                    )
                    if nested:
                        source, added = source.nest(), added.nest()
                    if added_as_dict:
                        added = dict(added)
                    before, added_before = copy.deepcopy(source), copy.deepcopy(added)
                    expected = self.table(Result({**source, **added}), {})
                    actual = self.table(source, added)
                    assert_frame_equal(
                        actual.as_table(sort=False), expected.as_table(sort=False)
                    )
                    self.assert_mapping_unchanged(source, before)
                    self.assert_mapping_unchanged(added, added_before)

    def test_repeated_tables_use_the_original_results_after_override(self):
        source_before = copy.deepcopy(self.result)
        initial = self.table(self.result, {})
        key = self.scenarios[0].name + ".tend.classify.xy"
        changed = self.table(self.result, {key: 100.0})
        self.assertEqual(changed["sum_xy"][(1.0,)], 100.0)
        later = self.table(self.result, {})
        assert_frame_equal(initial.as_table(sort=False), later.as_table(sort=False))
        self.assert_mapping_unchanged(self.result, source_before)
        rerun, _ = propagate.fault_sample(self.model, self.sample, showprogress=False)
        self.assert_mapping_unchanged(self.result, rerun)

    def test_failed_metric_construction_does_not_leave_partial_input_updates(self):
        added = {self.scenarios[0].name + ".tend.classify.note": 9.0}
        before = copy.deepcopy(self.result)
        with self.assertRaises(Exception):
            FMEA(self.result, self.sample, add_res=added, sum_metric="missing_metric")
        self.assert_mapping_unchanged(self.result, before)
        self.assertEqual(added, {self.scenarios[0].name + ".tend.classify.note": 9.0})

    def test_empty_additions_self_additions_and_retained_values_are_unchanged(self):
        for added in ({}, Result(), self.result):
            with self.subTest(kind=type(added).__name__):
                before = copy.deepcopy(self.result)
                actual = self.table(self.result, added)
                expected = self.table(copy.deepcopy(before), {})
                self.assertEqual(actual.data, expected.data)
                self.assert_mapping_unchanged(self.result, before)
        array = np.arange(3.0)
        self.result["metadata"] = array
        self.table(self.result, {"note": "value"})
        self.assertIs(self.result["metadata"], array)
        self.assertNotIn("note", self.result)
        np.testing.assert_array_equal(array, [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
