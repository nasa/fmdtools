#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for histories of string states with declared value sets.

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

import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class StatusState(State):
    status: str = "idle"
    status_set = ("idle", "awaiting operator authorization", "recovered α")


class StatusFunction(Function):
    container_s = StatusState

    def dynamic_behavior(self):
        self.s.status = self.s.status_set[int(self.t.time)]

    def classify(self, **kwargs):
        return {"status": self.s.status}


class UnboundedStringState(State):
    status: str = "idle"
    empty: str = "ready"
    empty_set = ()


class TestStringStateHistories(unittest.TestCase):
    def test_declared_values_are_recorded_without_truncation(self):
        values = StatusState.status_set
        for timerange in (None, np.arange(len(values))):
            with self.subTest(timerange=timerange):
                state = StatusState()
                history = state.create_hist(timerange, default_str_size="<U2")
                if timerange is None:
                    self.assertIsInstance(history.status, list)
                    values_to_log = values[1:]
                else:
                    expected_width = max(len(value) for value in values)
                    self.assertEqual(
                        history.status.dtype, np.dtype(f"<U{expected_width}")
                    )
                    values_to_log = values
                for index, value in enumerate(values_to_log):
                    state.status = value
                    history.log(state, index)
                np.testing.assert_array_equal(history.status, values)

    def test_absent_and_empty_sets_keep_the_requested_default_width(self):
        for width in (20, 40):
            with self.subTest(width=width):
                state = UnboundedStringState()
                history = state.create_hist([0, 1], default_str_size=f"<U{width}")
                history.log(state, 0)
                state.status = "s" * width
                state.empty = "e" * width
                history.log(state, 1)
                for field, initial in (("status", "idle"), ("empty", "ready")):
                    self.assertEqual(history[field].dtype, np.dtype(f"<U{width}"))
                    np.testing.assert_array_equal(
                        history[field], [initial, getattr(state, field)]
                    )

    def test_real_simulation_records_categorical_state_transitions(self):
        model = StatusFunction(sp={"end_time": 2.0})
        result, history = Simulation(mdl=model)()
        np.testing.assert_array_equal(history.time, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(history["s.status"], StatusState.status_set)
        self.assertEqual(result["tend.classify.status"], "recovered α")
        self.assertEqual(model.s.status, "idle")


if __name__ == "__main__":
    unittest.main()
