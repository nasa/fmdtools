#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for explicit zero timer increments.

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
from fmdtools.define.container.time import Time
from fmdtools.define.object.timer import Timer
from fmdtools.sim.propagate import Simulation


class CountdownTime(Time):
    timernames = ("delay",)


class CountdownState(State):
    remaining: float = 3.0
    finished: bool = False


class PausedCountdown(Function):
    container_t = CountdownTime
    container_s = CountdownState

    def init_block(self, **kwargs):
        self.t.delay.set_timer(3.0)

    def dynamic_behavior(self):
        self.t.delay.inc(0.0 if self.t.time <= 2.0 else None)
        self.s.remaining = self.t.delay.time
        self.s.finished = self.t.delay.indicate_complete()

    def classify(self, **kwargs):
        return {"remaining": self.s.remaining, "finished": self.s.finished}


class TestZeroTimerIncrement(unittest.TestCase):
    """Zero is an explicit increment, while omission uses the configured step."""

    def test_zero_does_not_use_or_change_the_configured_increment(self):
        for zero in (0, 0.0, -0.0, np.int64(0), np.float64(0.0)):
            for configured in (-1.0, -0.25, 0.5):
                with self.subTest(zero=type(zero).__name__, configured=configured):
                    timer = Timer("held", tstep=configured)
                    timer.set_timer(3.0)
                    for _ in range(3):
                        self.assertIsNone(timer.inc(zero))
                        self.assertEqual(timer.time, 3.0)
                        self.assertEqual(timer.tstep, configured)
                        self.assertTrue(timer.indicate_ticking())
                        self.assertFalse(timer.indicate_complete())
                    timer.inc()
                    self.assertEqual(timer.time, 3.0 + configured)

    def test_default_sentinels_nonzero_overrides_and_completion_remain_supported(self):
        for argument in (None, [], ()):
            with self.subTest(argument=argument):
                timer = Timer("default", time=3.0, tstep=-0.5)
                timer.inc(argument)
                self.assertEqual(timer.time, 2.5)
                self.assertEqual(timer.mode, "ticking")
        for increment, expected, mode in (
            (-0.25, 0.75, "ticking"),
            (0.5, 1.5, "ticking"),
            (-1.0, 0.0, "complete"),
            (-2.0, 0.0, "complete"),
        ):
            with self.subTest(increment=increment):
                timer = Timer("override", time=1.0, tstep=-0.1)
                timer.inc(increment)
                self.assertEqual(
                    (timer.time, timer.mode, timer.tstep), (expected, mode, -0.1)
                )
        for initial in (0.0, -1.0):
            with self.subTest(initial=initial):
                timer = Timer("ended", time=initial, tstep=-1.0)
                timer.inc(0.0)
                self.assertEqual(timer.time, 0.0)
                self.assertTrue(timer.indicate_complete())

    def test_copies_and_logged_history_keep_zero_increments_independent(self):
        timer = Timer("original", time=2.0, tstep=-0.5)
        clone = timer.copy(name="clone")
        history = clone.create_hist(timerange=np.arange(4))
        history.log(clone, 0)
        for index, increment in enumerate((0.0, -0.5, 0.0), start=1):
            clone.inc(increment)
            history.log(clone, index)
        np.testing.assert_array_equal(history["time"], [2.0, 2.0, 1.5, 1.5])
        self.assertEqual(timer.time, 2.0)
        clone.reset()
        self.assertEqual((clone.time, clone.mode, clone.tstep), (0.0, "standby", -0.5))
        np.testing.assert_array_equal(history["time"], [2.0, 2.0, 1.5, 1.5])

    def test_real_countdown_simulation_completes_after_zero_increment_steps(self):
        model = PausedCountdown(sp={"end_time": 5.0})
        result, history = Simulation(mdl=model)()
        np.testing.assert_array_equal(
            history["s.remaining"], [3.0, 3.0, 3.0, 2.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(history["s.finished"], [False] * 5 + [True])
        self.assertEqual(result["tend.classify.remaining"], 0.0)
        self.assertTrue(result["tend.classify.finished"])
        self.assertEqual(model.t.delay.time, 3.0)
        again_result, again_history = Simulation(mdl=model)()
        self.assertEqual(again_result, result)
        self.assertEqual(again_history, history)


if __name__ == "__main__":
    unittest.main()
