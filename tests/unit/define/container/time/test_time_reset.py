#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for resetting pending time execution.

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

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.time import ExtendedTime, Time


class TestTimeReset(unittest.TestCase):
    def test_reset_matches_a_fresh_clock_for_all_execution_flag_combinations(self):
        for cls in (Time, ExtendedTime):
            for executing, static, dynamic in itertools.product(
                (False, True), repeat=3
            ):
                with self.subTest(
                    cls=cls.__name__,
                    executing=executing,
                    static=static,
                    dynamic=dynamic,
                ):
                    clock = cls()
                    clock.update_time(3.0)
                    clock.assign(
                        {
                            "executing": executing,
                            "executed_static": static,
                            "executed_dynamic": dynamic,
                            "t_ind": 3,
                        }
                    )
                    clock.reset()
                    self.assertEqual(clock.return_mutables(), cls().return_mutables())
                    self.assertFalse(clock.has_executed())
                    clock.reset()
                    self.assertEqual(clock.return_mutables(), cls().return_mutables())

    def test_timer_state_resets_while_configuration_and_copies_remain_independent(self):
        clock = ExtendedTime(dt=0.25, use_local=False)
        clock.t1.set_timer(2.0, tstep=-0.125)
        clock.t1.inc()
        clock.t2.set_timer(3.0)
        clock.update_time(1.0)
        clone = clock.copy()
        timer_ids = {name: id(timer) for name, timer in clock.timers.items()}
        clock.reset()
        self.assertFalse(clock.executing)
        self.assertEqual(clock.dt, 0.25)
        self.assertFalse(clock.use_local)
        self.assertEqual(clock.t1.tstep, -0.125)
        self.assertEqual(clock.t2.tstep, -0.25)
        for name, timer in clock.timers.items():
            self.assertEqual(id(timer), timer_ids[name])
            self.assertEqual(timer.time, 0.0)
            self.assertEqual(timer.mode, "standby")
        self.assertTrue(clone.executing)
        self.assertEqual(clone.t1.time, 1.875)
        self.assertEqual(clone.t2.time, 3.0)
        clock.update_time(0.0)
        self.assertTrue(clock.executing)
        self.assertEqual(clock.time, 0.0)
        self.assertFalse(clock.has_executed())

    def test_actual_function_reset_clears_initial_and_incomplete_timestep_flags(self):
        for time, increment in ((0.0, "all"), (1.0, "none"), (2.0, "all")):
            with self.subTest(time=time, increment=increment):
                model = ExampleFunction(sp={"end_time": 3.0})
                model(time=time, inc_at=increment)
                self.assertEqual(
                    bool(model.t.executing), increment == "none" or time == 0.0
                )
                model.reset()
                fresh = ExampleFunction(sp={"end_time": 3.0})
                # Block reset uses container defaults, not constructor overrides.
                fresh.reset()
                self.assertEqual(model.t.return_mutables(), fresh.t.return_mutables())
                self.assertFalse(model.t.executing)
                model(time="end", end_of_simulation=True)
                fresh(time="end", end_of_simulation=True)
                self.assertEqual(model.s, fresh.s)
                self.assertEqual(model.h, fresh.h)
                self.assertEqual(model.t.return_mutables(), fresh.t.return_mutables())


if __name__ == "__main__":
    unittest.main()
