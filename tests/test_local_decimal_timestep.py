#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for decimal local timestep alignment.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from decimal import Decimal

import numpy as np
import pytest

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.time import Time


@pytest.mark.parametrize("cast", [float, np.float64])
@pytest.mark.parametrize(
    "global_dt, local_dt",
    [
        (1.0, 0.1),
        (0.3, 0.1),
        (0.1, 0.3),
        (1.0, 0.2),
        (0.03, 0.01),
        (0.01, 0.03),
        (0.001, 0.0001),
        (0.7, 0.1),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.25, 0.5),
        (1.0, 2.0),
    ],
)
def test_compatible_local_steps_and_timer_increments(cast, global_dt, local_dt):
    """Use the decimal settings as the independent alignment reference."""
    larger, smaller = sorted(
        (Decimal(str(global_dt)), Decimal(str(local_dt))), reverse=True
    )
    assert larger % smaller == 0

    step = cast(local_dt)

    class LocalTime(Time):
        local_dt = step
        timernames = ("sample",)

    clock = LocalTime(dt=cast(global_dt))
    assert clock.dt == local_dt
    assert clock.timers["sample"].tstep == -local_dt
    clock.set_timestep(dt=global_dt, use_local=False)
    assert clock.dt == global_dt
    assert clock.timers["sample"].tstep == -global_dt
    clock.set_timestep(dt=global_dt, use_local=True)
    assert clock.dt == local_dt
    assert clock.timers["sample"].tstep == -local_dt


@pytest.mark.parametrize(
    "global_dt, local_dt", [(1.0, 0.3), (0.3, 0.2), (0.2, 0.3), (0.1, 0.15)]
)
def test_incompatible_local_steps_still_raise(global_dt, local_dt):
    step = local_dt

    class LocalTime(Time):
        local_dt = step

    with pytest.raises(Exception, match="doesn't line up"):
        LocalTime(dt=global_dt)


@pytest.mark.parametrize("local_dt", [0.1, 0.2, 0.01])
def test_function_simulation_with_decimal_local_step(local_dt):
    """Construct and run the real Function, comparing its global-step control."""
    step = local_dt

    class LocalTime(Time):
        local_dt = step

    class LocalFunction(ExampleFunction):
        container_t = LocalTime

    local = LocalFunction(
        "local", sp={"dt": local_dt, "end_time": 1.0, "use_local": True}
    )
    control = ExampleFunction(
        "control", sp={"dt": local_dt, "end_time": 1.0, "use_local": False}
    )
    local(time=1.0, proptype="dynamic")
    control(time=1.0, proptype="dynamic")
    assert local.t.dt == local_dt
    assert local.t.time == control.t.time
    assert local.s.x == control.s.x
    np.testing.assert_array_equal(local.h["s.x"], control.h["s.x"])
    np.testing.assert_array_equal(local.h["time"], control.h["time"])
