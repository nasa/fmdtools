#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for phase lookup at the configured simulation timestep.

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

import numpy as np
import pytest

from fmdtools.analyze.history import History
from fmdtools.analyze.phases import PhaseMap, from_hist
from fmdtools.define.container.mode import Fault


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "relative_time, expected",
    [
        (0.0, "on"),
        (1.5, "on"),
        (2.0, "off"),
        (4.5, "off"),
    ],
)
def test_find_phase_uses_map_timestep(dt, relative_time, expected):
    """Phase endpoints are inclusive sample times, each with width dt."""
    phases = PhaseMap({"on": [0.0, dt], "off": [2 * dt, 4 * dt]}, dt=dt)
    assert phases.find_phase(relative_time * dt) == expected


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0, 2.0])
def test_lookup_rejects_times_outside_sampled_phases(dt):
    phases = PhaseMap({"on": [0.0, dt], "off": [2 * dt, 4 * dt]}, dt=dt)
    for time in (-dt, 5 * dt):
        with pytest.raises(Exception, match="not in phases"):
            phases.find_phase(time)


@pytest.mark.parametrize("dt", [0.25, 0.5, 1.0, 2.0])
def test_history_phases_and_fault_rates_use_same_timestep(dt):
    """Exercise history extraction, sample counting and actual fault rates."""
    hist = History(
        {"pump.m.mode": ["on", "on", "off", "off", "off"], "time": np.arange(5) * dt}
    )
    phases = from_hist(hist, dt=dt)["pump"]
    assert phases.calc_samples_in_phases(*hist["time"]) == {"on": 2, "off": 3}
    assert phases.find_base_phase(2 * dt) == "off"
    assert phases.calc_scen_exposure_time(2 * dt) == pytest.approx(3 * dt)
    fault = Fault(prob=0.1, phases=(("on", 0.2), ("off", 0.8)), units="sec")
    assert fault.calc_rate(
        2 * dt, phases, sim_time=5 * dt, sim_units="sec"
    ) == pytest.approx(0.1 * 0.8 * 3 * dt)


def test_explicit_timestep_override_remains_supported():
    phases = PhaseMap({"on": [0.0, 0.25], "off": [0.5, 1.0]}, dt=0.25)
    assert phases.find_phase(0.5, 1.0) == "on"
    assert phases.find_phase(0.5, dt=0.25) == "off"
    with pytest.raises(Exception, match="not in phases"):
        phases.find_phase(0.375, dt=0.0)
