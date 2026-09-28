#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for unique phase identifiers extracted from mode histories.

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

import collections
import itertools

import numpy as np
import pytest

from fmdtools.analyze.history import History
from fmdtools.analyze.phases import from_hist
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.mode import Fault
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.sample import FaultDomain, FaultSample


MODE_SEQUENCES = [
    ["on", "off", "on", "on1"],
    ["on1", "on", "off", "on"],
    ["on", "on1", "on", "on2", "on"],
    ["on1", "on", "on1", "on", "on11"],
    ["phase", "phase1", "phase", "phase2", "phase", "phase11", "phase1"],
    ["on", "on", "off", "on", "on", "on1", "on1"],
    ["a", "b"] * 12 + ["a1", "b", "a1"],
]


def expected_intervals(modes, times):
    """Independent contiguous-run oracle with no phase-name generation."""
    intervals = []
    start = 0
    for _, group in itertools.groupby(modes):
        count = len(list(group))
        intervals.append([times[start], times[start + count - 1]])
        start += count
    return intervals


def assert_partition(mapping, modes, times):
    assert list(mapping.phases.values()) == expected_intervals(modes, times)
    counts = collections.Counter(modes)
    phase_ids = [phase for group in mapping.modephases.values() for phase in group]
    assert len(phase_ids) == len(set(phase_ids))
    assert set(phase_ids) == set(mapping.phases)
    assert mapping.calc_samples_in_phases(*times) == counts
    for mode in counts:
        expected_times = [time for time, value in zip(times, modes) if value == mode]
        assert mapping.get_phase_times(mode) == expected_times
        assert mapping.calc_modephase_time(mode) == pytest.approx(
            counts[mode] * mapping.dt
        )
    for time, mode in zip(times, modes):
        assert mapping.find_base_phase(time) == mode


@pytest.mark.parametrize("modes", MODE_SEQUENCES)
@pytest.mark.parametrize("dt", [0.25, 1.0])
def test_numbered_mode_names_preserve_every_interval_and_membership(modes, dt):
    times = np.arange(len(modes)) * dt + 10.0
    history = History({"plant.m.mode": modes.copy(), "time": times.copy()})
    mapping = from_hist(history, dt=dt)["plant"]
    assert_partition(mapping, modes, times)
    assert history["plant.m.mode"] == modes
    np.testing.assert_array_equal(history.time, times)


def test_exhaustive_short_histories_partition_even_when_labels_overlap():
    labels = ("on", "on1", "on2")
    for count in range(1, 6):
        for modes in itertools.product(labels, repeat=count):
            times = np.arange(count)
            mapping = from_hist(History({"plant.m.mode": modes, "time": times}))[
                "plant"
            ]
            assert_partition(mapping, modes, times)


def test_suffix_allocation_reserves_real_modes_before_generating_names():
    modes = ["on", "off", "on", "on1", "on2", "off", "on"]
    mapping = from_hist(History({"plant.m.mode": modes, "time": list(range(7))}))[
        "plant"
    ]
    assert list(mapping.phases) == ["on", "off", "on3", "on1", "on2", "off1", "on4"]
    assert mapping.modephases["on"] == {"on", "on3", "on4"}
    assert mapping.modephases["on1"] == {"on1"}
    assert mapping.modephases["on2"] == {"on2"}


def test_existing_noncolliding_labels_and_empty_histories_are_unchanged():
    mapping = from_hist(
        History(
            {"plant.m.mode": ["on", "off", "on", "off", "on"], "time": [0, 1, 2, 3, 4]}
        )
    )["plant"]
    assert mapping.phases == {
        "on": [0, 0],
        "off": [1, 1],
        "on1": [2, 2],
        "off1": [3, 3],
        "on2": [4, 4],
    }
    assert mapping.modephases == {"on": {"on", "on1", "on2"}, "off": {"off", "off1"}}
    assert from_hist(History({"plant.m.mode": [], "time": []})) == {}
    with pytest.raises(Exception, match="Value m.mode not in Result keys"):
        from_hist(History({"time": [0, 1, 2]}))


@pytest.mark.parametrize("selection", ["all", ["first"], []])
def test_each_function_has_independent_labels_with_optional_mode_grouping(selection):
    modes = MODE_SEQUENCES[0]
    times = np.arange(len(modes))
    history = History(
        {"first.m.mode": modes, "second.m.mode": np.array(modes[::-1]), "time": times}
    )
    mappings = from_hist(history, fxn_modephases=selection)
    assert list(mappings) == ["first", "second"]
    for name, values in [("first", modes), ("second", modes[::-1])]:
        mapping = mappings[name]
        assert list(mapping.phases.values()) == expected_intervals(values, times)
        if selection == "all" or name in selection:
            assert_partition(mapping, values, times)
        else:
            assert mapping.modephases == {}
            assert len(mapping.phases) == 4


def test_correct_times_flow_into_fault_sampling_and_exposure_rates():
    modes = ["on", "off", "on", "on1", "on1", "on1"]
    times = np.arange(len(modes), dtype=float)
    mapping = from_hist(History({"plant.m.mode": modes, "time": times}))["plant"]
    model = ExampleFunction(sp={"end_time": 5.0})
    domain = FaultDomain(model)
    domain.add_fault("examplefunction", "low")
    sample = FaultSample(domain, phasemap=mapping, def_mdl_phasemap=False)
    sample.add_fault_phases("on", method="all")
    assert sample.get_times() == [0.0, 2.0]
    assert [scenario.time for scenario in sample.scenarios()] == [0.0, 2.0]
    factors = {"on": 0.2, "off": 0.3, "on1": 0.5}
    fault = Fault(prob=0.1, phases=tuple(factors.items()), units="sec")
    counts = collections.Counter(modes)
    for time, mode in zip(times, modes):
        assert fault.calc_rate(
            time, mapping, sim_time=len(times), sim_units="sec"
        ) == pytest.approx(0.1 * factors[mode] * counts[mode])


class ScheduledModes(ExampleFunction):
    """Expose numbered operating modes through an actual model history."""

    def dynamic_behavior(self):
        self.m.set_mode(("nominal", "wait", "nominal", "nominal1")[int(self.t.time)])


def test_real_simulated_mode_history_preserves_revisited_mode():
    model = ScheduledModes(sp={"end_time": 3.0}, m={"mode": "nominal"})
    _, history = Simulation(mdl=model)()
    modes = list(history["m.mode"])
    assert modes == ["nominal", "wait", "nominal", "nominal1"]
    # A block history is prefixed just as it is in a function architecture.
    tagged = History({"plant.m.mode": history["m.mode"], "time": history.time})
    mapping = from_hist(tagged)["plant"]
    assert_partition(mapping, modes, history.time)
    assert model.m.mode == "nominal"
