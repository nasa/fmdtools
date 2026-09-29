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
import unittest

import numpy as np

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


class ScheduledModes(ExampleFunction):
    """Expose numbered operating modes through an actual model history."""

    def dynamic_behavior(self):
        self.m.set_mode(("nominal", "wait", "nominal", "nominal1")[int(self.t.time)])


class TestHistoryPhaseIdentifiers(unittest.TestCase):
    """Run the regression cases with standard unittest discovery."""

    def assert_partition(self, mapping, modes, times):
        self.assertEqual(
            list(mapping.phases.values()), expected_intervals(modes, times)
        )
        counts = collections.Counter(modes)
        phase_ids = [phase for group in mapping.modephases.values() for phase in group]
        self.assertEqual(len(phase_ids), len(set(phase_ids)))
        self.assertEqual(set(phase_ids), set(mapping.phases))
        self.assertEqual(mapping.calc_samples_in_phases(*times), counts)
        for mode in counts:
            expected_times = [
                time for time, value in zip(times, modes) if value == mode
            ]
            self.assertEqual(mapping.get_phase_times(mode), expected_times)
            np.testing.assert_allclose(
                mapping.calc_modephase_time(mode),
                counts[mode] * mapping.dt,
                rtol=1e-06,
                atol=1e-12,
            )
        for time, mode in zip(times, modes):
            self.assertEqual(mapping.find_base_phase(time), mode)

    def test_numbered_mode_names_preserve_every_interval_and_membership(self):
        for dt in [0.25, 1.0]:
            for modes in MODE_SEQUENCES:
                with self.subTest(dt=dt, modes=modes):
                    times = np.arange(len(modes)) * dt + 10.0
                    history = History(
                        {"plant.m.mode": modes.copy(), "time": times.copy()}
                    )
                    mapping = from_hist(history, dt=dt)["plant"]
                    self.assert_partition(mapping, modes, times)
                    self.assertEqual(history["plant.m.mode"], modes)
                    np.testing.assert_array_equal(history.time, times)

    def test_exhaustive_short_histories_partition_even_when_labels_overlap(self):
        labels = ("on", "on1", "on2")
        for count in range(1, 6):
            for modes in itertools.product(labels, repeat=count):
                times = np.arange(count)
                mapping = from_hist(History({"plant.m.mode": modes, "time": times}))[
                    "plant"
                ]
                self.assert_partition(mapping, modes, times)

    def test_suffix_allocation_reserves_real_modes_before_generating_names(self):
        modes = ["on", "off", "on", "on1", "on2", "off", "on"]
        mapping = from_hist(History({"plant.m.mode": modes, "time": list(range(7))}))[
            "plant"
        ]
        self.assertEqual(
            list(mapping.phases), ["on", "off", "on3", "on1", "on2", "off1", "on4"]
        )
        self.assertEqual(mapping.modephases["on"], {"on", "on3", "on4"})
        self.assertEqual(mapping.modephases["on1"], {"on1"})
        self.assertEqual(mapping.modephases["on2"], {"on2"})

    def test_existing_noncolliding_labels_and_empty_histories_are_unchanged(self):
        mapping = from_hist(
            History(
                {
                    "plant.m.mode": ["on", "off", "on", "off", "on"],
                    "time": [0, 1, 2, 3, 4],
                }
            )
        )["plant"]
        self.assertEqual(
            mapping.phases,
            {"on": [0, 0], "off": [1, 1], "on1": [2, 2], "off1": [3, 3], "on2": [4, 4]},
        )
        self.assertEqual(
            mapping.modephases, {"on": {"on", "on1", "on2"}, "off": {"off", "off1"}}
        )
        self.assertEqual(from_hist(History({"plant.m.mode": [], "time": []})), {})
        with self.assertRaisesRegex(Exception, "Value m.mode not in Result keys"):
            from_hist(History({"time": [0, 1, 2]}))

    def test_each_function_has_independent_labels_with_optional_mode_grouping(self):
        for selection in ["all", ["first"], []]:
            with self.subTest(selection=selection):
                modes = MODE_SEQUENCES[0]
                times = np.arange(len(modes))
                history = History(
                    {
                        "first.m.mode": modes,
                        "second.m.mode": np.array(modes[::-1]),
                        "time": times,
                    }
                )
                mappings = from_hist(history, fxn_modephases=selection)
                self.assertEqual(list(mappings), ["first", "second"])
                for name, values in [("first", modes), ("second", modes[::-1])]:
                    mapping = mappings[name]
                    self.assertEqual(
                        list(mapping.phases.values()), expected_intervals(values, times)
                    )
                    if selection == "all" or name in selection:
                        self.assert_partition(mapping, values, times)
                    else:
                        self.assertEqual(mapping.modephases, {})
                        self.assertEqual(len(mapping.phases), 4)

    def test_correct_times_flow_into_fault_sampling_and_exposure_rates(self):
        modes = ["on", "off", "on", "on1", "on1", "on1"]
        times = np.arange(len(modes), dtype=float)
        mapping = from_hist(History({"plant.m.mode": modes, "time": times}))["plant"]
        model = ExampleFunction(sp={"end_time": 5.0})
        domain = FaultDomain(model)
        domain.add_fault("examplefunction", "low")
        sample = FaultSample(domain, phasemap=mapping, def_mdl_phasemap=False)
        sample.add_fault_phases("on", method="all")
        self.assertEqual(sample.get_times(), [0.0, 2.0])
        self.assertEqual([scenario.time for scenario in sample.scenarios()], [0.0, 2.0])
        factors = {"on": 0.2, "off": 0.3, "on1": 0.5}
        fault = Fault(prob=0.1, phases=tuple(factors.items()), units="sec")
        counts = collections.Counter(modes)
        for time, mode in zip(times, modes):
            np.testing.assert_allclose(
                fault.calc_rate(time, mapping, sim_time=len(times), sim_units="sec"),
                0.1 * factors[mode] * counts[mode],
                rtol=1e-06,
                atol=1e-12,
            )

    def test_real_simulated_mode_history_preserves_revisited_mode(self):
        model = ScheduledModes(sp={"end_time": 3.0}, m={"mode": "nominal"})
        _, history = Simulation(mdl=model)()
        modes = list(history["m.mode"])
        self.assertEqual(modes, ["nominal", "wait", "nominal", "nominal1"])
        # A block history is prefixed just as it is in a function architecture.
        tagged = History({"plant.m.mode": history["m.mode"], "time": history.time})
        mapping = from_hist(tagged)["plant"]
        self.assert_partition(mapping, modes, history.time)
        self.assertEqual(model.m.mode, "nominal")


if __name__ == "__main__":
    unittest.main()
