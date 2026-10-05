#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent phase-map defaults.

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

from fmdtools.analyze.phases import PhaseMap, join_phasemaps
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestPhaseMapDefaultIsolation(unittest.TestCase):
    """Default mode registries must not leak across unrelated phase maps."""

    def test_default_mode_dictionaries_are_independent(self):
        first = PhaseMap({"first": [0, 1]})
        second = PhaseMap({"second": [2, 3]})
        self.assertIsNot(first.modephases, second.modephases)

    def test_editing_one_map_does_not_break_other_samples(self):
        first = PhaseMap({"first": [0, 1]})
        second = PhaseMap({"second": [2, 3]})
        original = copy.deepcopy(first.modephases)
        try:
            first.modephases["operating"] = {"first"}
            self.assertEqual(first.get_sample_times(), {"operating": [0.0, 1.0]})
            for independent in (second, PhaseMap({"second": [2, 3]})):
                self.assertEqual(independent.get_sample_times(), {"second": [2.0, 3.0]})
                self.assertEqual(independent.find_base_phase(2), "second")
        finally:
            first.modephases.clear()
            first.modephases.update(original)

    def test_explicit_mode_maps_keep_their_existing_identity(self):
        modes = {"operating": {"first", "second"}}
        first = PhaseMap({"first": [0, 1], "second": [2, 3]}, modes)
        second = PhaseMap({"first": [0, 1], "second": [2, 3]}, modes)
        self.assertIs(first.modephases, modes)
        self.assertIs(second.modephases, modes)
        self.assertEqual(first.get_sample_times(), {"operating": [0.0, 1.0, 2.0, 3.0]})

    def test_joint_map_does_not_retain_another_maps_modes(self):
        first = PhaseMap({"first": [0, 1]})
        original = copy.deepcopy(first.modephases)
        try:
            first.modephases["operating"] = {"first"}
            joined = join_phasemaps(
                PhaseMap({"a": [0, 1]}, {}), PhaseMap({"b": [0, 1]}, {})
            )
            self.assertEqual(joined.modephases, {})
            self.assertEqual(joined.get_sample_times(), {("a", "b"): [0.0, 1.0]})
        finally:
            first.modephases.clear()
            first.modephases.update(original)

    def test_default_fault_sample_phase_metadata_stays_local(self):
        model = ExFxnArch(sp={"end_time": 2.0})
        domain = FaultDomain(model)
        domain.add_fault("ex_fxn", "low")
        first, second = FaultSample(domain), FaultSample(domain)
        original = copy.deepcopy(first.phasemap.modephases)
        try:
            phase = next(iter(first.phasemap.phases))
            first.phasemap.modephases["regrouped"] = {phase}
            first.add_fault_times([1.0])
            second.add_fault_times([1.0])
            self.assertEqual(first.scenarios()[0].phase, "regrouped")
            self.assertEqual(second.scenarios()[0].phase, phase)
        finally:
            first.phasemap.modephases.clear()
            first.phasemap.modephases.update(original)


if __name__ == "__main__":
    unittest.main()
