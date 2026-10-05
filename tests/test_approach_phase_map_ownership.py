#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for ownership of SampleApproach phase-map dictionaries.

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

from types import MappingProxyType
import unittest

from fmdtools.analyze.phases import PhaseMap
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, SampleApproach


class TestApproachPhaseMapOwnership(unittest.TestCase):
    def test_default_approaches_keep_their_own_model_phase_maps(self):
        first = SampleApproach(ExampleFunction(sp={"end_time": 3.0}))
        saved = first.phasemaps["mdl"]
        second = SampleApproach(ExampleFunction(sp={"end_time": 9.0}))
        self.assertIsNot(first.phasemaps, second.phasemaps)
        self.assertIs(first.phasemaps["mdl"], saved)
        self.assertEqual(first.phasemaps["mdl"].phases, {"na": [0.0, 3.0]})
        self.assertEqual(second.phasemaps["mdl"].phases, {"na": [0.0, 9.0]})
        first.phasemaps["local"] = PhaseMap({"local": [0.0, 1.0]})
        self.assertNotIn("local", second.phasemaps)
        without_model = SampleApproach(ExampleFunction(), def_mdl_phasemap=False)
        self.assertEqual(without_model.phasemaps, {})

    def test_explicit_mapping_keys_are_copied_but_phase_objects_remain_shared(self):
        for readonly in (False, True):
            for include_model in (False, True):
                with self.subTest(readonly=readonly, include_model=include_model):
                    custom = PhaseMap({"custom": [0.0, 2.0]})
                    original_model_map = PhaseMap({"provided": [0.0, 8.0]})
                    source = {"custom": custom, "mdl": original_model_map}
                    argument = MappingProxyType(source) if readonly else source
                    approach = SampleApproach(
                        ExampleFunction(sp={"end_time": 3.0}),
                        phasemaps=argument,
                        def_mdl_phasemap=include_model,
                    )
                    self.assertEqual(list(source), ["custom", "mdl"])
                    self.assertIs(source["mdl"], original_model_map)
                    self.assertIs(approach.phasemaps["custom"], custom)
                    if include_model:
                        self.assertEqual(
                            approach.phasemaps["mdl"].phases, {"na": [0.0, 3.0]}
                        )
                    else:
                        self.assertIs(approach.phasemaps["mdl"], original_model_map)
                    source["added_after"] = PhaseMap({"after": [0.0, 1.0]})
                    self.assertNotIn("added_after", approach.phasemaps)
                    approach.phasemaps.pop("custom")
                    self.assertIs(source["custom"], custom)

    def test_reusing_an_explicit_mapping_does_not_replace_an_earlier_model_map(self):
        for include_model in (False, True):
            with self.subTest(include_model=include_model):
                source = {}
                first = SampleApproach(
                    ExampleFunction(sp={"end_time": 2.0}),
                    phasemaps=source,
                    def_mdl_phasemap=include_model,
                )
                before = dict(first.phasemaps)
                second = SampleApproach(
                    ExampleFunction(sp={"end_time": 8.0}),
                    phasemaps=source,
                    def_mdl_phasemap=include_model,
                )
                self.assertEqual(source, {})
                self.assertEqual(first.phasemaps, before)
                self.assertIsNot(first.phasemaps, second.phasemaps)
                first.phasemaps["new"] = PhaseMap({"test": [0.0, 1.0]})
                self.assertNotIn("new", second.phasemaps)

    def test_actual_phase_sampling_and_simulation_ignore_later_approach_creation(self):
        model = ExampleFunction(
            sp={"end_time": 4.0, "phases": (("early", 0.0, 2.0), ("late", 3.0, 4.0))}
        )
        approach = SampleApproach(model)
        approach.add_faultdomain("faults", "fault", model.name, "low")
        SampleApproach(
            ExampleFunction(sp={"end_time": 9.0, "phases": (("other", 0.0, 9.0),)})
        )
        approach.add_faultsample(
            "early_sample",
            "fault_phases",
            "faults",
            "early",
            phasemap="mdl",
            method="all",
        )
        domain = FaultDomain(model)
        domain.add_fault(model.name, "low")
        reference = FaultSample(domain, phasemap=PhaseMap(model.sp.phases))
        reference.add_fault_phases("early", method="all")
        self.assertEqual(
            [s.asdict() for s in approach.scenarios()],
            [s.asdict() for s in reference.scenarios()],
        )
        self.assertEqual(sorted(approach.get_times()), [0.0, 1.0, 2.0])
        self.assertAlmostEqual(sum(s.rate for s in approach.scenarios()), 1.0)
        actual_result, actual_history = propagate.fault_sample(
            model, approach, showprogress=False
        )
        expected_result, expected_history = propagate.fault_sample(
            model, reference, showprogress=False
        )
        self.assertEqual(actual_result, expected_result)
        self.assertEqual(actual_history, expected_history)


if __name__ == "__main__":
    unittest.main()
