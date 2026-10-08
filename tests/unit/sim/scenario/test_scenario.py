#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for scenario behavior and parameter lookup.

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
from collections import UserDict
from types import MappingProxyType

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim.sample import ParameterDomain, ParameterSample
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import Injection, ParameterScenario, Scenario, Sequence


class TestParameterScenario(unittest.TestCase):
    """Test parameter scenario field lookup behavior."""

    def test_existing_dictionary_entries_are_returned(self):
        fields = ("p", "r", "sp", "inputparams")
        values = (
            ("value", 3.5),
            ("zero", 0),
            ("enabled", False),
            ("label", "nominal"),
        )
        for field in fields:
            for key, value in values:
                with self.subTest(field=field, key=key, value=value):
                    scenario = ParameterScenario(**{field: {key: value}})
                    self.assertEqual(scenario.get_param(f"{field}.{key}"), value)
                    self.assertEqual(scenario.get_params(f"{field}.{key}"), [value])

    def test_missing_dictionary_entry_preserves_default(self):
        for field in ("p", "r", "sp", "inputparams"):
            with self.subTest(field=field):
                scenario = ParameterScenario(**{field: {"present": 1}})
                marker = object()
                self.assertEqual(scenario.get_param(f"{field}.absent"), "NA")
                self.assertIs(
                    scenario.get_param(f"{field}.absent", default=marker), marker
                )

    def test_ordered_parameter_lookup_and_probability(self):
        scenario = ParameterScenario(
            p={"x": 3.5},
            r={"seed": 42},
            sp={"end_time": 10.0},
            prob=0.25,
        )
        before = copy.deepcopy(scenario.asdict())
        self.assertEqual(
            scenario.get_params("p.x", "sp.end_time", "r.seed", "prob"),
            [3.5, 10.0, 42, 0.25],
        )
        self.assertEqual(
            scenario.get_params("r.seed", "p.x", "p.x"),
            [42, 3.5, 3.5],
        )
        self.assertEqual(scenario.get_param("unrecognized"), "NA")
        self.assertEqual(scenario.get_params(), [])
        self.assertEqual(scenario.asdict(), before)

    def test_remaining_path_is_kept_as_a_dictionary_key(self):
        scenario = ParameterScenario(
            p={"component.gain": 2.0, "component": {"gain": 9.0}}
        )
        self.assertEqual(scenario.get_param("p.component.gain"), 2.0)

    def test_lookup_from_actual_parameter_sample_scenarios(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        sample = ParameterSample(domain)
        sample.add_variable_scenario(
            2.0,
            3.0,
            seed=123,
            sp={"end_time": 10.0},
            inputparams={"command": "hold"},
        )
        sample.add_variable_scenario(
            4.0,
            1.0,
            seed=456,
            sp={"end_time": 20.0},
            inputparams={"command": "move"},
        )
        fields = ("p.x", "p.y", "r.seed", "sp.end_time", "inputparams.command")
        self.assertEqual(
            [scenario.get_params(*fields) for scenario in sample.scenarios()],
            [
                [2.0, 3.0, 123, 10.0, "hold"],
                [4.0, 1.0, 456, 20.0, "move"],
            ],
        )


class TestInjectionUpdates(unittest.TestCase):
    """Dictionary updates must have the same effects as Injection updates."""

    def test_mapping_and_injection_forms_apply_identical_updates(self):
        payloads = (
            {},
            {"faults": {"pump": ["low"]}},
            {"disturbances": {"s.x": 0.0}},
            {
                "faults": {"pump": ["low"], "valve": ["stuck"]},
                "disturbances": {"s.x": 0.0, "s.enabled": False},
            },
        )
        for payload in payloads:
            for wrapper in (dict, UserDict, MappingProxyType):
                with self.subTest(fields=tuple(payload), mapping=wrapper.__name__):
                    raw = copy.deepcopy(payload)
                    before = copy.deepcopy(raw)
                    actual = Injection(
                        faults={"pump": ["old"], "motor": ["off"]},
                        disturbances={"s.x": 4.0, "s.y": 2.0},
                    )
                    expected = Injection(
                        faults=copy.deepcopy(actual.faults),
                        disturbances=copy.deepcopy(actual.disturbances),
                    )
                    expected.update(Injection(**copy.deepcopy(payload)))
                    self.assertIsNone(actual.update(wrapper(raw)))
                    self.assertEqual(actual.asdict(), expected.asdict())
                    self.assertEqual(raw, before)

    def test_repeated_updates_replace_same_scope_and_preserve_other_fields(self):
        injection = Injection(faults={"pump": ["old"]}, disturbances={"s.y": 3.0})
        injection.update({"faults": {"pump": {"low": {"prob": 0.2}}}})
        injection.update({"faults": {"motor": ["off"]}, "disturbances": {"s.x": 0.0}})
        injection.update({"faults": {}, "disturbances": {}})
        self.assertEqual(
            injection.faults, {"pump": {"low": {"prob": 0.2}}, "motor": ["off"]}
        )
        self.assertEqual(injection.disturbances, {"s.y": 3.0, "s.x": 0.0})
        injection.update({"faults": {"pump": []}, "unused": {"ignored": True}})
        self.assertEqual(injection.faults, {"pump": [], "motor": ["off"]})
        self.assertEqual(injection.disturbances, {"s.y": 3.0, "s.x": 0.0})

    def test_sequence_existing_time_updates_accept_dictionary_injections(self):
        actual = Sequence({1.0: {"pump": ["old"]}}, {2.0: {"s.y": 1.0}})
        reference = Sequence({1.0: {"pump": ["old"]}}, {2.0: {"s.y": 1.0}})
        updates = {
            1.0: {"faults": {"pump": ["low"]}, "disturbances": {"s.x": 5.0}},
            2.0: {"disturbances": {"s.y": 0.0}},
        }
        before = copy.deepcopy(updates)
        actual.update_sequence(updates)
        reference.update_sequence(
            {time: Injection(**payload) for time, payload in before.items()}
        )
        for time in actual:
            self.assertEqual(actual[time].asdict(), reference[time].asdict())
        self.assertEqual(updates, before)

    def test_real_fault_and_disturbance_simulations_match_explicit_injections(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        for faults, disturbances in (
            ({model.name: ["no_charge"]}, {}),
            ({}, {"s.x": 20.0}),
            ({model.name: ["no_charge"]}, {"s.x": 20.0}),
        ):
            with self.subTest(faults=bool(faults), disturbances=bool(disturbances)):
                actual = Sequence(disturbances={1.0: {}})
                payload = {
                    "faults": copy.deepcopy(faults),
                    "disturbances": copy.deepcopy(disturbances),
                }
                actual[1.0].update(payload)
                reference = Sequence({1.0: faults}, {1.0: disturbances})
                result, history = Simulation(
                    mdl=model,
                    scen=Scenario(name="updated", sequence=actual, times=(1.0,)),
                )()
                expected_result, expected_history = Simulation(
                    mdl=model,
                    scen=Scenario(name="direct", sequence=reference, times=(1.0,)),
                )()
                self.assertEqual(result, expected_result)
                self.assertEqual(history, expected_history)
                _, nominal = Simulation(mdl=model)()
                self.assertFalse(np.array_equal(history["s.x"], nominal["s.x"]))
                self.assertEqual(
                    payload, {"faults": faults, "disturbances": disturbances}
                )
                self.assertEqual(model.s.x, 0.0)
                self.assertFalse(model.m.any_faults())


if __name__ == "__main__":
    unittest.main()
