#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for independent mutable defaults in scenario objects.

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
from contextlib import contextmanager
import pickle
import unittest

from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation
from fmdtools.sim.scenario import (
    BaseScenario,
    Injection,
    JointFaultScenario,
    ParameterScenario,
    Scenario,
    Sequence,
    SingleFaultScenario,
)


@contextmanager
def preserved_mapping(mapping):
    """Avoid contaminating other tests when reproducing shared defaults."""
    original = copy.deepcopy(mapping)
    try:
        yield
    finally:
        mapping.clear()
        mapping.update(original)


class TestScenarioDefaultIndependence(unittest.TestCase):
    def test_injections_have_independent_default_mappings(self):
        for field, change in (
            ("faults", {"examplefunction": ["low"]}),
            ("disturbances", {"s.x": 12.0}),
        ):
            with self.subTest(field=field):
                first, second = Injection(), Injection()
                mapping = getattr(first, field)
                with preserved_mapping(mapping):
                    first.update(Injection(**{field: change}))
                    self.assertEqual(getattr(first, field), change)
                    self.assertEqual(getattr(second, field), {})
                    self.assertEqual(getattr(Injection(), field), {})
                    self.assertIsNot(mapping, getattr(second, field))

    def test_partial_injection_arguments_do_not_share_omitted_fields(self):
        factories = (
            (lambda: Injection(faults={"examplefunction": ["low"]}), "disturbances"),
            (lambda: Injection(disturbances={"s.x": 1}), "faults"),
        )
        for factory, field in factories:
            with self.subTest(field=field):
                first, second = factory(), factory()
                with preserved_mapping(getattr(first, field)):
                    getattr(first, field)["sentinel"] = 1
                    self.assertEqual(getattr(second, field), {})
                    self.assertEqual(getattr(factory(), field), {})

    def test_scenario_classes_have_independent_default_sequences(self):
        classes = (
            BaseScenario,
            Scenario,
            SingleFaultScenario,
            JointFaultScenario,
            ParameterScenario,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                first, second = cls(), cls()
                with preserved_mapping(first.sequence):
                    first.sequence.update_sequence(
                        Sequence(faultseq={1: {"examplefunction": ["low"]}})
                    )
                    self.assertEqual(second.sequence, {})
                    self.assertEqual(cls().sequence, {})
                    self.assertIsInstance(first.sequence, Sequence)
                    self.assertIsNot(first.sequence, second.sequence)
                    for other in classes:
                        self.assertEqual(other().sequence, {})

    def test_parameter_scenarios_have_independent_default_dictionaries(self):
        for field in ("p", "r", "sp", "inputparams"):
            with self.subTest(field=field):
                first, second = ParameterScenario(), ParameterScenario()
                with preserved_mapping(getattr(first, field)):
                    getattr(first, field)["value"] = 7
                    self.assertEqual(getattr(second, field), {})
                    self.assertEqual(getattr(ParameterScenario(), field), {})
                    self.assertIsNot(getattr(first, field), getattr(second, field))

    def test_joint_fault_factories_do_not_share_default_disturbances(self):
        faults = (("first", "low"), ("second", "high"))
        first = JointFaultScenario.from_faults(faults, 1)
        second = JointFaultScenario.from_faults(faults, 2)
        with preserved_mapping(first.sequence[1].disturbances):
            first.sequence[1].update(Injection(disturbances={"s.x": 12}))
            self.assertEqual(second.sequence[2].disturbances, {})
            self.assertEqual(
                JointFaultScenario.from_faults(faults, 3).sequence[3].disturbances, {}
            )
            self.assertEqual(
                second.sequence[2].faults, {"first": ["low"], "second": ["high"]}
            )

    def test_fault_added_to_one_scenario_does_not_change_another_simulation(self):
        faulty, nominal = Scenario(name="faulty"), Scenario(name="nominal")
        with preserved_mapping(faulty.sequence):
            faulty.sequence.update_sequence(
                Sequence(faultseq={1: {"examplefunction": ["low"]}})
            )
            model = ExampleFunction(sp={"end_time": 3.0})
            expected, _ = Simulation(mdl=model, scen=Scenario(sequence=Sequence()))()
            actual, _ = Simulation(mdl=model, scen=nominal)()
            fault_result, _ = Simulation(mdl=model, scen=faulty)()
            self.assertEqual(dict(actual.flatten()), dict(expected.flatten()))
            self.assertNotEqual(dict(fault_result.flatten()), dict(expected.flatten()))

    def test_explicit_inputs_and_shallow_copies_keep_existing_identity(self):
        faults = {"fxn": ["low"]}
        disturbances = {"s.x": 3}
        injection = Injection(faults, disturbances)
        self.assertIs(injection.faults, faults)
        self.assertIs(injection.disturbances, disturbances)
        sequence = Sequence(faultseq={1: faults})
        scenario = ParameterScenario(sequence=sequence, p={"gain": 2}, r={"seed": 0})
        clone = scenario.copy_with(name="clone")
        self.assertIs(scenario.sequence, sequence)
        self.assertIs(clone.sequence, sequence)
        self.assertIs(clone.p, scenario.p)
        self.assertIs(clone.r, scenario.r)
        self.assertEqual(clone.name, "clone")

    def test_default_records_still_serialize_and_preserve_scalar_fields(self):
        for record in (
            Injection(),
            Scenario(),
            SingleFaultScenario(),
            JointFaultScenario(),
            ParameterScenario(),
        ):
            with self.subTest(cls=type(record).__name__):
                restored = pickle.loads(pickle.dumps(record))
                self.assertIs(type(restored), type(record))
                self.assertEqual(restored.asdict(), record.asdict())
                self.assertEqual(record.copy_with().asdict(), record.asdict())


if __name__ == "__main__":
    unittest.main()
