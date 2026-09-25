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

from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim.sample import ParameterDomain, ParameterSample
from fmdtools.sim.scenario import ParameterScenario


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


if __name__ == "__main__":
    unittest.main()
