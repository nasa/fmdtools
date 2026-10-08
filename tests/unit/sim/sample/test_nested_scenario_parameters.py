#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for nested scenario parameter lookup.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample
from fmdtools.sim.scenario import ParameterScenario


class InnerParameters(Parameter, readonly=True):
    gain: float = 2.0
    gain_set = (0.0, 2.0, 4.0)


class OuterParameters(Parameter, readonly=True):
    inner: InnerParameters = InnerParameters()


class TotalState(State):
    total: np.float64 = 0.0


class NestedFunction(Function):
    container_p = OuterParameters
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.p.inner.gain

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestNestedScenarioParameters(unittest.TestCase):
    def test_nested_values_are_found_in_each_scenario_mapping(self):
        for field in ("p", "r", "sp", "inputparams"):
            for wrapper in (dict, UserDict, MappingProxyType):
                data = {
                    "inner": wrapper(
                        {
                            "deep": {
                                "gain": 0.0,
                                "flag": False,
                                "label": "",
                                "unset": None,
                            }
                        }
                    )
                }
                scenario = ParameterScenario(**{field: data})
                with self.subTest(field=field, mapping=wrapper.__name__):
                    self.assertEqual(
                        scenario.get_params(
                            *(
                                field + ".inner.deep." + k
                                for k in ("gain", "flag", "label", "unset")
                            )
                        ),
                        [0.0, False, "", None],
                    )
                    self.assertEqual(
                        dict(getattr(scenario, field)["inner"]),
                        {
                            "deep": {
                                "gain": 0.0,
                                "flag": False,
                                "label": "",
                                "unset": None,
                            }
                        },
                    )

    def test_literal_dotted_keys_keep_precedence_at_every_mapping_level(self):
        data = {
            "component.gain": 2.0,
            "component": {"gain": 9.0, "deep.factor": 4.0, "deep": {"factor": 7.0}},
        }
        scenario = ParameterScenario(p=data)
        before = copy.deepcopy(scenario.asdict())
        self.assertEqual(scenario.get_param("p.component.gain"), 2.0)
        self.assertEqual(scenario.get_param("p.component.deep.factor"), 4.0)
        self.assertEqual(scenario.get_param("p.component.deep"), {"factor": 7.0})
        self.assertEqual(scenario.asdict(), before)

    def test_missing_roots_leaves_and_nonmapping_intermediates_use_the_default(self):
        scenario = ParameterScenario(
            p={"inner": {"gain": 2.0}, "none": None, "sequence": [1, 2]}, prob=0.25
        )
        marker = object()
        for name in (
            "absent.gain",
            "p.absent.gain",
            "p.inner.absent",
            "p.inner.gain.child",
            "p.none.child",
            "p.sequence.0",
            "prob.child",
            "name.child",
        ):
            with self.subTest(name=name):
                self.assertIs(scenario.get_param(name, default=marker), marker)
                self.assertEqual(scenario.get_param(name), "NA")
        self.assertEqual(scenario.get_param("prob"), 0.25)
        self.assertEqual(scenario.get_param("unknown"), "NA")

    def test_ordered_and_repeated_queries_do_not_change_parameter_values(self):
        scenario = ParameterScenario(
            p={"inner": {"gain": 2.0, "factor": 3.0}}, r={"seed": 0}
        )
        before = copy.deepcopy(scenario.asdict())
        self.assertEqual(
            scenario.get_params(
                "p.inner.factor", "r.seed", "p.inner.gain", "p.inner.factor"
            ),
            [3.0, 0, 2.0, 3.0],
        )
        self.assertEqual(scenario.get_params(), [])
        self.assertEqual(scenario.asdict(), before)

    def test_sampled_nested_parameters_match_the_simulated_state_totals(self):
        domain = ParameterDomain(OuterParameters)
        domain.add_variable("inner.gain")
        sample = ParameterSample(domain, seed=0)
        sample.add_variable_ranges()
        result, history = propagate.parameter_sample(
            NestedFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        observed = []
        for scenario in sample.scenarios():
            gain = scenario.get_param("p.inner.gain")
            self.assertIsInstance(gain, (float, np.floating))
            observed.append(gain)
            self.assertEqual(result[scenario.name + ".tend.classify.total"], 3 * gain)
            np.testing.assert_allclose(
                history[scenario.name + ".s.total"], np.arange(4) * gain
            )
        self.assertEqual(sorted(observed), [0.0, 2.0, 4.0])
        self.assertAlmostEqual(sum(s.prob for s in sample.scenarios()), 1.0)


if __name__ == "__main__":
    unittest.main()
