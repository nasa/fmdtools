#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for align parameter mappings with their named variables.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
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

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class InputParameter(Parameter, readonly=True):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Accumulation(State):
    total: float = 0.0


class MappedFunction(Function):
    container_p = InputParameter
    container_s = Accumulation

    def dynamic_behavior(self):
        self.s.total += self.p.x + 10 * self.p.y + 100 * self.p.z

    def classify(self, **kwargs):
        return {"total": self.s.total}


def mapped_domain(factory=dict):
    domain = ParameterDomain(factory)
    domain.add_variables("x", "y", "z", var_map=False)
    domain.add_variables("z", "x", var_map=lambda z, x: (z + 30, x + 10))
    domain.add_variable("y", var_map=lambda y: (y + 20,))
    return domain


class TestNamedParameterMaps(unittest.TestCase):
    def test_independent_groups_follow_names_not_registration_order(self):
        domain = mapped_domain()
        self.assertEqual(domain.get_map_vars(1, 2, 3), [11, 22, 33])
        self.assertEqual(domain.get_param_kwargs(1, 2, 3), {"x": 11, "y": 22, "z": 33})
        self.assertEqual(domain(1, 2, 3), {"x": 11, "y": 22, "z": 33})
        maps = domain.var_maps.copy()
        domain.var_maps = dict(reversed(list(maps.items())))
        self.assertEqual(domain.get_map_vars(1, 2, 3), [11, 22, 33])
        self.assertEqual(domain.variables, {"x": (), "y": (), "z": ()})

    def test_replacing_a_group_does_not_shift_or_duplicate_variables(self):
        domain = ParameterDomain(dict)
        domain.add_variables("x", "y")
        domain.add_variable("x", var_map=lambda x: (2 * x,))
        self.assertEqual(domain.get_map_vars(3, 4), [6, 4])
        self.assertEqual(domain(3, 4), {"x": 6, "y": 4})
        domain.add_variables("y", "x", var_map=lambda y, x: (y + 10, x + 20))
        self.assertEqual(domain.get_map_vars(3, 4), [23, 14])

    def test_passthrough_partial_inputs_and_constants_keep_order(self):
        domain = ParameterDomain(dict)
        domain.add_variables("x", "y", "z")
        domain.add_constant("fixed", 99)
        for values in [(), (1,), (1, 2), (1, 2, 3)]:
            with self.subTest(values=values):
                self.assertEqual(domain.get_map_vars(*values), list(values))
        self.assertEqual(
            domain.get_param_kwargs(1, 2, 3), {"fixed": 99, "x": 1, "y": 2, "z": 3}
        )
        domain = ParameterDomain(dict)
        domain.add_variable("x", var_map=False)
        domain.add_variable("y", var_map=lambda y: (2 * y,))
        self.assertEqual(domain(3, 4), {"x": 3, "y": 8})

    def test_mapping_arity_is_checked_without_mutating_domains(self):
        for mapper in [lambda x, y: (x,), lambda x, y: (x, y, 99)]:
            domain = ParameterDomain(dict)
            domain.add_variables("x", "y", var_map=mapper)
            before = copy.deepcopy(domain.variables)
            with self.assertRaisesRegex(ValueError, "mapped value"):
                domain.get_map_vars(1, 2)
            self.assertEqual(domain.variables, before)
        domain = ParameterDomain(dict)
        domain.add_variables("x", "y", var_map=lambda x, y: (item for item in (y, x)))
        self.assertEqual(domain.get_map_vars(1, 2), [2, 1])

    def test_sample_simulations_receive_the_intended_transformed_parameters(self):
        domain = mapped_domain(InputParameter)
        sample = ParameterSample(domain, seed=11)
        sample.add_variable_replicates([(1, 2, 3), (4, 5, 6)], replicates=2, weight=0.8)
        original = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        model = MappedFunction(sp={"end_time": 3.0})
        result, history = propagate.parameter_sample(model, sample, showprogress=False)
        for scenario, values in zip(
            sample.scenarios(), [(11, 22, 33)] * 2 + [(14, 25, 36)] * 2
        ):
            self.assertEqual(tuple(scenario.p[k] for k in ("x", "y", "z")), values)
            step = values[0] + 10 * values[1] + 100 * values[2]
            self.assertEqual(result[scenario.name + ".tend.classify.total"], 3 * step)
            np.testing.assert_allclose(
                history[scenario.name + ".s.total"], np.arange(4) * step
            )
        self.assertAlmostEqual(sum(s.prob for s in sample.scenarios()), 0.8)
        self.assertEqual([s.asdict() for s in sample.scenarios()], original)
        self.assertEqual(model.s.total, 0.0)


if __name__ == "__main__":
    unittest.main()
