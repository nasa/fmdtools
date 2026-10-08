#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for callable parameter factories.

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

import unittest
from dataclasses import dataclass
from functools import partial

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim.sample import ParameterDomain, ParameterSample
from fmdtools.sim.search import ParameterSimProblem


def mapping_factory(x=1.0, y=2.0, label="nominal"):
    return {"x": x, "y": y, "label": label}


def model_factory(x=1.0, y=2.0):
    return ExampleParameter(x=float(x), y=float(y))


class Factory:
    def make(self, **kwargs):
        return mapping_factory(**kwargs)

    def __call__(self, **kwargs):
        return mapping_factory(**kwargs)


@dataclass
class PlainParameters:
    x: float = 1.0
    y: float = 2.0


class FactorySimulation(ParameterSimProblem):
    def init_problem(self, **kwargs):
        domain = ParameterDomain(model_factory)
        domain.add_variables("x", "y")
        self.add_parameterdomain(domain)
        self.add_sim(ExampleFunction(sp={"end_time": 3.0}), "nominal")
        self.add_result_objective("cost", "s.x", time=3.0)
        self.add_result_constraint(
            "bound", "s.x", time=3.0, threshold=7.0, comparator="less"
        )


class TestCallableParameterDomains(unittest.TestCase):
    def test_functions_methods_partials_and_callable_objects_accept_variables(self):
        owner = Factory()
        for factory in (
            mapping_factory,
            owner.make,
            owner,
            partial(mapping_factory, label="custom"),
        ):
            with self.subTest(factory=type(factory).__name__):
                domain = ParameterDomain(factory)
                domain.add_variables("x", "y")
                self.assertEqual(domain.variables, {"x": (), "y": ()})
                self.assertEqual(domain.get_x_defaults(), (1.0, 2.0))
                self.assertEqual(domain.get_set_constraints(9.0, -8.0), (False, False))
                actual = domain(0.0, 3.0)
                self.assertEqual((actual["x"], actual["y"]), (0.0, 3.0))
                self.assertEqual(
                    actual["label"],
                    "custom" if isinstance(factory, partial) else "nominal",
                )

    def test_variable_mapping_constants_and_kwargs_reach_the_factory(self):
        seen = []

        def factory(**kwargs):
            seen.append(kwargs.copy())
            return kwargs

        domain = ParameterDomain(factory)
        domain.add_variable("x", var_map=lambda x: (x * 2,))
        domain.add_variable("y")
        domain.add_constant("label", "mapped")
        self.assertEqual(domain(2.0, 3.0), {"x": 4.0, "y": 3.0, "label": "mapped"})
        self.assertEqual(seen, [{"x": 4.0, "y": 3.0, "label": "mapped"}])
        self.assertEqual(domain.variables, {"x": (), "y": ()})

    def test_explicit_domains_and_parameter_class_inference_remain_unchanged(self):
        domain = ParameterDomain(mapping_factory)
        domain.add_variables("x", "y", lims={"x": (0.0, 2.0)}, sets={"y": (1.0, 3.0)})
        self.assertEqual(domain.variables, {"x": (0.0, 2.0), "y": {1.0, 3.0}})
        sample = ParameterSample(domain)
        self.assertEqual(
            set(sample.combine_product()),
            {(x, y) for x in (0.0, 1.0, 2.0) for y in (1.0, 3.0)},
        )
        typed = ParameterDomain(ExampleParameter)
        typed.add_variables("x", "y")
        self.assertEqual(typed.variables, {"x": (0, 10), "y": {1.0, 2.0, 3.0, 4.0}})
        self.assertIsInstance(typed(2.0, 3.0), ExampleParameter)
        plain = ParameterDomain(PlainParameters)
        plain.add_variables("x", "y")
        self.assertEqual(plain(4.0, 5.0), PlainParameters(4.0, 5.0))

    def test_manual_replicates_keep_inputs_probabilities_and_seed(self):
        domain = ParameterDomain(mapping_factory)
        domain.add_variables("x", "y")
        sample = ParameterSample(domain, seed=0)
        sample.add_variable_replicates([(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(
            [s.p for s in sample.scenarios()],
            [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        )
        self.assertEqual([s.prob for s in sample.scenarios()], [0.5, 0.5])
        self.assertTrue(all(s.r == {"seed": 0} for s in sample.scenarios()))

    def test_factory_backed_simulation_matches_the_parameter_class_path(self):
        problem = FactorySimulation()
        self.assertEqual(problem.cost(2.0, 3.0), 6.0)
        self.assertEqual(problem.bound(2.0, 3.0), -1.0)
        np.testing.assert_array_equal(problem.hist["s.x"], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(problem.cost(1.0, 2.0), 3.0)
        np.testing.assert_array_equal(problem.hist["s.x"], [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
