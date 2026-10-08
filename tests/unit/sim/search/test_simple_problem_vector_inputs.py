#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for sampling and optimization inputs.

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

import numpy as np
from scipy.optimize import minimize

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import ParameterSimProblem, SimpleProblem


class QuadraticProblem(SimpleProblem):
    def init_problem(self, **kwargs):
        self.calls = 0
        self.add_variables("x", "y")
        self.add_objective("cost", self.evaluate_cost)
        self.add_objective("affine", lambda x, y: 2 * x + y)
        self.add_constraint(
            "balance",
            lambda x, y: x + y,
            threshold=2.0,
            comparator="greater",
            negative=True,
        )

    def evaluate_cost(self, x, y):
        self.calls += 1
        return (x - 2.0) ** 2 + (y + 1.0) ** 2


class SingleVariableProblem(SimpleProblem):
    def init_problem(self, **kwargs):
        self.add_variables("x")
        self.add_objective("cost", lambda x: x * x)


class VectorParameterProblem(ParameterSimProblem):
    def init_problem(self, **kwargs):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        self.add_sim(ExampleFunction(sp={"end_time": 3.0}), "nominal")
        self.add_parameterdomain(domain)
        self.add_result_objective("at_two", "s.x", time=2.0)
        self.add_result_constraint(
            "limit", "s.x", time=2.0, threshold=5.0, comparator="less"
        )


class TestSimpleProblemVectorInputs(unittest.TestCase):
    def test_vector_and_expanded_calls_share_values_and_cache(self):
        for constructor in (list, tuple, np.array):
            for entry in ("cost", "balance", "call_outputs"):
                with self.subTest(constructor=constructor.__name__, entry=entry):
                    problem = QuadraticProblem()
                    vector = constructor([3.0, 2.0])
                    before = np.array(vector, copy=True)
                    actual = getattr(problem, entry)(vector)
                    expected = {
                        "cost": 10.0,
                        "balance": 3.0,
                        "call_outputs": ([10.0, 8.0], [3.0]),
                    }[entry]
                    self.assertEqual(actual, expected)
                    self.assertEqual(problem.current_x(), [3.0, 2.0])
                    self.assertFalse(problem.new_x(vector))
                    self.assertFalse(problem.new_x(3.0, 2.0))
                    self.assertEqual(problem.cost(3.0, 2.0), 10.0)
                    self.assertEqual(problem.affine(vector), 8.0)
                    self.assertEqual(problem.balance(vector), 3.0)
                    self.assertEqual(problem.calls, 1)
                    self.assertEqual(len(problem.iter_hist.time), 1)
                    np.testing.assert_array_equal(vector, before)

    def test_changed_inputs_and_forced_updates_recompute_all_outputs(self):
        problem = QuadraticProblem()
        vector = np.array([3.0, 2.0])
        problem.cost(vector)
        vector[0] = 4.0
        self.assertEqual(problem.current_x(), [3.0, 2.0])
        self.assertTrue(problem.new_x(vector))
        self.assertEqual(problem.call_outputs(vector), ([13.0, 10.0], [4.0]))
        self.assertEqual(problem.calls, 2)
        problem.call_outputs(vector, force_update=True)
        self.assertEqual(problem.calls, 3)
        np.testing.assert_array_equal(problem.iter_hist.variables.x, [3.0, 4.0, 4.0])
        np.testing.assert_array_equal(
            problem.iter_hist.objectives.cost, [10.0, 13.0, 13.0]
        )
        read_only = np.array([4.0, 999.0, 2.0, 999.0])[::2]
        read_only.setflags(write=False)
        self.assertEqual(problem.cost(read_only), 13.0)
        self.assertEqual(problem.calls, 3)

    def test_single_variable_and_direct_update_forms_are_preserved(self):
        for value in (2.0, np.float64(2.0), [2.0], (2.0,), np.array([2.0])):
            with self.subTest(value_type=type(value).__name__):
                problem = SingleVariableProblem()
                self.assertEqual(problem.cost(value), 4.0)
                self.assertEqual(problem.cost(2.0), 4.0)
                self.assertEqual(len(problem.iter_hist.time), 1)
        problem = QuadraticProblem()
        problem.update_objectives(np.array([3.0, 2.0]))
        self.assertEqual(problem.get_objectives(), [10.0, 8.0])
        self.assertEqual(problem.get_constraints(), [3.0])

    def test_scipy_optimizers_reach_analytic_unconstrained_and_constrained_minima(self):
        for constrained in (False, True):
            with self.subTest(constrained=constrained):
                problem = QuadraticProblem()
                constraints = (
                    ({"type": "ineq", "fun": problem.balance},) if constrained else ()
                )
                result = minimize(
                    problem.cost,
                    np.array([5.0, 3.0]),
                    method="SLSQP" if constrained else "BFGS",
                    constraints=constraints,
                    options={"maxiter": 100},
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(
                    result.x, [2.5, -0.5] if constrained else [2.0, -1.0], atol=1e-5
                )
                self.assertAlmostEqual(
                    problem.cost(result.x), 0.5 if constrained else 0.0, places=8
                )
                if constrained:
                    self.assertGreaterEqual(problem.balance(result.x), -1e-6)

    def test_parameter_simulation_reuses_its_result_for_an_equal_vector(self):
        problem, reference = VectorParameterProblem(), VectorParameterProblem()
        vector = np.array([2.0, 3.0])
        self.assertEqual(problem.at_two(vector), reference.at_two(*vector))
        self.assertEqual(problem.res, reference.res)
        self.assertEqual(problem.hist, reference.hist)
        count = len(problem.iter_hist.time)
        saved_result = problem.res
        self.assertEqual(problem.limit(vector), -1.0)
        self.assertEqual(problem.at_two(*vector), 4.0)
        self.assertIs(problem.res, saved_result)
        self.assertEqual(len(problem.iter_hist.time), count)


if __name__ == "__main__":
    unittest.main()
