#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault summaries and model calculations.

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

from fmdtools.sim.search import (
    ExampleProblemArchitecture,
    ProblemArchitecture,
    SimpleProblem,
)


class UpstreamProblem(SimpleProblem):
    def init_problem(self, **kwargs):
        self.calls = 0
        self.add_variables("x", "y")
        self.add_objective("cost", self.evaluate)

    def evaluate(self, x, y):
        self.calls += 1
        return x + y


class DownstreamProblem(SimpleProblem):
    def init_problem(self, **kwargs):
        self.calls = 0
        self.add_variables("x", "y", "z")
        self.add_objective("cost", self.evaluate)
        self.add_constraint(
            "balance",
            lambda x, y, z: x + y,
            threshold=2.0,
            comparator="greater",
            negative=True,
        )

    def evaluate(self, x, y, z):
        self.calls += 1
        return (x - 2.0) ** 2 + (y + 1.0) ** 2 + (z - 3.0) ** 2


class CoupledProblem(ProblemArchitecture):
    def init_problem(self, **kwargs):
        self.add_connector_variable("shared_x", "x")
        self.add_connector_variable("shared_y", "y")
        self.add_problem(
            "upstream", UpstreamProblem, outputs={"shared_x": ["x"], "shared_y": ["y"]}
        )
        self.add_problem(
            "downstream",
            DownstreamProblem,
            inputs={"shared_x": ["x"], "shared_y": ["y"]},
        )


class TestArchitectureVectorInputs(unittest.TestCase):
    def test_packed_and_expanded_full_callbacks_evaluate_the_same_point(self):
        for container in (list, tuple, np.array):
            for callback in ("downstream_cost_full", "downstream_balance_full"):
                with self.subTest(container=container.__name__, callback=callback):
                    problem = CoupledProblem()
                    vector = container([3.0, 2.0, 4.0])
                    actual = getattr(problem, callback)(vector)
                    self.assertEqual(
                        actual, 11.0 if callback.endswith("cost_full") else 3.0
                    )
                    self.assertEqual(actual, getattr(problem, callback)(*vector))
                    self.assertEqual(
                        problem.problems["upstream"].current_x(), [3.0, 2.0]
                    )
                    self.assertEqual(
                        problem.problems["downstream"].current_x(), [3.0, 2.0, 4.0]
                    )
                    np.testing.assert_array_equal(
                        problem.variables["upstream_xloc"].values, [3.0, 2.0]
                    )
                    np.testing.assert_array_equal(
                        problem.variables["downstream_xloc"].values, [4.0]
                    )
                    np.testing.assert_array_equal(vector, [3.0, 2.0, 4.0])

    def test_partial_execution_splits_local_groups_and_leaves_later_problems_unrun(
        self,
    ):
        problem = CoupledProblem()
        problem.update_full_problem(np.array([1.0, 2.0]), probname="upstream")
        self.assertEqual(problem.upstream_cost(1.0, 2.0), 3.0)
        self.assertEqual(problem.problems["upstream"].calls, 1)
        self.assertEqual(problem.problems["downstream"].calls, 0)
        self.assertFalse(problem.problems["downstream"].consistent)
        problem.update_full_problem([3.0, 2.0, 4.0])
        self.assertEqual(problem.objectives["downstream_cost"], 11.0)
        self.assertEqual(problem.constraints["downstream_balance"], 3.0)
        self.assertTrue(all(p.consistent for p in problem.problems.values()))

    def test_cache_forcing_and_readonly_strided_vectors_preserve_values(self):
        problem = CoupledProblem()
        vector = np.array([3.0, 99.0, 2.0, 99.0, 4.0, 99.0])[::2]
        vector.setflags(write=False)
        problem.update_full_problem(vector)
        problem.downstream_balance_full(vector)
        self.assertEqual([p.calls for p in problem.problems.values()], [1, 1])
        problem.update_full_problem(vector, force_update=True)
        self.assertEqual([p.calls for p in problem.problems.values()], [2, 2])
        changed = np.array([4.0, 2.0, 4.0])
        problem.update_objectives(changed)
        self.assertEqual(problem.objectives["downstream_cost"], 14.0)
        changed[0] = 100.0
        self.assertEqual(problem.problems["upstream"].variables["x"], 4.0)
        np.testing.assert_array_equal(vector, [3.0, 2.0, 4.0])
        self.assertGreater(len(problem.iter_hist.time), 0)

    def test_scipy_optimizes_full_architecture_callbacks_to_analytic_minima(self):
        for constrained in (False, True):
            with self.subTest(constrained=constrained):
                problem = CoupledProblem()
                constraints = (
                    ({"type": "ineq", "fun": problem.downstream_balance_full},)
                    if constrained
                    else ()
                )
                result = minimize(
                    problem.downstream_cost_full,
                    np.array([5.0, 3.0, 6.0]),
                    method="SLSQP" if constrained else "BFGS",
                    constraints=constraints,
                    options={"maxiter": 100},
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(
                    result.x,
                    [2.5, -0.5, 3.0] if constrained else [2.0, -1.0, 3.0],
                    atol=1e-5,
                )
                self.assertAlmostEqual(
                    result.fun, 0.5 if constrained else 0.0, places=8
                )
                if constrained:
                    self.assertGreaterEqual(
                        problem.downstream_balance_full(result.x), -1e-6
                    )

    def test_full_vector_inputs_drive_fault_and_disturbance_simulations(self):
        problem, reference = ExampleProblemArchitecture(), ExampleProblemArchitecture()
        problem.update_full_problem(np.array([2.0, 3.0]))
        reference.update_full_problem(2.0, 3.0)
        self.assertEqual(problem.objectives, reference.objectives)
        self.assertEqual(problem.constraints, reference.constraints)
        for name in ("ex_scenprob", "ex_dp"):
            self.assertEqual(problem.problems[name].res, reference.problems[name].res)
            self.assertEqual(problem.problems[name].hist, reference.problems[name].hist)
        self.assertEqual(problem.objectives["ex_sp_f1"], 5.0)
        self.assertEqual(problem.objectives["ex_dp_f1"], 3.0)


if __name__ == "__main__":
    unittest.main()
