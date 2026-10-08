"""Regression tests for explicit time-zero optimization objectives."""

import unittest

import numpy as np

from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import BaseSimProblem, ParameterSimProblem
from fmdtools.sim.search import ResultConstraint, ResultObjective


class ConfigurableSimProblem(BaseSimProblem):
    """Allow the test to configure the real simulation problem incrementally."""

    def init_problem(self, **kwargs):
        pass


class ConfigurableParameterProblem(ParameterSimProblem):
    """Use the production parameter simulation with test-specific objectives."""

    def init_problem(self, **kwargs):
        pass


class TestZeroTimeObjectives(unittest.TestCase):
    """Distinguish zero from the unspecified-time sentinel throughout search."""

    def test_zero_time_only_selects_matching_results(self):
        result = Result(
            {
                "a.t0p0.cost": 1.0,
                "a.t2p0.cost": 90.0,
                "b.t0p0.cost": 3.0,
                "b.t2p0.cost": 80.0,
            }
        )
        for time in (0, 0.0, np.float64(0.0)):
            for nested in (False, True):
                for method, expected in ((np.sum, 4.0), (np.mean, 2.0)):
                    with self.subTest(time=time, nested=nested, method=method):
                        source = result.nest() if nested else result
                        obj = ResultObjective("cost", time=time, method=method)
                        self.assertEqual(obj.get_result_value(source), expected)
                        obj.update(source)
                        self.assertEqual(obj.value, expected)
        self.assertEqual(ResultObjective("cost").get_result_value(result), 174.0)
        self.assertEqual(
            ResultObjective("cost", time=2.0).get_result_value(result), 170.0
        )

    def test_zero_time_constraints_and_negative_objectives(self):
        result = Result({"t0p0.cost": 1.0, "t2p0.cost": 90.0})
        obj = ResultObjective("cost", time=0.0, negative=True)
        obj.update(result)
        self.assertEqual(obj.value, -1.0)
        constraint = ResultConstraint(
            "cost", time=0.0, threshold=2.0, comparator="less"
        )
        constraint.update(result)
        self.assertTrue(constraint.satisfied)
        self.assertEqual(constraint.value, -1.0)

    def test_output_requests_keep_zero_and_unspecified_times_separate(self):
        for keep_ec in (False, True):
            with self.subTest(keep_ec=keep_ec):
                problem = ConfigurableSimProblem()
                problem.add_sim(ExampleFunction(), "nominal", keep_ec=keep_ec)
                problem.add_result_objective("initial", "s.x", time=0.0)
                problem.add_result_constraint("initial_limit", "s.y", time=0.0)
                problem.add_result_objective("later", "s.x", time=2.0)
                problem.add_result_objective("final", "classify.xy")
                expected = {
                    0.0: ["s.x", "s.y"],
                    2.0: ["s.x"],
                    "end": (["classify"] if keep_ec else []) + ["classify.xy"],
                }
                self.assertEqual(problem.obj_con_des_res(), expected)
                self.assertEqual(problem.get_end_time(), problem.mdl.sp.end_time)

    def test_all_zero_objectives_stop_at_zero(self):
        problem = ConfigurableSimProblem()
        problem.add_sim(ExampleFunction(), "nominal")
        problem.add_result_objective("initial", "s.x", time=0.0)
        problem.add_result_constraint("limit", "s.y", time=0.0)
        self.assertEqual(problem.get_end_time(), 0.0)
        problem.add_result_objective("later", "s.x", time=2.0)
        self.assertEqual(problem.get_end_time(), 2.0)

    def test_parameter_simulation_returns_initial_and_later_states(self):
        for only_initial in (False, True):
            with self.subTest(only_initial=only_initial):
                domain = ParameterDomain(ExampleParameter)
                domain.add_variable("x")
                problem = ConfigurableParameterProblem()
                problem.add_sim(ExampleFunction(sp={"end_time": 3.0}), "nominal")
                problem.add_parameterdomain(domain)
                problem.add_result_objective("initial", "s.x", time=0.0)
                problem.add_result_constraint(
                    "initial_limit", "s.x", time=0.0, threshold=1.0, comparator="less"
                )
                if not only_initial:
                    problem.add_result_objective("later", "s.x", time=2.0)
                self.assertEqual(problem.initial(2.0), 0.0)
                self.assertEqual(problem.initial_limit(2.0), -1.0)
                self.assertTrue(problem.constraints["initial_limit"].satisfied)
                self.assertEqual(problem.hist.time[-1], 0.0 if only_initial else 2.0)
                self.assertEqual(problem.res.get("t0p0.s.x"), 0.0)
                if not only_initial:
                    self.assertEqual(problem.later(2.0), 4.0)


if __name__ == "__main__":
    unittest.main()
