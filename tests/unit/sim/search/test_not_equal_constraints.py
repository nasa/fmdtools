"""Regression tests for not-equal optimization constraint feasibility."""

import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.sim.search import (
    Constraint,
    HistoryConstraint,
    ResultConstraint,
    SimpleProblem,
)


class ConfigurableSimpleProblem(SimpleProblem):
    """Configure a real callable-backed optimization problem in the test."""

    def init_problem(self, **kwargs):
        pass


class TestNotEqualConstraints(unittest.TestCase):
    """Keep numeric residual signs consistent with exact inequality."""

    def test_exact_inequality_under_both_sign_and_boundary_conventions(self):
        for negative in (False, True):
            for or_equal in (False, True):
                for threshold in (0.0, 2.0, -3.0):
                    for value in (
                        threshold - 1.0,
                        threshold,
                        np.nextafter(threshold, np.inf),
                        threshold + 1.0,
                    ):
                        with self.subTest(
                            negative=negative,
                            or_equal=or_equal,
                            threshold=threshold,
                            value=value,
                        ):
                            constraint = Constraint(
                                threshold=threshold,
                                comparator="notequal",
                                negative=negative,
                                or_equal=or_equal,
                            )
                            constraint.update(value)
                            expected = value != threshold
                            self.assertEqual(constraint.satisfied, expected)
                            residual = -1 if expected else 1
                            self.assertEqual(
                                constraint.value, -residual if negative else residual
                            )

    def test_result_and_history_constraints_share_inequality_semantics(self):
        for cls in (ResultConstraint, HistoryConstraint):
            for negative in (False, True):
                for or_equal in (False, True):
                    for value in (2.0, 3.0):
                        with self.subTest(
                            cls=cls, negative=negative, or_equal=or_equal, value=value
                        ):
                            if cls is HistoryConstraint:
                                result = History(
                                    {"run.cost": np.array([value / 2, value / 2])}
                                )
                            else:
                                result = Result({"run.t1p0.cost": value})
                            constraint = cls(
                                "cost",
                                threshold=2.0,
                                comparator="notequal",
                                negative=negative,
                                or_equal=or_equal,
                            )
                            constraint.update(result)
                            self.assertEqual(constraint.satisfied, value != 2.0)
                            residual = -1 if value != 2.0 else 1
                            self.assertEqual(
                                constraint.value, -residual if negative else residual
                            )

    def test_simple_problem_callable_and_cached_updates(self):
        for negative in (False, True):
            with self.subTest(negative=negative):
                problem = ConfigurableSimpleProblem()
                problem.add_variables("x")
                problem.add_constraint(
                    "different",
                    lambda x: x,
                    threshold=2.0,
                    comparator="notequal",
                    negative=negative,
                )
                for value in (3.0, 2.0, 2.0, 1.0):
                    result = problem.different(value)
                    expected = -1 if value != 2.0 else 1
                    self.assertEqual(result, -expected if negative else expected)
                    self.assertEqual(
                        problem.constraints["different"].satisfied, value != 2.0
                    )

    def test_other_comparators_keep_existing_residuals(self):
        for comparator in ("less", "greater", "equal"):
            for value in (1.0, 2.0, 3.0):
                for negative in (False, True):
                    for or_equal in (False, True):
                        with self.subTest(
                            comparator=comparator,
                            value=value,
                            negative=negative,
                            or_equal=or_equal,
                        ):
                            residual = {
                                "less": value - 2.0,
                                "greater": 2.0 - value,
                                "equal": abs(value - 2.0),
                            }[comparator]
                            constraint = Constraint(
                                threshold=2.0,
                                comparator=comparator,
                                negative=negative,
                                or_equal=or_equal,
                            )
                            constraint.update(value)
                            self.assertEqual(
                                constraint.value, -residual if negative else residual
                            )
                            self.assertEqual(
                                constraint.satisfied,
                                residual <= 0 if or_equal else residual < 0,
                            )


if __name__ == "__main__":
    unittest.main()
