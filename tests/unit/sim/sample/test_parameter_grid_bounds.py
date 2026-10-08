#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for parameter grids staying inside declared limits.

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

from fractions import Fraction
import unittest

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestParameterGridBounds(unittest.TestCase):
    """Check inclusive limits without inventing an off-grid endpoint."""

    def test_requested_steps_stay_in_bounds_and_preserve_aligned_endpoints(self):
        cases = [
            (0, 1, 0.3),
            (-1, 1, 0.6),
            (0.1, 0.9, 0.3),
            (2, 3, 2),
            (2, 2, 0.5),
            (0, 1, 0.1),
            (-0.3, 0.3, 0.1),
            (0, 3, 2),
            (0.0, 1e-12, 3e-13),
            (0.0, 0.3, 0.1),
            (1.5, 2.5, 0.25),
        ]
        for low, high, step in cases:
            with self.subTest(low=low, high=high, step=step):
                domain = ParameterDomain(ExampleParameter)
                domain.add_variable("x", var_lim=(low, high))
                start, stop, stride = map(lambda x: Fraction(str(x)), (low, high, step))
                expected = []
                point = start
                while point <= stop:
                    expected.append(float(point))
                    point += stride
                actual = domain.get_var_iters(step)["x"]
                self.assertEqual(actual.ndim, 1)
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(np.all(actual >= low))
                self.assertTrue(np.all(actual <= high))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-14,
                    atol=4 * np.finfo(float).eps * max(abs(low), abs(high), abs(step)),
                )
                self.assertEqual(domain.variables["x"], (low, high))
                if (stop - start) % stride == 0:
                    self.assertEqual(actual[-1], high)

    def test_per_variable_steps_and_discrete_choices_keep_their_own_values(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x", var_lim=(0.0, 1.0))
        domain.add_variable("y", var_lim=(1.0, 2.0))
        domain.add_variable("z", var_set=(1, 4, 7))
        overrides = {"x": 0.3, "y": 0.4}
        values = domain.get_var_iters(2.0, resolutions=overrides)
        self.assertEqual(list(values), ["x", "y", "z"])
        np.testing.assert_allclose(values["x"], [0.0, 0.3, 0.6, 0.9])
        np.testing.assert_allclose(values["y"], [1.0, 1.4, 1.8])
        self.assertEqual(set(values["z"]), {1, 4, 7})
        self.assertEqual(overrides, {"x": 0.3, "y": 0.4})
        integer = ParameterDomain(ExampleParameter)
        integer.add_variable("x", var_lim=(0, 5))
        grid = integer.get_var_iters(2)["x"]
        self.assertTrue(np.issubdtype(grid.dtype, np.integer))
        np.testing.assert_array_equal(grid, [0, 2, 4])

    def test_invalid_steps_are_rejected_before_creating_scenarios(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x", var_lim=(0.0, 1.0))
        for step in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(step=step):
                sample = ParameterSample(domain, seed=7)
                with self.assertRaisesRegex(ValueError, "resolution"):
                    sample.add_variable_ranges(comb_kwargs={"resolution": step})
                self.assertEqual(sample.scenarios(), [])

    def test_product_and_orthogonal_sampling_never_leave_domain_limits(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x", var_lim=(0.0, 1.0))
        domain.add_variable("y", var_lim=(1.0, 4.0))
        for method in ("product", "orthogonal"):
            with self.subTest(method=method):
                sample = ParameterSample(domain, seed=0)
                values = getattr(sample, "combine_" + method)(resolution=0.6)
                self.assertTrue(values)
                for value in values:
                    self.assertFalse(any(domain.get_set_constraints(*value)))

    def test_actual_parameter_simulations_use_only_the_bounded_grid(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variable("x", var_lim=(0.0, 1.0))
        sample = ParameterSample(domain, seed=19)
        sample.add_variable_ranges(comb_kwargs={"resolution": 0.3})
        scenarios = sample.scenarios()
        np.testing.assert_allclose(
            [scenario.p["x"] for scenario in scenarios], [0.0, 0.3, 0.6, 0.9]
        )
        self.assertEqual(len(scenarios), 4)
        self.assertAlmostEqual(sum(scenario.prob for scenario in scenarios), 1.0)
        model = ExampleFunction(sp={"end_time": 3.0})
        results, histories = propagate.parameter_sample(
            model, sample, showprogress=False
        )
        for scenario in scenarios:
            with self.subTest(name=scenario.name):
                result = results.get(scenario.name)
                history = histories.get(scenario.name)
                np.testing.assert_allclose(
                    history["s.x"], scenario.p["x"] * history.time
                )
                self.assertAlmostEqual(result["tend.classify.xy"], 3 * scenario.p["x"])
        self.assertEqual(model.s.x, 0.0)


if __name__ == "__main__":
    unittest.main()
