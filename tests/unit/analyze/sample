#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for model calculation correctness.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class InnerParameters(Parameter, readonly=True):
    gain: float = 2.0
    factor: float = 1.0
    free: float = 0.0
    gain_lim = (0.0, 4.0)
    factor_set = (1.0, 3.0)


class OuterParameters(Parameter, readonly=True):
    inner: InnerParameters = InnerParameters()
    count: int = 1


class DeepParameters(Parameter, readonly=True):
    outer: OuterParameters = OuterParameters()


class AccumulatedState(State):
    total: np.float64 = 0.0


class NestedParameterFunction(Function):
    container_p = OuterParameters
    container_s = AccumulatedState

    def dynamic_behavior(self):
        self.s.total += self.p.inner.gain * self.p.inner.factor

    def classify(self, **kwargs):
        return {"total": self.s.total}


class TestNestedParameterConstraints(unittest.TestCase):
    def test_limits_and_sets_resolve_through_nested_parameter_classes(self):
        for cls, prefix in (
            (InnerParameters, ""),
            (OuterParameters, "inner."),
            (DeepParameters, "outer.inner."),
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.get_set_const(prefix + "gain"), (0.0, 4.0))
                self.assertEqual(cls.get_set_const(prefix + "factor"), {1.0, 3.0})
                self.assertEqual(cls.get_set_const(prefix + "free"), ())
                options = cls.get_set_const(prefix + "factor")
                options.add(99.0)
                self.assertEqual(cls.get_set_const(prefix + "factor"), {1.0, 3.0})
        self.assertEqual(OuterParameters.get_set_const("count.value"), ())
        with self.assertRaises(KeyError):
            OuterParameters.get_set_const("missing.value")

    def test_domain_bounds_reject_invalid_values_at_each_depth(self):
        for cls, prefix in (
            (OuterParameters, "inner."),
            (DeepParameters, "outer.inner."),
        ):
            domain = ParameterDomain(cls)
            domain.add_variables(prefix + "gain", prefix + "factor")
            with self.subTest(cls=cls.__name__):
                self.assertEqual(
                    domain.variables,
                    {prefix + "gain": (0.0, 4.0), prefix + "factor": {1.0, 3.0}},
                )
                self.assertEqual(domain.get_set_constraints(2.0, 1.0), (False, False))
                self.assertEqual(domain.get_set_constraints(-1.0, 2.0), (True, True))
                self.assertEqual(domain.get_set_constraints(5.0, 3.0), (True, False))

    def test_product_and_orthogonal_sampling_use_inherited_domains(self):
        domain = ParameterDomain(OuterParameters)
        domain.add_variables("inner.gain", "inner.factor")
        sample = ParameterSample(domain, seed=7)
        grid = domain.get_var_iters(resolution=2.0)
        np.testing.assert_array_equal(grid["inner.gain"], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(np.sort(grid["inner.factor"]), [1.0, 3.0])
        self.assertEqual(
            set(map(tuple, sample.combine_product(resolution=2.0))),
            {(gain, factor) for gain in (0.0, 2.0, 4.0) for factor in (1.0, 3.0)},
        )
        orthogonal = sample.combine_orthogonal(resolution=2.0)
        self.assertEqual(len(orthogonal), 5)
        self.assertTrue(
            all(not any(domain.get_set_constraints(*values)) for values in orthogonal)
        )

    def test_explicit_overrides_remain_authoritative(self):
        domain = ParameterDomain(OuterParameters)
        domain.add_variable("inner.gain", var_lim=(-2.0, 8.0))
        domain.add_variable("inner.factor", var_set=(2.0, 4.0))
        self.assertEqual(
            domain.variables, {"inner.gain": (-2.0, 8.0), "inner.factor": {2.0, 4.0}}
        )
        self.assertEqual(domain.get_set_constraints(6.0, 2.0), (False, False))
        self.assertEqual(InnerParameters.gain_lim, (0.0, 4.0))
        self.assertEqual(InnerParameters.factor_set, (1.0, 3.0))

    def test_real_parameter_sweep_matches_analytic_state_totals(self):
        domain = ParameterDomain(OuterParameters)
        domain.add_variables("inner.gain", "inner.factor")
        sample = ParameterSample(domain, seed=11)
        sample.add_variable_ranges(comb_kwargs={"resolution": 2.0})
        self.assertEqual(sample.num_scenarios(), 6)
        result, history = propagate.parameter_sample(
            NestedParameterFunction(sp={"end_time": 3.0}), sample, showprogress=False
        )
        for scenario in sample.scenarios():
            with self.subTest(scenario=scenario.name):
                inner = scenario.p["inner"]
                increment = inner["gain"] * inner["factor"]
                self.assertEqual(
                    result[scenario.name + ".tend.classify.total"], 3 * increment
                )
                np.testing.assert_allclose(
                    history[scenario.name + ".s.total"], np.arange(4) * increment
                )
        self.assertAlmostEqual(sum(s.prob for s in sample.scenarios()), 1.0)


if __name__ == "__main__":
    unittest.main()
