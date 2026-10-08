#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for applying a joint scenario sampling weight once.

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
import itertools
import math
import unittest

import numpy as np

from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, JointFaultSample
from fmdtools.sim.scenario import JointFaultScenario


class RiskMode(Mode):
    fault_a = Fault(prob=0.2, cost=10.0)
    fault_b = Fault(prob=0.3, cost=20.0)
    fault_c = Fault(prob=0.4, cost=30.0)


class LossState(State):
    loss: np.float64 = 0.0


class RiskFunction(Function):
    container_m = RiskMode
    container_s = LossState

    def dynamic_behavior(self):
        self.s.loss = sum(self.m.get_fault(name).cost for name in self.m.faults)

    def classify(self, **kwargs):
        return {"cost": self.s.loss}


class PhaseRiskMode(Mode):
    fault_a = Fault(prob=0.2, units="hr", phases=(("early", 0.75), ("late", 0.25)))
    fault_b = Fault(prob=0.3, units="hr", phases=(("early", 0.8), ("late", 0.2)))


class PhaseRiskFunction(RiskFunction):
    container_m = PhaseRiskMode


def sample_for(model=None, count=2, **kwargs):
    model = model or RiskFunction(sp={"end_time": 4.0})
    domain = FaultDomain(model)
    for name in ("a", "b", "c")[:count]:
        domain.add_fault(model.name, name)
    return FaultSample(domain, def_mdl_phasemap=False, **kwargs)


class TestJointScenarioQuadrature(unittest.TestCase):
    def test_direct_joint_rates_apply_sampling_weight_once(self):
        model = RiskFunction(sp={"end_time": 4.0})
        for count, weight, conditional in itertools.product(
            (1, 2, 3), (0.0, 0.1, 0.5, 1.0, 2.0), (0.0, 0.4, 1.0)
        ):
            faults = tuple((model.name, name) for name in ("a", "b", "c")[:count])
            probabilities = (0.2, 0.3, 0.4)[:count]
            for basis, unweighted in (
                ("ind", math.prod(probabilities)),
                ("max", max(probabilities)),
                (faults[0], probabilities[0]),
            ):
                with self.subTest(
                    count=count, weight=weight, conditional=conditional, basis=basis
                ):
                    scenario = JointFaultScenario.from_faults(
                        faults,
                        2.0,
                        mdl=model,
                        weight=weight,
                        baserate=basis,
                        p_cond=conditional,
                        starttime=0.0,
                    )
                    self.assertAlmostEqual(
                        scenario.rate, unweighted * weight * conditional
                    )
                    control = JointFaultScenario.from_faults(
                        faults, 2.0, mdl=model, baserate=basis, starttime=0.0
                    )
                    actual, expected = scenario.asdict(), control.asdict()
                    actual.pop("rate")
                    expected.pop("rate")
                    self.assertEqual(actual, expected)
        self.assertFalse(model.m.any_faults())

    def test_model_free_scenarios_keep_one_explicit_weight(self):
        for count, weight in itertools.product((1, 2, 3), (0.0, 0.25, 0.75, 1.0)):
            faults = tuple(("system", name) for name in ("a", "b", "c")[:count])
            for basis in ("ind", "max", faults[0]):
                with self.subTest(count=count, weight=weight, basis=basis):
                    scenario = JointFaultScenario.from_faults(
                        faults, 1.0, weight=weight, p_cond=0.6, baserate=basis
                    )
                    self.assertAlmostEqual(scenario.rate, 0.6 * weight)

    def test_sampling_resolution_does_not_dilute_the_total_joint_rate(self):
        for count in (2, 3):
            expected = math.prod((0.2, 0.3, 0.4)[:count])
            for number in (1, 2, 4):
                sample = sample_for(count=count)
                times = list(range(1, number + 1))
                weights = np.full(number, 1 / number)
                before = weights.copy()
                sample.add_fault_times(times, weights=weights, n_joint=count)
                with self.subTest(count=count, number=number):
                    self.assertAlmostEqual(
                        sum(s.rate for s in sample.scenarios()), expected
                    )
                    np.testing.assert_allclose(
                        [s.rate for s in sample.scenarios()], expected * weights
                    )
                    np.testing.assert_array_equal(weights, before)
                    self.assertEqual(set(sample.get_times()), set(times))
            for method, arguments in (
                ("even", (1,)),
                ("even", (3,)),
                ("all", ()),
                ("quad", ([-0.5, 0.5], [1.0, 3.0])),
            ):
                sample = sample_for(count=count, phasemap=PhaseMap({"run": [0.0, 4.0]}))
                sample.add_fault_phases(
                    "run", method=method, args=arguments, n_joint=count
                )
                with self.subTest(count=count, method=method, args=arguments):
                    self.assertAlmostEqual(
                        sum(s.rate for s in sample.scenarios()), expected
                    )

    def test_phase_opportunities_and_exposure_are_combined_before_quadrature(self):
        model = PhaseRiskFunction(sp={"end_time": 2.5, "dt": 0.5, "units": "hr"})
        phase_map = PhaseMap({"early": [0.0, 1.0], "late": [1.5, 2.5]}, dt=0.5)
        # Each inclusive phase represents 1.5 hours of exposure.
        expected = (0.2 * 1.5 * 0.75) * (0.3 * 1.5 * 0.8) + (0.2 * 1.5 * 0.25) * (
            0.3 * 1.5 * 0.2
        )
        for method, args in (("even", (1,)), ("even", (2,)), ("all", ())):
            with self.subTest(method=method, args=args):
                sample = sample_for(model, phasemap=phase_map)
                sample.add_fault_phases(method=method, args=args, n_joint=2)
                self.assertAlmostEqual(
                    sum(s.rate for s in sample.scenarios()), expected
                )

    def test_joint_domain_wrapper_and_nonindependent_bases_are_preserved(self):
        sample = sample_for()
        domains = []
        for fault in sample.faultdomain.faults:
            domain = FaultDomain(sample.faultdomain.mdl)
            domain.add_fault(*fault)
            domains.append(domain)
        for basis, expected in (
            ("ind", 0.06),
            ("max", 0.3),
            ((sample.faultdomain.mdl.name, "a"), 0.2),
        ):
            with self.subTest(basis=basis):
                joint = JointFaultSample(*domains, def_mdl_phasemap=False)
                joint.add_fault_times(
                    [1.0, 2.0, 3.0],
                    weights=[0.0, 0.25, 0.75],
                    n_joint=2,
                    baserate=basis,
                    p_cond=0.4,
                )
                self.assertAlmostEqual(
                    sum(s.rate for s in joint.scenarios()), expected * 0.4
                )
                self.assertEqual(joint.scenarios()[0].rate, 0.0)

    def test_simulated_fmea_expected_loss_is_invariant_to_time_sample_count(self):
        for number in (1, 2, 4):
            with self.subTest(number=number):
                sample = sample_for()
                sample.add_fault_times(
                    list(range(1, number + 1)), weights=[1 / number] * number, n_joint=2
                )
                before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
                result, history = propagate.fault_sample(
                    sample.faultdomain.mdl, sample, showprogress=False
                )
                table = FMEA(
                    result,
                    sample,
                    group_by=(),
                    expected_metric="cost",
                    rates="scenario_rate",
                    round_value=False,
                )
                self.assertAlmostEqual(table["expected_cost"][()], 0.2 * 0.3 * 30.0)
                for scenario in sample.scenarios():
                    self.assertEqual(
                        result[scenario.name + ".tend.classify.cost"], 30.0
                    )
                    expected = np.where(np.arange(5) >= scenario.times[0], 30.0, 0.0)
                    np.testing.assert_array_equal(
                        history[scenario.name + ".s.loss"], expected
                    )
                self.assertEqual([s.asdict() for s in sample.scenarios()], before)
                self.assertFalse(sample.faultdomain.mdl.m.any_faults())


if __name__ == "__main__":
    unittest.main()
