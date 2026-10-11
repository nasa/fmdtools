#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical rate and risk invariants across fault-sampling approaches.

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

from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    JointFaultSample,
    SampleApproach,
)


class ReferenceMode(Mode):
    fault_sensor = Fault(
        prob=0.01,
        units="hr",
        cost=10.0,
        phases=(("early", 0.75), ("late", 0.25)),
    )
    fault_power = Fault(
        prob=0.02,
        units="hr",
        cost=100.0,
        phases=(("early", 0.20), ("late", 0.80)),
    )


class DifferentPhaseMode(ReferenceMode):
    fault_power = Fault(
        prob=0.02,
        units="hr",
        cost=100.0,
        phases=(("load", 0.60), ("idle", 0.40)),
    )


class TupleOpportunityMode(ReferenceMode):
    fault_sensor = Fault(
        prob=0.01,
        units="hr",
        cost=10.0,
        phases=((("early", "early"), 0.60), (("late", "late"), 0.40)),
    )


class ReferenceState(State):
    loss: float = 0.0


class ReferenceFunction(Function):
    container_m = ReferenceMode
    container_s = ReferenceState

    def dynamic_behavior(self):
        self.s.loss = sum(self.m.get_fault(name).cost for name in self.m.faults)

    def classify(self, **kwargs):
        return {"cost": self.s.loss, "hazard": bool(self.s.loss)}


class DifferentPhaseFunction(ReferenceFunction):
    container_m = DifferentPhaseMode


class TupleOpportunityFunction(ReferenceFunction):
    container_m = TupleOpportunityMode


def make_model(function_class=ReferenceFunction):
    # Inclusive timestep lengths: early = 3 hours; late = 4 hours.
    return function_class(
        "risk",
        sp={
            "end_time": 6.0,
            "dt": 1.0,
            "units": "hr",
            "phases": (("early", 0.0, 2.0), ("late", 3.0, 6.0)),
        },
    )


def make_domain(model, *modes):
    domain = FaultDomain(model)
    for mode in modes:
        domain.add_fault(model.name, mode)
    return domain


METHODS = (
    ("all", (), {}, {}),
    ("even", (2,), {}, {}),
    ("quad", ([-0.6, 0.6], [1.0, 3.0]), {}, {}),
    (
        "all",
        (),
        {"early": "even", "late": "quad"},
        {"early": (1,), "late": ([-0.6, 0.6], [3.0, 1.0])},
    ),
)


class TestSamplingRiskInvariants(unittest.TestCase):
    """Use independent exposure/rate arithmetic, not implementation-derived totals."""

    def _assert_risk(self, sample, expected, costs):
        self.assertAlmostEqual(
            sample.get_metric("rate", method="sum", round_value=False),
            sum(expected.values()),
            places=13,
        )
        result, history = propagate.fault_sample(self.model, sample, showprogress=False)
        table = FMEA(
            result,
            sample,
            group_by=("fault",),
            expected_metric=["cost", "hazard"],
            rates="scenario_rate",
            round_value=False,
        )
        for mode, rate in expected.items():
            with self.subTest(mode=mode):
                self.assertAlmostEqual(
                    sum(s.rate for s in sample.scenarios() if s.fault == mode),
                    rate,
                    places=13,
                )
                self.assertAlmostEqual(
                    table["expected_cost"][(mode,)], rate * costs[mode]
                )
                self.assertAlmostEqual(table["expected_hazard"][(mode,)], rate)
        self.assertAlmostEqual(
            sum(table["expected_cost"].values()),
            sum(rate * costs[mode] for mode, rate in expected.items()),
        )
        self.assertTrue(history)
        self.assertFalse(self.model.m.any_faults())

    def test_two_phase_rates_and_fmea_survive_sampling_methods_and_approach(self):
        # Integrated sensor rate: .01 * (3*.75 + 4*.25) = .0325
        # Integrated power rate: .02 * (3*.20 + 4*.80) = .076
        expected = {"sensor": 0.0325, "power": 0.076}
        costs = {"sensor": 10.0, "power": 100.0}
        for method, args, phase_methods, phase_args in METHODS:
            for wrapper in ("direct", "approach"):
                with self.subTest(method=method, wrapper=wrapper):
                    self.model = make_model()
                    phase_map = PhaseMap(self.model.sp.phases, dt=1.0)
                    if wrapper == "direct":
                        sample = FaultSample(
                            make_domain(self.model, *expected), phasemap=phase_map
                        )
                        sample.add_fault_phases(
                            method=method,
                            args=args,
                            phase_methods=phase_methods,
                            phase_args=phase_args,
                        )
                    else:
                        sample = SampleApproach(self.model)
                        sample.add_faultdomain(
                            "both",
                            "faults",
                            (self.model.name, "sensor"),
                            (self.model.name, "power"),
                        )
                        sample.add_faultsample(
                            "phase_scenarios",
                            "fault_phases",
                            "both",
                            phasemap="mdl",
                            method=method,
                            args=args,
                            phase_methods=phase_methods,
                            phase_args=phase_args,
                        )
                    self._assert_risk(sample, expected, costs)

    def test_two_phase_joint_rates_match_between_wrappers_and_sampling_methods(self):
        # Joint independent rates per phase:
        # early: (.01 * 3 * .75) * (.02 * 3 * .20) = .00027
        # late:  (.01 * 4 * .25) * (.02 * 4 * .80) = .00064
        expected_rate = 0.00091
        for method, args, _, _ in METHODS[:3]:
            for wrapper in ("direct", "joint", "approach"):
                with self.subTest(method=method, wrapper=wrapper):
                    self.model = make_model()
                    phase_map = PhaseMap(self.model.sp.phases, dt=1.0)
                    if wrapper == "direct":
                        sample = FaultSample(
                            make_domain(self.model, "sensor", "power"),
                            phasemap=phase_map,
                        )
                        sample.add_fault_phases(method=method, args=args, n_joint=2)
                    elif wrapper == "joint":
                        sample = JointFaultSample(
                            make_domain(self.model, "sensor"),
                            make_domain(self.model, "power"),
                            phasemaps=[phase_map],
                        )
                        sample.add_fault_phases(method=method, args=args, n_joint=2)
                    else:
                        sample = SampleApproach(self.model)
                        sample.add_faultdomain(
                            "left", "fault", self.model.name, "sensor"
                        )
                        sample.add_faultdomain(
                            "right", "fault", self.model.name, "power"
                        )
                        sample.add_faultsample(
                            "joint",
                            "fault_phases",
                            ["left", "right"],
                            phasemap="mdl",
                            method=method,
                            args=args,
                            n_joint=2,
                        )
                    self.assertAlmostEqual(
                        sample.get_metric("rate", method="sum", round_value=False),
                        expected_rate,
                        places=13,
                    )
                    result, _ = propagate.fault_sample(
                        self.model, sample, showprogress=False
                    )
                    table = FMEA(
                        result,
                        sample,
                        group_by=(),
                        expected_metric="cost",
                        rates="scenario_rate",
                        round_value=False,
                    )
                    self.assertAlmostEqual(
                        table["expected_cost"][()], expected_rate * 110.0, places=13
                    )
                    self.assertFalse(self.model.m.any_faults())

    def test_different_source_maps_keep_opportunities_and_overlap_exposure(self):
        # Only times 1..6 overlap. Independent joined exposure intervals:
        # (early, load): .01 * 2 * .75 * .02 * 2 * .60 = .00036
        # (late, load):  .01 * 2 * .25 * .02 * 2 * .60 = .00012
        # (late, idle):  .01 * 2 * .25 * .02 * 2 * .40 = .00008
        joint_rates = {
            ("early", "load"): 0.00036,
            ("late", "load"): 0.00012,
            ("late", "idle"): 0.00008,
        }
        single_rates = {"sensor": 0.025, "power": 0.064}
        for method, args, _, _ in METHODS[:3]:
            for n_joint in (1, 2):
                for wrapper in ("joint", "approach"):
                    with self.subTest(method=method, n_joint=n_joint, wrapper=wrapper):
                        model = make_model(DifferentPhaseFunction)
                        maps = [
                            PhaseMap({"early": [0.0, 2.0], "late": [3.0, 6.0]}),
                            PhaseMap({"load": [1.0, 4.0], "idle": [5.0, 6.0]}),
                        ]
                        if wrapper == "joint":
                            sample = JointFaultSample(
                                make_domain(model, "sensor"),
                                make_domain(model, "power"),
                                phasemaps=maps,
                            )
                            sample.add_fault_phases(
                                method=method, args=args, n_joint=n_joint
                            )
                        else:
                            sample = SampleApproach(
                                model,
                                phasemaps={"sensor": maps[0], "power": maps[1]},
                                def_mdl_phasemap=False,
                            )
                            sample.add_faultdomain(
                                "sensor", "fault", model.name, "sensor"
                            )
                            sample.add_faultdomain(
                                "power", "fault", model.name, "power"
                            )
                            sample.add_faultsample(
                                "overlap",
                                "fault_phases",
                                ["sensor", "power"],
                                phasemap=["sensor", "power"],
                                method=method,
                                args=args,
                                n_joint=n_joint,
                            )
                        if n_joint == 1:
                            for mode, expected_rate in single_rates.items():
                                actual_rate = sum(
                                    s.rate
                                    for s in sample.scenarios()
                                    if s.fault == mode
                                )
                                self.assertAlmostEqual(
                                    actual_rate, expected_rate, places=13
                                )
                            expected_cost = 0.025 * 10.0 + 0.064 * 100.0
                            total_rate = sum(single_rates.values())
                        else:
                            for phase, expected_rate in joint_rates.items():
                                actual_rate = sum(
                                    s.rate
                                    for s in sample.scenarios()
                                    if s.phase == phase
                                )
                                self.assertAlmostEqual(
                                    actual_rate, expected_rate, places=13
                                )
                            expected_cost = sum(joint_rates.values()) * 110.0
                            total_rate = sum(joint_rates.values())
                        self.assertAlmostEqual(
                            sample.get_metric("rate", method="sum", round_value=False),
                            total_rate,
                            places=13,
                        )
                        result, _ = propagate.fault_sample(
                            model, sample, showprogress=False
                        )
                        table = FMEA(
                            result,
                            sample,
                            group_by=(),
                            expected_metric="cost",
                            rates="scenario_rate",
                            round_value=False,
                        )
                        self.assertAlmostEqual(
                            table["expected_cost"][()], expected_cost, places=13
                        )
                        self.assertFalse(model.m.any_faults())

    def test_explicit_joined_phase_opportunity_keys_remain_supported(self):
        # Existing tuple-keyed opportunities and source-local opportunities can
        # coexist; each fault should use the form declared for that fault.
        model = make_model(TupleOpportunityFunction)
        mapping = PhaseMap(model.sp.phases, dt=1.0)
        sample = JointFaultSample(
            make_domain(model, "sensor"),
            make_domain(model, "power"),
            phasemaps=[mapping, mapping],
        )
        sample.add_fault_phases(method="all", n_joint=2)
        # early: (.01*3*.60)*(.02*3*.20) = .000216
        # late:  (.01*4*.40)*(.02*4*.80) = .001024
        expected = {("early", "early"): 0.000216, ("late", "late"): 0.001024}
        for phase, rate in expected.items():
            actual = sum(s.rate for s in sample.scenarios() if s.phase == phase)
            self.assertAlmostEqual(actual, rate, places=13)
        self.assertAlmostEqual(
            sample.get_metric("rate", method="sum", round_value=False),
            0.00124,
            places=13,
        )


if __name__ == "__main__":
    unittest.main()
