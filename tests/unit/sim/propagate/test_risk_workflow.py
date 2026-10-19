#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical checks from fault simulation through histories and risk/FMEA.

The five-point, hourly exposure convention gives total sampled rate 0.6.
The final rate-weighted loss is 4.5; its normalized average is 7.5. These
are different quantities. The nominal run is a reference, not an inferred
complementary probability. A zero-rate fault still runs but adds no risk.

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
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import FMEA
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, SampleApproach


class WorkflowMode(Mode):
    fault_slow = Fault(prob=0.08, units="hr")
    fault_stop = Fault(prob=0.04, units="hr")
    fault_zero = Fault(prob=0.0, units="hr")


class WorkflowState(State):
    loss: float = 0.0
    hazardous: bool = False


class WorkflowModel(Function):
    container_m = WorkflowMode
    container_s = WorkflowState

    def dynamic_behavior(self):
        """Accumulate loss on each faulty step, including the injection step."""
        if self.m.has_fault("slow"):
            self.s.loss += 2.0
        if self.m.has_fault("stop"):
            self.s.loss += 5.0
        if self.m.has_fault("zero"):
            self.s.loss += 100.0
        self.s.hazardous = self.s.loss >= 10.0

    def classify(self, **kwargs):
        return {"cost": float(self.s.loss), "hazard": bool(self.s.hazardous)}


# Rates are 0.08 or 0.04 per hour * 5 hours * time weights (0.25, 0.75).
# Each tuple contains the independent rate and the full loss trajectory.
# End risk: .10*8 + .30*4 + .05*20 + .15*10 = 4.5.
# Hazard rate: .05 + .15 = .2; normalized end loss: 4.5/.6 = 7.5.
REFERENCE = {
    ("slow", 1.0): (0.10, (0.0, 2.0, 4.0, 6.0, 8.0)),
    ("slow", 3.0): (0.30, (0.0, 0.0, 0.0, 2.0, 4.0)),
    ("stop", 1.0): (0.05, (0.0, 5.0, 10.0, 15.0, 20.0)),
    ("stop", 3.0): (0.15, (0.0, 0.0, 0.0, 5.0, 10.0)),
    ("zero", 1.0): (0.0, (0.0, 100.0, 200.0, 300.0, 400.0)),
    ("zero", 3.0): (0.0, (0.0, 0.0, 0.0, 100.0, 200.0)),
}
RISK_HISTORY = (0.0, 0.45, 0.90, 2.70, 4.50)
MEAN_HISTORY = (0.0, 0.75, 1.50, 4.50, 7.50)
HAZARD_RATES = (0.0, 0.0, 0.05, 0.05, 0.20)


def make_workflow(wrapper=FaultSample):
    model = WorkflowModel("risk", sp={"end_time": 4.0, "units": "hr"})
    modes = ("slow", "stop", "zero")
    if wrapper is SampleApproach:
        sample = SampleApproach(model)
        sample.add_faultdomain("faults", "faults", *((model.name, m) for m in modes))
        sample.add_faultsample(
            "times", "fault_times", "faults", [1.0, 3.0], weights=[0.25, 0.75]
        )
    else:
        domain = FaultDomain(model)
        domain.add_faults(*((model.name, m) for m in modes))
        sample = FaultSample(domain)
        sample.add_fault_times([1.0, 3.0], weights=[0.25, 0.75])
    return model, sample


class TestRiskWorkflow(unittest.TestCase):
    def assert_workflow(self, result, history, sample, include_nominal):
        """Check every layer against the reference, not just against another API."""
        raw_result, raw_history = result, history
        result, history = result.flatten(), history.flatten()
        names = sample.scen_names()
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(
            {(s.fault, s.time) for s in sample.scenarios()}, set(REFERENCE)
        )
        expected_names = set(names) | ({"nominal"} if include_nominal else set())
        self.assertEqual(set(result.nest(levels=1)), expected_names)
        self.assertEqual(set(history.nest(levels=1)), expected_names)
        for scenario in sample.scenarios():
            rate, losses = REFERENCE[(scenario.fault, scenario.time)]
            self.assertAlmostEqual(scenario.rate, rate, places=14)
            trace = history.get(scenario.name)
            np.testing.assert_array_equal(trace["time"], np.arange(5))
            np.testing.assert_array_equal(trace["s.loss"], losses)
            np.testing.assert_array_equal(trace["s.hazardous"], np.array(losses) >= 10)
            for mode in ("slow", "stop", "zero"):
                expected = (np.arange(5) >= scenario.time) & (mode == scenario.fault)
                np.testing.assert_array_equal(trace["m.faults." + mode], expected)
            self.assertEqual(result[scenario.name + ".tend.classify.cost"], losses[-1])
            self.assertEqual(
                result[scenario.name + ".tend.classify.hazard"], losses[-1] >= 10
            )
        if include_nominal:
            self.assertEqual(result["nominal.tend.classify.cost"], 0.0)
            np.testing.assert_array_equal(history["nominal.s.loss"], np.zeros(5))
            self.assertFalse(np.any(history["nominal.s.hazardous"]))

        # Reverse the factor mapping independently of result/history ordering.
        rates = {s.name: s.rate for s in reversed(sample.scenarios())}
        faults = result.get_scens(*names)
        traces = history.get_scens(*names)
        self.assertAlmostEqual(sample.get_metric("rate", method="sum"), 0.6)
        np.testing.assert_allclose(
            traces.get_metric(
                "s.loss", method="expected", rates=rates, axis=0, round_value=False
            ),
            RISK_HISTORY,
            rtol=1e-13,
            atol=0.0,
        )
        np.testing.assert_allclose(
            traces.get_metric(
                "s.hazardous", method="rate", rates=rates, axis=0, round_value=False
            ),
            HAZARD_RATES,
            rtol=1e-13,
            atol=0.0,
        )
        np.testing.assert_allclose(
            raw_history.get_expected(sample, round_value=False)["s.loss"],
            MEAN_HISTORY,
            rtol=1e-13,
            atol=0.0,
        )
        self.assertAlmostEqual(
            faults.get_metric(
                "cost",
                method="expected",
                rates=rates,
                prefix="tend.classify.",
                round_value=False,
            ),
            4.5,
            places=13,
        )
        self.assertAlmostEqual(
            raw_result.get_expected(sample, round_value=False)["tend.classify.cost"],
            7.5,
            places=13,
        )
        self.assertAlmostEqual(
            faults.get_metric(
                "hazard",
                method="rate",
                rates=rates,
                prefix="tend.classify.",
                round_value=False,
            ),
            0.2,
            places=13,
        )
        for factors, reference in (
            (("fault",), {("slow",): 2.0, ("stop",): 2.5, ("zero",): 0.0}),
            (("time",), {(1.0,): 1.8, (3.0,): 2.7}),
        ):
            table = FMEA(
                raw_result,
                sample,
                group_by=factors,
                expected_metric="cost",
                rate_metric="hazard",
                rates="scenario_rate",
                round_value=False,
            )
            self.assertEqual(set(table["expected_cost"]), set(reference))
            for group, expected in reference.items():
                self.assertAlmostEqual(table["expected_cost"][group], expected, 13)
            self.assertAlmostEqual(sum(table["expected_cost"].values()), 4.5, 13)
            self.assertAlmostEqual(sum(table["rate_hazard"].values()), 0.2, 13)
            frame = table.as_table(sort_by="expected_cost", ascending=True)
            self.assertTrue(frame["expected_cost"].notna().all())
            self.assertTrue(frame["expected_cost"].is_monotonic_increasing)
            for group, expected in reference.items():
                self.assertAlmostEqual(frame["expected_cost"][group], expected, 13)

    def test_simulation_histories_and_risk_agree_across_execution_paths(self):
        for wrapper, staged, include_nominal in itertools.product(
            (FaultSample, SampleApproach), (False, True), (False, True)
        ):
            with self.subTest(
                wrapper=wrapper.__name__, staged=staged, include_nominal=include_nominal
            ):
                model, sample = make_workflow(wrapper)
                before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
                result, history = propagate.fault_sample(
                    model,
                    sample,
                    staged=staged,
                    include_nominal=include_nominal,
                    showprogress=False,
                )
                self.assert_workflow(result, history, sample, include_nominal)
                self.assertEqual([s.asdict() for s in sample.scenarios()], before)
                self.assertEqual(model.s.loss, 0.0)
                self.assertFalse(model.m.any_faults())

    def test_reordered_nested_outputs_keep_scenario_alignment(self):
        model, sample = make_workflow()
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        original_result, original_history = (
            copy.deepcopy(result),
            copy.deepcopy(history),
        )
        keys = list(result)
        result = Result({k: result[k] for k in keys[1::2] + keys[::2]})
        history = History(dict(reversed(list(history.items()))))
        for nested in (False, True):
            with self.subTest(nested=nested):
                outputs = (
                    (result.nest(), history.nest()) if nested else (result, history)
                )
                self.assert_workflow(*outputs, sample, True)
        np.testing.assert_equal(result.flatten().data, original_result.data)
        np.testing.assert_equal(history.flatten().data, original_history.data)

    def test_nested_averages_preserve_nominal_reference_options(self):
        model, sample = make_workflow()
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        for source, key, expected, with_reference in (
            (result, "tend.classify.cost", 7.5, 2.8125),
            (history, "s.loss", MEAN_HISTORY, (0.0, 0.28125, 0.5625, 1.6875, 2.8125)),
        ):
            # with_nominal=True uses the API's reference weight of one:
            # 4.5 / (1 + .6) = 2.8125, not a complementary probability.
            for nested, nominal, difference in itertools.product(
                (False, True), repeat=3
            ):
                with self.subTest(
                    kind=type(source).__name__,
                    nested=nested,
                    nominal=nominal,
                    difference=difference,
                ):
                    values = source.nest() if nested else source
                    before = copy.deepcopy(values)
                    average = values.get_expected(
                        sample,
                        with_nominal=nominal,
                        difference_from_nominal=difference,
                        round_value=False,
                    )
                    reference = np.asarray(with_reference if nominal else expected)
                    if difference:
                        reference = -reference
                    self.assertIsInstance(average, type(source))
                    self.assertIn(key, average)
                    np.testing.assert_allclose(
                        average[key], reference, rtol=1e-13, atol=0.0
                    )
                    np.testing.assert_equal(
                        values.flatten().data, before.flatten().data
                    )

    def test_intermediate_classifications_do_not_enter_terminal_risk(self):
        model, sample = make_workflow()
        result, history = propagate.fault_sample(
            model,
            sample,
            to_return={3.0: ["classify"], "end": ["classify"]},
            showprogress=False,
        )
        for scenario in sample.scenarios():
            losses = REFERENCE[(scenario.fault, scenario.time)][1]
            self.assertEqual(result[scenario.name + ".t3p0.classify.cost"], losses[3])
        # Request t=3, after both injections; its risk is 2.7, not the end risk 4.5.
        table = FMEA(
            result,
            sample,
            group_by=("fault",),
            prefix="t3p0.classify.",
            expected_metric="cost",
            rates="scenario_rate",
            round_value=False,
        )
        self.assertAlmostEqual(sum(table["expected_cost"].values()), 2.7, 13)
        self.assert_workflow(result, history, sample, True)

    def test_saved_results_and_histories_reproduce_the_risk_workflow(self):
        model, sample = make_workflow(SampleApproach)
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("npz", "json"):
                with self.subTest(extension=extension):
                    result_path = str(Path(directory, "result." + extension))
                    history_path = str(Path(directory, "history." + extension))
                    result.save(result_path)
                    history.save(history_path)
                    restored_result = Result.load(result_path)
                    restored_history = History.load(history_path)
                    self.assert_workflow(
                        restored_result, restored_history, sample, True
                    )


if __name__ == "__main__":
    unittest.main()
