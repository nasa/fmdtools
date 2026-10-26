#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent reference checks for the satellite risk/FMEA demonstration.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools_examples.planetary_observation.script_risk_analysis import (
    run_risk_analysis,
)


class TestSatelliteRiskAnalysis(unittest.TestCase):
    def assert_same(self, actual, expected):
        actual, expected = actual.flatten(), expected.flatten()
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_deterministic_scenario_severity_and_fmea_match_hand_calculations(self):
        reference = {
            ("capture", "failed"): (0.01, (2, 1), 1.25, 0.0125, 0.01),
            ("capture", "biased"): (0.02, (0, 0), 0.0, 0.0, 0.0),
            ("process", "failed"): (0.01, (2, 1), 1.25, 0.0125, 0.01),
            ("downlink", "failed"): (0.01, (2, 2), 2.0, 0.02, 0.01),
        }
        for staged in (False, True):
            with self.subTest(staged=staged):
                analysis = run_risk_analysis(staged=staged)
                self.assertEqual(len(analysis.sample.scenarios()), 8)
                self.assertEqual(
                    set(analysis.fmea["expected_shortfall"]), set(reference)
                )
                for scenario in analysis.sample.scenarios():
                    group = (scenario.obj, scenario.fault)
                    probability, losses, _, _, _ = reference[group]
                    index = (1, 3).index(scenario.time)
                    weight = (0.25, 0.75)[index]
                    self.assertAlmostEqual(scenario.rate, probability * weight)
                    prefix = scenario.name + ".tend.classify."
                    self.assertEqual(
                        analysis.mean_result[prefix + "shortfall"], losses[index]
                    )
                    history = analysis.histories.get("seed_0").get(scenario.name)
                    self.assertEqual(
                        max(2 - history["fxns.downlink.s.downlinked"][-1], 0),
                        losses[index],
                    )
                for group, (rate, _, severity, risk, event_weight) in reference.items():
                    table = analysis.fmea
                    self.assertAlmostEqual(table["sum_scenario_rate"][group], rate)
                    self.assertAlmostEqual(table["average_shortfall"][group], severity)
                    self.assertAlmostEqual(table["expected_shortfall"][group], risk)
                    self.assertAlmostEqual(
                        table["expected_delivery_failed"][group], event_weight
                    )
                self.assertAlmostEqual(
                    sum(analysis.fmea["expected_shortfall"].values()), 0.045
                )
                self.assertAlmostEqual(
                    sum(analysis.fmea["expected_delivery_failed"].values()), 0.03
                )
                self.assertEqual(
                    analysis.mean_result["nominal.tend.classify.shortfall"], 0
                )

    def test_exact_contact_distribution_has_known_background_and_fault_contributions(
        self,
    ):
        # With these seeds and no other randomness the contact indicators cover
        # 10, 11, 00, 01 exactly once. This is an exhaustive four-path reference,
        # not a claim that four arbitrary Monte Carlo samples are sufficient.
        analysis = run_risk_analysis(
            (0, 1, 2, 8),
            stochastic=True,
            params={
                "noise_std": 0.0,
                "obscuration_prob": 0.0,
                "contact_loss_prob": 0.5,
            },
        )
        patterns = {
            tuple(
                analysis.histories.get("seed_" + str(seed))[
                    "nominal.fxns.schedule.r.s.contact_lost"
                ][-2:]
            )
            for seed in analysis.seeds
        }
        self.assertEqual(patterns, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertEqual(analysis.mean_result["nominal.tend.classify.shortfall"], 1.0)
        self.assertEqual(
            analysis.mean_result["nominal.tend.classify.delivery_failed"], 0.75
        )
        # Failed capture/processor: .25*2 + .75*(2-.75) = 1.4375.
        expected = {
            ("capture", "failed"): 0.014375,
            ("process", "failed"): 0.014375,
            ("capture", "biased"): 0.02,
            ("downlink", "failed"): 0.02,
        }
        for group, risk in expected.items():
            self.assertAlmostEqual(analysis.fmea["expected_shortfall"][group], risk)
        self.assertAlmostEqual(
            sum(analysis.fmea["expected_shortfall"].values()), 0.06875
        )
        self.assertAlmostEqual(
            sum(analysis.fmea["expected_delivery_failed"].values()), 0.045
        )
        self.assertAlmostEqual(
            sum(analysis.fmea["expected_delta_shortfall"].values()), 0.01875
        )

    def test_seed_averaging_is_equal_weighted_and_keeps_paired_differences(self):
        seeds = (0, 1, 31)
        combined = run_risk_analysis(seeds, stochastic=True)
        singles = [run_risk_analysis((seed,), stochastic=True) for seed in seeds]
        for key in combined.mean_result:
            self.assertAlmostEqual(
                combined.mean_result[key],
                sum(single.mean_result[key] for single in singles) / len(seeds),
            )
        for metric, groups in combined.fmea.items():
            for group, value in groups.items():
                self.assertAlmostEqual(
                    value, np.mean([a.fmea[metric][group] for a in singles])
                )
        for seed in seeds:
            results = combined.results.get("seed_" + str(seed))
            baseline = results["nominal.tend.classify.shortfall"]
            history = combined.histories.get("seed_" + str(seed))
            for scenario in combined.sample.scenarios():
                prefix = scenario.name + ".tend.classify."
                self.assertEqual(
                    results[prefix + "delta_shortfall"],
                    results[prefix + "shortfall"] - baseline,
                )
                for field in ("noise", "obscured", "contact_lost"):
                    suffix = ".fxns.schedule.r.s." + field
                    np.testing.assert_array_equal(
                        history[scenario.name + suffix], history["nominal" + suffix]
                    )
        # More realizations do not multiply the declared fault mass.
        self.assertAlmostEqual(sum(combined.fmea["sum_scenario_rate"].values()), 0.05)
        # A bias can improve delivery counts. Signed deltas must not be clipped.
        self.assertLess(
            singles[1].fmea["expected_delta_shortfall"][("capture", "biased")], 0
        )

    def test_execution_order_and_input_ownership_do_not_change_results(self):
        params = {"noise_std": 0.1, "obscuration_prob": 0.2}
        before = copy.deepcopy(params)
        staged = run_risk_analysis((0, 8), stochastic=True, params=params)
        direct = run_risk_analysis((0, 8), stochastic=True, params=params, staged=False)
        self.assert_same(staged.results, direct.results)
        self.assert_same(staged.histories, direct.histories)
        self.assertEqual(staged.fmea.data, direct.fmea.data)
        reordered = run_risk_analysis(
            (8, 0),
            stochastic=True,
            params=params,
            times=(3, 1),
            time_weights=(0.75, 0.25),
        )
        self.assert_same(staged.mean_result, reordered.mean_result)
        self.assertEqual(staged.fmea.data, reordered.fmea.data)
        self.assertEqual(params, before)
        self.assertEqual(set(staged.sample.faultdomain.mdl.classify().values()), {0})

    def test_zero_time_weights_late_faults_and_zero_target(self):
        for times, weights, total in (
            ((1, 3), (1.0, 0.0), 0.06),
            ((1, 3), (0.0, 1.0), 0.04),
            ((6,), (1.0,), 0.01),
        ):
            with self.subTest(times=times, weights=weights):
                analysis = run_risk_analysis(times=times, time_weights=weights)
                self.assertAlmostEqual(
                    sum(analysis.fmea["expected_shortfall"].values()), total
                )
                self.assertAlmostEqual(
                    sum(analysis.fmea["sum_scenario_rate"].values()), 0.05
                )
        analysis = run_risk_analysis((0, 1), stochastic=True, required_packets=0)
        for metric in ("shortfall", "delivery_failed", "delta_shortfall"):
            self.assertEqual(set(analysis.fmea["expected_" + metric].values()), {0.0})
        blackout = run_risk_analysis(
            (0, 31), stochastic=True, params={"obscuration_prob": 1.0}
        )
        self.assertEqual(blackout.mean_result["nominal.tend.classify.shortfall"], 2)
        self.assertAlmostEqual(sum(blackout.fmea["expected_shortfall"].values()), 0.1)
        self.assertEqual(set(blackout.fmea["expected_delta_shortfall"].values()), {0.0})

    def test_tables_and_serializable_results_retain_the_same_totals(self):
        analysis = run_risk_analysis((0, 2), stochastic=True)
        before = copy.deepcopy(analysis.fmea.data)
        frame = analysis.fmea.as_table(sort_by="expected_shortfall")
        restored = pd.read_csv(io.StringIO(frame.to_csv()), index_col=[0, 1])
        assert_frame_equal(restored, frame, check_names=False)
        self.assertEqual(analysis.fmea.data, before)
        names = analysis.sample.scen_names()
        direct = analysis.mean_result.get_scens(*names).get_metric(
            "shortfall",
            method="expected",
            prefix="tend.classify.",
            rates=analysis.sample.get_scen_values("rate"),
            round_value=False,
        )
        self.assertAlmostEqual(direct, frame["expected_shortfall"].sum())
        nested = Result(analysis.results).nest().get_expected(round_value=False)
        self.assert_same(nested, analysis.mean_result)
        with tempfile.TemporaryDirectory() as directory:
            result_path = str(Path(directory, "results.npz"))
            history_path = str(Path(directory, "histories.npz"))
            analysis.results.save(result_path)
            analysis.histories.save(history_path)
            self.assert_same(Result.load(result_path), analysis.results)
            self.assert_same(History.load(history_path), analysis.histories)

    def test_invalid_inputs_are_rejected_before_propagation(self):
        cases = (
            {"seeds": ()},
            {"seeds": (0, 0)},
            {"seeds": (True,)},
            {"seeds": (-1,)},
            {"seeds": (1.5,)},
            {"times": (), "time_weights": ()},
            {"times": (1, 1)},
            {"times": (0, 3)},
            {"times": (1, 7)},
            {"times": (1, 2.5)},
            {"times": (1, float("nan"))},
            {"time_weights": (1,)},
            {"time_weights": (0.2, 0.2)},
            {"time_weights": (-0.1, 1.1)},
            {"time_weights": (0, float("inf"))},
            {"time_weights": (0, float("nan"))},
            {"required_packets": -1},
            {"required_packets": True},
            {"required_packets": 1.5},
        )
        with patch("fmdtools.sim.propagate.fault_sample") as simulate:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_risk_analysis(**options)
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
