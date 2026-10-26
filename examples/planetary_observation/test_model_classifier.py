#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checks for image inference, fault propagation and classification accounting.

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

import importlib.util
import pickle
import unittest
from unittest.mock import patch

from fmdtools_examples.planetary_observation import model_classifier as ml
from fmdtools_examples.planetary_observation.script_risk_analysis import (
    make_fault_sample,
)
from fmdtools_examples.planetary_observation.script_classification import (
    run_classification_analysis,
)
from fmdtools.analyze.tabulate import FMEA
from fmdtools.sim import propagate

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "requires fmdtools[examples]")
class TestSatelliteClassifier(unittest.TestCase):
    def assert_same(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def assert_counts(self, result, history):
        c = {k.removeprefix("tend.classify."): v for k, v in result.items()}
        self.assertEqual(
            c["processed"],
            sum(
                c[k]
                for k in (
                    "true_positives",
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                )
            ),
        )
        self.assertEqual(c["selected"], c["true_positives"] + c["false_positives"])
        self.assertEqual(c["opportunities"], c["processed"] + c["unclassified"])
        self.assertEqual(
            c["target_opportunities"], c["true_positives"] + c["missed_targets"]
        )
        self.assertEqual(c["unclassified"], c["missed"] + c["unprocessed"])
        np.testing.assert_array_equal(
            history["fxns.process.s.selected"],
            history["fxns.downlink.s.downlinked"]
            + history["flows.buffer.s.queued"]
            + history["fxns.process.s.dropped"],
        )

    def test_bundled_training_is_offline_disjoint_and_cached(self):
        from sklearn.datasets import load_digits
        from sklearn.svm import SVC

        data, labels = load_digits(return_X_y=True)
        split = len(labels) // 2
        ml._digit_data.cache_clear()
        ml._digit_classifier.cache_clear()
        original_fit = SVC.fit
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            with patch.object(
                SVC, "fit", autospec=True, side_effect=original_fit
            ) as fit:
                classifier = ml._digit_classifier()
                self.assertIs(ml._digit_classifier(), classifier)
                self.assertEqual(fit.call_count, 1)
                np.testing.assert_array_equal(fit.call_args.args[1], data[:split])
                np.testing.assert_array_equal(fit.call_args.args[2], labels[:split])
        images, truth = ml.heldout_digits()
        np.testing.assert_array_equal(images * 16, data[split:])
        np.testing.assert_array_equal(truth, labels[split:])
        self.assertFalse(images.flags.writeable)
        self.assertFalse(truth.flags.writeable)
        reference = SVC(gamma=0.001).fit(data[:split], labels[:split])
        actual = [ml.predict_digit(image) for image in images]
        np.testing.assert_array_equal(actual, reference.predict(data[split:]))
        self.assertGreater(np.mean(np.asarray(actual) == truth), 0.85)

    def test_nominal_classification_matches_known_heldout_images(self):
        model = ml.ClassifyingSatellite()
        result, history = propagate.nominal(model)
        np.testing.assert_array_equal(ml.heldout_digits()[1][:4], [8, 8, 4, 9])
        np.testing.assert_array_equal(
            history["fxns.process.s.predicted_label"], [-1, 8, 8, 4, 9, -1, -1]
        )
        np.testing.assert_array_equal(
            history["s.true_positives"], [0, 1, 2, 2, 2, 2, 2]
        )
        np.testing.assert_array_equal(
            history["flows.observation.s.pixels"][1:5], ml.heldout_digits()[0][:4]
        )
        self.assertEqual(result["tend.classify.downlinked"], 2)
        self.assertEqual(result["tend.classify.classification_errors"], 0)
        self.assertEqual(result["tend.classify.missed_targets"], 0)
        self.assertEqual(set(model.classify().values()), {0})
        self.assert_counts(result, history)
        # Scalar scores and threshold no longer control image-based selection.
        other = ml.ClassifyingSatellite(
            p={"signals": (0.0, 0.0, 0.0, 0.0), "threshold": 1.0}
        )
        other_result, other_history = propagate.nominal(other)
        self.assertEqual(other_result, result)
        np.testing.assert_array_equal(
            other_history["fxns.process.s.predicted_label"],
            history["fxns.process.s.predicted_label"],
        )

    def test_errors_are_counted_and_truth_never_changes_the_decisions(self):
        model = ml.ClassifyingSatellite()
        with patch.object(ml, "predict_digit", side_effect=[8, 1, 8, 9]) as predict:
            result, history = propagate.nominal(model)
        for metric in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ):
            self.assertEqual(result["tend.classify." + metric], 1)
        self.assertEqual(result["tend.classify.classification_errors"], 2)
        self.assertEqual(result["tend.classify.missed_targets"], 1)
        self.assertEqual(predict.call_count, 4)
        for call in predict.call_args_list:
            self.assertEqual(call.args[0].shape, (64,))
            self.assertFalse(call.kwargs)
        self.assert_counts(result, history)
        reference, ref_history = propagate.nominal(model)
        images, truth = ml.heldout_digits()
        with patch.object(
            ml, "heldout_digits", return_value=(images, (truth + 1) % 10)
        ):
            changed, changed_history = propagate.nominal(model)
        for key in ref_history:
            if key.startswith(("flows.", "fxns.")):
                np.testing.assert_array_equal(changed_history[key], ref_history[key])
        self.assertEqual(
            changed["tend.classify.selected"], reference["tend.classify.selected"]
        )
        self.assertEqual(changed["tend.classify.classification_errors"], 4)

    def test_faults_and_random_loss_skip_inference_or_corrupt_pixels(self):
        for block, fault in (
            ("capture", "failed"),
            ("process", "failed"),
            ("downlink", "failed"),
        ):
            with self.subTest(block=block):
                model = ml.ClassifyingSatellite()
                with patch.object(
                    ml, "predict_digit", wraps=ml.predict_digit
                ) as predict:
                    result, history = propagate.one_fault(
                        model, block, fault, time=1.0, include_nominal=False
                    )
                self.assertEqual(predict.call_count, 4 if block == "downlink" else 0)
                self.assertEqual(
                    result["tend.classify.missed_targets"],
                    0 if block == "downlink" else 2,
                )
                self.assertEqual(result["tend.classify.downlinked"], 0)
                self.assert_counts(result, history)
        for bias in (-1.0, 1.0):
            model = ml.ClassifyingSatellite(p={"bias": bias})
            result, history = propagate.one_fault(
                model, "capture", "biased", time=1.0, include_nominal=False
            )
            expected = np.full((4, 64), float(bias > 0))
            np.testing.assert_array_equal(
                history["flows.observation.s.pixels"][1:5], expected
            )
            np.testing.assert_array_equal(
                history["fxns.process.s.predicted_label"][1:5],
                [ml.predict_digit(x) for x in expected],
            )
            self.assert_counts(result, history)
        model = ml.ClassifyingSatellite(
            p={"obscuration_prob": 1.0}, sp={"run_stochastic": True}
        )
        with patch.object(ml, "predict_digit", side_effect=AssertionError("no image")):
            result, history = propagate.nominal(model)
        self.assertEqual(result["tend.classify.unclassified"], 4)
        self.assertEqual(result["tend.classify.classification_errors"], 0)
        self.assert_counts(result, history)

    def test_seeded_fault_paths_are_paired_and_staging_independent(self):
        outputs = []
        for seed in (0, 31):
            model = ml.ClassifyingSatellite(
                r={"seed": seed}, sp={"run_stochastic": True}
            )
            sample = make_fault_sample(model)
            a, ah = propagate.fault_sample(
                model, sample, staged=False, showprogress=False
            )
            b, bh = propagate.fault_sample(
                model, sample, staged=True, showprogress=False
            )
            self.assert_same(a, b)
            self.assert_same(ah, bh)
            for scenario in sample.scenarios():
                h = ah.get(scenario.name)
                self.assert_counts(a.get(scenario.name), h)
                for variable in ("noise", "obscured", "contact_lost"):
                    key = "fxns.schedule.r.s." + variable
                    np.testing.assert_array_equal(h[key], ah["nominal." + key])
            clean = ml.heldout_digits()[0]
            for time in model.p.observation_times:
                if ah["nominal.flows.observation.s.valid"][time]:
                    idx = model.p.image_indices[model.p.observation_times.index(time)]
                    noise = ah["nominal.fxns.schedule.r.s.noise"][time]
                    np.testing.assert_allclose(
                        ah["nominal.flows.observation.s.pixels"][time],
                        np.clip(clean[idx] + noise, 0, 1),
                    )
            outputs.append(ah["nominal.fxns.schedule.r.s.noise"])
        self.assertFalse(np.array_equal(*outputs))

    def test_copy_pickle_reset_and_repeat_runs_never_refit(self):
        from sklearn.svm import SVC

        model = ml.ClassifyingSatellite(r={"seed": 31}, sp={"run_stochastic": True})
        before_estimator = pickle.dumps(ml._digit_classifier())
        with patch.object(SVC, "fit", side_effect=AssertionError("unexpected refit")):
            expected, hist = propagate.nominal(model)
            model(time=3.0)
            original = model.classify()
            for clone in (model.copy(), pickle.loads(pickle.dumps(model))):
                clone(time=6.0)
                self.assert_same(clone.h.flatten(), hist)
                self.assertEqual(model.classify(), original)
                self.assertIsNot(
                    clone.flows["observation"].s.pixels,
                    model.flows["observation"].s.pixels,
                )
            model.reset()
            model(time=6.0)
            self.assert_same(model.h.flatten(), hist)
            self.assertEqual(
                model.classify(),
                {k.removeprefix("tend.classify."): v for k, v in expected.items()},
            )
        self.assertEqual(pickle.dumps(ml._digit_classifier()), before_estimator)

    def test_native_fmea_distinguishes_missed_targets_from_link_failure(self):
        model = ml.ClassifyingSatellite(p={"bias": 0.0})
        sample = make_fault_sample(model)
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        metrics = [
            "missed_targets",
            "classification_errors",
            "false_positives",
            "unclassified",
        ]
        table = FMEA(
            result,
            sample,
            expected_metric=metrics,
            rates="scenario_rate",
            round_value=False,
        )
        # The two true targets are acquired at t=1 and t=2. A t=1 outage
        # misses both; a t=3 outage misses neither: .01*(.25*2+.75*0)=.005.
        expected = {
            ("capture", "failed"): 0.005,
            ("process", "failed"): 0.005,
            ("capture", "biased"): 0.0,
            ("downlink", "failed"): 0.0,
        }
        for group, value in expected.items():
            self.assertAlmostEqual(table["expected_missed_targets"][group], value)
        self.assertAlmostEqual(sum(table["expected_missed_targets"].values()), 0.01)
        self.assertAlmostEqual(sum(table["expected_unclassified"].values()), 0.05)
        self.assertEqual(sum(table["expected_classification_errors"].values()), 0.0)
        self.assertEqual(sum(table["expected_false_positives"].values()), 0.0)
        self.assertEqual(len(table.as_table()), 4)

    def test_analysis_script_preserves_inputs_and_explicit_time_weights(self):
        params = {"bias": 0.0}
        result, history, sample, table = run_classification_analysis(
            params=params, time_weights=(1.0, 0.0)
        )
        self.assertEqual(params, {"bias": 0.0})
        self.assertEqual(len(sample.scenarios()), 8)
        self.assertEqual(sum(s.rate == 0 for s in sample.scenarios()), 4)
        self.assertAlmostEqual(sum(table["expected_missed_targets"].values()), 0.04)
        self.assert_counts(result.get("nominal"), history.get("nominal"))
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                run_classification_analysis(seed=seed)
        first = run_classification_analysis(seed=31, stochastic=True, staged=True)
        second = run_classification_analysis(seed=31, stochastic=True, staged=False)
        self.assert_same(first[0], second[0])
        self.assert_same(first[1], second[1])
        self.assertEqual(first[3].data, second[3].data)

    def test_invalid_images_parameters_and_partial_runs(self):
        good = ml.heldout_digits()[0][0].copy()
        self.assertEqual(ml.predict_digit(good), ml.predict_digit(good.reshape(8, 8)))
        np.testing.assert_array_equal(good, ml.heldout_digits()[0][0])
        for pixels in (
            np.zeros(63),
            np.zeros((2, 32)),
            np.full(64, np.nan),
            np.full(64, -0.1),
            np.full(64, 1.1),
        ):
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                ml.predict_digit(pixels)
        for params in (
            {"image_indices": (0,)},
            {"image_indices": (0, 1, 2, 99999)},
            {"image_indices": (True, 1, 2, 3)},
            {"image_indices": (0.5, 1, 2, 3)},
            {"target_digits": (10,)},
            {"target_digits": (8, 8)},
            {"target_digits": (True,)},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                ml.ImageMissionParam(**params)
        for end in (0, 2, 6, 8):
            for targets in ((), (8,), tuple(range(10))):
                with self.subTest(end=end, targets=targets):
                    model = ml.ClassifyingSatellite(
                        p={"target_digits": targets}, sp={"end_time": float(end)}
                    )
                    result, history = propagate.nominal(model)
                    self.assert_counts(result, history)
                    self.assertEqual(result["tend.classify.opportunities"], min(end, 4))


if __name__ == "__main__":
    unittest.main()
