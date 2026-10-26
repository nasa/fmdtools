#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline image classification in the planetary-observation mission example.

Handwritten digits are a software-integration proxy, not planetary imagery.
The estimator and dataset come from scikit-learn; no model files are downloaded.

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

from functools import lru_cache
from numbers import Integral

from fmdtools_examples.planetary_observation.model_disturbances import (
    DisturbanceParam,
    DisturbedCapture,
    DisturbedDownlink,
    DisturbedObservationSatellite,
    DisturbedProcessingState,
    DisturbedScene,
    DisturbedSchedule,
    FaultyProcessor,
)
from fmdtools_examples.planetary_observation.model_main import (
    DataBuffer,
    Observation,
    ObservationState,
)
from fmdtools.define.container.state import State

import numpy as np


@lru_cache(maxsize=1)
def _digit_data():
    """Load immutable normalized pixels and labels from scikit-learn's data."""
    from sklearn.datasets import load_digits

    pixels, labels = load_digits(return_X_y=True)
    pixels = pixels / 16.0
    pixels.setflags(write=False)
    labels.setflags(write=False)
    return pixels, labels


def heldout_digits():
    """Return read-only mission pixels/labels, disjoint from the training half.

    Rows are flattened 8-by-8 images normalized to [0, 1]. Indices in
    ImageMissionParam refer to this held-out partition, not the full dataset.
    Labels are available for evaluation, never as classifier input.
    """
    pixels, labels = _digit_data()
    split = len(labels) // 2
    return pixels[split:], labels[split:]


@lru_cache(maxsize=1)
def _digit_classifier():
    """Fit the upstream SVC recipe once per process, using only the first half.

    The private estimator is used read-only for inference. It is not simulator
    state, is never fitted inside a propagation step, and needs no checkpoint.
    """
    from sklearn.svm import SVC

    pixels, labels = _digit_data()
    split = len(labels) // 2
    return SVC(gamma=0.001).fit(pixels[:split] * 16.0, labels[:split])


def predict_digit(pixels):
    """Classify one finite normalized 8-by-8 image (or its 64-vector).

    This is the example's inference boundary: only observed pixels enter the
    estimator. Invalid inputs raise rather than silently returning a class.
    """
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape not in ((64,), (8, 8)):
        raise ValueError("Expected one 8-by-8 image or a 64-element pixel vector")
    if not np.all(np.isfinite(pixels)) or np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("Pixels must be finite and normalized to [0, 1]")
    return int(_digit_classifier().predict(pixels.reshape(1, 64) * 16.0)[0])


class ImageMissionParam(DisturbanceParam, readonly=True):
    """Held-out image indices and the digit classes worth selecting.

    image_indices contains one held-out row index per scheduled observation.
    target_digits defines a binary detection task on the ten predicted classes.
    Inherited signals/threshold do not drive image selection. noise_std and
    bias are common-mode brightness offsets in normalized pixel units.
    """

    image_indices: tuple = (0, 1, 2, 3)
    target_digits: tuple = (8,)

    def check_lim(self, k, v):
        """Keep native mission limits and validate the image-selection tuples."""
        super().check_lim(k, v)
        if k == "image_indices":
            if len(v) != len(self.observation_times):
                raise ValueError("Provide one image index per observation time")
            count = len(heldout_digits()[1])
            if any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < count
                for i in v
            ):
                raise ValueError("Image indices must refer to held-out rows")
        elif k == "target_digits":
            if any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i <= 9
                for i in v
            ) or len(set(v)) != len(v):
                raise ValueError("target_digits must contain distinct digits 0 to 9")


class ImageObservationState(ObservationState):
    """Captured normalized pixels; invalid observations contain a zero vector.

    Inherited signal records mean observed brightness for inspection only.
    scene_id identifies the scheduled observation, not its ground-truth class.
    """

    pixels: np.array = np.zeros(64)


class ImageObservation(Observation):
    """Pixel-valued payload output consumed by the onboard classifier."""

    container_s = ImageObservationState


class ImageCapture(DisturbedCapture):
    """Apply existing availability faults and common-mode image corruption."""

    container_p = ImageMissionParam
    flow_observation = ImageObservation

    def dynamic_behavior(self):
        super().dynamic_behavior()
        pixels = np.zeros(64)
        if self.observation.s.valid:
            index = self.p.image_indices[self.scene.s.scene_id]
            clean = heldout_digits()[0][index]
            bias = self.p.bias if self.m.has_fault("biased") else 0.0
            pixels = np.clip(clean + self.scene.s.noise + bias, 0.0, 1.0)
        self.observation.s.pixels = pixels
        self.observation.s.signal = float(np.mean(pixels))


class ClassifierState(DisturbedProcessingState):
    """Add the current prediction: 0..9 after inference, -1 when not classified.

    Inherited processed/selected/dropped/unprocessed fields are cumulative.
    No ground-truth labels are supplied to this function.
    """

    predicted_label: int = -1


class ClassifyObservation(FaultyProcessor):
    """Call scikit-learn for valid arrivals, retaining native outage/buffer logic."""

    container_p = ImageMissionParam
    container_s = ClassifierState
    flow_observation = ImageObservation

    def init_block(self, **kwargs):
        """Prepare the fixed estimator before propagation, without custom init."""
        _digit_classifier()

    def select_observation(self, signal):
        """Replace the scalar threshold with pixel-only class membership."""
        self.s.predicted_label = predict_digit(self.observation.s.pixels)
        return self.s.predicted_label in self.p.target_digits

    def dynamic_behavior(self):
        self.s.predicted_label = -1
        super().dynamic_behavior()


class ClassificationCounts(State):
    """Cumulative evaluation counts, separate from the decision-making function.

    Confusion counts apply only to classified observations. unclassified counts
    unavailable acquisitions/processing, with unclassified_targets their target
    subset. target_opportunities counts scheduled true targets. Errors measure
    ten-class prediction errors; detection uses membership in target_digits.
    """

    target_opportunities: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    unclassified: int = 0
    unclassified_targets: int = 0
    classification_errors: int = 0


class ClassifyingSatellite(DisturbedObservationSatellite):
    """Image-classification variant; the original satellite models are unchanged.

    The architecture evaluates predictions after all mission functions run.
    Its truth labels update counters only and cannot alter selection or storage.
    Delivered packets remain count-only, so their target identities are unknown.
    """

    container_p = ImageMissionParam
    container_s = ClassificationCounts
    default_track = (*DisturbedObservationSatellite.default_track, "s")

    def init_architecture(self, **kwargs):
        self.check_values()
        self.add_flow("scene", DisturbedScene)
        self.add_flow("observation", ImageObservation)
        self.add_flow("buffer", DataBuffer)
        self.add_fxn("schedule", DisturbedSchedule, "scene", p=self.p)
        self.add_fxn("capture", ImageCapture, "scene", "observation", p=self.p)
        self.add_fxn("process", ClassifyObservation, "observation", "buffer", p=self.p)
        self.add_fxn("downlink", DisturbedDownlink, "scene", "buffer", p=self.p)

    def dynamic_behavior(self):
        """Evaluate this step after inference; never feed truth back to the model."""
        scene = self.flows["scene"].s
        if not scene.visible:
            return
        truth = int(heldout_digits()[1][self.p.image_indices[scene.scene_id]])
        target = truth in self.p.target_digits
        self.s.target_opportunities += int(target)
        predicted = self.fxns["process"].s.predicted_label
        if predicted < 0:
            self.s.unclassified += 1
            self.s.unclassified_targets += int(target)
            return
        self.s.classification_errors += int(predicted != truth)
        selected = predicted in self.p.target_digits
        if target:
            if selected:
                self.s.true_positives += 1
            else:
                self.s.false_negatives += 1
        elif selected:
            self.s.false_positives += 1
        else:
            self.s.true_negatives += 1

    def classify(self, **kwargs):
        """Return raw mission and classification counts, suitable for native FMEA."""
        return {
            **super().classify(**kwargs),
            **{name: int(value) for name, value in self.s.asdict().items()},
            "missed_targets": int(self.s.false_negatives + self.s.unclassified_targets),
        }


if __name__ == "__main__":
    from fmdtools.sim import propagate

    result, history = propagate.nominal(ClassifyingSatellite())
    print(result)
