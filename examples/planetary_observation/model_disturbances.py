#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeded disturbances and injected faults for the planetary-observation example.

All distribution parameters and per-run fault probabilities are illustrative.
They are not calibrated spacecraft reliability or detection-performance data.

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

import math

from fmdtools_examples.planetary_observation.model_main import (
    CaptureObservation,
    CaptureState,
    DataBuffer,
    DownlinkData,
    MissionParam,
    Observation,
    ObservationSatellite,
    ProcessObservation,
    ProcessingState,
    Scene,
    SceneState,
    ScheduleObservation,
)
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.rand import Rand
from fmdtools.define.container.state import State


class DisturbanceParam(MissionParam, readonly=True):
    """Synthetic disturbance settings; schedules and capacities remain inherited.

    noise_std is the dimensionless Gaussian standard deviation before clipping
    an observed score to [0, 1]. obscuration_prob applies per scheduled capture;
    contact_loss_prob applies per scheduled contact, independently between steps.
    bias is the additive score offset while the payload's biased fault is active.
    Random effects require sp.run_stochastic=True; faults require injection.
    """

    noise_std: float = 0.05
    obscuration_prob: float = 0.1
    contact_loss_prob: float = 0.1
    bias: float = 0.4
    noise_std_lim = (0.0, math.inf)
    obscuration_prob_lim = (0.0, 1.0)
    contact_loss_prob_lim = (0.0, 1.0)
    bias_lim = (-1.0, 1.0)

    def check_lim(self, k, v):
        """Retain native bounds and reject an infinite noise scale."""
        super().check_lim(k, v)
        if k == "noise_std" and not math.isfinite(v):
            raise ValueError("noise_std must be finite")


class DisturbanceState(State):
    """Latent draws for the current step, zero outside their scheduled events.

    noise is the unbounded, dimensionless additive Gaussian draw. obscured and
    contact_lost are Bernoulli indicators (0 or 1), not persistent fault flags.
    """

    noise: float = 0.0
    obscured: int = 0
    contact_lost: int = 0


class DisturbanceRand(Rand):
    """One native RNG stream for all exogenous mission disturbances."""

    s: DisturbanceState = DisturbanceState()

    def reset(self):
        """Reset the behavior-maintained density along with the native draws."""
        super().reset()
        self.probdens = 1.0


class DisturbedSceneState(SceneState):
    """Prescribed scene plus current noise and temporary availability losses.

    noise is added at capture. obscured blocks only this observation, while
    contact_lost blocks only this contact. Neither changes the planned schedule.
    """

    noise: float = 0.0
    obscured: bool = False
    contact_lost: bool = False


class DisturbedScene(Scene):
    """Shared exogenous inputs; the original scene score stays unchanged."""

    container_s = DisturbedSceneState


class DisturbedSchedule(ScheduleObservation):
    """Draw inputs independently of fault state to preserve paired scenarios.

    At an observation draw noise (only for positive scale), then obscuration;
    at a contact draw link loss. There are no draws at time zero or idle steps.
    The same schedule and seed therefore retain inputs across injected faults.
    """

    container_p = DisturbanceParam
    container_r = DisturbanceRand
    flow_scene = DisturbedScene

    def dynamic_behavior(self):
        super().dynamic_behavior()
        self.r.s.put(noise=0.0, obscured=0, contact_lost=0)
        if self.scene.s.visible:
            if self.p.noise_std > 0.0:
                self.r.set_rand_state("noise", "normal", 0.0, self.p.noise_std)
            self.r.set_rand_state("obscured", "binomial", 1, self.p.obscuration_prob)
        if self.scene.s.contact:
            self.r.set_rand_state(
                "contact_lost", "binomial", 1, self.p.contact_loss_prob
            )
        self.scene.s.put(
            noise=self.r.s.noise,
            obscured=bool(self.r.s.obscured),
            contact_lost=bool(self.r.s.contact_lost),
        )
        # These draws occur in behavior, after the automatic Rand update.
        if self.r.track_pdf:
            self.r.probdens = self.r.return_probdens()


class PayloadMode(Mode):
    """Persistent no-capture and score-bias faults with synthetic per-run rates.

    Failed takes precedence if both are present; neither is randomly activated.
    """

    fault_failed = Fault(prob=0.01, units="sim")
    fault_biased = Fault(prob=0.02, units="sim")


class UnavailableMode(Mode):
    """Persistent processing or downlink outage, injected explicitly.

    The 0.01 per-run probability is a demonstration placeholder, not mission data.
    """

    fault_failed = Fault(prob=0.01, units="sim")


class DisturbedCaptureState(CaptureState):
    """acquired counts captures; missed counts scheduled but blocked acquisitions."""

    missed: int = 0


class DisturbedProcessingState(ProcessingState):
    """Add unprocessed, the captured packets discarded while processing is failed."""

    unprocessed: int = 0


class DisturbedCapture(CaptureObservation):
    """Apply temporary obscuration, persistent failure/bias and score clipping."""

    container_p = DisturbanceParam
    container_m = PayloadMode
    container_s = DisturbedCaptureState
    flow_scene = DisturbedScene

    def dynamic_behavior(self):
        valid = self.scene.s.visible and not (
            self.scene.s.obscured or self.m.has_fault("failed")
        )
        signal = 0.0
        if valid:
            bias = self.p.bias if self.m.has_fault("biased") else 0.0
            signal = min(1.0, max(0.0, self.scene.s.signal + self.scene.s.noise + bias))
        self.observation.s.put(
            valid=valid, scene_id=self.scene.s.scene_id if valid else -1, signal=signal
        )
        if self.scene.s.visible:
            if valid:
                self.s.acquired += 1
            else:
                self.s.missed += 1


class FaultyProcessor(ProcessObservation):
    """Discard arrivals during an injected processing outage; keep prior backlog."""

    container_m = UnavailableMode
    container_s = DisturbedProcessingState

    def dynamic_behavior(self):
        if self.m.has_fault("failed"):
            if self.observation.s.valid:
                self.s.unprocessed += 1
        else:
            super().dynamic_behavior()


class DisturbedDownlink(DownlinkData):
    """Retain queued packets during temporary or persistent link outages."""

    container_m = UnavailableMode
    flow_scene = DisturbedScene

    def dynamic_behavior(self):
        if not (self.scene.s.contact_lost or self.m.has_fault("failed")):
            super().dynamic_behavior()


class DisturbedObservationSatellite(ObservationSatellite):
    """Example-only variant using one seeded disturbance source and native modes.

    Set r={'seed': N} and sp={'run_stochastic': True} to enable random effects.
    Leave randomness off to reproduce the scaffold or isolate an injected fault.
    The original execution order and packet-buffer semantics are unchanged.
    """

    container_p = DisturbanceParam
    container_r = Rand

    def init_architecture(self, **kwargs):
        self.check_values()
        self.add_flow("scene", DisturbedScene)
        self.add_flow("observation", Observation)
        self.add_flow("buffer", DataBuffer)
        self.add_fxn("schedule", DisturbedSchedule, "scene", p=self.p)
        self.add_fxn("capture", DisturbedCapture, "scene", "observation", p=self.p)
        self.add_fxn("process", FaultyProcessor, "observation", "buffer", p=self.p)
        self.add_fxn("downlink", DisturbedDownlink, "scene", "buffer", p=self.p)

    def classify(self, **kwargs):
        """Return counts, including failed acquisition and processing outcomes."""
        return {
            **super().classify(**kwargs),
            "missed": int(self.fxns["capture"].s.missed),
            "unprocessed": int(self.fxns["process"].s.unprocessed),
        }


if __name__ == "__main__":
    from fmdtools.sim import propagate

    model = DisturbedObservationSatellite(r={"seed": 0}, sp={"run_stochastic": True})
    result, history = propagate.nominal(model)
    print(result)
