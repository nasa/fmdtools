#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministic planetary-observation mission scaffold with bounded data storage.

This synthetic example has scheduled scalar observations, threshold selection,
and packet-count downlink. It is not a physical spacecraft or trained ML model.

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
from numbers import Integral, Real

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow


class MissionParam(Parameter, readonly=True):
    """Immutable synthetic schedule and capacities for one-minute execution steps.

    observation_times and contact_times are strictly increasing positive integer
    minutes. signals contains one dimensionless value in [0, 1] per observation.
    threshold is inclusive. Capacities count whole packets, not bytes or bit rate.
    """

    observation_times: tuple = (1, 2, 3, 4)
    signals: tuple = (0.2, 0.8, 0.1, 0.9)
    contact_times: tuple = (5, 6)
    threshold: float = 0.5
    buffer_capacity: int = 4
    downlink_capacity: int = 1

    threshold_lim = (0.0, 1.0)
    buffer_capacity_lim = (0, math.inf)
    downlink_capacity_lim = (0, math.inf)

    def get_true_fields(self, *args, force_kwargs=False, **kwargs):
        """Reject unsafe capacity coercions before BaseContainer types fields.

        The normal type conversion turns ``True`` or ``1.5`` into integer 1
        before check_lim runs. Keep only this type guard in field resolution;
        declared limits handle ranges without duplicating their checks.
        """
        values = super().get_true_fields(*args, force_kwargs=force_kwargs, **kwargs)
        for name in ("buffer_capacity", "downlink_capacity"):
            value = values[self.__fields__.index(name)]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(name + " must be an integer packet count")
        return values

    def check_lim(self, k, v):
        """Apply declared scalar limits, then check mission-specific sequences."""
        super().check_lim(k, v)
        if k in ("observation_times", "contact_times"):
            if any(
                isinstance(t, bool) or not isinstance(t, Integral) or t <= 0 for t in v
            ):
                raise ValueError(k + " must contain positive integer minutes")
            if any(a >= b for a, b in zip(v, v[1:])):
                raise ValueError(k + " must be strictly increasing")
        elif k == "signals":
            if len(v) != len(self.observation_times):
                raise ValueError("signals must contain one value per observation time")
            if any(
                not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                for value in v
            ):
                raise ValueError("signals must be finite values in [0, 1]")


class SceneState(State):
    """Scene opportunity during the current one-minute step.

    ``visible`` is a scheduled acquisition opportunity; ``contact`` marks
    a scheduled ground link. ``scene_id`` is the zero-based observation index
    (or -1 when absent); ``signal`` is its dimensionless [0, 1] score.
    """

    visible: bool = False
    contact: bool = False
    scene_id: int = -1
    signal: float = 0.0


class Scene(Flow):
    """Environment-to-payload/link schedule flow."""

    container_s = SceneState


class ObservationState(State):
    """Current captured observation, cleared on steps without an opportunity.

    ``valid`` indicates a capture this step; ``scene_id`` is its schedule
    index (-1 if invalid); ``signal`` is the captured dimensionless score.
    """

    valid: bool = False
    scene_id: int = -1
    signal: float = 0.0


class Observation(Flow):
    """Payload-to-processor current observation."""

    container_s = ObservationState


class BufferState(State):
    """Pending selected-packet count.

    ``queued`` is the number of whole packets awaiting contact; the scaffold
    stores no packet content, identity or bytes.
    """

    queued: int = 0


class DataBuffer(Flow):
    """Shared packet count between processing and downlink."""

    container_s = BufferState


class ScheduleState(State):
    """Cumulative schedule counts.

    ``opportunities`` counts elapsed scheduled visibility steps, not
    completed acquisitions or downlinks.
    """

    opportunities: int = 0


class CaptureState(State):
    """Cumulative payload captures.

    ``acquired`` counts valid one-packet acquisitions since mission start.
    """

    acquired: int = 0


class ProcessingState(State):
    """Cumulative processing counters for the current mission.

    ``processed`` counts valid signals checked by the selector; ``selected``
    counts scores meeting the threshold. ``dropped`` counts selected packets
    rejected because the count-only buffer was full at arrival.
    """

    processed: int = 0
    selected: int = 0
    dropped: int = 0


class DownlinkState(State):
    """Cumulative contact-time transmissions.

    ``downlinked`` counts whole packets drained from the shared buffer.
    """

    downlinked: int = 0


class ScheduleObservation(Function):
    """Supply a prescribed scene/contact schedule, without propagating an orbit."""

    container_p = MissionParam
    container_s = ScheduleState
    flow_scene = Scene

    def dynamic_behavior(self):
        time = self.t.time
        visible = time in self.p.observation_times
        index = self.p.observation_times.index(time) if visible else -1
        self.scene.s.put(
            visible=visible,
            contact=time in self.p.contact_times,
            scene_id=index,
            signal=self.p.signals[index] if visible else 0.0,
        )
        if visible:
            self.s.opportunities += 1


class CaptureObservation(Function):
    """Copy the current visible scalar scene into an observation packet."""

    container_s = CaptureState
    flow_scene = Scene
    flow_observation = Observation

    def dynamic_behavior(self):
        valid = self.scene.s.visible
        self.observation.s.put(
            valid=valid,
            scene_id=self.scene.s.scene_id if valid else -1,
            signal=self.scene.s.signal if valid else 0.0,
        )
        if valid:
            self.s.acquired += 1


class ProcessObservation(Function):
    """Select observations and enqueue them, counting overflow separately.

    select_observation is the local extension point for a later external model.
    This baseline uses only a threshold and has no ML accuracy interpretation.
    """

    container_p = MissionParam
    container_s = ProcessingState
    flow_observation = Observation
    flow_buffer = DataBuffer

    def select_observation(self, signal):
        """Return the deterministic keep/discard decision for a valid observation."""
        return signal >= self.p.threshold

    def dynamic_behavior(self):
        if self.observation.s.valid:
            self.s.processed += 1
            if self.select_observation(self.observation.s.signal):
                self.s.selected += 1
                if self.buffer.s.queued < self.p.buffer_capacity:
                    self.buffer.s.queued += 1
                else:
                    self.s.dropped += 1


class DownlinkData(Function):
    """Deliver at most downlink_capacity queued packets per contact step."""

    container_p = MissionParam
    container_s = DownlinkState
    flow_scene = Scene
    flow_buffer = DataBuffer

    def dynamic_behavior(self):
        if self.scene.s.contact:
            sent = min(self.buffer.s.queued, self.p.downlink_capacity)
            self.buffer.s.queued -= sent
            self.s.downlinked += sent


class ObservationSatellite(FunctionArchitecture):
    """Nominal observation, selection and store-and-forward mission architecture.

    Time zero is the initial condition. Each positive integer minute executes
    schedule -> capture -> process -> downlink, in that order. A newly selected
    packet can be downlinked in the same step if it fits in the buffer first.
    """

    container_p = MissionParam
    default_sp = dict(end_time=6.0, dt=1.0, units="min")

    def check_values(self):
        """Check the example-specific one-minute schedule requirements."""
        if (
            self.sp.start_time != 0.0
            or self.sp.dt != 1.0
            or self.t.dt != 1.0
            or self.sp.units != "min"
            or not math.isfinite(self.sp.end_time)
            or self.sp.end_time < 0
            or self.sp.end_time != int(self.sp.end_time)
        ):
            raise ValueError(
                "The scaffold requires start_time=0, dt=1 minute and "
                "a nonnegative integer end_time"
            )

    def init_architecture(self, **kwargs):
        self.check_values()
        self.add_flow("scene", Scene)
        self.add_flow("observation", Observation)
        self.add_flow("buffer", DataBuffer)
        # Registration order defines the causal order of each discrete step.
        self.add_fxn("schedule", ScheduleObservation, "scene", p=self.p)
        self.add_fxn("capture", CaptureObservation, "scene", "observation")
        self.add_fxn("process", ProcessObservation, "observation", "buffer", p=self.p)
        self.add_fxn("downlink", DownlinkData, "scene", "buffer", p=self.p)

    def classify(self, **kwargs):
        """Return raw mission counts, without assigning probabilities or cost."""
        return {
            "opportunities": int(self.fxns["schedule"].s.opportunities),
            "acquired": int(self.fxns["capture"].s.acquired),
            "processed": int(self.fxns["process"].s.processed),
            "selected": int(self.fxns["process"].s.selected),
            "downlinked": int(self.fxns["downlink"].s.downlinked),
            "queued": int(self.flows["buffer"].s.queued),
            "dropped": int(self.fxns["process"].s.dropped),
        }


if __name__ == "__main__":
    from fmdtools.sim import propagate

    result, history = propagate.nominal(ObservationSatellite())
    print(result)
