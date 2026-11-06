#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Connect an independent calculation to a native fmdtools Function.

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Fault, Mode
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.rand import Rand
from fmdtools.define.container.state import State

from .adapter import _finite_scalar, call_step
from .external_sim import StepRequest, reference_step


class RateParam(Parameter, readonly=True):
    """Nominal change per second; signed rates are allowed."""

    rate: float = 2.0

    def check_lim(self, k, v):
        """Use native field validation and require a finite rate."""
        super().check_lim(k, v)
        _finite_scalar(v, k)


class RateState(State):
    """Independent uniform multiplier, sampled by Rand at each dynamic step."""

    multiplier: float = 1.0
    multiplier_update = ("uniform", (0.9, 1.1))


class RateRand(Rand):
    """FMDTools owns the seed and disturbance sequence, outside the backend."""

    s: RateState = RateState()


class ActuatorMode(Mode):
    """An injected stopped actuator supplies rate zero; probability is synthetic."""

    fault_stopped = Fault(prob=0.01, units="sim")


class CoupledState(State):
    """Committed value, most recent input rate, and successful update count."""

    value: float = 0.0
    last_rate: float = 0.0
    calls: int = 0


class CoupledFunction(Function):
    """Wrap an explicit-state calculation using native containers and hooks.

    Override external_step with a stateless module-level callable, using
    staticmethod, to substitute another implementation of this contract.
    Mutable simulation state remains in native containers for copy/reset and
    staged propagation. No external resource handle is stored on the model.
    """

    container_p = RateParam
    container_s = CoupledState
    container_m = ActuatorMode
    container_r = RateRand
    default_sp = {"end_time": 3.0, "dt": 1.0, "units": "sec"}
    external_step = staticmethod(reference_step)

    def init_block(self, **kwargs):
        """Require the example's zero-origin, second-based time convention."""
        if self.sp.start_time != 0 or self.sp.units != "sec":
            raise ValueError("This example requires start_time=0 and units='sec'")

    def dynamic_behavior(self):
        """Apply model effects, evaluate, then commit validated results only."""
        rate = 0.0 if self.m.has_fault("stopped") else self.p.rate * self.r.s.multiplier
        response = call_step(
            StepRequest(self.t.time, self.t.dt, self.s.value, rate),
            self.external_step,
        )
        self.s.put(value=response.value, last_rate=rate, calls=self.s.calls + 1)

    def classify(self, **kwargs):
        """Expose committed results through the ordinary propagation API."""
        return {"value": self.s.value, "calls": self.s.calls}


if __name__ == "__main__":
    from fmdtools.sim import propagate

    result, history = propagate.nominal(CoupledFunction())
    print(result)
    print(history["s.value"])
