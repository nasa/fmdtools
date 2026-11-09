"""
Standalone reservoir update used as a simple external calculation.

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

from dataclasses import dataclass


@dataclass(frozen=True)
class StepRequest:
    """Owned scalar inputs for one update, with no live model references.

    time is the update's end time in seconds, dt its positive duration in
    seconds, value the preceding accumulated quantity, and rate its change
    per second. The caller supplies any fault/noise effects in rate.
    """

    time: float
    dt: float
    value: float
    rate: float


@dataclass(frozen=True)
class StepResult:
    """The proposed accumulated value after one update."""

    value: float


def reference_step(request: StepRequest) -> StepResult:
    """Return value + rate * dt without changing inputs or retaining state."""
    return StepResult(request.value + request.rate * request.dt)
