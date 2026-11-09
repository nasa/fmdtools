"""
Check request and response data before updating the fmdtools model.

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
from collections.abc import Callable
from numbers import Real

from .external_sim import StepRequest, StepResult, reference_step


def _finite_scalar(value, name):
    """Reject ambiguous/nonfinite values before calling or committing outputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real scalar")  # noqa: TRY004
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " must be a finite real scalar") from error
    if not math.isfinite(number):
        raise ValueError(name + " must be a finite real scalar")
    return number


def call_step(
    request: StepRequest,
    backend: Callable[[StepRequest], StepResult] = reference_step,
) -> StepResult:
    """Validate a snapshot, call once, and validate the complete response.

    Backend exceptions propagate unchanged; there are no retries, implicit
    fault injection or substitute outputs. A successful return contains only
    validated scalar data. This is an example-specific contract, not a sandbox
    or a generic runner for arbitrary input/output schemas.
    """
    if not isinstance(request, StepRequest):
        raise TypeError("Expected StepRequest")
    request = StepRequest(
        **{
            name: _finite_scalar(getattr(request, name), "request." + name)
            for name in ("time", "dt", "value", "rate")
        }
    )
    if request.dt <= 0 or request.time < request.dt:
        raise ValueError(
            "Require time >= dt > 0 for an update starting at or after zero"
        )
    response = backend(request)
    if not isinstance(response, StepResult):
        raise TypeError("External calculation must return StepResult")
    return StepResult(_finite_scalar(response.value, "response.value"))
