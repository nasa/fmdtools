"""
Use a fitted flow estimator for one reservoir update.

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

from dataclasses import replace

import numpy as np

from .adapter import _finite_scalar
from .external_sim import StepRequest, StepResult, reference_step


def pump_step(request: StepRequest, estimator) -> StepResult:
    """Predict delivered flow for one command and integrate it over the step.

    Pass this backend to call_step for request and result validation. The
    estimator takes one row with one feature and returns one flow in litres/s.
    """
    prediction = estimator.predict([[request.rate]])
    if np.shape(prediction) != (1,):
        raise ValueError("Expected one flow prediction with shape (1,)")
    delivered_rate = _finite_scalar(prediction[0], "predicted flow")
    return reference_step(replace(request, rate=delivered_rate))
