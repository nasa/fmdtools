"""
Check how a single regression prediction updates reservoir volume.

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

from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
from fmdtools_examples.external_simulation.adapter import call_step
from fmdtools_examples.external_simulation.external_sim import StepRequest, StepResult
from fmdtools_examples.external_simulation.pump_adapter import pump_step


def test_one_prediction_for_one_reservoir_step():
    estimator = Mock()
    estimator.predict.return_value = np.array([-1.5])
    request = StepRequest(time=3.0, dt=0.5, value=5.0, rate=-2.0)
    result = call_step(request, partial(pump_step, estimator=estimator))
    estimator.predict.assert_called_once_with([[-2.0]])
    assert result == StepResult(4.25)
    assert request.rate == -2.0


@pytest.mark.parametrize(
    "prediction", [1.0, [], [1, 2], [[1.0]], [True], ["1.0"], [np.nan], [np.inf]]
)
def test_invalid_predictions(prediction):
    estimator = Mock()
    estimator.predict.return_value = prediction
    request = StepRequest(1.0, 1.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="prediction|predicted flow"):
        call_step(request, partial(pump_step, estimator=estimator))


def test_estimator_failure_reaches_caller():
    estimator = Mock()
    estimator.predict.side_effect = RuntimeError("pump calibration unavailable")
    request = StepRequest(1.0, 1.0, 0.0, 2.0)
    with pytest.raises(RuntimeError, match="pump calibration unavailable"):
        call_step(request, partial(pump_step, estimator=estimator))
    estimator.predict.assert_called_once()
