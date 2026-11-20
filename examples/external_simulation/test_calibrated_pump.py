"""
Check how the fitted pump advances the reservoir.

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

from unittest.mock import Mock

import numpy as np
import pytest
from fmdtools_examples.external_simulation.calibrated_pump import CalibratedPump

from fmdtools.sim import propagate


def test_reverse_flow_with_half_second_steps():
    pytest.importorskip("sklearn")
    model = CalibratedPump(p={"rate": -2.0}, sp={"dt": 0.5}, t={"dt": 0.5})
    result, history = propagate.nominal(model)
    np.testing.assert_allclose(history["s.value"], -1.6 * history["time"])
    assert result["tend.classify.value"] == pytest.approx(-4.8)


def test_one_fit_serves_multiple_steps(monkeypatch):
    estimator = Mock()
    estimator.predict.return_value = [1.25]
    factory = Mock(return_value=estimator)
    monkeypatch.setattr(CalibratedPump, "fit_estimator", staticmethod(factory))
    model = CalibratedPump()
    model(time=1.0)
    model(time=2.0)
    factory.assert_called_once_with()
    assert estimator.predict.call_count == 2
    assert model.s.value == 2.5
