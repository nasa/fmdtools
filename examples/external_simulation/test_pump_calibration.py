"""
Check the fitted pump at settings outside its training table.

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

import numpy as np
import pytest
from fmdtools_examples.external_simulation.pump_calibration import fit_pump


def test_held_out_commands_and_stopped_pump():
    pytest.importorskip("sklearn")
    pump = fit_pump()
    flow = pump.predict([[-2.5], [0.0], [2.0], [3.5]])
    np.testing.assert_allclose(flow, [-2.0, 0.0, 1.6, 2.8])
