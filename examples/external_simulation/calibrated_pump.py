"""
Run the reservoir with a fitted pump calibration.

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

from .model_main import CoupledFunction
from .pump_adapter import pump_step
from .pump_calibration import fit_pump


class CalibratedPump(CoupledFunction):
    """Accumulate delivered volume from the commanded pump rate."""

    __slots__ = ("_estimator",)
    fit_estimator = staticmethod(fit_pump)

    def init_block(self, **kwargs):
        """Fit once per instance, including fresh simulation copies."""
        super().init_block(**kwargs)
        self._estimator = self.fit_estimator()

    def external_step(self, request):
        """Predict delivery using the current command after faults and noise."""
        return pump_step(request, self._estimator)
