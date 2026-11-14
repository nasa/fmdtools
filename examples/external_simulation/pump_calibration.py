"""
Fit a pump calibration curve from synthetic flow measurements.

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


def fit_pump():
    """Fit delivered flow against commanded flow, both in litres per second.

    Each row pairs a command with a synthetic measurement. A negative command
    reverses the flow. Fixing the intercept at zero keeps a stopped pump at zero.
    """
    from sklearn.linear_model import LinearRegression

    calibration = np.array(
        [
            [-4.0, -3.2],
            [-1.0, -0.8],
            [0.0, 0.0],
            [1.0, 0.8],
            [4.0, 3.2],
        ]
    )
    return LinearRegression(fit_intercept=False).fit(
        calibration[:, :1], calibration[:, 1]
    )
