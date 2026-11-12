# -*- coding: utf-8 -*-
"""
Common aircraft controls constructs.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow

class ControlsState(State):
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.6
    flaps: float = 0.0
    brakes: float = 0.0

    def set_steady_flight(self, angle=0.0, throttle=0.6):
        """Set controls for steady, level flight (trimmed)"""
        self.aileron = 0.0
        self.elevator = 0.029 + 1.5*angle
        self.rudder = 0.0
        self.throttle = throttle  # Typical cruise throttle
        self.flaps = 0.0

    def set_turn(self, direction: str, amount: float = 1.0):
        """Set controls for turning left or right at a specified intensity (amount)."""
        # Positive aileron/rudder is assumed right, negative is left
        if direction.lower() == 'left':
            self.aileron = -1.0 * amount
            self.rudder = -0.5 * amount
        elif direction.lower() == 'right':
            self.aileron = 1.0 * amount
            self.rudder = 0.5 * amount
        else:
            raise ValueError("direction must be 'left' or 'right'")
        self.flaps = 0.0
        # Keep elevator/throttle unchanged (or optionally decrease elevator for coordinated turn)

    def set_descent(self, angle = 0.0, throttle=0.4, flaps=0.0):
        self.elevator = 0.01 + angle
        self.throttle = throttle
        self.put(aileron=0.0, rudder=0.0)
        self.flaps = self.flaps

    def set_ascent(self, angle=0.0, throttle=0.6):
        """
        Set controls to achieve an ascent (climb) at the specified rate (m/s).

        Elevator and throttle are increased for climb.
        """
        # self.elevator = 0.029-0.1
        self.elevator = -0.1 + angle
        self.throttle = throttle
        self.put(aileron=0.0, rudder=0.0)
        self.flaps = 0.0


class ControlsFlow(Flow):
    container_s = ControlsState
