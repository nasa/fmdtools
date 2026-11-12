# -*- coding: utf-8 -*-
"""
ControlFlight function used to control the aircraft.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.container.state import State

class ControlState(State):
    """
    State of ControlFlight Function planning.

    Parameters
    ----------
    flightplan: tuple
        Set of x/y points the drone is to fly through.
    height: float
        Height the drone is to fly at.
    pt: int
        Point within the flight plan that the drone is to fly to.
    """

    flightplan: tuple = ((0.0, 0.0), (25.0, 25.0))
    height: float = 25.0
    pt: int = 0

    def get_goal(self):
        """Get the current goal point (x,y) for the drone."""
        if self.pt < len(self.flightplan):
            return self.flightplan[self.pt]
        else:
            return self.flightplan[-1]

    def inc_goal(self):
        """Increment the current goal."""
        if self.pt < len(self.flightplan):
            self.pt += 1

    def is_start(self):
        """Determine whether the drone is as the start of its flightplan."""
        return self.pt == 0

    def is_end(self):
        """Determine whether the drone is at the end of its flightplan."""
        return self.pt >= len(self.flightplan)-1