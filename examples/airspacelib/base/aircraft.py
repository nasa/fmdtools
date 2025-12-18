# -*- coding: utf-8 -*-
"""
Base aircraft for use in Airspace Library.

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
from fmdtools.define.block.function import Function

from examples.airspacelib.base.state import AircraftState
from examples.airspacelib.base.parameter import AircraftParameter


class BaseAircraft(Function):
    """Base Aircraft function to be used by other models."""

    __slots__ = ()
    container_s = AircraftState
    container_p = AircraftParameter

    def indicate_at_goal(self):
        """Indicate whether the aircraft is at its goal location."""
        return self.s.at_goal()

    def indicate_in_range(self):
        """Indicate whether the aircraft is in range of its goal location."""
        return self.s.in_range()

    def fly_to_goal(self):
        """Fly to a pre-determined goal."""
        if not self.indicate_at_goal():
            self.s.update_position(maxvel=self.p.max_speed)
            dist = self.s.calc_dist_to_travel()
            self.s.inc(fuel_status=-dist/self.p.max_range)


if __name__ == "__main__":
    ba = BaseAircraft()
