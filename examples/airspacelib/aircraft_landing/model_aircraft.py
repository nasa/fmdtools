# -*- coding: utf-8 -*-
"""
Simple model of aircraft landing at an airport.

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

from fmdtools_examples.airspacelib.base.state import AircraftPosition3
from fmdtools_examples.airspacelib.base.aircraft import BaseAircraft


from model_environment import AirportEnvironment


class Aircraft(BaseAircraft):
    container_s = AircraftPosition3
    default_s = {'x': 1865527.46, 'y': 596499.47, 'goal_x': 1863436.83, 'goal_y': 601624.24}
    flow_environment = AirportEnvironment

    def dynamic_behavior(self):
        if not self.s.at_goal():
            self.s.update_position(maxvel=200) # speed - depends on aircraft, choosing 200 ft/s



# Runway 32R approach from southeast: 6,119,798.14, 1,959,233.13

# Runway 14L approach from northwest: 6,104,090.56 1,996,976.09

# 1000 ft up 3 miles out, 300-500 feet 1 mile out

# Moffett Location:  'goal_x': 6112017.97, 'goal_y': 1977296.34


if __name__ == "__main__":
    ba = Aircraft()
    ba(100)

    fig, ax = ba.environment.ga.show_infrastructure()
    fig, ax = ba.h.plot_trajectories('s.x', 's.y', fig=fig, ax=ax, time_groups=['nominal'])