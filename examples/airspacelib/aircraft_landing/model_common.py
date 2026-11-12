# -*- coding: utf-8 -*-
"""
Common aircraft modeling constructs.

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


from fmdtools.define.flow.base import Flow

from fmdtools_examples.airspacelib.base.state import Position3





class ControlsFlow(Flow):
    container_s = ControlsState



class AircraftDOFs(Flow):
    container_s = AeroState


class Position3(Position3):
    distance: float = 0.0


class AircraftTrajectory(Flow):
    container_s = Position3


if __name__ == "__main__":
    pass
