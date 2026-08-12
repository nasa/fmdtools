"""
Top-level model for the water-rescue simulation.

Wires together the shared beach environment with the three functions:
the lifeguard responder, the swimmer behavior, and the patrol drone.

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
from fmdtools_examples.airspacelib.water_rescue.model_environment import BeachEnvironment, BeachBehavior
from fmdtools_examples.airspacelib.water_rescue.model_responder import Responder
from fmdtools_examples.airspacelib.water_rescue.model_drone import BeachAircraft
from fmdtools.define.architecture.function import FunctionArchitecture


class Beach(FunctionArchitecture):
    """Function architecture connecting all functions through the environment flow."""

    def init_architecture(self, **kwargs):
        self.add_flow("environment", BeachEnvironment)
        self.add_fxn("responder", Responder, "environment")
        self.add_fxn("swimmers", BeachBehavior, "environment")
        self.add_fxn("beach_aircraft", BeachAircraft, "environment")
