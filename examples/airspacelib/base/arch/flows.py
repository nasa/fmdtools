# -*- coding: utf-8 -*-
"""
Base flows used in the models.

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
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.environment import Environment
from fmdtools.define.container.parameter import Parameter


from examples.airspacelib.base.state import AircraftPosition3


class ForceState(State):
    """State of Force Flow."""

    weight: float = 1.0
    contact_support: float = 0.0
    lift_support: float = 1.0


class Force(Flow):
    """Flow of force through the aircraft functions."""

    __slots__ = ()
    container_s = ForceState


class ElectricityState(State):
    """State of electricity - assumes high and low volate lines."""

    charge: float = 100.0
    voltage_high: float = 1.0
    current_high: float = 1.0
    power_high: bool = False
    voltage_low: float = 1.0
    current_low: float = 1.0
    power_low: bool = True



class Electricity(Flow):
    """Flow of electricity through the aircraft functions."""

    __slots__ = ()
    container_s = ElectricityState


class AircraftEnvironment(Environment):
    "Placeholder for environment flow TBD."


class AicraftControlParameter(Parameter):
    """Parameter determining aircraft control."""

    max_vel: float = 10.0


class Trajectories(MultiFlow):
    """Degrees of freedom of the aircraft."""

    __slots__ = ()
    container_s = AircraftPosition3
    container_p = AicraftControlParameter


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

