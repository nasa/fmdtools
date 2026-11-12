# -*- coding: utf-8 -*-
"""
Basic force characteristics.

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


class ForceState(State):
    """State of Force Flow."""

    weight: float = 1.0
    contact_support: float = 0.0
    lift_support: float = 1.0


class Force(Flow):
    """Flow of force through the aircraft functions."""

    __slots__ = ()
    container_s = ForceState
