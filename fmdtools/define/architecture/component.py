#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`ComponentArchitecture` class to represent component architectures.

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

from fmdtools.define.architecture.base import Architecture
from fmdtools.define.container.mode import Mode


class ComponentArchitecture(Architecture):
    """Class defining Component Architectures."""

    __slots__ = ['comps']
    flexible_roles = ['flows', 'comps']
    roletypes = ['container', 'flow', 'comp']
    rolename = 'ca'
    container_m = Mode

    def __init__(self, **kwargs):
        Architecture.__init__(self, **kwargs)

    def add_comp(self, name, compclass, *flownames, **kwargs):
        """
        Associate a Component with the architecture. Called after add_flow.

        Parameters
        ----------
        name : str
            Internal Name for the Component
        compclass : Component
            Component class to instantiate
        *flownames : flow
            Flows (optional) which connect the components
        **kwargs : any
            kwargs to instantiate the Component with
        """
        self.add_sim('comps', name, compclass, *flownames, **kwargs)

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'comps', faults)
