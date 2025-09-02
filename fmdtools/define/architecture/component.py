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
from fmdtools.define.block.component import ExampleComponent


class ComponentArchitecture(Architecture):
    """
    Class defining Component Architectures.

    ComponentArchitectures represent the physical realization of a system, which may
    combine multiple components to fulfil given functions.

    Component Architectures are meant to be able to be called externally by functions
    to determine how individual component behaviors aggregate as a given function.

    However, they can also simulate via their own static/dynamic behaviors.

    Examples
    --------
    >>> exc = ExampleComponentArchitecture()
    >>> exc
    examplecomponentarchitecture ExampleComponentArchitecture
    - t=Time(time=-0.1, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    COMPS:
    - c1=ExampleComponent(s=(x=5.0, y=5.0))
    - c2=ExampleComponent(s=(x=10.0, y=10.0))
    >>> exc()
    >>> exc
    examplecomponentarchitecture ExampleComponentArchitecture
    - t=Time(time=1.0, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    COMPS:
    - c1=ExampleComponent(s=(x=7.0, y=6.0))
    - c2=ExampleComponent(s=(x=12.0, y=11.0))
    >>> exc()
    >>> exc
    examplecomponentarchitecture ExampleComponentArchitecture
    - t=Time(time=2.0, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    COMPS:
    - c1=ExampleComponent(s=(x=7.0, y=7.0))
    - c2=ExampleComponent(s=(x=12.0, y=12.0))
    """

    __slots__ = ['comps']
    flexible_roles = ['flow', 'comp']
    roletypes = ['container']
    default_track = ('comps', 'flows', 'i')
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

    def build(self, construct_graph=True, require_connections=False, **kwargs):
        """Build the function architecture - connections should be enforced."""
        super().build(construct_graph=construct_graph,
                      require_connections=require_connections, **kwargs)


class ExampleComponentArchitecture(ComponentArchitecture):
    """
    Example Component Architecture Class demonstrating individual simulation.

    In this architecture, custom static_behavior and dynamic_behavior methods are used
    to enable the individual simulation of the ComponentArchitecture.
    """

    __slots__ = ()

    def init_architecture(self, *args, **kwargs):
        self.add_comp("c1", ExampleComponent, s={'x': 5.0, 'y': 5.0})
        self.add_comp("c2", ExampleComponent, s={'x': 10.0, 'y': 10.0})


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    exca = ExampleComponentArchitecture()
