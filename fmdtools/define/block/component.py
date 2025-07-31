# -*- coding: utf-8 -*-
"""
Defines :class:`Component` class for representing system components.

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

from fmdtools.define.block.base import Block
from fmdtools.define.container.state import ExampleState


class Component(Block):
    """
    Superclass for components (most attributes inherited from Block superclass).

    Components may be called from external Component Architectures or on their own.

    Typically, components are meant to represent physical realizations of systems
    that maybe multifunctional.

    Examples
    --------
    >>> c = ExampleComponent()
    >>> c
    examplecomponent ExampleComponent
    - s=ExampleState(x=0.0, y=0.0)
    - t=Time(time=-0.1, timers={})
    >>> c()
    >>> c()
    >>> c
    examplecomponent ExampleComponent
    - s=ExampleState(x=2.0, y=1.0)
    - t=Time(time=2.0, timers={})
    >>> c()
    >>> c
    examplecomponent ExampleComponent
    - s=ExampleState(x=2.0, y=2.0)
    - t=Time(time=3.0, timers={})
    >>> c(proptype="dynamic")
    >>> c
    examplecomponent ExampleComponent
    - s=ExampleState(x=2.0, y=3.0)
    - t=Time(time=4.0, timers={})
    """

    __slots__ = ()

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Component


class ExampleComponent(Component):
    """
    Example component used for testing.

    The desired behavior is for x and y to start at 0, for y to increase at the second
    timestep, causing x to leapfrog it (since it increments by 2).
    """

    __slots__ = ()
    container_s = ExampleState
    default_s = {'x': 0.0, 'y': 0.0}

    def static_behavior(self, time):
        """Increment x if y is higher."""
        if self.s.x < self.s.y:
            self.s.x += 2.0

    def dynamic_behavior(self, time):
        """Increment y."""
        if self.t.time >= 0.0:
            self.s.y += 1.0


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
