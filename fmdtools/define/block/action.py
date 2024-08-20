#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Action` class for representing discrete actions.

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
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.define.flow.base import ExampleFlow


class Action(Block):
    """
    Superclass for actions.

    Actions are blocks which have behaviors and live in an ActionArchitecture.

    Examples
    --------
    >>> exa = ExampleAction()
    >>> exa.exf
    exampleflow ExampleFlow flow: ExampleState(x=1.0, y=1.0)
    >>> exa(1.0)
    >>> exa.exf
    exampleflow ExampleFlow flow: ExampleState(x=2.0, y=1.0)
    >>> exa.indicate_done()
    True
    """

    __slots__ = ('duration',)

    def __init__(self, name=None, duration=0.0, **kwargs):
        self.duration = duration
        super().__init__(name=name, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Action

    def __call__(self, time=0, run_stochastic=False, proptype='dynamic', dt=1.0):
        """
        Update the behaviors, faults, times, etc of the action.

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if time > self.t.time:
            if hasattr(self, 'r'):
                self.r.update_stochastic_states()
        if not proptype == 'dynamic' or self.t.time < time:
            self.update_behavior(time, dt)
        self.t.time = time

    def update_behavior(self, time, dt):
        """Update the behavior of the Action."""
        if hasattr(self, 'behavior'):
            self.behavior(time)
        self.t.t_loc += dt

    def copy(self, *args, **kwargs):
        cop = super().copy(*args, **kwargs)
        cop.duration = self.duration
        return cop


class ExampleAction(Action):
    """Example action for use in testing/docs."""
    __slots__ = ('exf')
    container_p = ExampleParameter
    flow_exf = ExampleFlow

    def behavior(self, time):
        """The Action increases x when executed."""
        if not self.indicate_done():
            self.exf.s.inc(x=1.0)

    def indicate_done(self):
        """When it reaches the threshold, it enters 'done' status."""
        return self.exf.s.x > self.p.x


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
