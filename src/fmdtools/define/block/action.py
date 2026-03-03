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
from fmdtools.define.container.time import Time
from fmdtools.define.flow.base import ExampleFlow
import numpy as np


class ActionTime(Time):
    """
    Time class with extra attributes for Actions.

    Attributes
    ----------
    out_delay: float
        Time after which the duration is complete before moving to the next action(s)
    duration : float
        Duration of the Action.
    t_loc : float
        local time (e.g., for actions with durations)
    """

    out_delay: np.float64 = np.float64(0.0)
    duration: np.float64 = np.float64(0.0)
    t_loc: np.float64 = np.float64(0.0)

    def return_mutables(self):
        """Return mutable attributes."""
        return *super().return_mutables(), self.duration, self.t_loc

    def reset(self):
        """Reset the time."""
        super().reset()
        self.t_loc = 0.0

    def duration_complete(self):
        """Return True if the local time is over the given duration."""
        return self.duration+self.dt <= self.t_loc

    def complete(self):
        """Return True if the duration and delay are over."""
        return self.duration+self.out_delay <= self.t_loc


class Action(Block):
    """
    Superclass for actions.

    Actions are blocks which have behaviors and live in an ActionArchitecture.

    Examples
    --------
    >>> exa = ExampleAction()
    >>> exa
    exampleaction ExampleAction
    - t=ActionTime(time=-0.1, timers={})
    - exf=ExampleFlow(s=(x=1.0, y=1.0))
    >>> exa(1.0)
    >>> exa
    exampleaction ExampleAction
    - t=ActionTime(time=1.0, timers={})
    - exf=ExampleFlow(s=(x=2.0, y=1.0))
    >>> exa.indicate_done()
    True
    """

    __slots__ = ()
    container_t = ActionTime

    def __init__(self, name=None, duration=0.0, out_delay=0.0, **kwargs):
        kwargs['t'] = {'duration': duration, 'out_delay': out_delay,
                       **kwargs.get('t', {})}
        super().__init__(name=name, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Action

    def update_dynamic_behaviors(self, proptype="dynamic"):
        """Update the behavior of the Action provided the action is not complete."""
        if not self.t.duration_complete():
            super().update_dynamic_behaviors(proptype=proptype)

    def inc_sim_time(self, time=None, **kwargs):
        """Increment the simulation time (update from external time)."""
        if time is not None:
            self.t.update_time(time)
        super().inc_sim_time(time=time, **kwargs)


class ExampleAction(Action):
    """Example action for use in testing/docs."""
    __slots__ = ('exf')
    container_p = ExampleParameter
    flow_exf = ExampleFlow

    def dynamic_behavior(self):
        """Increase x when executed."""
        if not self.indicate_done():
            self.exf.s.inc(x=1.0)

    def indicate_done(self):
        """When it reaches the threshold, it enters 'done' status."""
        return bool(self.exf.s.x > self.p.x)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
