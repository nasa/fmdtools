#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Function` class for representing system functional behaviors.

Has classes:

- :class:`Function`: Class for defining model Functions.
- :class:`GenericFxn`: Function class to use as a placeholder.

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
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.define.container.mode import ExampleMode
from fmdtools.define.flow.base import ExampleFlow


class Function(Block):
    """
    Superclass for representing system functions.

    Functions are distinguished from other blocks in their ability to contain
    architectures, which may be used to hold multiple components or an action sequence
    graph within the function.


    Additional role types
    ---------------------
    arch : Architecture
        component, action, function architectures at ca, aa, fa, etc.

    Examples
    --------
    >>> exf = ExampleFunction("exf")
    >>> exf
    exf ExampleFunction
    - t=Time(time=-0.1, timers={})
    - s=ExampleState(x=0.0, y=0.0)
    - m=ExampleMode(mode='standby', faults=set(), sub_faults=False)
    - exf=ExampleFlow(s=(x=1.0, y=1.0))

    Behavior can be called using __call__ or the user-defined behavior method:

    >>> exf(time=1.0, proptype="dynamic")
    >>> exf
    exf ExampleFunction
    - t=Time(time=1.0, timers={})
    - s=ExampleState(x=1.0, y=0.0)
    - m=ExampleMode(mode='standby', faults=set(), sub_faults=False)
    - exf=ExampleFlow(s=(x=2.0, y=1.0))

    Which can also be used to inject faults:

    >>> exf(time=2.0, proptype="dynamic", faults=['no_charge'])
    >>> exf
    exf ExampleFunction
    - t=Time(time=2.0, timers={})
    - s=ExampleState(x=1.0, y=3.0)
    - m=ExampleMode(mode='no_charge', faults={'no_charge'}, sub_faults=False)
    - exf=ExampleFlow(s=(x=3.0, y=1.0))
    """

    __slots__ = ["ca", "aa", "fa", "args_f"]
    default_track = ["ca", "aa", "fa"]+Block.default_track
    roletypes = ['container', 'flow', 'arch']

    def __init__(self, name=None, args_f=dict(), **kwargs):
        """
        Instantiate the function superclass with the relevant parameters.

        Parameters
        ----------
        args_f : dict, optional
            arguments to pass to custom __init__ function.
        """
        super().__init__(name=name, **kwargs)
        self.args_f = args_f
        if hasattr(self, 'behavior'):
            raise Exception("Invalid behavioral method: behavior(). Behavior must be"
                            " specified as dynamic_behavior() or static_behavior()")
        if hasattr(self, 'condfaults'):
            raise Exception("Use of condfaults() is deprecated.")

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Function


class ExampleFunction(Function):
    """Example Function block for testing."""

    __slots__ = ('exf',)
    container_p = ExampleParameter
    container_s = ExampleState
    container_m = ExampleMode
    flow_exf = ExampleFlow
    default_s = {'x': 0.0, 'y': 0.0}

    def dynamic_behavior(self):
        """Increment x if nominal, else increment y."""
        if not self.m.any_faults():
            self.s.x += self.p.x
        else:
            self.s.y += self.p.y
        self.exf.s.inc(x=self.s.x)

    def classify(self, **kwargs):
        """Classify via metric xy = s.x + s.y."""
        return {"xy": self.s.x + self.s.y}


class GenericFxn(Function):
    """
    Generic function block.

    For use when the user has not yet defined a class for the given (to be implemented)
    function block. Acts as a placeholder that enables simulation.
    """
    __slots__ = ['__dict__']
    check_dict_creation = False

    def __init__(self, name='', flows={}, args_f={}, **kwargs):
        super().__init__(name=name, flows={}, **kwargs)
        self.flows = tuple([*flows.keys()])
        for fl, flobj in flows.items():
            setattr(self, fl, flobj)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    exf = ExampleFunction("exf")
    # from fmdtools.sim import propagate
    # res, hist = propagate.one_fault(exf, "exf", "short", 2)