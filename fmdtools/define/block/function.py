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

from fmdtools.define.container.mode import Mode
from fmdtools.define.block.base import Block
from fmdtools.define.container.state import ExampleState
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.define.container.mode import ExampleMode
from fmdtools.define.flow.base import ExampleFlow

from decimal import Decimal

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
    - ExampleState(x=1.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())

    Behavior can be called using __call__ or the user-defined behavior method:

    >>> exf("dynamic", time=1.0)
    >>> exf
    exf ExampleFunction
    - ExampleState(x=2.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())

    Which can also be used to inject faults:

    >>> exf("dynamic", time=2.0, faults=['no_charge'])
    >>> exf
    exf ExampleFunction
    - ExampleState(x=2.0, y=4.0)
    - ExampleMode(mode=no_charge, faults={'no_charge'})
    """

    __slots__ = ["ca", "aa", "fa", "args_f", "archs"]
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

    def return_faultmodes(self):
        """
        Get the fault modes present in the simulation (for propagate/model).

        Returns
        -------
        ms : list
            List of faults present.
        modeprops : dict
            Dict of corresponding fault mode properties.
        """
        if hasattr(self, 'm'):
            ms = [m for m in self.m.faults.copy() if m != 'nom']
            modeprops = dict.fromkeys(ms)
            for mode in ms:
                modeprops[mode] = self.m.faultmodes.get(mode)
                if mode not in self.m.faultmodes:
                    raise Exception("Mode " + mode + " not in m.faultmodes for fxn " +
                                    self.__class__.__name__+" and may not be tracked.")
        else:
            ms = []
            modeprops = {}
        return ms, modeprops

    def update_seed(self, seed=[]):
        """
        Update seed and propagates update to contained actions/components.

        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        super().update_seed(seed)
        for at in self.get_roles('arch'):
            arch = getattr(self, at)
            if hasattr(arch, 'r'):
                arch.update_seed(self.r.seed)

    def prop_arch_behaviors(self, proptype, time, run_stochastic):
        """Propagate behaviors into contained architectures."""
        for objname in self.get_roles('arch'):
            try:
                obj = getattr(self, objname)
                # TODO: this should be more general
                if objname == 'aa':
                    obj(proptype, time, run_stochastic, self.t.dt)
                elif objname == 'fa':
                    obj.propagate(time, proptype=proptype, run_stochastic=run_stochastic)
            except TypeError as e:
                raise Exception("Poorly specified Architecture: "
                                + str(obj.__class__)) from e

    def prop_arch_faults_up(self):
        """Get faults from contained components and add to .m."""
        for objname in self.get_roles('arch'):
            obj = getattr(self, objname)
            self.m.faults.difference_update(obj.m.sub_modes)
            self.m.faults.update(obj.get_faults())

    def __call__(self, proptype, faults=[], time=0, run_stochastic=False):
        """
        Update the state of the function at a given time and injects faults.

        Parameters
        ----------
        proptype : str
            Type of propagation step to update
            ('static_behavior', or 'dynamic_behavior')
        faults : list, optional
            Faults to inject in the function. The default is [].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if hasattr(self, 'r'):
            self.r.run_stochastic = run_stochastic
        # if there is a fault, it is instantiated
        self.inject_faults(faults)

        if time > self.t.time:
            if hasattr(self, 'r'):
                self.r.update_stochastic_states()
        self.prop_arch_behaviors(proptype, time, run_stochastic)

        if proptype == 'static' and hasattr(self, 'static_behavior'):
            self.static_behavior(time)
        elif proptype == 'dynamic' and hasattr(self, 'dynamic_behavior') and time > self.t.time:
            if self.t.run_times >= 1:
                for i in range(self.t.run_times):
                    self.dynamic_behavior(time)
            elif not Decimal(str(time)) % Decimal(str(self.t.dt)):
                self.dynamic_behavior(time)

        self.prop_arch_faults_up()

        self.t.time = time
        if run_stochastic == 'track_pdf':
            if hasattr(self, 'r'):
                self.r.probdens = self.r.return_probdens()
        if hasattr(self, 'm') and self.m.exclusive is True and len(self.m.faults) > 1:
            raise Exception("More than one fault present in " + self.name +
                            "\n at t= " + str(time) +
                            "\n faults: " + str(self.m.faults) +
                            "\n Is the mode representation nonexclusive?")
        return

    def return_probdens(self):
        """Get the probability density associated with FxnBlock and its archs."""
        pd = super().return_probdens()
        for arch in self.archs:
            pd *= getattr(self, arch).return_probdens()
        return pd


class ExampleFunction(Function):
    """Example Function block for testing."""

    __slots__ = ('exf',)
    container_p = ExampleParameter
    container_s = ExampleState
    container_m = ExampleMode
    flow_exf = ExampleFlow

    def dynamic_behavior(self, time):
        """Increment x if nominal, else increment y."""
        if not self.m.any_faults():
            self.s.x += self.p.x
        else:
            self.s.y += self.p.y
        if time < 1.0:
            self.s.put(x=0.0, y=0.0)
        self.exf.s.inc(x=self.s.x)

    def find_classification(self, scen, hist):
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