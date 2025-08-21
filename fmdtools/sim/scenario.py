#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes for defining scenarios to simulate.

Classes:

- :class:`Injection`: Defines faults and disturbances to inject at a specific time.
- :class:`Scenario`: Defines a generic scenario to simulation.
- :class:`SingleFaultScenario`: Defines the scenario of a single fault injected in a
  function.
- :class:`JointFaultScenario`: Defines the scenario of multiple faults injected in a
  function at the same time.
- :class:`ParameterScenario`: Defines the scenario of a model having parameter values.
- :class:`Sequence`: Creates an overall sequence of Injections from a given sequence of
  faults and disturbances.

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

from fmdtools.define.base import t_key

from recordclass import dataobject, asdict
from collections import UserDict
from typing import ClassVar
import numpy as np


def create_scenname(faulttup, time):
    """Create a scenario name for a given fault scenario."""
    sn = '__'.join(['_'.join(fm) for fm in faulttup])+'_'+t_key(time)
    return'_'.join(sn.split("."))


class BaseScenObj(dataobject, readonly=True, mapping=True):
    """Base class for Scenarios and injections."""

    def get(self, entry, fallback):
        """
        Get entry if present, otherwise return fallback.

        Parameters
        ----------
        entry : str
            Name of field to get.
        fallback : val
            Fallback value

        Returns
        -------
        val_to_get: val
            Value of entry (or fallback, if no entry)
        """
        if not hasattr(self, entry):
            return fallback
        else:
            return self[entry]

    def copy_with(self, **kwargs):
        """
        Copy the Scen with the given kwargs.

        Parameters
        ----------
        **kwargs : kwargs
            Fields to change in the copy

        Returns
        -------
        copy : BaseScenObj
            Copy of self with optional kwargs.

        """
        existing_kwargs = asdict(self)
        return self.__class__(**{**existing_kwargs, **kwargs})

    def asdict(self):
        """Return as dictionary."""
        return asdict(self)


class Injection(BaseScenObj):
    """
    Defines a single fault/disturbance injection.

    Fields
    ------
    faults : dict
        Faults to inject with structure {'fxnname': [faults]}
        or {'fxnname': {'faultname': Fault()}}
    disturbances : dict
        Disturbances to inject with structure {'path.to.state': statevalue}
    """

    faults: dict = {}
    disturbances: dict = {}

    def update(self, inj):
        """
        Update the injection with the given faults/disturbances.

        Parameters
        ----------
        inj : dict
            Dict with structure {'faults': faults, 'disturbances': disturbances}
        """
        if hasattr(inj, 'faults'):
            self.faults.update(inj['faults'])
        if hasattr(inj, 'disturbances'):
            self.disturbances.update(inj['disturbances'])


class Sequence(UserDict):
    """
    Defines a dict with a sequence of faults and disturbances.

    Examples
    --------
    >>> s = Sequence(faultseq={1: {'fxnname':['mode1']}},
    ...              disturbances={2: {'state.name': 1.0}})
    >>> s[1]
    Injection(faults={'fxnname': ['mode1']}, disturbances={})
    >>> s[2]
    Injection(faults={}, disturbances={'state.name': 1.0})

    >>> Sequence({1: {'fxnname': ['mode1']}}, {1: {'state.name': 1.0}})
    {1: Injection(faults={'fxnname': ['mode1']}, disturbances={'state.name': 1.0})}
    """

    def __init__(self, faultseq={}, disturbances={}):
        times = {*faultseq, *disturbances}
        self.data = {t: Injection(faults=faultseq.get(t, {}),
                                  disturbances=disturbances.get(t, {}))
                     for t in times}

    def update_sequence(self, new_sequence):
        """
        Update the sequence given a different sequence.

        Parameters
        ----------
        new_sequence : Sequence
            Sequence to update from

        Examples
        --------
        >>> s = Sequence(faultseq={1: {'fxnname':['mode1']}},
        ...              disturbances={2: {'state.name': 1.0}})
        >>> s2 = Sequence({1: {'fxnname': ['mode2']}}, {1: {'state.name': 1.0}})
        >>> s.update_sequence(s2)
        >>> s
        {1: Injection(faults={'fxnname': ['mode2']}, disturbances={'state.name': 1.0}), 2: Injection(faults={}, disturbances={'state.name': 1.0})}
        """
        for i in new_sequence:
            if i not in self:
                self[i] = new_sequence[i]
            else:
                self[i].update(new_sequence[i])

    @classmethod
    def from_fault(cls, faulttup, time, **kwargs):
        """Generate sequence from single fault mode faulttup (fxn, mode)."""
        if kwargs:
            return cls(faultseq={float(time): {faulttup[0]: {faulttup[1]: kwargs}}})
        else:
            return cls(faultseq={float(time): {faulttup[0]: [faulttup[1]]}})


class BaseScenario(BaseScenObj):
    """
    Base class for scenarios.

    Parameters
    ----------
    sequence: Sequence
        Sequence of faults and distrubances over time defined by Sequence
    times : tuple
        Times faunts/distrubances will occur.

    Class Variables
    ---------------
    prob : float
        Scenario probability
    rate : float
        Scenario rate.
    name : str
        Scenario name.
    time : float
        Start of scenario.
    """

    sequence: Sequence = Sequence()
    times: tuple = ()
    prob: ClassVar[float] = 1.0
    rate: ClassVar[float] = 1.0
    name: ClassVar[str] = "nominal"
    time: ClassVar[float] = 0.0

    def start_time(self):
        """Get the first time in the scenario."""
        return np.min([*self.sequence])

    def get_times(self):
        """Get time that the scenario branches off."""
        return [self.start_time()]

    def scenarios(self):
        """Return list with scenario."""
        return [self]


class Scenario(BaseScenario):
    """Class defining generic scenarios. Extends BaseScenario."""

    rate: float = 1.0
    name: str = "nominal"
    time: float = 0.0


class SingleFaultScenario(BaseScenario):
    """
    Class defining a single fault occuring. Extends BaseScenario.

    Parameters
    ----------
    function : str
        Function where the fault occurs
    fault : str
        Name of the fault
    """

    function: str = ''
    fault: str = ''
    rate: float = 1.0
    name: str = 'faulty'
    time: float = 0.0
    phase: str = ''

    @classmethod
    def from_fault(cls, faulttup, time, mdl=None, weight=1.0, phasemap=None,
                   starttime=None, **kwargs):
        """
        Generate the fault scenario for faulttup at time.

        Examples
        --------
        >>> SingleFaultScenario.from_fault(('fxn', 'fault'), 10)
        SingleFaultScenario(sequence={10.0: Injection(faults={'fxn': ['fault']}, disturbances={})}, times=(10,), function='fxn', fault='fault', rate=1.0, name='fxn_fault_t10', time=10, phase='')
        >>> SingleFaultScenario.from_fault(('fxn', 'fault'), 10, prob=10)
        SingleFaultScenario(sequence={10.0: Injection(faults={'fxn': {'fault': {'prob': 10}}}, disturbances={})}, times=(10,), function='fxn', fault='fault', rate=1.0, name='fxn_fault_t10', time=10, phase='')
        """
        if len(faulttup) == 1:
            faulttup = faulttup[0]
        if phasemap:
            phase = phasemap.find_base_phase(time)
        else:
            phase = ''
        if mdl:
            rate = mdl.get_scen_rate(faulttup[0], faulttup[1], time,
                                     phasemap=phasemap, weight=weight, **kwargs)
        else:
            rate = weight
        if not starttime:
            starttime = time
        scen = cls(sequence=Sequence.from_fault(faulttup, time, **kwargs),
                   function=faulttup[0],
                   fault=faulttup[1],
                   rate=rate,
                   name=create_scenname((faulttup,), time),
                   time=starttime,
                   times=(time,),
                   phase=phase)
        return scen


class JointFaultScenario(BaseScenario):
    """
    Class defining a joint fault occuring. Extends BaseScenario.

    Parameters
    ----------
    joint_faults : int
        Joint Faults in the scenario
    functions : tuple
        Functions where the faults are to occur
    modes : tuple
        Names of the fault modes
    """

    joint_faults: int = 1
    functions: tuple = ()
    modes: tuple = ()
    rate: float = 1.0
    name: str = 'faulty'
    time: float = 0.0
    phase: tuple = ()

    @classmethod
    def from_faults(cls, faulttups, time, mdl=None, phasemap=None, weight=1.0,
                    baserate='ind', p_cond=1.0, starttime=None, **kwargs):
        """Generate JointFaultScenario given fault names, time."""
        if phasemap:
            phase = phasemap.find_base_phase(time)
        else:
            phase = ''
        # calculate rate
        rates = {}
        for i, faulttup in enumerate(faulttups):
            if mdl:
                rate = mdl.get_scen_rate(*faulttup, time,
                                         phasemap=phasemap, weight=weight, **kwargs)
            else:
                rate = weight
            rates[faulttup] = rate
        if baserate == 'ind':
            rate = np.prod([*rates.values()])
        elif baserate == 'max':
            rate = np.max([*rates.values()])
        else:
            rate = rates[baserate]
        rate *= p_cond
        # create sequence
        faults = {}
        for faulttup in faulttups:
            if faulttup[0] not in faults:
                faults[faulttup[0]] = [faulttup[1]]
            else:
                faults[faulttup[0]].append(faulttup[1])
        sequence = {time: Injection(faults=faults)}
        if not starttime:
            starttime = time
        # add fault scenario
        scen = cls(sequence=sequence,
                   joint_faults=len(faulttups),
                   functions=tuple(set([f[0] for f in faulttups])),
                   modes=tuple(set([f[1] for f in faulttups])),
                   rate=rate,
                   name=create_scenname(faulttups, time),
                   time=starttime,
                   times=(time,),
                   phase=phase)
        return scen


class ParameterScenario(BaseScenario, readonly=True):
    """
    Class defining a nominal (non-fault-injection) Scenario.

    Parameters
    ----------
    p : dict
        non-default Parameter fields
    r : dict
        non-default Rand fields
    sp : dict
        non-default SimParam fields
    prob : float
        probability
    inputparams : dict
        inputs to parameter generator
    rangeid : str
        name of containing range
    name : str
        scenario name
    """

    p: dict = {}
    r: dict = {}
    sp: dict = {}
    prob: float = 1.0
    inputparams: dict = {}
    rangeid: str = ''
    name: str = 'nominal'

    def get_param(self, param, default="NA"):
        """
        Get a parameter from the scenario.

        Parameters
        ----------
        param : str
            Name of parameter (e.g., "p.size")
        default : val, optional
            Default value for field (if not present). The default is "NA".

        Returns
        -------
        pval : val
            Value of the parameter.
        """
        if "." in param:
            p_index = param.split(".")
            p_field = p_index[0]
            p_entry = ".".join(p_field[1:])
            pval = self.get(p_field, default).get(p_entry, default)
        elif param == 'prob':
            pval = self.prob
        else:
            pval = default
        return pval

    def get_params(self, *params, default="NA"):
        """
        Get a list of parameters for the scenarios.

        Parameters
        ----------
        *params : str
            Name of parameters
        default : val, optional
            Default value for parameters (if not present). The default is "NA".

        Returns
        -------
        params : list
            List of parameters.
        """
        return [self.get_param(param, default=default) for param in params]

    def start_time(self):
        """Return the start time for a parameter scenario--0.0."""
        return 0.0


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    a = Scenario()

    b = SingleFaultScenario()

    seq = Sequence(faultseq={1:"fault"})
