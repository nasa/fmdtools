#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`Mode` class for defining (faulty and otherwise) :term:`mode` s.

Has classes:

- :class:`Fault`: Class for defining fault parameters
- :class:`Mode`: Class for defining the mode property (and associated probability model)
  held in Blocks.

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

from fmdtools.define.base import eq_units
from fmdtools.define.container.base import BaseContainer
from fmdtools.analyze.history import History

from typing import ClassVar
import numpy as np
import itertools
import copy


class Fault(BaseContainer, readonly=True):
    """
    Stores Default Attributes for modes to use in Mode.faultmodes.

    Fields
    ------
    prob : float
        Mode probability or rate per operation
    cost : float
        Individual mode cost (e.g., of repair)
    phases : dict
        Opportunity vector mapping phases of operation to relative probability.
        i.e., phase_probability = Fault.prob * Fault.phases[phase]
    units : str
        Units on probability. Can be a unit of continuous ('sec', 'min', 'hr', 'day') or
        discrete ('sim') time. Default is 'sim'.
    """

    prob: float = 1.0
    cost: float = 0.0
    phases: dict = {}
    units: str = 'sim'

    def calc_rate(self, time, phasemap={}, sim_time=1.0, sim_units='hr', weight=1.0):
        """
        Calculate the rate of a given fault mode.

        Parameters
        ----------
        time : float
            Time the fault will be injected.
        phasemap : PhaseMap, optional
            Map of phases/modephases that define operations the mode will be injected
            during (and maps to the opportunity vector phases). The default is {}.
        sim_time : float, optional
            Duration of the simulation. Used to determine fault exposure time when
            time-units ('sec', 'min', etc) define the fault rate. The default is 1.0.
        sim_units : float, optional
            Simulation time units. Used to determine fault exposure time when time-units
            define the fault rate. The default is 'hr'.
        weight : float, optional
            Weight for the fault/scenario (e.g., for multiple scens.). The default is 1.

        Returns
        -------
        rate : float
            Calculated rate of the scenario with the given information.

        Examples
        --------
        >>> # Calculating the rate of a mode in the 'on phase':
        >>> from fmdtools.analyze.phases import PhaseMap
        >>> pm = PhaseMap({'on': [0, 5], 'off': [6, 10]})
        >>> exfault = Fault(prob=0.5, phases = {'on': 0.9, 'off': 0.1}, units='hr')
        >>> rate = exfault.calc_rate(4, pm, sim_time=10.0, sim_units='min')
        >>> # note that this rate is the same as what we would calculate:
        >>> manual_calc = 0.5 * 6 * 0.9 / 60
        >>> manual_calc == rate
        True
        >>> rate_off = exfault.calc_rate(7, pm, sim_time=10.0, sim_units='min')
        >>> # note that the interval for off (6-10) is 5 while (0-5) is 6
        >>> manual_calc_off = 0.5 * 5 * 0.1 / 60
        >>> manual_calc_off == rate_off
        True
        """
        if self.units == 'sim':
            sim_exposure_time = 1.0
            baserate = self['prob']
        else:
            sim_exposure_time = eq_units(self['units'], sim_units) * sim_time
            baserate = self['prob'] * sim_exposure_time

        if not phasemap:
            return baserate * weight
        else:
            if self.units == 'sim':
                t_factor = 1.0
            else:
                t_exposure = phasemap.calc_scen_exposure_time(time)
                t_exposure *= eq_units(self['units'], sim_units)
                t_factor = t_exposure/sim_exposure_time
            if self.phases:
                phase = phasemap.find_base_phase(time)
                opp_factor = self.phases.get(phase, 0.0)
            else:
                opp_factor = 1.0
            return baserate * opp_factor * t_factor * weight


class Mode(BaseContainer, readonly=False):
    """
    Class for defining the mode property (and probability model) held in Blocks.

    Mode is meant to be inherited in order to define the specific faults related to a
    given Block.

    Class Variables
    ---------------
    opermodes : tuple
        Names of non-faulty operational modes.
    failrate : float
        Overall failure rate for the block. The default is 1.0. Note that if a failrate
        is provided, the prob argument in faultparams is a conditional probability
        (e.g. Fault.prob = Mode.failrate * Mode.faultparams['mode']['prob']).
    probtype : str, optional
        Type of probability in the probability model, a per-time 'rate' or
        per-run 'prob'. The default is 'rate'.
    units : str, optional
        Type of units ('sec'/'min'/'hr'/'day') used for the rates.
        Default is 'sim', which is unitless (prob/simulation).
    phases : dict
        Phases to inject faults in.
    exclusive : True/False
        Whether fault modes are exclusive of each other or not.
        Default is False (i.e. more than one can be present).
    longnames : dict
        Longer names for the faults (if desired). {faultname: longname}
    faults : set
        Set of faults present (or not) at any given time.
    mode : str
        Name of the current mode. the default is 'nominal'.
    fm_args : dict
        Arguments to Mode.init_faultmodes().
    he_args : tuple
        Arguments for add_he_rate defining a human error probability model.
    sfs_args : tuple
        Arguments for self.init_single_faultstates (franges, {kwargs}).
    nfs_args : tuple
        Arguments for self.init_n_faultstates (*args, {kwargs}).
    fsm_args : tuple
        Arguments for self.init_faultstates_modes (manual_modes, {kwargs}).

    These fields are then used in simulation and elsewhere.

    Fields
    ------
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    mode_state_dict: dict
        Maps modes to states. Assigned by init_faultstates methods.
    sub_modes: dict
        Maps modes to internal architecture elements.
    faultmodes : dict
            Dictionary of :class:`Fault` defining possible fault modes and
            their properties

    Examples
    --------
    >>> class ExampleMode(Mode):
    ...    fm_args = {"no_charge": (1e-5, 100, {'standby': 1.0}),
    ...              "short": (1e-5, 100, {'supply': 1.0})}
    ...    opermodes = ("supply", "charge", "standby")
    ...    exclusive = True
    ...    mode: str = "standby"
    >>> exm = ExampleMode()
    >>> exm.mode
    'standby'
    >>> exm.any_faults()
    False
    >>> exm.faultmodes
    {'no_charge': Fault(prob=1e-05, cost=100, phases={'standby': 1.0}, units='sim'), 'short': Fault(prob=1e-05, cost=100, phases={'supply': 1.0}, units='sim')}
    """

    rolename = "m"
    mode: ClassVar[str] = 'nominal'
    failrate: ClassVar[float] = 1.0
    faults: set = set()
    faultmodes: dict = {}
    mode_state_dict: dict = {}
    sub_modes: dict = {}
    fm_args = {}
    he_args = tuple()
    sfs_args = tuple()
    nfs_args = tuple()
    fsm_args = tuple()
    opermodes = ('nominal',)
    units = 'sim'
    units_set = ('sec', 'min', 'hr', 'day', 'sim')
    phases = {}
    exclusive = False
    longnames = {}
    default_track = ('mode', 'faults')

    def __init__(self, *args, s_kwargs={}, **kwargs):
        if self.he_args:
            if 'failrate' not in self.__fields__:
                raise Exception("failrate must be added to " + self.__class__.__name__ +
                                " Mode definition to calculate failrate from he_args")
            kwargs['failrate'] = self.add_he_rate(*self.he_args)
        args = self.get_true_fields(*args, **kwargs)
        super().__init__(*args)
        if not self.mode:
            self.mode = 'nominal'
        if not self.faults:
            self.faults = set()
        if not self.faultmodes:
            self.faultmodes = dict()
        if not self.mode_state_dict:
            self.mode_state_dict = dict()

        if 's' in self.__fields__:
            self.s.set_atts(**s_kwargs)

        if self.fm_args:
            self.init_faultmodes(self.fm_args)
        if self.sfs_args:
            self.init_single_faultstates(self.sfs_args[0], **self.sfs_args[1])
        if self.nfs_args:
            self.init_n_faultstates(self.nfs_args[0], **self.nfs_args[1])
        if self.fsm_args:
            self.init_faultstates_modes(self.fsm_args[0], **self.fsm_args[1])

    def __repr__(self):
        reprstr = (self.__class__.__name__ +
                   "(mode=" +
                   self.mode +
                   ", faults=" +
                   str(self.faults) +
                   ")")
        return reprstr

    def add_he_rate(self, gtp, EPCs={'na': [1, 0]}):
        """
        Calculate self.failrate based on a human error probability model.

        Parameters
        ----------
        gtp : float
            Generic Task Probability. (from HEART)
        EPCs : Dict or list
            Error producing conditions (and respective factors) for a given task
            (from HEART). Used in format:

            - Dict {'name':[EPC factor, Effect proportion]} or

            - list [[EPC factor, Effect proportion],[[EPC factor, Effect proportion]]]
        """
        if type(EPCs) == dict:
            EPC_f = np.prod([((epc-1)*x+1) for _, [epc, x] in EPCs.items()])
        elif type(EPCs) == list:
            EPC_f = np.prod([((epc-1)*x+1) for [epc, x] in EPCs])
        else:
            raise Exception("Invalid type for EPCs: " + str(type(EPCs)))
        return gtp*EPC_f

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Mode

    def return_mutables(self):
        return (self.mode, copy.copy(self.faults))

    def init_faultmodes(self, fm_args):
        """
        Initialize the self.faultmodes dictionary from the parameters of the Mode.

        Parameters
        ----------
        fm_args : dict or tuple
                Dictionary/tuple of arguments defining faultmodes, which can have forms:
                    - tuple ('fault1', 'fault2', 'fault3') (just the respective faults)
                    - dict {'fault1': args, 'fault2': kwargs}, where args and kwargs are
                    args or kwargs to Fault (prob, phases, cost, units, etc).
        """
        default_kwargs = {'prob': 1.0 / max(len(fm_args), 1),
                          'phases': self.phases,
                          'units': self.units}
        for mode in fm_args:
            if type(fm_args) is tuple:
                args = ()
                kwargs = {**default_kwargs}
            elif type(fm_args[mode]) is dict:
                args = ()
                kwargs = {**default_kwargs, **fm_args[mode]}
            elif type(fm_args[mode]) is tuple:
                args = fm_args[mode]
                kwargs = {**default_kwargs}
            else:
                raise Exception("Invalid mode definition in " +
                                str(self.__class__) + ", " + mode +
                                " modeparams values should be dict or tuple")
            args = Fault().get_true_fields(*args, **kwargs)
            args[0] *= self.failrate
            if type(args[2]) in [tuple, list, set]:
                args[2] = {ph: 1.0 for ph in args[1] for ph in args[1]}
            self.faultmodes[mode] = Fault(*args)

    def init_single_faultstates(self, franges, **kwargs):
        """
        Associate modes with given faultstates as faults.

        Modes generated for each frange (but not combined).

        Parameters
        ----------
        franges : dict, optional
            Dictionary of form {'state':{val1, val2...}) of ranges for each health state
            (if used to generate modes). The default is {}.
        **kwargs : kwargs
            Entries for the Fault (e.g., phases, etc)
        """
        tot_faults = len([f for s in franges.values() for f in s])
        prob = kwargs.get('prob', 1/tot_faults) * self.failrate
        loc_kwargs = {**kwargs, 'prob': prob}
        for state in franges:
            modes = {state + '_' + str(value): Fault(*loc_kwargs)
                     for value in franges[state]}
            modestates = {state + '_' + str(value):
                          {state: value} for value in franges[state]}
            self.faultmodes.update(modes)
            self.mode_state_dict.update(modestates)

    def init_n_faultstates(self, franges, n='all', seed=42, **kwargs):
        """
        Associate n faultstate mode combinations as faults.

        Parameters
        ----------
        franges : dict, optional
            Dictionary of form {'state':{val1, val2...}) of ranges for each health state
                                (if used to generate modes). The default is {}.
        n : int, optional
            Number of faultstate combinations to sample. The default is 'all'.
        seed : int, optional
            Seed used in selection (if n!='all). The default is 42.
        **kwargs : kwargs
            Entries for the Fault (e.g., phases, etc)
        """
        nom_fstates = {state:
                       copy.copy(self.s.__default_vals__[self.s.__fields__.index(state)])
                       for state in franges}
        for state in franges:
            franges[state].add(nom_fstates[state])
        nomvals = tuple([*nom_fstates.values()])
        statecombos = [i for i in itertools.product(*franges.values())
                       if i != nomvals]
        if type(n) == int and len(statecombos) > 0:
            rng = np.random.default_rng(seed)
            full_list = [i for i, _ in enumerate(statecombos)]
            sample = rng.choice(full_list, size=n, replace=False)
            statecombos = [statecombos[i] for i in sample]

        prob = kwargs.get('prob', 1/len(statecombos)) * self.failrate
        loc_kwargs = {**kwargs, 'prob': prob}
        self.faultmodes.update({'hmode_' + str(i): Fault(**loc_kwargs)
                                for i in range(len(statecombos))})
        self.mode_state_dict.update({'hmode_'+str(i):
                                     {list(franges)[j]: state
                                      for j, state in enumerate(statecombos[i])}
                                     for i in range(len(statecombos))})

    def init_faultstate_modes(self, manual_modes, **kwargs):
        """
        Associate modes manual_modes with provided faultstates.

        Parameters
        ----------
        manual_modes : dict, optional
            Dictionary/Set of faultmodes with structure, which has the form:
                - dict {'fault1': [atts], 'fault2': atts}, where atts may be of form:
                    - states: {state1: val1, state2, val2}
        **kwargs : kwargs
            Entries for the Fault (e.g., phases, etc)
        """
        p_def = 1/len(manual_modes)
        for mode, atts in manual_modes.items():
            if type(atts) is list:
                states = atts[0]
                loc_kwargs = {**kwargs, **atts[1]}
            elif type(atts) is dict:
                states = atts
                loc_kwargs = {**kwargs}
            loc_kwargs['prob'] = loc_kwargs.get('prob', p_def) * self.failrate
            self.mode_state_dict.update({mode: states})
            self.faultmodes.update({mode: Fault(**loc_kwargs)})

    def update_modestates(self):
        """
        Update states of the model associated with a specific fault mode.

        (see init_faultstates)
        """
        num_update = 0
        for fault in self.faults:
            if fault in self.mode_state_dict:
                self.s.put(**self.mode_state_dict[fault])
                num_update += 1
                if num_update > 1:
                    raise Exception("Exclusive fault mode scenarios" +
                                    " present at the same time")

    def set_mode(self, mode):
        """
        Set a mode in the block.

        Parameters
        ----------
        mode : str
            name of the mode to enter.
        """
        if 'mode' not in self.__fields__:
            raise Exception("mode not defined in class " + self.__class__.__name__)
        if self.exclusive:
            if self.any_faults():
                raise Exception("Cannot set mode from fault state w/o removing faults.")
            elif mode in self.faultmodes:
                self.to_fault(mode)
            else:
                self._assign_mode(mode)
        else:
            self._assign_mode(mode)

    def in_mode(self, *modes):
        """
        Check if the system is in a given operational mode.

        Parameters
        ----------
        *modes : strs
            names of the mode to check
        """
        return self.mode in modes

    def has_fault(self, *faults):
        """
        Check if the block has fault (a str).

        Parameters
        ----------
        *faults : strs
            names of the fault to check.
        """
        return any(self.faults.intersection(set(faults)))

    def no_fault(self, fault):
        """
        Check if the block does not have fault (a str).

        Parameters
        ----------
        fault : str
            name of the fault to check.
        """
        return not (any(self.faults.intersection(set([fault]))))

    def any_faults(self):
        """Check if the block has any fault modes."""
        return any(self.faults)

    def to_fault(self, fault):
        """
        Move from the current fault mode to a new fault mode.

        Parameters
        ----------
        fault : str
            name of the fault mode to switch to
        """
        self.faults.clear()
        self.faults.add(fault)
        if self.exclusive:
            self._assign_mode(fault)

    def add_fault(self, *faults):
        """
        Add fault (a str) to the block.

        Parameters
        ----------
        *fault : str(s)
            name(s) of the fault to add to the black
        """
        self.faults.update(faults)
        if self.exclusive:
            if len(faults) > 1:
                raise Exception("Multiple fault modes added to function with" +
                                " exclusive fault representation")
            elif len(faults) == 0 and self.mode in self.faultmodes:
                raise Exception("In " + str(self.__class__)
                                + "--no faults but mode is still in faultmode "
                                + self.mode)
            elif faults:
                self._assign_mode(faults[0])

    def replace_fault(self, fault_to_replace, fault_to_add):
        """
        Replace fault_to_replace with fault_to_add in the set of faults.

        Parameters
        ----------
        fault_to_replace : str
            name of the fault to replace
        fault_to_add : str
            name of the fault to add in its place
        """
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
        if self.exclusive:
            self._assign_mode(fault_to_add)

    def remove_fault(self, fault_to_remove, opermode=False, warnmessage=False):
        """
        Remove fault in the set of faults and returns to given operational mode.

        Parameters
        ----------
        fault_to_replace : str
            name of the fault to remove
        opermode : str (optional)
            operational mode to return to when the fault mode is removed
        warnmessage : str/False
            Warning to give when performing operation. Default is False (no warning)
        """
        self.faults.discard(fault_to_remove)
        if opermode:
            self._assign_mode(opermode)
        if self.exclusive and not (opermode):
            raise Exception("Unclear which operational mode to enter w- fault removed")
        if warnmessage:
            self.warn(warnmessage, "Fault mode `" +
                      fault_to_remove + "' removed.", stacklevel=3)

    def remove_any_faults(self, opermode=False, warnmessage=False):
        """
        Reset fault mode to nominal and returns to the given operational mode.

        Parameters
        ----------
        opermode : str (optional)
            operational mode to return to when the fault mode is removed
        warnmessage : str/False
            Warning to give when performing operation. Default is False (no warning)
        """
        self.faults.clear()
        self._assign_mode(opermode)

        if self.exclusive and not (self.mode):
            raise Exception("In " + str(self.__class__) + ": Unclear which operational"
                            + " mode to enter with fault removed--no default or"
                            + " opermode specified")
        if warnmessage:
            self.warn(warnmessage, "All faults removed.")

    def set_field(self, fieldname, value, as_copy=True):
        """Extend BaseContainer.assign to not set faultmodes (always the same)."""
        if fieldname != 'faultmodes' or self.faultmodes != value:
            BaseContainer.set_field(self, fieldname, value, as_copy=as_copy)

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates faults and mode tracking."""
        if att == 'faults':
            fh = History()
            for faultmode in self.faultmodes:
                fh.init_att(faultmode, False, timerange, track='all', dtype=bool)
            hist['faults'] = fh
        elif att == 'mode':
            fm_lens = [len(fm) for fm in self.faultmodes]
            om_lens = [len(m) for m in self.opermodes]
            modelength = max(fm_lens+om_lens)
            str_size = '<U'+str(modelength)
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)

    def _assign_mode(self, mode):
        if 'mode' in self.__fields__:
            if mode is None or mode is False:
                mode = self.__defaults__[self.__fields__.index('mode')]
            self.mode = mode


class ExampleMode(Mode):
    fm_args = {"no_charge": (1e-5, 100, {'standby': 1.0}),
               "short": (1e-5, 100, {'supply': 1.0})}
    opermodes = ("supply", "charge", "standby")
    exclusive = True
    mode: str = "standby"


if __name__ == "__main__":
    from fmdtools.analyze.phases import PhaseMap
    exfault = Fault(prob=0.5, phases={'on': 0.9, 'off': 0.1}, units='hr')
    phases = PhaseMap({'on': [0, 5], 'off': [6, 10]})
    rate = exfault.calc_rate(4, phases, sim_time=10.0, sim_units='min')

    import doctest
    doctest.testmod(verbose=True)
