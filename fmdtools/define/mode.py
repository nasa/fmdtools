# -*- coding: utf-8 -*-
"""
Description: Module for helping define Modes (faulty and otherwise).

Has classes:

- :class:`Fault`: Class for defining fault parameters
- :class:`Mode`: Class for defining the mode property (and associated probability model)
  held in Blocks.
"""
from recordclass import dataobject
from typing import ClassVar
import numpy as np
import itertools
import copy
from .common import get_true_fields, get_true_field, get_dataobj_track
from fmdtools.analyze.result import History, init_hist_iter


class Fault(dataobject, readonly=True, mapping=True):
    """
    Stores Default Attributes for for modes to use in Mode.faultmodes.

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



class Mode(dataobject, readonly=False):
    """
    Description: Class for defining the mode property (and associated probability model)
    held in Blocks.

    Mode is meant to be inherited in order to define the specific faults related to a
    given Block.

    e.g.,
    class ExampleMode(Mode):
        fm_args = {"high_heat", "low_heat"}
        mode = "start"
    Will create a Mode structure where m.mode = 'start' that can enter the given
    fault modes 'high_heat' and 'low_heat'.

    Mode has the following class variables  which can be modified to define the
    representation and probability model:

    Class Variables
    ---------------
    opermodes : tuple
        Names of non-faulty operational modes.
    failrate : float
        Overall failure rate for the block. The default is 1.0. Note that if a failrate
        is provided, the prob argument in faultparams is a conditional probability
        (e.g. Fault.prob = Mode.failrate * Mode.faultparams['mode']['prob'])
    probtype : str, optional
        Type of probability in the probability model, a per-time 'rate' or
        per-run 'prob'.The default is 'rate'
    units : str, optional
        Type of units ('sec'/'min'/'hr'/'day') used for the rates.
        Default is 'sim', which is unitless (prob/simulation)
    phases : dict
        Phases to inject faults in.
    exclusive : True/False
        Whether fault modes are exclusive of each other or not.
        Default is False (i.e. more than one can be present).
    longnames : dict
        Longer names for the faults (if desired). {faultname: longname}
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    fm_args : dict
        Arguments to Mode.init_faultmodes()
    he_args : tuple
        Arguments for add_he_rate defining a human error probability model.
    sfs_args : tuple
        Arguments for self.init_single_faultstates (franges, {kwargs})
    nfs_args : tuple
        Arguments for self.init_n_faultstates (*args, {kwargs})
    fsm_args : tuple
        Arguments for self.init_faultstates_modes (manual_modes, {kwargs})

    These properties can then be used in simulation
    ------------
    faults : set
        Set of faults present (or not) at any given time
    mode : str
        Name of the current mode. the default is 'nominal'
    mode_state_dict: dict
        Maps modes to states. Assigned by assoc_faultstates

    While these properties are used for determining scenario information
    ------------
    faultmodes : dict
            Dictionary of :class:`Fault` defining possible fault modes and
            their properties
    """

    mode: ClassVar[str] = 'nominal'
    failrate: ClassVar[float] = 1.0
    faults: set = set()
    faultmodes: dict = {}
    mode_state_dict: dict = {}
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
        args = get_true_fields(self, *args, **kwargs)
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
            if type(fm_args) == tuple:
                args = ()
                kwargs = {**default_kwargs}
            elif type(fm_args[mode]) == dict:
                args = ()
                kwargs = {**default_kwargs, **fm_args[mode]}
            elif type(fm_args[mode]) == tuple:
                args = fm_args[mode]
                kwargs = {**default_kwargs}
            else:
                raise Exception("Invalid mode definition in " +
                                str(self.__class__) + ", " + mode +
                                " modeparams values should be dict or tuple")
            args = get_true_fields(Fault, *args, **kwargs)
            args[0] *= self.failrate
            if type(args[2]) in [tuple, list, set]:
                args[2] = {ph: 1.0 for ph in args[1] for ph in args[1]}
            self.faultmodes[mode] = Fault(*args)

    def assoc_single_faultstates(self, franges, **kwargs):
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

    def assoc_n_faultstates(self, franges, n='all', seed=42, **kwargs):
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
        nom_fstates = {state: self.s.__defaults__[self.s.__fields__.index(state)]
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

    def assoc_faultstate_modes(self, manual_modes, **kwargs):
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
            if type(atts) == list:
                states = atts[0]
                loc_kwargs = {**kwargs, **atts[1]}
            elif type(atts) == dict:
                states = atts
                loc_kwargs = {**kwargs}
            loc_kwargs['prob'] = loc_kwargs.get('prob', p_def) * self.failrate
            self.mode_state_dict.update({mode: states})
            self.faultmodes.update({mode: Fault(**loc_kwargs)})

    def update_modestates(self):
        """
        Update states of the model associated with a specific fault mode.

        (see assoc_faultstates)
        """
        num_update = 0
        for fault in self.faults:
            if fault in self.mode_state_dict:
                for state, value in self.mode_state_dict[fault].items():
                    setattr(self, state, value)
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
        return not(any(self.faults.intersection(set([fault]))))

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

    def mirror(self, mode_to_mirror):
        if 'mode' in self.__fields__:
            self.mode = mode_to_mirror.mode
        self.faults.clear()
        self.faults.update(mode_to_mirror.faults)

    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)

    def get_true_fields(self, *args, **kwargs):
        return get_true_fields(self, *args, **kwargs)

    def create_hist(self, timerange, track):
        h = History()
        track = get_dataobj_track(self, track)
        if 'faults' in track:
            fh = History()
            for faultmode in self.faultmodes:
                fh.init_att(faultmode, False, timerange, track='all', dtype=bool)
            h['faults'] = fh
        fm_lens = [len(fm) for fm in self.faultmodes]
        om_lens = [len(m) for m in self.opermodes]
        modelength = max(fm_lens+om_lens)
        str_size = '<U'+str(modelength)
        h.init_att('mode', self.mode, timerange, track, str_size=str_size)
        return h

    def _assign_mode(self, mode):
        if 'mode' in self.__fields__:
            if mode is None or mode is False:
                mode = self.__defaults__[self.__fields__.index('mode')]
            self.mode = mode
