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
import copy


class Fault(BaseContainer, readonly=True):
    """
    Stores Default Attributes for individual fault modes.

    Fields
    ------
    prob : float
        Mode probability or rate per operation
    cost : float
        Individual mode cost (e.g., of repair)
    phases : tuple
        Opportunity vector mapping phases of operation to relative probability.
        i.e., phase_probability = Fault.prob * Fault.phases[phase]
        Has format ('phasename1', value, 'phasename2', value2)
    disturbances : tuple
        Disturbances caused by the fault to aspect(s) of the containing simulable.
        e.g. ('s.x', 1.0, 's.y', 2.0)
    units : str
        Units on probability. Can be a unit of continuous ('sec', 'min', 'hr', 'day') or
        discrete ('sim') time. Default is 'sim'.
    """

    prob: np.float64 = np.float64(1.0)
    cost: np.float64 = np.float64(0.0)
    phases: tuple = ()
    disturbances: tuple = ()
    units: np.str_ = 'sim'

    def __init__(self, *args, failrate=1.0, **kwargs):
        args = self.get_true_fields(*args, force_kwargs=True, **kwargs)
        args[0] = failrate * args[0]
        super().__init__(*args)

    @classmethod
    def valid_fault(cls, fault):
        """Check if the external fault is a valid fault (has same keys)."""
        if isinstance(fault, cls):
            return True
        else:
            for kwarg in fault:
                if kwarg not in cls.__fields__:
                    return False
            return True

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
                opp_factor = dict(self.phases).get(phase, 0.0)
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
    exclusive : True/False
        Whether fault modes are exclusive of each other or not.
        Default is False (i.e. more than one can be present).
    default_xx : float
        Default values for Fault fields (e.g., prob, phases, etc)

    These fields are then used in simulation and elsewhere.

    Fields
    ------
    faults : set
        Set of faults present (or not) at any given time
    sub_faults : bool
        Whether objects contained by the object are faulty.
    mode : str
        Name of the current mode. the default is 'nominal'
    faultmodes : dict
            Dictionary of :class:`Fault` defining possible fault modes and
            their properties

    Examples
    --------
    >>> class ExampleMode(Mode):
    ...    fault_no_charge = Fault(1e-5, 100, (('standby', 1.0),))
    ...    fault_short = (1e-5, 100, (('supply', 1.0),))
    ...    opermodes = ("supply", "charge", "standby")
    ...    exclusive = True
    ...    mode: str = "standby"
    >>> exm = ExampleMode()
    >>> exm.mode
    'standby'
    >>> exm.any_faults()
    False
    >>> exm.get_faults()
    {'no_charge': Fault(prob=1e-05, cost=100, phases=(('standby', 1.0),), disturbances=(), units='sim'), 'short': Fault(prob=1e-05, cost=100, phases=(('supply', 1.0),), disturbances=(), units='sim')}
    """

    rolename = "m"
    mode: ClassVar[str] = 'nominal'
    failrate: ClassVar[float] = 1.0
    faults: set = set()
    sub_faults: bool = False
    opermodes = ('nominal',)
    exclusive = False
    default_track = ('mode', 'faults', 'sub_faults')

    def __init__(self, *args, s_kwargs={}, **kwargs):
        args = self.get_true_fields(*args, **kwargs)
        super().__init__(*args)
        if not self.mode:
            self.mode = 'nominal'
        if not self.faults:
            self.faults = set()

    def create_repr(self, fields=["mode", "faults", "sub_faults"], **kwargs):
        """Limit default repr to relevant fields."""
        return super().create_repr(fields=fields, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Mode

    def return_mutables(self):
        return (self.mode, copy.copy(self.faults))

    def get_fault(self, faultname, **kwargs):
        """
        Get the Fault object associated with the given faultname.

        Parameters
        ----------
        faultname : str
            Name of the fault (if defined as a part of the mode at value fault_name).
            Can also be parameters of the fault.

        Returns
        -------
        fault: Fault
            Fault container with given fields.

        Examples
        --------
        >>> exm = ExampleMode()
        >>> exm.get_fault('short')
        Fault(prob=1e-05, cost=100, phases=(('supply', 1.0),), disturbances=(), units='sim')
        >>> exm.get_fault('short', prob=0.1)
        Fault(prob=0.1, cost=100, phases=(('supply', 1.0),), disturbances=(), units='sim')
        """
        if isinstance(faultname, str):
            fault = getattr(self, 'fault_'+faultname)
        else:
            fault = faultname
        if isinstance(fault, Fault):
            return fault
        else:
            defaults = self.get_pref_attrs("default")
            try:
                if isinstance(fault, tuple):
                    arg = fault
                    kwar = {**defaults, 'failrate': self.failrate, **kwargs}
                elif isinstance(fault, dict):
                    arg = ()
                    kwar = {**defaults, **fault, 'failrate': self.failrate, **kwargs}
                return Fault(*arg, **kwar)
            except Exception as e:
                raise Exception("Poorly specified fault mode: " + faultname + " in "
                                + self.__class__.__name__) from e

    def get_all_faultnames(self):
        """Get all names of faults."""
        return tuple([*self.get_pref_attrs("fault")])

    def get_faults(self, *faults):
        """Get Fault objects for all associated faults."""
        if not faults:
            faults = self.get_all_faultnames()
        return {k: self.get_fault(k) for k in faults}

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
            elif mode in self.get_all_faultnames():
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

        Examples
        --------
        >>> exm = ExampleMode()
        >>> exm.get_fault('low')
        Fault(prob=1.0, cost=0.0, phases=(), disturbances={'s.x': 20.0}, units='sim')
        >>> exm.add_fault(dict(low={'disturbances': {'s.x': 40.0}}))
        >>> exm
        ExampleMode(faults={'low'}, sub_faults=False, fault_low={'disturbances': {'s.x': 40.0}}, mode='low')
        >>> exm.get_fault('low')
        Fault(prob=1.0, cost=0.0, phases=(), disturbances={'s.x': 40.0}, units='sim')
        """
        if len(faults) == 1 and (isinstance(faults[0], list) or isinstance(faults[0], dict)):
            faults = faults[0]
        self.faults.update(faults)
        if isinstance(faults, dict):
            for faultname, fault in faults.items():
                setattr(self, 'fault_'+faultname, fault)
        if self.exclusive:
            if len(faults) > 1:
                raise Exception("Multiple fault modes added to function with" +
                                " exclusive fault representation")
            elif len(faults) == 0 and self.mode in self.get_all_faultnames():
                raise Exception("In " + str(self.__class__)
                                + "--no faults but mode is still in faultmode "
                                + self.mode)
            elif faults:
                self._assign_mode(*faults)

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

    def get_fault_disturbances(self, *faults):
        """Get all disturbances caused by present (or specified) faults."""
        if not faults:
            faults = self.faults
        dists = {}
        for fault in faults:
            faultobj = self.get_fault(fault)
            if faultobj:
                dists.update(faultobj.disturbances)
        return dists

    def init_hist_att(self, hist, att, timerange, track, str_size='<U20'):
        """Add field 'att' to history. Accommodates faults and mode tracking."""
        if att == 'faults':
            fh = History()
            for faultmode in self.get_all_faultnames():
                fh.init_att(faultmode, False, timerange, track='all', dtype=bool)
            hist['faults'] = fh
        elif att == 'mode':
            fm_lens = [len(fm) for fm in self.get_all_faultnames()]
            om_lens = [len(m) for m in self.opermodes]
            modelength = max(fm_lens+om_lens)
            str_size = '<U'+str(modelength)
            BaseContainer.init_hist_att(self, hist, att, timerange, track, str_size)
        elif att == 'sub_faults':
            hist.init_att(att, False, timerange, track='all', dtype=bool)

    def _assign_mode(self, mode):
        if 'mode' in self.__fields__:
            if mode is None or mode is False:
                mode = self.__defaults__[self.__fields__.index('mode')]
            self.mode = mode


class HumanErrorMode(Mode):
    """
    Mode for Human Errors using HEART-based model.

    Overall failrate of human error determined by given:

    gtp : float
        Generic task probability
    epc_XX : tuple/list
        Error producing condition factors. May be specified as performance shaping
        factors or tuple (factor, effect proportion).

    Examples
    --------
    >>> class ExHMode(HumanErrorMode):
    ...     gtp : float = 0.01
    ...     epc_1 : tuple = (2, 0.5)
    >>> exh = ExHMode()
    >>> exh.failrate
    np.float64(0.015)
    >>> class ExHMode2(HumanErrorMode):
    ...     gtp : float = 0.01
    ...     epc_1 : float = 2.0
    ...     epc_2 : float = 3.0
    >>> exh2 = ExHMode2()
    >>> exh2.failrate
    np.float64(0.06)
    """

    failrate: np.float64 = 1.0
    gtp: ClassVar[np.float64] = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failrate = self.calc_he_rate()

    def calc_he_rate(self):
        """
        Calculate self.failrate based on a human error probability model.

        This model uses an overall generic task probability (set by self.gtp) as well as
        error producing conditions (EPCs) to determine the overall failure rate of the
        task.
        """
        EPCs = self.get_pref_attrs("epc")
        if EPCs:
            EPC_f = np.prod([((epc[0]-1)*epc[1]+1) if isinstance(epc, tuple) else epc
                             for _, epc in EPCs.items()])
            return self.gtp*EPC_f
        else:
            return self.gtp


class ExampleMode(Mode):
    """Example mode for testing/docs."""

    fault_no_charge = Fault(1e-5, 100, (('standby', 1.0),))
    fault_short = (1e-5, 100, (('supply', 1.0),))
    fault_low: dict = {'disturbances': {'s.x': 20.0}}
    opermodes = ("supply", "charge", "standby")
    exclusive = True
    mode: str = "standby"


if __name__ == "__main__":
    from fmdtools.analyze.phases import PhaseMap
    exfault = Fault(prob=0.5, phases=(('on', 0.9), ('off', 0.1)), units='hr')
    phases = PhaseMap({'on': [0, 5], 'off': [6, 10]})
    rate = exfault.calc_rate(4, phases, sim_time=10.0, sim_units='min')

    import doctest
    doctest.testmod(verbose=True)
