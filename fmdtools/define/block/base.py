#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines base :class:`Simulable` and :class:`Block` classes for defining simulations.

Classes:
- :class:`Simulable`: Superclass for architectures and blocks.
- :class:`Block`: Superclass for Functions, Components, Actions, etc.

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

from fmdtools.define.base import set_var
from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.time import Time
from fmdtools.analyze.result import Result

import itertools
import copy
import warnings
import numpy as np


class SimParam(Parameter, readonly=True):
    """
    Class defining Simulation parameters.

    Parameters
    ----------
    phases : tuple
        phases (('name', start, end)...) that the simulation progresses through
    start_time : float
        Start time of the simulation.
    end_time : float
        End time of the simulation.
    track_times : tuple
        Defines what times to include in the history.
        Options are:
        - ('all',)--all simulated times
        - ('interval', n)--includes every nth time in the history
        - ('times', [t1, ... tn])--only includes times defined in the
          vector [t1 ... tn]
    dt : float
        time-step used in the simulation. default is 1.0
    units : str
        time-units. default is hours`
    end_condition : str
        Name of indicator method to use to end the simulation. If not provided (''),
        the simulation ends at the final time. Default is ''
    use_local : bool
        Whether to use locally-defined time-steps in functions (if any).
        Default is True.
    """

    rolename = "sp"
    phases: tuple = (('na', 0, 100),)
    start_time: float = 0.0
    end_time: float = 100.0
    track_times: tuple = ('all',)
    dt: float = 1.0
    units: str = "hr"
    units_set = ('sec', 'min', 'hr', 'day', 'wk', 'month', 'year')
    end_condition: str = ''
    use_local: bool = True

    def __init__(self, *args, **kwargs):
        if 'phases' not in kwargs:
            p = (('na', kwargs.get('start_time', 0.0),
                  kwargs.get('end_time', SimParam.__defaults__['end_time'])), )
            kwargs['phases'] = p
        super().__init__(*args, **kwargs)
        self.find_any_phase_overlap()

    def base_type(self):
        return SimParam

    def find_any_phase_overlap(self):
        """Check that simparam phases don't overlap."""
        phase_dict = {v[0]: [v[1], v[2]] for v in self.phases}
        intervals = [*phase_dict.values()]
        int_low = np.sort([i[0] for i in intervals])
        int_high = np.sort([i[1] if len(i) == 2 else i[0] for i in intervals])
        for i, il in enumerate(int_low):
            if i+1 == len(int_low):
                break
            if int_low[i+1] <= int_high[i]:
                raise Exception("Global phases overlap in " + self.__class__.__name__ +
                                ": " + str(self.phases) +
                                " Ensure max of each phase < min of each other phase")

    def get_timerange(self, start_time=None, end_time=None):
        """
        Generate the timerange to simulate over.

        Examples
        --------
        >>> SimParam(end_time=5.0).get_timerange()
        array([0., 1., 2., 3., 4., 5.])
        """
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        return np.round(np.arange(start_time, end_time + self.dt, self.dt), 7)

    def get_histrange(self, start_time=0.0, end_time=None):
        """Get the history range associated with the SimParam."""
        timerange = self.get_timerange(start_time, end_time)
        if self.track_times[0] == "all":
            histrange = timerange
        elif self.track_times[0] == 'interval':
            histrange = timerange[0:len(timerange):self.track_times[1]]
        elif self.track_times[0] == 'times':
            histrange = self.track_times[1]
        return histrange

    def get_shift(self, old_start_time=0.0):
        """
        Get the shift between the sim timerange in the history.

        Examples
        --------
        >>> SimParam().get_shift(6.0)
        6
        >>> SimParam(track_times=('interval', 2)).get_shift(2)
        1
        """
        prevrange = self.get_timerange(start_time=0.0, end_time=old_start_time)[:-1]
        if self.track_times[0] == "all":
            shift = len(prevrange)
        elif self.track_times[0] == 'interval':
            shift = len(prevrange[0:len(prevrange):self.track_times[1]])
        elif self.track_times[0] == 'times':
            shift = 0
        return shift

    def get_hist_ind(self, t_ind, t, shift):
        """
        Get the index of the history given the simulation time/index and shift.

        Examples
        --------
        >>> SimParam().get_hist_ind(2, 2.0, 1)
        3
        >>> SimParam(track_times=('interval', 2)).get_hist_ind(4, 4.0, 0)
        2
        """
        if self.track_times[0] == 'all':
            t_ind_rec = t_ind + shift
        elif self.track_times[0] == 'interval':
            t_ind_rec = t_ind//self.track_times[1] + shift
        elif self.track_times[0] == 'times':
            t_ind_rec = self.track_times[1].index(t)
        else:
            raise Exception("Invalid argument, track_times=" + str(self.track_times))
        return t_ind_rec


class Simulable(BaseObject):
    """
    Base class for object which simulate (blocks and architectures).

    Note that classes solely based on Simulable may not be able to be simulated.

    Parameters
    ----------
    t : Time
        Time tracker and options.
    sp : SimParam
        Parameters defining the simulation.
    mut_kwargs : dict
        Non-default kwargs for mutable containers/roles (to use for reset)
    """

    __slots__ = ('p', 'sp', 'r', 't', 'h', 'track', 'flows', 'mut_kwargs')
    container_t = Time
    default_track = ["all"]
    immutable_roles = BaseObject.immutable_roles + ['sp']
    default_sp = {}
    container_sp = SimParam

    def __init__(self, sp={}, **kwargs):
        """
        Instantiate internal Simulable attributes.

        Parameters
        ----------
        sp : dict
            SimParam arguments
        **kwargs : kwargs
            Keyword arguments to BaseObject
        """
        loc_kwargs = {**kwargs, 'sp': {**self.default_sp, **sp}}
        BaseObject.__init__(self, **loc_kwargs)
        self.mut_kwargs = {role: kwargs.get(role)
                           for role in self.get_roles('container', with_immutable=False)
                           if role in kwargs}

    def init_hist(self, h={}):
        """Initialize the history of the sim using SimParam parameters and track."""
        if not h:
            timerange = self.sp.get_histrange()
            self.h = self.create_hist(timerange)
        else:
            self.h = h.copy()

    def init_time_hist(self):
        """Add time history to the model (only done at top level)."""
        if 'time' not in self.h:
            timerange = self.sp.get_histrange()
            self.h['time'] = timerange

    def log_hist(self, t_ind, t, shift):
        """Log the history over time."""
        if self.sp.track_times:
            t_ind_rec = self.sp.get_hist_ind(t_ind, t, shift)
            self.h.log(self, t_ind_rec, time=t)

    def update_seed(self, seed=[]):
        """
        Update seed and propogates update to contained actions/components.

        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        if seed and hasattr(self, 'r'):
            self.r.update_seed(seed)

    def find_classification(self, scen, mdlhists):
        """
        Classify the results of the simulation (placeholder).

        Parameters
        ----------
        scen     : Scenario
            Scenario defining the model run.
        mdlhists : History
            History for the simulation(s)

        Returns
        -------
        endclass: Result
            Result dictionary with rate, cost, and expecte_cost values
        """
        return Result({'rate': scen.rate, 'cost': 1, 'expected_cost': scen.rate})

    def new_params(self, name='', p={}, sp={}, r={}, track={}, **kwargs):
        """
        Create a copy of the defining immutable parameters for use in a new Simulable.

        Parameters
        ----------
        p     : dict
            Parameter args to update
        sp    : dict
            SimParam args to update
        r     : dict
            Rand args to update
        track : dict
            track kwargs to update.

        Returns
        -------
        param_dict: dict
            Dict with immutable parameters/options. (e.g., 'p', 'sp', 'track')
        """
        param_dict = {**copy.deepcopy(self.mut_kwargs)}

        if hasattr(self, 'p'):
            param_dict['p'] = self.p.copy_with_vals(**p)
        if hasattr(self, 'sp'):
            param_dict['sp'] = self.sp.copy_with_vals(**sp)
        if not r and hasattr(self, 'r'):
            param_dict['r'] = {'seed': self.r.seed}
        elif r:
            param_dict['r'] = r
        if not track:
            param_dict['track'] = copy.deepcopy(self.track)
        return param_dict

    def new(self, **kwargs):
        """
        Create a new Model with the same parameters as the current model.

        Can initiate with with changes to mutable parameters (p, sp, track, rand etc.).
        """
        return self.__class__(name=self.name, **self.new_params(**kwargs))

    def get_fxns(self):
        """
        Get fxns associated with the Simulable (self if Function, self.fxns if Model).

        Returns
        -------
        fxns: dict
            Dict with structure {fxnname: fxnobj}
        """
        if hasattr(self, 'fxns'):
            fxns = self.fxns
        else:
            fxns = {self.name: self}
        return fxns

    def get_scen_rate(self, fxnname, faultmode, time, phasemap={}, weight=1.0):
        """
        Get the scenario rate for the given single-fault scenario.

        Parameters
        ----------
        fxnname: str
            Name of the function with the fault
        faultmode: str
            Name of the fault mode
        time: int
            Time when the scenario is to occur
        phasemap : PhaseMap, optional
            Map of phases/modephases that define operations the mode will be injected
            during (and maps to the opportunity vector phases). The default is {}.
        weight : int, optional
            Scenario weight (e.g., if more than one scenario is sampled for the fault).
            The default is 1.

        Returns
        -------
        rate: float
            Rate of the scenario
        """
        fxn = self.get_fxns()[fxnname]
        fm = fxn.m.faultmodes.get(faultmode, False)
        if not fm:
            raise Exception("faultmode "+faultmode+" not in "+str(fxn.m.__class__))
        else:
            sim_time = self.sp.start_time - self.sp.end_time + self.sp.dt
            rate = fm.calc_rate(time, phasemap=phasemap, sim_time=sim_time,
                                sim_units=self.sp.units, weight=weight)
        return rate

    def return_probdens(self):
        """Get the probability density associated with Block and things it contains."""
        if hasattr(self, 'r'):
            state_pd = self.r.return_probdens()
        else:
            state_pd = 1.0
        return state_pd


class Block(Simulable):
    """
    Superclass for Function, Component, and Action subclasses.

    Has functions for model setup, querying state, and reseting the model.

    Attributes
    ----------
    p : Parameter
        Internal Parameter for the block. Instanced from container_p
    s : State
        Internal State of the block. Instanced from container_s.
    m : Mode
        Internal Mode for the block. Instanced from container_m
    r : Rand
        Internal Rand for the block. Instanced from container_r
    t : Time
        Internal Time for the block. Instanced from container_t
    name : str
        Block name
    flows : dict
        Dictionary of flows included in the Block (if any are added via flow_flowname)
    """

    __slots__ = ['s', 'm']
    default_track = ['s', 'm', 'r', 't', 'i']
    roletypes = ['container', 'flow']
    check_dict_creation = True

    def __init__(self, name='', flows={}, h={}, **kwargs):
        """
        Instance superclass. Called by Function and Component classes.

        Parameters
        ----------
        name : str
            Name for the Block instance.
        flows :dict
            Flow objects passed from the model level.
        kwargs : kwargs
            Roles and tracking to override the defaults. See Simulable.__init__
        """
        # use aliases for flows
        if hasattr(self, 'flownames'):
            flows = {self.flownames.get(fn, fn): flow for fn, flow in flows.items()}
        flows = {**flows}

        Simulable.__init__(self, name=name, roletypes=['container', 'flow'],
                           **flows, **kwargs)
        # send flows from block level to arch level
        if 'arch' in self.roletypes:
            self.init_roletypes('arch', **self.create_arch_kwargs(**kwargs))
            self.update_contained_modes()
        self.check_flows(flows=flows)
        self.update_seed()
        # finally, allow for user-defined role/state changing
        self.init_block(**kwargs)
        self.init_hist(h=h)
        self.check_slots()

    def init_block(self, **kwargs):
        """Placeholder initialization method to set initial states etc."""
        return

    def create_arch_kwargs(self, **kwargs):
        """
        Create keyword arguments for contained architectures.

        Enables the passing of flows from block to contained architecture level.
        """
        archs = self.find_roletype_initiators("arch")
        b_flows = {f: getattr(self, f) for f in self.flows}
        arch_kwargs = {}
        for k in archs:
            if k in kwargs and isinstance(kwargs[k], dict):
                arch_kwargs[k] = {**kwargs[k], **{'flows': b_flows}, 'name': k, 'sp': self.sp}
            elif k in kwargs and isinstance(kwargs[k], BaseObject):
                arch_kwargs[k] = kwargs[k].copy(flows=b_flows, name=k, sp=self.sp)
            else:
                arch_kwargs[k] = {'flows': b_flows, 'name': k, 'sp': self.sp}

        return {'flows': b_flows, **arch_kwargs}

    def update_contained_modes(self):
        """
        Add contained faultmodes for the container at to the Function model.

        Parameters
        ----------
        at : str ('ca' or 'aa')
            Role to update (for ComponentArchitecture or ActionArchitecture roles)
        """
        for at in self.get_roles('arch'):
            arch = getattr(self, at)
            try:
                for flex_role in arch.flexible_roles:
                    role = getattr(arch, flex_role)
                    for block in role.values():
                        if hasattr(block, 'm'):
                            fms = {block.name + '_' + fname: vals
                                   for fname, vals in block.m.faultmodes.items()}
                            self.m.faultmodes.update(fms)
            except AttributeError as e:
                raise Exception("Class " + self.__class__.__name__ + " missing mode" +
                                "containter despite containing arch" + arch.name) from e

    def check_flows(self, flows={}):
        """
        Associate flows with the given Simulable.

        Flows must be defined with the flow_flowname class variable pointing to the
        class to initialize (e.g., flow_flowname = FlowClass).

        Parameters
        ----------
        flows : dict, optional
            If flows is provided AND it contains a flowname corresponding to the
            function's flowname, it will be used instead (so that it can act as a
            connection to the rest of the model)
        """
        # check if any sent but not attached
        unattached_flows = [f for f in flows if not hasattr(self, f)]
        if unattached_flows:
            warnings.warn("these flows sent from model "+str(unattached_flows)
                          + " not added to class "+str(self.__class__))
        # check that hashes match
        unhashed_flows = [f for f, obj in flows.items()
                          if getattr(self, f).__hash__() != obj.__hash__()]
        if unhashed_flows:
            raise Exception("Flows in " + str(self.__class__) +
                            "copied instead of added: " + str(unhashed_flows))

    def get_flows(self):
        """Return a dictionary of the Block's flows."""
        return {f: getattr(self, f) for f in self.flows}

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Block

    def is_static(self):
        """Check if Block has static execution step."""
        return (getattr(self, 'behavior', False) or
                getattr(self, 'static_behavior', False) or
                (hasattr(self, 'aa') and getattr(self.aa, 'proptype', '') == 'static'))

    def is_dynamic(self):
        """Check if Block has dynamic execution step."""
        return (hasattr(self, 'dynamic_behavior') or
                (hasattr(self, 'aa') and getattr(self.aa, 'proptype', '') == 'dynamic'))

    def __repr__(self):
        """
        Provide a repl-friendly string showing the states of the Block.

        Returns
        -------
        repr: str
            console string
        """
        if hasattr(self, 'name'):
            fxnstr = getattr(self, 'name', '')+' '+self.__class__.__name__
            for at in ['s', 'm']:
                at_container = getattr(self, at, False)
                if at_container:
                    fxnstr = fxnstr+'\n'+"- "+at_container.__repr__()
            return fxnstr
        else:
            return 'New uninitialized '+self.__class__.__name__

    def get_rand_states(self, auto_update_only=False):
        """
        Get dict of random states from block and associated actions/components.

        Parameters
        ----------
        auto_update_only

        Returns
        -------
        rand_states : dict
            Random states from the block and associated actions/components.
        """
        if hasattr(self, 'r'):
            rand_states = self.r.get_rand_states(auto_update_only)
        else:
            rand_states = {}
        if hasattr(self, 'ca'):
            rand_states.update(self.ca.get_rand_states(auto_update_only=auto_update_only))
        if hasattr(self, 'aa'):
            for actname, act in self.aa.actions.items():
                if act.get_rand_states(auto_update_only=auto_update_only):
                    rand_states[actname] = act.get_rand_states(auto_update_only=auto_update_only)
        return rand_states

    def choose_rand_fault(self, faults, default='first', combinations=1):
        """
        Randomly chooses a fault or combination of faults to insert in fxn.m.

        Parameters
        ----------
        faults : list
            list of fault modes to choose from
        default : str/list, optional
            Default fault to inject when model is run deterministically.
            The default is 'first', which chooses the first in the list.
            Can provide a mode as a str or a list of modes
        combinations : int, optional
            Number of combinations of faults to elaborate and select from.
            The default is 1, which just chooses single fault modes.
        """
        if hasattr(self, 'r') and getattr(self.r, 'run_stochastic', True):
            faults = [list(x) for x in itertools.combinations(faults, combinations)]
            self.m.add_fault(*self.r.rng.choice(faults))
        elif default == 'first':
            self.m.add_fault(faults[0])
        elif isinstance(default, str):
            self.m.add_fault(default)
        else:
            self.m.add_fault(*default)

    def get_flowtypes(self):
        """
        Return the names of the flow types in the model.

        Returns
        -------
        flowtypes : set
            Set of flow type names in the model.
        """
        return {obj.__class__.__name__ for name, obj in self.get_flows().items()}

    def copy(self, *args, flows={}, **kwargs):
        """
        Copy the block with its current attributes.

        Parameters
        ----------
        args   : tuple
            New arguments to use to instantiate the block, (e.g., flows, p, s)
        kwargs :
            New kwargs to use to instantiate the block.

        Returns
        -------
        cop : Block
            copy of the exising block
        """
        try:
            paramdict = self.new_params(**kwargs)
            if 'arch' in self.roletypes:
                arch_dict = {role: getattr(self, role)
                             for role in self.get_roles('arch')}
                paramdict = {**paramdict, **arch_dict}
            cop = self.__class__(self.name, *args, flows=flows, **paramdict)
            cop.assign_roles('container', self)
        except TypeError as e:
            raise Exception("Poor specification of "+str(self.__class__)) from e
        if 'arch' in self.roletypes:
            cop.update_contained_modes()
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def propagate(self, time, faults={}, disturbances={}, run_stochastic=False):
        """
        Inject and propagates faults through the graph at one time-step.

        Parameters
        ----------
        time : float
            The current time-step.
        faults : dict
            Faults to inject during this propagation step.
            With structure {fname: ['fault1', 'fault2'...]}
        disturbances : dict
            Variables to change during this propagation step.
            With structure {'var1': value}
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of
            random states over time.
        """
        # Step 0: Update block states with disturbances
        for var, val in disturbances.items():
            set_var(self, var, val)
        faults = faults.get(self.name, [])

        # Step 1: Run Dynamic Propagation Methods in Order Specified
        # and Inject Faults if Applicable
        if hasattr(self, 'dynamic_loading_before'):
            self.dynamic_loading_before(self, time)
        if self.is_dynamic():
            self("dynamic", time=time, faults=faults, run_stochastic=run_stochastic)
        if hasattr(self, 'dynamic_loading_after'):
            self.dynamic_loading_after(self, time)

        # Step 2: Run Static Propagation Methods
        active = True
        oldmutables = self.return_mutables()
        flows_mutables = {f: fl.return_mutables() for f, fl in self.get_flows().items()}
        while active:
            if self.is_static():
                self("static", time=time, faults=faults, run_stochastic=run_stochastic)

            if hasattr(self, 'static_loading'):
                self.static_loading(time)
            # Check to see what flows now have new values and add connected functions
            # (done for each because of communications potential)
            active = False
            newmutables = self.return_mutables()
            if oldmutables != newmutables:
                active = True
                oldmutables = newmutables
            for flowname, fl in self.get_flows().items():
                newflowmutables = fl.return_mutables()
                if flows_mutables[flowname] != newflowmutables:
                    active = True
                    flows_mutables[flowname] = newflowmutables


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
