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

from fmdtools.define.base import gen_timerange, is_iter, get_var, filter_kwargs
from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.time import Time
from fmdtools.define.container.mode import Fault
from fmdtools.analyze.result import Result
from fmdtools.analyze.history import History
from fmdtools.analyze.graph.base import Graph

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
    with_loadings : bool
        Whether to execute simulable with defined loading methods. Default is False.
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
    track_pdf : bool
        If run_stochastic=True, track_pdf is used to calculate/track the probability
        densities of random states over time.
    """

    rolename = "sp"
    phases: tuple = (('na', 0, 100),)
    start_time: np.float64 = np.float64(0.0)
    end_time: np.float64 = np.float64(100.0)
    track_times: tuple = ('all',)
    dt: np.float64 = np.float64(1.0)
    units: str = "hr"
    units_set = ('sec', 'min', 'hr', 'day', 'wk', 'month', 'year')
    end_condition: str = ''
    use_local: bool = True
    with_loadings: bool = False
    run_stochastic: bool = False
    track_pdf: bool = False

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

    def get_timerange(self, start_time=None, end_time=None, min_r=7):
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
        return gen_timerange(start_time, end_time, self.dt, min_r)

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

    def get_hist_ind(self, t_ind, t):
        """
        Get the index of the history given the simulation time/index and shift.

        Examples
        --------
        >>> SimParam().get_hist_ind(2, 2.0)
        2
        >>> SimParam(track_times=('interval', 2)).get_hist_ind(4, 4.0)
        2
        """
        if self.track_times[0] == 'all':
            t_ind_rec = t_ind
        elif self.track_times[0] == 'interval':
            t_ind_rec = t_ind//self.track_times[1]
        elif self.track_times[0] == 'times':
            t_ind_rec = self.track_times[1].index(t)
        else:
            raise Exception("Invalid argument, track_times=" + str(self.track_times))
        return t_ind_rec

    def get_sub_kwargs(self):
        """Get keyword arguments for contained sim from larger sim."""
        kwargs = self.asdict()
        if not self.use_local:
            kwargs.pop("dt")
        kwargs['end_condition'] = ""
        return kwargs


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

    __slots__ = ('p', 'sp', 'r', 't', 'h', 'track', 'mut_kwargs', '_sims')
    container_t = Time
    default_track = ["all"]
    immutable_roles = BaseObject.immutable_roles + ['sp']
    default_sp = {}
    container_sp = SimParam

    def __init__(self, **kwargs):
        """
        Instantiate internal Simulable attributes.

        Parameters
        ----------
        sp : dict
            SimParam arguments
        **kwargs : kwargs
            Keyword arguments to BaseObject
        """
        BaseObject.__init__(self, **kwargs)
        self.set_time()
        self._sims = {}

    def is_static(self):
        """Check if Block has static execution step."""
        return (getattr(self, 'behavior', False) or
                getattr(self, 'static_behavior', False) or
                any([r.is_static() for r in self.get_sims().values()]))

    def is_dynamic(self):
        """Check if Block has dynamic execution step."""
        return (hasattr(self, 'dynamic_behavior') or
                any([r.is_dynamic() for r in self.get_sims().values()]))

    def set_time(self):
        """Set time from SimParam."""
        self.t.set_timestep(dt=self.sp.dt, use_local=self.sp.use_local)

    def init_hist(self, h={}):
        """Initialize the history of the sim using SimParam parameters and track."""
        if not isinstance(h, History):
            timerange = self.sp.get_histrange()
            self.h = self.create_hist(timerange)
        else:
            self.h = h.copy()
        if not self.root:
            self.init_time_hist()

    def init_time_hist(self):
        """Add time history to the model (only done at top level)."""
        if 'time' not in self.h:
            timerange = self.sp.get_histrange()
            self.h['time'] = timerange

    def update_seed(self, seed=[]):
        """
        Update seed and propogates update to contained actions/components.

        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        if hasattr(self, 'r'):
            self.r.assign(self.sp, 'run_stochastic', 'track_pdf')
            if seed:
                self.r.update_seed(seed)
            else:
                seed = self.r.seed
        for simname, sim in self.get_sims().items():
            sim.update_seed(seed)

    def classify(self, scen={}, **kwargs):
        """Classify the results of the simulation (placeholder)."""
        return {}

    def get_result(self, to_return={}, nomresult={}, **kwargs):
        """
        Get the Result for the simulation corresponding to to_return.

        Parameters
        ----------
        to_return : iterable, optional
            Iterable of stings to get. The default is {}.
        **kwargs : kwargs
            Keyword arguments to self.classify or self.as_modelgraph.

        Returns
        -------
        result : TYPE
            DESCRIPTION.
        """
        result = Result()
        self.get_endclass(to_return=to_return, result=result, nomresult=nomresult,
                          **kwargs)
        if 'faults' in to_return:
            result['faults'] = [*self.return_faultmodes()]
        self.get_resgraph(to_return=to_return, result=result, nomresult=nomresult)
        self.get_resvars(to_return=to_return, result=result)
        return result

    def get_endclass(self, to_return={}, result={}, **kwargs):
        """
        Get the end-state classification from self.classify().

        Parameters
        ----------
        to_return : iterable, optional
            Iterable with attributes to get from model. If "class" is in to_return,
            it returns the whole dictionary. If "class.att" is in to_return, it gets
            the attribute "att" from dict returned by classify(). The default is {}.
        result : Result, optional
            Result to add the result to. The default is Result().
        **kwargs : kwargs
            Keyword arguments to self.classify().

        Returns
        -------
        result : Result
            Result with returned class attributes.
        """
        sub_returns = [x.split('.')[1] for x in to_return
                       if isinstance(x, str) and 'classify.' in x]
        get_endclass = 'classify' in to_return
        if get_endclass or sub_returns:
            res = self.classify(**filter_kwargs(self.classify, **kwargs))
            if get_endclass:
                result['classify'] = Result(res)
            for sub_r in sub_returns:
                result['classify.'+sub_r] = res[sub_r]
        return result

    def get_resgraph(self, to_return={}, result={}, nomresult={}):
        """
        Get the graph corresponding to the state of the simulation.

        Parameters
        ----------
        to_return : iterable, optional
            Iterable with attributes to get from the model. If "graph" is in to_return,
            it returns the graph from self.get_modelgraph. If "graph.obj" is in
            to_return, it returns the graph from obj.get_modelgraph. If to_return is a
            dict, the values of the dict can specify arguments to get_modelgraph. The
            default is {}.
        result : Result, optional
            Result to add the graph to. The default is Result().
        nomresult : dict, optional
            Nominal result. If provided, set_resgraph() adds degradations to the graph
            attributes. The default is {}.

        Returns
        -------
        result : Result
            Result with graph added.
        """
        if not isinstance(to_return, dict):
            to_return = dict.fromkeys(to_return)
        graphs_to_get = {g: arg for g, arg in to_return.items()
                         if type(g) is str and (g.startswith('graph')
                                                or g.startswith('Graph'))}
        for g, arg in graphs_to_get.items():
            if '.' in g:
                strs = g.split(".")
                obj = get_var(self, strs[1:])
            else:
                obj = self
            kwar = {}
            if isinstance(arg, tuple):
                kwar = {**kwar, **arg[1], 'gtype': arg[0]}
            elif isinstance(arg, Graph):
                kwar = {**kwar, 'gtype': arg}

            rgraph = obj.as_modelgraph(time=self.t.time, **kwar)

            if nomresult and g in nomresult:
                rgraph.set_resgraph(nomresult[g])
            else:
                rgraph.set_resgraph()
            result[g] = rgraph
        return result

    def get_resvars(self, to_return={}, result={}):
        """
        Get variables specified in to_return.

        Parameters
        ----------
        to_return : Iterable, optional
            Iterable of values to get from the object. Keys could start with "vars" or
            simply not be "class", "faults", or "graph". The default is {}.
        result : Result, optional
            REsult to add the variables to. The default is Result().

        Returns
        -------
        result : Result
            Result with variables added.
        """
        vars_to_get = [v for v in to_return if isinstance(v, str) and
                       (not 'class' in v and not 'vars' in v and not 'faults' in v
                        and not 'graph' in v and not 'Graph' in v)]
        if 'vars' in to_return:
            vars_to_get.append(to_return['vars'])
        var_result = self.get_vars(*vars_to_get, trunc_tuple=False)
        for i, var in enumerate(vars_to_get):
            result[var] = var_result[i]
        return result

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
        param_dict = {**copy.deepcopy(getattr(self, 'mut_kwargs', {}))}

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
        return {**param_dict, **kwargs}

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

    def get_fault(self, scope, faultmode, **kwargs):
        """
        Get the named Fault for the Simulable in the given scope of the model.

        Will also get a Fault for a contained object at the given scope.

        Parameters
        ----------
        scope : str
            Scope to get the fault from (e.g., self.name, 'self', or 'global')
        faultmode: str
            Name of the fault (e.g., "short").
        **kwargs: kwargs
            Keyword arguments to Mode.get_fault.
        """
        name = self.get_full_name()
        if name.endswith(scope) or scope in ['self', 'global']:
            obj = self
        else:
            if scope.startswith(name):
                scope = scope[len(name)+1:]
            obj = self.get_vars(scope)
        fm = obj.m.get_fault(faultmode, **kwargs)
        if not fm:
            raise Exception("faultmode "+faultmode+" not in "+str(obj.m.__class__))
        else:
            return fm

    def get_scen_rate(self, scope, faultmode, time, phasemap={}, weight=1.0, **kwargs):
        """
        Get the scenario rate for the given single-fault scenario.

        Parameters
        ----------
        scope: str
            Name of the block or sub-block with the fault
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
        **kwargs : kwargs
            Non-default keyword arguments for Fault/Mode.

        Returns
        -------
        rate: float
            Rate of the scenario
        """
        fm = self.get_fault(scope, faultmode, **kwargs)
        sim_time = self.sp.start_time - self.sp.end_time + self.sp.dt
        rate = fm.calc_rate(time, phasemap=phasemap, sim_time=sim_time,
                            sim_units=self.sp.units, weight=weight)
        return rate

    def inject_faults(self, faults=[]):
        """
        Inject faults into the block and its contained architectures.

        Parameters
        ----------
        faults : str/list/dict, optional
            Faults to inject. The default is [].
        """
        if faults:
            if isinstance(faults, str):
                faults = [faults]
            if faults and (isinstance(faults, list)
                           or Fault.valid_fault([*faults.values()][0])):
                faults = {self.name: faults}
            full_name = self.get_full_name()
            for faultscope, fault in faults.items():
                if faultscope == self.name or faultscope == full_name:
                    self.m.add_fault(fault)
                    self.set_fault_disturbances(fault)
                else:
                    if faultscope.startswith(self.name):
                        faultscope = faultscope[len(self.name)+1:]
                    if faultscope.startswith(full_name):
                        faultscope = faultscope[len(full_name)+1:]
                    fs_split = faultscope.split(".")
                    roles = self.get_roles()
                    if fs_split[0] in roles or fs_split[1] in roles:
                        obj = self.get_vars(faultscope)
                        obj.inject_faults(fault)
            self.set_sub_faults()

    def set_fault_disturbances(self, *faults):
        """Set Mode-based disturbances (if present)."""
        if len(faults) == 1 and is_iter(faults[0]):
            faults = faults[0]
        if isinstance(faults, dict):
            faults = [*faults.keys()]
        if hasattr(self, 'm'):
            disturbances = self.m.get_fault_disturbances(*faults)
            if disturbances:
                self.set_vars(**disturbances)

    def set_sub_faults(self):
        """Set the mode to have sub_faults if contained objs have faults."""
        if hasattr(self, 'm'):
            sub_faults = self.get_faults(with_base_faults=False)
            self.m.sub_faults = bool(sub_faults)
        if hasattr(self, 'm') and self.m.exclusive is True and len(self.m.faults) > 1:
            raise Exception("More than one fault present in " + self.name +
                            "\n faults: " + str(self.m.faults) +
                            "\n Is the mode representation nonexclusive?")

    def get_faults(self, with_sub_faults=True, with_base_faults=True,
                   only_present=True):
        """
        Get faults associated with the given block.

        Parameters
        ----------
        with_sub_faults : bool, optional
            Whether to get faults from the block. The default is True.
        with_base_faults : bool, optional
            Whether to get faults from objects contained by the block. The default is
            True.
        only_present: bool, optional
            Whether to get only present faults (if True) or all possible faults. The
            default is True.

        Returns
        -------
        all_faults : dict
            Dictionary of blocks and faults present in the block. Of structure
            {blockname: faults}
        """
        all_faults = dict()
        if with_base_faults and hasattr(self, 'm'):
            if only_present:
                if self.m.faults:
                    faults = self.m.get_faults(*self.m.faults)
                    all_faults[self.name] = faults
            else:
                all_faults[self.get_full_name()] = self.m.get_faults()

        if with_sub_faults:
            if not isinstance(with_sub_faults, bool):
                with_sub_faults -= 1
            sub_kwar = dict(only_present=only_present, with_sub_faults=with_sub_faults)
            for objname, obj in self.get_sims().items():
                if hasattr(obj, 'get_faults'):
                    all_faults.update(obj.get_faults(**sub_kwar))
        return all_faults

    def return_faultmodes(self):
        """Return all faults as a single dictionary."""
        endfaultprops = {scope+"."+mode: f for scope, modes in self.get_faults().items()
                         for mode, f in modes.items()}
        return endfaultprops

    def return_probdens(self):
        """Get the probability density associated with Block and things it contains."""
        if hasattr(self, 'r'):
            state_pd = self.r.return_probdens()
        else:
            state_pd = 1.0
        for arch, sim in self.get_sims().items():
            state_pd *= sim.return_probdens()
        return state_pd

    def update_stochastic_states(self):
        """Update stochastic states if in run_stochastic configuration."""
        if not self.t.has_executed() and self.sp.run_stochastic and hasattr(self, 'r'):
            self.r.update_stochastic_states()
            if self.sp.track_pdf:
                self.r.probdens = self.r.return_probdens()

    def get_sims(self):
        """Return dict of simulable objects within the object."""
        if not self._sims:
            all_roles = self.get_roles_as_dict(with_immutable=False, exclude=['flow'])
            self._sims = {rolename: roleobj for rolename, roleobj in all_roles.items()
                          if isinstance(roleobj, Simulable)}
        return self._sims

    def update_arch_behaviors(self, proptype):
        """
        Propagate behaviors into contained architectures.

        Iterates through the Block's contained architectures and calls the corresponding
        propagation type.

        Parameters
        ----------
        time : float
            Time to update the architecture behaviors at.
        proptype : str
            Propagation step to perform ('dynamic' or 'static').
        """
        for objname, obj in self.get_sims().items():
            try:
                obj(time=self.t.time, proptype=proptype, inc_at="")
            except TypeError as e:
                raise Exception("Poorly specified Architecture: "
                                + str(obj.__class__)) from e

    def update_dynamic_behaviors(self, proptype="dynamic"):
        """
        Run the dynamic behaviors of the block, including loadings (if specified).

        Runs in the defined order:
            1.) running dynamic_loading_before() (if sp.with_loadings is set to True)
            2.) updating contained architecture dynamic behaviors/propagation
            3.) running dynamic_behavior())
            4.) running dynamic_loading_after (if sp.with_loadings is set to True)

        Parameters
        ----------
        proptype : str
            Propagation the system is in. Skips if proptype="static."
            Default is "dynamic"
        """
        if not self.t.executed_dynamic and proptype in ['dynamic', 'both'] and self.t.time > 0.0:
            if self.sp.with_loadings and hasattr(self, 'dynamic_loading_before'):
                self.dynamic_loading_before()
            self.update_arch_behaviors("dynamic")
            if hasattr(self, "dynamic_behavior"):
                self.dynamic_behavior()
            if self.sp.with_loadings and hasattr(self, 'dynamic_loading_after'):
                self.dynamic_loading_after()
            self.t.executed_dynamic = True

    def update_static_behaviors(self, proptype="static"):
        """
        Run the static behaviors of the block, including loadings (if specified).

        Runs in the defined order:
            1.) updating contained architecture static behaviors/propagation
            2.) running static_behavior())
            3.) running static_loading (if sp.with_loadings is set to True)

        Parameters
        ----------
        time : float
            Time to update the static behavior at.
        proptype : str
            Propagation the system is in. Skips if proptype="dynamic."
            If "static-once", static behaviors are only executed once. This option is
            used when executing in architectures. Default is "static"
        """
        if proptype in ['static', 'both']:
            active = True
            self.set_mutables()
            while active:
                self.execute_static_behaviors()
                # determine if propagation should continue due to new states
                active = self.has_changed(update=True)
        elif proptype in ['static-once']:
            self.execute_static_behaviors()

    def execute_static_behaviors(self):
        """Execute static behaviors."""
        self.update_arch_behaviors("static")
        if hasattr(self, 'static_behavior'):
            self.static_behavior()
        if self.sp.with_loadings and hasattr(self, 'static_loading'):
            self.static_loading()
        self.t.executed_static = True

    def inc_sim_time(self, **kwargs):
        """Update the simulation time t_ind at the end of the timestep."""
        if self.t.time > 0.0:
            self.t.t_ind += 1
            self.t.executing = False
        for objname, obj in self.get_sims().items():
            obj.inc_sim_time(**kwargs)

    def cut_hist(self):
        """Cut the simulation history to the current time."""
        self.h.cut(end_ind=self.t.t_ind)

    def __call__(self, time=None, proptype="both", faults=[], disturbances={},
                 inc_at="all", end_of_simulation=False, copy=False):
        """
        Call the Simulable to propagate behavior at a single time-step.

        Parameters
        ----------
        time : float/'end'
            Time at which to simulate the Simulable. Default is None, which uses
            self.t.time. If 'end' is provided, will be simulated until the end of
            the simulation (sp.end_time or sp.end_condition)
        proptype : str
            Type of propagation step to update ('static', 'dynamic', or 'both')
        faults : dict/list
            Faults to inject during this propagation step. (to be injected at time.)
            With structure ['fault1', 'fault2'...]
        disturbances : dict
            Variables to change during this propagation step. (to be injected at time.)
            With structure {'var1': value}
        inc_at : bool
            When to increment the history. If "all", increment at each time the sim
            iterates through. If "time", increment only at t==time.
        end_of_simulation : bool
            Whether this is the end of the simulation (and history should be cut).
        copy : bool
            If true, returns a copy of the model just before fault injection and
            stops simulation
        """
        try:
            if time == 'end':
                time = self.sp.end_time
            start_time, end_time = self.t.get_sim_times(time)
            sim_times = self.sp.get_timerange(start_time, end_time)
            for t in sim_times:
                if t == end_time:
                    if copy:
                        return self.copy()
                    if t != self.t.time:  # faults are only injected if not run yet
                        self.set_vars(**disturbances)
                        self.inject_faults(faults)
                self.t.update_time(t)
                if self.t.executing:
                    self.update_stochastic_states()
                    self.update_dynamic_behaviors(proptype=proptype)
                    self.update_static_behaviors(proptype=proptype)
                    self.set_sub_faults()
                    if inc_at == "all" or (inc_at == "time" and t == time):
                        self.inc_sim_time()
                        t_ind = self.sp.get_hist_ind(self.t.t_ind, self.t.time)
                        self.h.log(self, t_ind, self.t.time)
                    if self.sp.end_condition and not copy:
                        if get_var(self, self.sp.end_condition)():
                            break

            if end_of_simulation:
                self.cut_hist()
        except Exception as e:
            raise Exception("Error simulating " + self.name +
                            " of class " + self.__class__.__name__ +
                            " at time=" + str(self.t.time)) from e


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
        self.mut_kwargs = {role: kwargs.get(role)
                           for role in self.get_roles(with_flex=False,
                                                      with_immutable=False)
                           if role in kwargs}
        self.check_flows(flows=flows)
        self.update_seed()
        # finally, allow for user-defined role/state changing
        self.init_block(**kwargs)
        self.init_hist(h=h)
        self.check_slots()

    def create_repr(self, rolenames=['s', 'm'], **kwargs):
        """Restricts default repr to state and mode."""
        return super().create_repr(rolenames=rolenames, **kwargs)

    def init_block(self, **kwargs):
        """Initilialization method to set initial states etc (placeholder)."""
        return

    def create_arch_kwargs(self, **kwargs):
        """
        Create keyword arguments for contained architectures.

        Enables the passing of flows from block to contained architecture level.
        """
        archs = self.find_roletype_initiators("arch")
        b_flows = {f: getattr(self, f) for f in self.flows}
        spk = self.sp.get_sub_kwargs()
        arch_kw = {}
        for k in archs:
            if k in kwargs and isinstance(kwargs[k], dict):
                arch_kw[k] = {**kwargs[k], **{'flows': b_flows}, 'name': k, 'sp': spk}
            elif k in kwargs and isinstance(kwargs[k], BaseObject):
                arch_kw[k] = kwargs[k].copy(flows=b_flows, name=k, sp=spk)
            else:
                arch_kw[k] = {'flows': b_flows, 'name': k, 'sp': spk}

        return {'flows': b_flows, **arch_kw}

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
                paramdict = {**paramdict, 'sp': self.sp, 'root': self.root, **arch_dict}
            cop = self.__class__(self.name, *args, flows=flows, **paramdict)
            cop.assign_roles('container', self)
        except TypeError as e:
            raise Exception("Poor specification of "+str(self.__class__)) from e
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
