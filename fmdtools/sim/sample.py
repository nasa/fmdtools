#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for defining samples of Faults and Parameters to simulate.

Has classes:

- :class:`ParameterDomain`: Defines domain for sampling from Parameters.
- :class:`FaultDomain`: Defines domain for sampling from Faults.
- :class:`FaultSample`: Defines a sample of fault scenarios.
- :class:`SampleApproach`: Defines a set of fault scenario samples.
- :Class:`ParameterSample`: Defines a sample of a set of parameters.
- :Class:`ParameterResultSample`: Defines a sample from Result values.
- :Class:`ParameterHistSample`: Defines a sample from History values.

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
from fmdtools.define.base import get_var, nest_dict, gen_timerange, is_numeric
from fmdtools.define.container.parameter import Parameter, ExampleParameter
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim.scenario import SingleFaultScenario, JointFaultScenario
from fmdtools.sim.scenario import ParameterScenario
from fmdtools.analyze.common import calc_metric
from fmdtools.analyze.phases import PhaseMap, join_phasemaps

import numpy as np
import itertools
import inspect
import copy


def pass_var(*x):
    return x


class ParameterDomain(object):
    """
    Defines a domain to sample from a Parameter.

    Attributes
    ----------
    parameter_init: Parameter/method
        Method which may be used to initialize a parameter. If a method is provided, it
        may only take kwargs as input (which will map to ParameterDomain variables).
    variables : dict
        Variables (inputs to parameter_init) and their range/set constraints.
    constants : dict
        Variables (inputs to parameter_init) with set parameter values.

    Examples
    --------
    Given the parameter:

    >>> class ExampleParameter(Parameter):
    ...    x: float = 1.0
    ...    y: float = 10.0
    ...    z: float = 0.0
    ...    x_lim = (0, 10)
    ...    y_set = (1.0, 2.0, 3.0, 4.0)

    We can then define the following domain, which by default gets set constraints:

    >>> expd = ParameterDomain(ExampleParameter)
    >>> expd.add_variables("y", "x")
    >>> expd.add_constants(z=20)
    >>> expd
    ParameterDomain with:
     - variables: {'y': {1.0, 2.0, 3.0, 4.0}, 'x': (0, 10)}
     - constants: {'z': 20}
     - parameter_initializer: ExampleParameter

    This ParameterDomain then becomes a callable with the given variables:

    >>> expd(1, 2)
    ExampleParameter(x=2.0, y=1.0, z=20.0)

    And we can separately check the set contraints for the variables:

    >>> expd.get_set_constraints(1, 2)
    (False, False)
    >>> expd.get_set_constraints(0, 20)
    (True, True)

    This can also work with nested parameters:

    >>> class ExNestedParam(Parameter):
    ...    ex_param: ExampleParameter = ExampleParameter()
    ...    k: float = 20.0

    >>> expd1 = ParameterDomain(ExNestedParam)
    >>> expd1.add_variables("ex_param.x", "ex_param.y", "k")
    >>> expd1(1,2, 3)
    ExNestedParam(ex_param=ExampleParameter(x=1.0, y=2.0, z=0.0), k=3.0)
    """

    def __init__(self, parameter_init):
        self.parameter_init = parameter_init
        self.variables = {}
        self.var_maps = {}
        self.constants = {}

    def add_variable(self, variable, var_set=(), var_lim=(), var_map=pass_var):
        """
        Add a variable to the ParameterDomain.

        Parameters
        ----------
        variable : str
            Name of the variable in parameter_init.
        var_set : tuple, optional
            Discrete set constraints for the variable. The default is ().
        var_lim : tuple, optional
            Variable limits. The default is ().
        var_map: function, optional
            Preprocessing variable mapping, mapping the input to the parameter domain
            to the actual variable value. May be used to e.g., encode different discrete
            options listed as strings ('arch1', 'arch2') as integers (1, 2).
            The default is pass_var, which passes the input variable.
        """
        if var_set:
            var_domain = set(var_set)
        elif var_lim:
            var_domain = var_lim
        elif issubclass(self.parameter_init, Parameter):
            var_domain = self.parameter_init.get_set_const(variable)
        else:
            var_domain = ()
        self.variables[variable] = var_domain
        if var_map:
            self.var_maps[(variable,)] = var_map

    def add_variables(self, *variables, sets={}, lims={}, var_map=pass_var):
        """
        Add a list of variables to the ParameterDomain.

        Parameters
        ----------
        *variables : str
            Names of the variables in parameter_init
        sets : dict, optional
            Set constraints for the variables. The default is {}.
        lims : dict, optional
            Variable limits. The default is {}.
        var_map: function, optional
            Preprocessing variable mapping, mapping the input to the parameter domain
            to the actual variable value. May be used to e.g., encode different discrete
            options listed as strings ('arch1', 'arch2') as integers (1, 2).
            The default is pass_var, which passes the input variable.
        """
        for variable in variables:
            self.add_variable(variable,
                              var_set=sets.get(variable, ()),
                              var_lim=lims.get(variable, ()),
                              var_map=False)
        if var_map:
            self.var_maps[variables] = var_map

    def add_constant(self, constant, value):
        """
        Add a value to be set to a constant value in the ParameterDomain.

        Parameters
        ----------
        constant : str
            Name of the parameter_init field to set to the constant value.
        value : value
            Value to set the constant to.
        """
        self.constants[constant] = value

    def add_constants(self, **constants):
        """
        Add a number of constant values to the ParameterDomain to be set as constant.

        Parameters
        ----------
        **constants : kwargs
            args to set as constant and their values.
        """
        for constant, value in constants.items():
            self.add_constant(constant, value)

    def get_set_constraints(self, *x):
        """
        Get the set constraints at a value of the variables x.

        Parameters
        ----------
        *x : values
            Values of the variables.

        Returns
        -------
        set_constraints : tuple
            Set constraint values (True or False) accross the variables.
            If False, the constraint is met. If True, the constraint is violated.
        """
        set_constraints = []
        for i, (variable, const) in enumerate(self.variables.items()):
            if not const:
                set_const = False
            else:
                if i < len(x):
                    arg = x[i]
                else:
                    raise Exception("x of invalid length: "+str(len(x)))
                if type(const) is tuple:
                    set_const = not (const[0] <= arg <= const[1])
                elif type(const) is set:
                    set_const = not (arg in const)
            set_constraints.append(set_const)
        return tuple(set_constraints)

    def get_map_vars(self, *x):
        """Get the mapped variables for x."""
        x_mapped = []
        i = 0
        for var_group in self.var_maps:
            x_map = self.var_maps[var_group](*x[i:i+len(var_group)])
            x_mapped.extend(x_map)
            i += len(var_group)
        return x_mapped

    def get_param_kwargs(self, *x):
        """Get kwargs for the parameter at the given value of x."""
        x_mapped = self.get_map_vars(*x)
        return {**x_to_kwargs(self.constants, self.variables, *x_mapped)}

    def __call__(self, *x):
        """Generate the parameter at a given value of the variables."""
        kwargs = self.get_param_kwargs(*x)

        if inspect.isclass(self.parameter_init) and issubclass(self.parameter_init, Parameter):
            kwargs['check_type'] = False
            kwargs['check_pickle'] = False
            kwargs['check_lim'] = False
            kwargs['set_type'] = True

        return self.parameter_init(**kwargs)

    def __repr__(self):
        rep = ("ParameterDomain with:" +
               "\n - variables: " + str(self.variables) +
               "\n - constants: " + str(self.constants) +
               "\n - parameter_initializer: " + self.parameter_init.__name__)
        return rep

    def get_var_iters(self, resolution=1.0, resolutions={}):
        """
        Get iterables for each variable (provided given resolution).

        Parameters
        ----------
        resolution : float, optional
            Default resolution for the ranges. The default is 1.
        resolutions : dict, optional
            Dict of resolutions for each variable e.g. {'var1': 0.1}.
            The default is {}.

        Returns
        -------
        var_iters : dict
            Dict or iterables for each variable.

        Examples
        --------
        >>> expd.get_var_iters()
        {'y': {1.0, 2.0, 3.0, 4.0}, 'x': array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])}
        """
        var_iters = dict.fromkeys(self.variables)
        for variable in var_iters:
            if isinstance(self.variables[variable], tuple):
                ran = self.variables[variable]
                if variable in resolutions:
                    res = resolutions[variable]
                else:
                    res = resolution
                var_iters[variable] = np.arange(ran[0], ran[1]+res, res)
            elif isinstance(self.variables[variable], set):
                var_iters[variable] = self.variables[variable]
            else:
                raise Exception("Invalid set constraint for variable " + variable)
        return var_iters

    def get_x_defaults(self):
        """Get default values for x from parameter."""
        default_param = self.parameter_init()
        x_defaults = []
        for variable in self.variables:
            x_def = get_var(default_param, variable)
            x_defaults.append(x_def)
        return tuple(x_defaults)


def x_to_kwargs(constants, variables, *x):
    """Convert x over the defined variables into a set of kwargs."""
    var_args = {}
    for i, variable in enumerate(variables):
        if i < len(x):
            var_args[variable] = x[i]
        else:
            raise Exception("x of invalid length: "+str(len(x)))
    return nest_dict({**var_args, **constants})


# example paramdomain/sample for testing
expd = ParameterDomain(ExampleParameter)
expd.add_variables("y", "x")
expd.add_constants(z=20)
expd


class ExNestedParam(Parameter):
    """Example nested parameter for testing ParamDomain."""

    ex_param: ExampleParameter = ExampleParameter()
    k: float = 20.0


def same_mode(modename1, modename2, exact=True):
    """Check if modename1 and modename2 are the same."""
    if exact:
        return modename1 == modename2
    else:
        return modename1 in modename2


def sample_times_even(times, numpts, dt=1.0):
    """
    Get sample time for the number of points from sampling evenly.

    Parameters
    ----------
    times : list
        Times to sample.
    numpts : int
        Number of points to sample.

    Returns
    -------
    sampletimes : list
        List of times to sample
    weights : list
        Weights.

    Examples
    --------
    >>> sample_times_even([0.0, 1.0, 2.0, 3.0, 4.0], 2)
    ([1.0, 3.0], [0.5, 0.5])
    """
    if numpts+2 > len(times):
        sampletimes = times
    else:
        pts = [np.quantile(times, p/(numpts+1)) for p in range(numpts+2)][1:-1]
        sampletimes = [round(pt/dt)*dt for pt in pts]
    weights = [1/len(sampletimes) for i in sampletimes]
    return sampletimes, weights


def sample_times_quad(times, nodes, weights):
    """
    Get the sample times for the given quadrature defined by nodes and weights.

    Parameters
    ----------
    times : list
        Times to sample.
    nodes : nodes
        quadrature nodes (ranging between -1 and 1)
    weights : weights
        corresponding quadrature weights

    Returns
    -------
    sampletimes : list
        List of times to sample
    weights : list
        Weights.

    Examples
    --------
    >>> sample_times_quad([0,1,2,3,4], [-0.5, 0.5], [0.5, 0.5])
    ([1, 3], [0.5, 0.5])
    """
    quantiles = np.array(nodes)/2 + 0.5
    if len(quantiles) > len(times):
        raise Exception("Nodes length " + str(len(nodes))
                        + "longer than times" + str(len(times)))
    else:
        sampletimes = [int(round(np.quantile(times, q))) for q in quantiles]
        weights = np.array(weights)/sum(weights)
    return sampletimes, list(weights)


class FaultDomain(object):
    """
    Defines the faults which will be sampled from in an approach.

    Attributes
    ----------
    fxns : dict
        Dict of fxns in the given Simulable (to simulate)
    faults : dict
        Dict of faults to inject in the simulable
    """

    def __init__(self, mdl):
        self.mdl = mdl
        self.fxns = mdl.get_fxns()
        self.faults = {}

    def __repr__(self):
        faultlist = [str(fault) for fault in self.faults]
        if len(faultlist) > 10:
            faultlist = faultlist[0:10] + ["...more"]
        modestr = "FaultDomain with faults:" + "\n -" + "\n -".join(faultlist)
        return modestr

    def add_fault(self, fxnname, faultmode):
        """
        Add a fault to the FaultDomain.

        Parameters
        ----------
        fxnname : str
            Name of the simulable to inject in
        faultmode : str
            Name of the faultmode to inject.
        """
        fault = self.fxns[fxnname].m.faultmodes[faultmode]
        self.faults[(fxnname, faultmode)] = fault

    def add_faults(self, *faults):
        """
        Add multiple faults to the FaultDomain.

        Parameters
        ----------
        *faults : tuple
            Faults (simname, faultmode) to inject

        Examples
        --------
        >>> exfd = FaultDomain(ExFxnArch())
        >>> exfd.add_faults(("ex_fxn", "no_charge"))
        >>> exfd
        FaultDomain with faults:
         -('ex_fxn', 'no_charge')
        """
        for fault in faults:
            self.add_fault(fault[0], fault[1])

    def add_all(self):
        """
        Add all faults in the Simulable to the FaultDomain.

        Examples
        --------
        >>> exfd = FaultDomain(ExFxnArch())
        >>> exfd.add_all()
        >>> exfd
        FaultDomain with faults:
         -('ex_fxn', 'no_charge')
         -('ex_fxn', 'short')
         -('ex_fxn2', 'no_charge')
         -('ex_fxn2', 'short')
        """
        faults = [(fxnname, mode) for fxnname, fxn in self.fxns.items()
                  if hasattr(fxn, 'm')
                  for mode in fxn.m.faultmodes]
        self.add_faults(*faults)

    def add_all_modes(self, *modenames, exact=True):
        """
        Add all modes with the given modenames to the FaultDomain.

        Parameters
        ----------
        *modenames : str
            Names of the modes
        exact : bool, optional
            Whether the mode name must be an exact match. The default is True.
        """
        for modename in modenames:
            faults = [(fxnname, mode) for fxnname, fxn in self.fxns.items()
                      if hasattr(fxn, 'm')
                      for mode in fxn.m.faultmodes
                      if same_mode(modename, mode, exact=exact)]
            self.add_faults(*faults)

    def add_all_fxnclass_modes(self, *fxnclasses):
        """
        Add all modes corresponding to the given fxnclasses.

        Parameters
        ----------
        *fxnclasses : str
            Name of the fxnclass (e.g., "AffectDOF", "MoveWater")

        Examples
        --------
        >>> exfd = FaultDomain(ExFxnArch())
        >>> exfd.add_all_fxnclass_modes("ExampleFunction")
        >>> exfd
        FaultDomain with faults:
         -('ex_fxn', 'no_charge')
         -('ex_fxn', 'short')
         -('ex_fxn2', 'no_charge')
         -('ex_fxn2', 'short')
        """
        for fxnclass in fxnclasses:
            faults = [(fxnname, mode)
                      for fxnname, fxn in self.mdl.fxns_of_class(fxnclass).items()
                      for mode in fxn.m.faultmodes if hasattr(fxn, 'm')]
            self.add_faults(*faults)

    def add_all_fxn_modes(self, *fxnnames):
        """
        Add all modes in the given simname.

        Parameters
        ----------
        *fxnnames : str
            Names of the functions (e.g., "affect_dof", "move_water").

        Examples
        --------
        >>> exfd = FaultDomain(ExFxnArch())
        >>> exfd.add_all_fxn_modes("ex_fxn2")
        >>> exfd
        FaultDomain with faults:
         -('ex_fxn2', 'no_charge')
         -('ex_fxn2', 'short')
        """
        for fxnname in fxnnames:
            faults = [(fxnname, mode) for mode in self.fxns[fxnname].m.faultmodes
                      if hasattr(self.fxns[fxnname], 'm')]
            self.add_faults(*faults)

    def add_singlecomp_modes(self, *fxns):
        """
        Add all single-component modes in functions.

        Parameters
        ----------
        *fxns : str
            Names of the functions containing the components.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> fd = FaultDomain(Drone())
        >>> fd.add_singlecomp_modes("affect_dof")
        >>> fd
        FaultDomain with faults:
         -('affect_dof', 'lf_short')
         -('affect_dof', 'lf_openc')
         -('affect_dof', 'lf_ctlup')
         -('affect_dof', 'lf_ctldn')
         -('affect_dof', 'lf_ctlbreak')
         -('affect_dof', 'lf_mechbreak')
         -('affect_dof', 'lf_mechfriction')
         -('affect_dof', 'lf_propwarp')
         -('affect_dof', 'lf_propstuck')
         -('affect_dof', 'lf_propbreak')
        """
        if not fxns:
            fxns = tuple(self.fxns)
        for fxn in fxns:
            if hasattr(self.fxns[fxn], 'ca'):
                firstcomp = list(self.fxns[fxn].ca.comps)[0]
                compfaults = [(fxn, fmode)
                              for fmode, comp in self.fxns[fxn].ca.m.sub_modes.items()
                              if firstcomp == comp]
                self.add_faults(*compfaults)


exfd = FaultDomain(ExFxnArch())
exfd.add_all()


class BaseSample():
    """
    Overarching sample class (for FaultSample and SampleApproach).

    Subclasses should have methods:
        - scenarios() for getting all scenarios from the sample
        - times() for getting all times from the sample
    """

    def get_scens(self, **kwargs):
        """
        Get scenarios with the values corresponding to **kwargs.

        Parameters
        ----------
        **kwargs : kwargs
            key-value pairs for the Scenario (e.g., fault='faultname')

        Returns
        -------
        scens : dict
            Dict of scenarios with the given properties.
        """
        scens = {i.name: i for i in self.scenarios()}
        for kwarg in kwargs:
            scens = {k: v for k, v in scens.items() if get_var(v, kwarg) == kwargs[kwarg]}
        return scens

    def get_scen_values(self, value):
        """
        Get a dict of information from the set of scenarios.

        Parameters
        ----------
        value : str
            Value to get from the scenarios.
        """
        return {scenname: getattr(scen, value)
                for scenname, scen in self.get_scens().items()}

    def get_metric(self, value, ids="all", **kwargs):
        """
        Get metrics from scenario(s).

        Parameters
        ----------
        value : str
            Scenario value to take the metric over.
        ids : list, optional
            List of scenarios to get the metric over. The default is "all".
        **kwargs : kwargs
            kwargs to calc_metric.

        Returns
        -------
        metric : float
            Metric (by default, average) over the set of scenarios.

        Examples
        --------
        >>> exfs.get_metric("rate")
        0.0
        >>> exfs2 = copy.deepcopy(exfs)
        >>> exfs2.scenarios()[0].rate=2
        >>> exfs2.get_metric("rate", method="sum")
        2.0
        >>> exfs2.get_metric("rate", method="average")
        0.25
        """
        if ids == "all":
            data = np.array([*self.get_scen_values(value).values()])
        else:
            data = np.array([j for i, j in self.get_scen_values(value).items()
                             if i in ids])
        return calc_metric(data, **kwargs)

    def get_groups_scens(self, groupnames, groups):
        """
        Get scenarios related to the given groups.

        Parameters
        ----------
        groupnames : list
            List of scenario properties to group (e.g., 'function'', 'fault')
        groups : list
            Groups to get e.g, [ ('fxnname1', 'fault1')]

        Returns
        -------
        scen_groups : dict
            dict of scenarios for each group with structure
            {(field1_val, field2_val) : [scenarios]}
        """
        scen_groups = {}
        for group in groups:
            group_kwargs = {groupname: group[i]
                            for i, groupname in enumerate(groupnames)}
            scen_groups[group] = list(self.get_scens(**group_kwargs))
        return scen_groups

    def group_scens(self, *groupnames):
        """
        Get the groups of scenario parameters corresponding to *groupnames.

        Parameters
        ----------
        *groupnames : str
            Fields of the scenarios to group. e.g., 'function' or 'fault'

        Returns
        -------
        groups : list
            List of tuples corresponding to the groups
        """
        groups = list(set([tuple([get_var(v, groupname) for groupname in groupnames])
                           for v in self.scenarios()]))
        return groups

    def get_scen_groups(self, *groupnames):
        """
        Get all groups of scenarios grouped by *groupnames.

        Parameters
        ----------
        *groupnames : str
            Fields of the underlying scenarios in self.scenarios()

        Returns
        -------
        scen_groups : dict
            Dict of scenarios
        """
        groups = self.group_scens(*groupnames)
        return self.get_groups_scens(groupnames, groups)

    def num_scenarios(self):
        """Get the number of scenarios in the sample."""
        return len(self.scenarios())

    def named_scenarios(self):
        """Get dict of scenarios by name."""
        return {scen.name: scen for scen in self.scenarios()}

    def scen_names(self):
        """Get list of scen names."""
        return [s.name for s in self.scenarios()]


class FaultSample(BaseSample):
    """
    Defines a sample of a given faultdomain.

    Parameters
    ----------
    faultdomain: FaultDomain
        Domain of faults to sample from
    phasemap: PhaseMap, (optional)
        Phases of operation to sample over.

    Attributes
    ----------
    _scenarios : list
        List of scenarios to sample.
    _times : set
        Set of times where the scenarios will occur

    Examples
    --------
    >>> exfs = FaultSample(exfd)
    >>> exfs.add_fault_times([1, 2])
    >>> exfs
    FaultSample of scenarios: 
     - ex_fxn_no_charge_t1
     - ex_fxn_no_charge_t2
     - ex_fxn_short_t1
     - ex_fxn_short_t2
     - ex_fxn2_no_charge_t1
     - ex_fxn2_no_charge_t2
     - ex_fxn2_short_t1
     - ex_fxn2_short_t2
    """

    def __init__(self, faultdomain, phasemap={}, def_mdl_phasemap=True):
        self.faultdomain = faultdomain
        if not phasemap and def_mdl_phasemap:
            phasemap = PhaseMap(faultdomain.mdl.sp.phases)
        self.phasemap = phasemap
        self._scenarios = []
        self._times = set()

    def __repr__(self):
        scens = self.scen_names()
        tot = len(scens)
        if tot > 10:
            scens = scens[0:10]+["... (" + str(tot) + " total)"]
        rep = "FaultSample of scenarios: " + "\n - " + "\n - ".join(scens)
        return rep

    def prune_scenarios(self, scen_var='rate', comparator=np.greater, value=0.0):
        """
        Prune scenarios from the FaultSample.

        Parameters
        ----------
        scen_var : str, optional
            Variable to prune. The default is 'rate'.
        comparator : method, optional
            Numpy method. The default is np.greater_equal.
        value : float, optional
            Value to compare against. The default is 0.0.
        """
        self._scenarios = [scen for scen in self._scenarios
                           if comparator(get_var(scen, scen_var), value)]

    def times(self):
        """Get all sampled times."""
        return list(self._times)

    def scenarios(self):
        """Get all sampled scenarios."""
        return [*self._scenarios]

    def add_single_fault_scenario(self, faulttup, time, weight=1.0):
        """
        Add a single fault scenario to the list of scenarios.

        Parameters
        ----------
        faulttup : tuple
            Fault to add ('blockname', 'faultname').
        time : float
            Time of the fault scenario.
        weight : float, optional
            Weighting factor for the scenario rate. The default is 1.0.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> mdl = Drone()
        >>> fd = FaultDomain(mdl)
        >>> fd.add_fault("affect_dof", "rf_propwarp")
        >>> fs = FaultSample(fd, phasemap=PhaseMap({"on": [0, 2], "off": [3, 5]}))
        >>> fs.add_single_fault_scenario(("affect_dof", "rf_propwarp"), 5)
        >>> fs
        FaultSample of scenarios: 
         - affect_dof_rf_propwarp_t5
        """
        self._times.add(time)
        scen = SingleFaultScenario.from_fault(faulttup, time, mdl=self.faultdomain.mdl,
                                              weight=weight, phasemap=self.phasemap)
        self._scenarios.append(scen)

    def add_joint_fault_scenario(self, faulttups, time, weight=1.0, baserate='ind',
                                 p_cond=1.0):
        """
        Add a single fault scenario to the list of scenarios.

        Parameters
        ----------
        faulttups : tuple
            Faults to add (('blockname', 'faultname'), ('blockname2', 'faultname2')).
        time : float
            Time of the fault scenario.
        weight : float, optional
            Weighting factor for the scenario rate. The default is 1.0.
        baserate : str/tuple
            Fault (fxn, mode) to get base rate for the scenario from (for joint faults).
            Default is 'ind' which calculates the rate as independent (rate1*rate2*...).
            Can also be 'max', which uses the max fault likelihood.
        p_cond : float
            Conditional fault probability for joint fault modes. Used if not using
            independent base rate assumptions to calculate. Default is 1.0.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> fd = FaultDomain(Drone())
        >>> fd.add_fault("affect_dof", "rf_propwarp")
        >>> fd.add_fault("affect_dof", "lf_propwarp")
        >>> fs = FaultSample(fd, phasemap=PhaseMap({"on": [0, 2], "off": [3, 5]}))
        >>> fs.add_joint_fault_scenario((("affect_dof", "rf_propwarp"),("affect_dof", "lf_propwarp")), 5)
        >>> fs
        FaultSample of scenarios: 
         - affect_dof_rf_propwarp__affect_dof_lf_propwarp_t5
        >>> fs.scenarios()[0].sequence[5].faults
        {'affect_dof': ['rf_propwarp', 'lf_propwarp']}
        >>> fs.add_single_fault_scenario(("affect_dof", "rf_propwarp"), 5)
        >>> fs.add_single_fault_scenario(("affect_dof", "lf_propwarp"), 5)
        >>> fs.scenarios()[0].rate == fs.scenarios()[1].rate*fs.scenarios()[2].rate
        True
        """
        self._times.add(time)
        scen = JointFaultScenario.from_faults(faulttups, time, mdl=self.faultdomain.mdl,
                                              weight=weight, baserate=baserate,
                                              p_cond=p_cond)
        self._scenarios.append(scen)

    def add_fault_times(self, times, weights=[], n_joint=1, **joint_kwargs):
        """
        Add all single-fault scenarios to the list of scenarios at the given times.

        Parameters
        ----------
        times : list
            List of times.
        weights : list, optional
            Weight factors corresponding to the times The default is [].
        n_joint : int
            Number of joint fault modes.
        **joint_kwargs : kwargs
            baserate and p_cond arguments to add_joint_fault_scenario.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> mdl = Drone()
        >>> fd = FaultDomain(mdl)
        >>> fd.add_fault("affect_dof", "rf_propwarp")
        >>> fs = FaultSample(fd, phasemap=PhaseMap({"on": [0, 2], "off": [3, 5]}))
        >>> fs.add_fault_times([1,2,3])
        >>> fs
        FaultSample of scenarios: 
         - affect_dof_rf_propwarp_t1
         - affect_dof_rf_propwarp_t2
         - affect_dof_rf_propwarp_t3
         >>> fd.add_fault("affect_dof", "lf_propwarp")
         >>> fd.add_fault("affect_dof", "rr_propwarp")
         >>> fs = FaultSample(fd)
         >>> fs.add_fault_times([5], n_joint=2)
         >>> fs
         FaultSample of scenarios: 
          - affect_dof_rf_propwarp__affect_dof_lf_propwarp_t5
          - affect_dof_rf_propwarp__affect_dof_rr_propwarp_t5
          - affect_dof_lf_propwarp__affect_dof_rr_propwarp_t5
         >>> fs = FaultSample(fd)
         >>> fs.add_fault_times([5], n_joint=3)
         >>> fs
         FaultSample of scenarios: 
          - affect_dof_rf_propwarp__affect_dof_lf_propwarp__affect_dof_rr_propwarp_t5
        """
        jointfaults = itertools.combinations(self.faultdomain.faults, n_joint)
        for faulttups in jointfaults:
            for i, time in enumerate(times):
                if weights:
                    weight = weights[i]
                elif self.phasemap:
                    phase_samples = self.phasemap.calc_samples_in_phases(*times)
                    phase = self.phasemap.find_base_phase(time)
                    weight = 1/phase_samples[phase]
                else:
                    weight = 1.0
                if n_joint == 1:
                    self.add_single_fault_scenario(faulttups[0], time, weight=weight)
                else:
                    self.add_joint_fault_scenario(faulttups, time, **joint_kwargs)

    def add_fault_phases(self, *phases_to_sample, method='even', args=(1,),
                         phase_methods={}, phase_args={},
                         n_joint=1, **joint_kwargs):
        """
        Sample scenarios in the given phases using a set sampling method.

        Parameters
        ----------
        *phases_to_sample : str
            Names of phases to sample. If no
        method : str, optional
            'even', 'quad', 'all', which selects whether to use sample_times_even or
            sample_times_quad, or gets all times, respectively. The default is 'even'.
        args : tuple, optional
            Arguments to the sampling method. The default is (1,).
        phase_methods : dict, optional
            Method ('even' or 'quad') to use of individual phases (if not default).
            The default is {}.
        phase_args : dict, optional
            Method args to use for individual phases (if not default).
            The default is {}.
        n_joint : int
            Number of joint fault modes to include in sample.
        **joint_kwargs : kwargs
            baserate and p_cond arguments to add_joint_fault_scenario.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> mdl = Drone()
        >>> fd = FaultDomain(mdl)
        >>> fd.add_fault("affect_dof", "rf_propwarp")
        >>> fs = FaultSample(fd, phasemap=PhaseMap({"on": [0, 2], "off": [3, 5]}))
        >>> fs.add_fault_phases("off")
        >>> fs
        FaultSample of scenarios: 
         - affect_dof_rf_propwarp_t4p0
        """
        if self.phasemap:
            phasetimes = self.phasemap.get_sample_times(*phases_to_sample)
        else:
            interval = [0, self.faultdomain.mdl.sp.times[-1]]
            tstep = self.faultdomain.mdl.sp.dt
            phasetimes = {'phase': gen_timerange(interval[0], interval[-1], tstep)}

        for phase, times in phasetimes.items():
            loc_method = phase_methods.get(phase, method)
            loc_args = phase_args.get(phase, args)
            if loc_method == 'even':
                sampletimes, weights = sample_times_even(times, *loc_args,
                                                         dt=self.faultdomain.mdl.sp.dt)
            elif loc_method == 'quad':
                sampletimes, weights = sample_times_quad(times, *loc_args)
            elif loc_method == 'all':
                sampletimes = times
                weights = [1/len(sampletimes) for i in sampletimes]
            else:
                raise Exception("Invalid method: "+loc_method)
            self.add_fault_times(sampletimes, weights, n_joint=n_joint, **joint_kwargs)


exfs = FaultSample(exfd)
exfs.add_fault_times([1, 2])


class JointFaultSample(FaultSample):
    """FaultSample for faults in multiple faultdomains and phasemaps."""

    def __init__(self, *faultdomains, phasemaps=[], def_mdl_phasemap=True):
        self.faultdomain = FaultDomain(faultdomains[0].mdl)
        for faultdomain in faultdomains:
            self.faultdomain.faults.update(faultdomain.faults)
        if phasemaps:
            self.phasemap = join_phasemaps(phasemaps)
        elif def_mdl_phasemap:
            self.phasemap = PhaseMap(faultdomains[0].mdl.sp.phases)


class SampleApproach(BaseSample):
    """
    Class for defining an agglomeration of fault samples accross an entire model.

    Attributes
    ----------
    mdl : Simulable
        Model
    phasemaps : dict
        Dict of phasemaps {'phasemapname': PhaseMap} which map to the various functions
        of the model.
    faultdomains : dict
        Dict of the faultdomains to sample {'domainname': FaultDomain}
    faultsamples: dict
        Dict of the FaultSamples making up the approach {'samplename': FaultSample}
    """

    def __init__(self, mdl, phasemaps={}, def_mdl_phasemap=True):
        self.mdl = mdl
        if def_mdl_phasemap:
            phasemaps['mdl'] = PhaseMap(self.mdl.sp.phases)
        self.phasemaps = phasemaps
        self.faultdomains = {}
        self.faultsamples = {}

    def __repr__(self):
        fd_str = ", ".join(self.faultdomains)
        fs_str = ", ".join(self.faultsamples)
        rep = "SampleApproach for " + self.mdl.name + " with:" +\
            " \n faultdomains: " + fd_str +\
            "\n faultsamples: " + fs_str
        return rep

    def add_faultdomain(self, name, add_method, *args, **kwargs):
        """
        Instantiate and associates a FaultDomain with the SampleApproach.

        Parameters
        ----------
        name : str
            Name to give the faultdomain (in faultdomains).
        add_method : str
            Method to add faults to the faultdomain with
            (e.g., to call Faultdomain.add_all, use "all")
        *args : args
            Arguments to add_method.
        **kwargs : kwargs
            Keyword arguments to add_method

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> s = SampleApproach(Drone())
        >>> s.add_faultdomain("all_faults", "all")
        >>> s
        SampleApproach for drone with: 
         faultdomains: all_faults
         faultsamples: 
        >>> s.faultdomains['all_faults']
        FaultDomain with faults:
         -('manage_health', 'lostfunction')
         -('store_ee', 'nocharge')
         -('store_ee', 'lowcharge')
         -('store_ee', 's1p1_short')
         -('store_ee', 's1p1_degr')
         -('store_ee', 's1p1_break')
         -('store_ee', 's1p1_nocharge')
         -('store_ee', 's1p1_lowcharge')
         -('dist_ee', 'short')
         -('dist_ee', 'degr')
         -...more
        """
        faultdomain = FaultDomain(self.mdl)
        meth = getattr(faultdomain, 'add_'+add_method)
        meth(*args, **kwargs)
        self.faultdomains[name] = faultdomain

    def add_faultsample(self, name, add_method, faultdomains, *args, phasemap={},
                        **kwargs):
        """
        Instantiate and associate a FaultSample with the SampleApproach.

        Parameters
        ----------
        name : str
            Name for the faultsample.
        add_method : str
            Method to add scenarios to the FaultSample with.
            (e.g., to call Faultdomain.add_fault_times, use "fault_times")
        faultdomain : str or list
            Name of faultdomain to sample from (must be in SampleApproach already).
        *args : args
            args to add_method.
        phasemap : str/PhaseMap/dict/tuple, optional
            Phasemap to instantiate the FaultSample with. If a dict/tuple is provided,
            uses a PhaseMap with the dict/tuple as phases. The default is {}.
            If a list, passes to JointFaultSample
        **kwargs : kwargs
            add_method kwargs.

        Examples
        --------
        >>> from examples.multirotor.drone_mdl_rural import Drone
        >>> s = SampleApproach(Drone())
        >>> s.add_faultdomain("all_faults", "all")
        >>> s.add_faultsample("start_times", "fault_times", "all_faults", [1,3,4])
        >>> s
        SampleApproach for drone with: 
         faultdomains: all_faults
         faultsamples: start_times
        >>> s.faultsamples['start_times']
        FaultSample of scenarios: 
         - manage_health_lostfunction_t1
         - manage_health_lostfunction_t3
         - manage_health_lostfunction_t4
         - store_ee_nocharge_t1
         - store_ee_nocharge_t3
         - store_ee_nocharge_t4
         - store_ee_lowcharge_t1
         - store_ee_lowcharge_t3
         - store_ee_lowcharge_t4
         - store_ee_s1p1_short_t1
         - ... (171 total)
        """
        if type(phasemap) is str:
            phasemap = self.phasemaps[phasemap]
        elif isinstance(phasemap, PhaseMap) or not phasemap:
            phasemap = phasemap
        elif isinstance(phasemap, dict) or isinstance(phasemap, tuple):
            phasemap = PhaseMap(phasemap)
        elif isinstance(phasemap, list):
            phasemap = [self.phasemaps[ph] for ph in phasemap]
        else:
            raise Exception("Invalid arg for phasemap: "+str(phasemap))
        if type(faultdomains) is list:
            if len(faultdomains) > 1:
                faultsample = JointFaultSample(faultdomains, phasemap)
            else:
                faultsample = FaultSample(self.faultdomains[faultdomains[0]], phasemap)
        else:
            faultsample = FaultSample(self.faultdomains[faultdomains], phasemap)
        meth = getattr(faultsample, 'add_'+add_method)
        meth(*args, **kwargs)
        self.faultsamples[name] = faultsample

    def add_faultdomains(self, **faultdomains):
        """
        Add dict of faultdomains to the SampleApproach.

        Parameters
        ----------
        **faultdomains : tuple
            FaultDomains to add to the SampleApproach and their arguments.
            Has structure::
            {'fd_name': (*args, **kwargs)}
            where args and kwargs are arguments/kwargs to SampleApproach.add_faultdomain
            (after name).
        """
        for fd in faultdomains:
            self.add_faultdomain(fd, *faultdomains[fd][0], **faultdomains[fd][1])

    def add_faultsamples(self, **faultsamples):
        """
        Add dict of faultsamples to the SampleApproach.

        Parameters
        ----------
        **faultsamples : tuple
            FaultSamples to add to othe SampleApproach and their arguments.
            Has structure::
            fs_name = (*args, **kwargs)}
            where args and kwargs are arguments/kwargs to SampleApproach.add_faultsample
            (after name).
        """
        for fs in faultsamples:
            self.add_faultsample(fs, *faultsamples[fs][0], **faultsamples[fs][1])

    def times(self):
        """Get all sampletimes covered by the SampleApproach."""
        return list(set(np.concatenate([list(samp.times())
                                        for samp in self.faultsamples.values()])))

    def scenarios(self):
        """Get all scenarios in the SampleApproach."""
        return [scen for faultsample in self.faultsamples.values()
                for scen in faultsample.scenarios()]

    def prune_scenarios(self, scen_var='rate', comparator=np.greater, value=0.0):
        """
        Prune scenarios from the FaultSample.

        Parameters
        ----------
        scen_var : str, optional
            Variable to prune. The default is 'rate'.
        comparator : method, optional
            Numpy method. The default is np.greater_equal.
        value : float, optional
            Value to compare against. The default is 0.0.
        """
        for fs in self.faultsamples.values():
            fs.prune_scenarios(scen_var=scen_var, comparator=comparator, value=value)


class ParameterSample(BaseSample):
    """
    Class for sampling parameters and other immutable model attributes.

    ...

    Attributes
    ----------
    seed : int
        Seed for random sampling.
    seedsequence : np.ramdom.SeedSequence
        Corresponding seed sequence
    sp : dict
        Non-default simparam arguments
    paramdomain: ParamDomain
        Parameter domain object to sample
    """

    def __init__(self, paramdomain=ParameterDomain(Parameter), seed=None, sp={}):
        self.seed = seed
        self.seedsequence = np.random.SeedSequence(seed)
        self.sp = sp
        self.paramdomain = paramdomain
        self._scenarios = []

    def __repr__(self):
        scens = self.scen_names()
        tot = len(scens)
        if tot > 10:
            scens = scens[0:10]+["... (" + str(tot) + " total)"]
        rep = "ParameterSample of scenarios:" + "\n - " + "\n - ".join(scens)
        return rep

    def scenarios(self):
        """Return list of scenarios that make up the ParameterSample."""
        return [*self._scenarios]

    def add_variable_scenario(self, *x, seed=False, sp={}, weight=1.0, name='var',
                              inputparams={}):
        """
        Add a scenario to the ParamSample.

        Parameters
        ----------
        *x : vars
            Values of the variables defined in the parameter domain.
        seed : int, optional
            Seed to include in the ParameterScenario. The default is False.
        sp : dict, optional
            SimParam arguments to the ParameterScenario. The default is {}.
        weight : float, optional
            Weight (probability) allocated to the scenario. The default is 1.0.
        name : str, optional
            Name to assign the scenario. The default is 'param'.
        inputparams : dict, optional
            Input parameters (if different from Parameter input).

        Examples
        --------
        >>> ex_ps = ParameterSample(expd)
        >>> ex_ps.add_variable_scenario(1, 2)
        >>> ex_ps
        ParameterSample of scenarios:
         - var_0
        >>> ex_ps.scenarios()[0].p['z']
        20
        >>> ex_ps.scenarios()[0].p['y']
        1
        >>> ex_ps.scenarios()[0].p['x']
        2
        """
        param_args = self.paramdomain.get_param_kwargs(*x)
        if seed:
            r = {'seed': seed}
        elif self.seed:
            r = {'seed': self.seed}
        else:
            r = {}
        if not sp:
            sp = self.sp
        name = name+"_"+str(len(self.scenarios()))
        if not inputparams:
            inputparams = {i: x[i] for i in range(len(x))}

        scen = ParameterScenario(p=param_args,
                                 r=r,
                                 sp=sp,
                                 prob=weight,
                                 name=name,
                                 inputparams=inputparams)
        self._scenarios.append(scen)

    def add_variable_replicates(self, x_combos, replicates=1, seed_comb='shared',
                                name='var', weight=1.0):
        """
        Add replicates for the given cominations of variables.

        Parameters
        ----------
        x_combos : list
            List of combinations of variable values. [x_1, x_2...]. If empty, the
            default values of x are used.
        replicates : int, optional
            Number of replicates to add of the variable. The default is 1.
        seed_comb : str, optional
            How to combine seeds ('shared' or 'independent'). 'shared' uses the same
            seed for each value while 'independent' uses different seeds over all
            variable values. The default is 'shared'.
        name : str, optional
            Name prefix for the set of replicates. The default is 'var'.
        weight : float, optional
            Total weight for the replicates (i.e., total probability to divide between
            samples). The default is 1.0.

        Examples
        --------
        >>> ex_ps = ParameterSample(expd)
        >>> ex_ps.add_variable_replicates([[1,1], [2,2]])
        >>> ex_ps
        ParameterSample of scenarios:
         - rep0_var_0
         - rep0_var_1
        >>> scen0 =ex_ps.scenarios()[0]
        >>> scen0.prob
        0.5
        >>> scen0.p['x']
        1
        >>> scen0.p['y']
        1
        >>> scen0.inputparams
        {0: 1, 1: 1}
        >>> scen1 =ex_ps.scenarios()[1]
        >>> scen1.prob
        0.5
        >>> scen1.p['x']
        2
        >>> scen1.p['y']
        2
        >>> scen1.inputparams
        {0: 2, 1: 2}
        """
        if len(x_combos) == 0:
            x_combos = [self.paramdomain.get_x_defaults()]
        n_scens = replicates * len(x_combos)
        weight = weight/n_scens
        if seed_comb == 'shared' and replicates > 1:
            seeds = self.seedsequence.generate_state(replicates)
        elif seed_comb == 'independent' and replicates > 1:
            seeds = self.seedsequence.generate_state(n_scens)
        else:
            seeds = []
        scen_num = 0
        for x_combo in x_combos:
            for i in range(replicates):
                if replicates > 1 and (seed_comb == 'shared'):
                    seed = seeds[i]
                elif replicates > 1 and (seed_comb == 'independent'):
                    seed = seeds[scen_num]
                else:
                    seed = False
                loc_name = "rep" + str(i) + "_" + name
                self.add_variable_scenario(*x_combo, name=loc_name,
                                           weight=weight, seed=seed)
                scen_num += 1

    def add_variable_ranges(self, combmethod='product', comb_kwargs={},
                            n_samp=False, name='range', **rep_kwargs):
        """
        Combine and add a set of ranges to the ParamSample.

        Parameters
        ----------
        combmethod : str, optional
            Name of the combination method ('product', 'orthogonal', 'random') for the
            class. The default is 'product'.
        comb_kwargs : dict, optional
            Keyword arguments to the self.combine_methodname. The default is {}.
        n_samp : int, optional
            Number of values to randomly sample from the set of combinations.
            The default is False, which samples all.
        name : str, optional
            Name for the range. The default is 'range'.
        **rep_kwargs : kwargs
            kwargs to add_variable_replicates (seed, weight, etc.).

        Examples
        --------
        >>> ex_ps = ParameterSample(expd)
        >>> ex_ps.add_variable_ranges()
        >>> ex_ps
        ParameterSample of scenarios:
         - rep0_range_0
         - rep0_range_1
         - rep0_range_2
         - rep0_range_3
         - rep0_range_4
         - rep0_range_5
         - rep0_range_6
         - rep0_range_7
         - rep0_range_8
         - rep0_range_9
         - ... (44 total)

        >>> ex_ps2 = ParameterSample(expd)
        >>> ex_ps2.add_variable_ranges(n_samp=5)
        >>> ex_ps2
        ParameterSample of scenarios:
         - rep0_range_0
         - rep0_range_1
         - rep0_range_2
         - rep0_range_3
         - rep0_range_4
        """
        if combmethod == 'product':
            x_combos = self.combine_product(**comb_kwargs)
        elif combmethod == 'orthogonal':
            x_combos = self.combine_orthogonal(**comb_kwargs)
        elif combmethod == 'random':
            x_combos = self.combine_random(**comb_kwargs)
        else:
            raise Exception("Invalid method: " + combmethod)
        if n_samp:
            rng = np.random.default_rng(self.seed)
            x_combos = rng.choice(x_combos, n_samp)

        self.add_variable_replicates(x_combos, name=name, **rep_kwargs)

    def combine_product(self, resolution=1, resolutions={}):
        """
        Find all combinations of possible range values.

        Parameters
        ----------
        resolution : float, optional
            Default resolution for the ranges. The default is 1.
        resolutions : dict, optional
            Resolution for each individual range if not default. The default is {}.

        Returns
        -------
        x_combos : list
            List of combinations of x.

        Examples
        --------
        >>> ex_ps = ParameterSample(expd, seed=1)
        >>> ex_ps.combine_product()
        [(1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3), (1.0, 4), (1.0, 5), (1.0, 6), (1.0, 7), (1.0, 8), (1.0, 9), (1.0, 10), (2.0, 0), (2.0, 1), (2.0, 2), (2.0, 3), (2.0, 4), (2.0, 5), (2.0, 6), (2.0, 7), (2.0, 8), (2.0, 9), (2.0, 10), (3.0, 0), (3.0, 1), (3.0, 2), (3.0, 3), (3.0, 4), (3.0, 5), (3.0, 6), (3.0, 7), (3.0, 8), (3.0, 9), (3.0, 10), (4.0, 0), (4.0, 1), (4.0, 2), (4.0, 3), (4.0, 4), (4.0, 5), (4.0, 6), (4.0, 7), (4.0, 8), (4.0, 9), (4.0, 10)]
        """
        var_iters = self.paramdomain.get_var_iters(resolution, resolutions=resolutions)
        x_combos = [*itertools.product(*var_iters.values())]
        return x_combos

    def combine_orthogonal(self, resolution=1, resolutions={}):
        """
        Create combinations of orthogonal arrays from the parameter domain.

        Parameters
        ----------
        resolution : float, optional
            Default resolution for the ranges. The default is 1.
        resolutions : dict, optional
            Resolution for each individual range if not default. The default is {}.

        Returns
        -------
        x_combos : list
            List of combinations of x.

        Examples
        --------
        >>> ex_ps = ParameterSample(expd, seed=1)
        >>> ex_ps.combine_orthogonal()
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [3.0, 0], [3.0, 1], [3.0, 2], [3.0, 3], [3.0, 4], [3.0, 5], [3.0, 6], [3.0, 7], [3.0, 8], [3.0, 9], [3.0, 10]]
        """
        var_iters = self.paramdomain.get_var_iters(resolution, resolutions=resolutions)
        x_def = self.paramdomain.get_x_defaults()
        x_combos = combine_orthogonal(x_def, var_iters.values())
        return x_combos

    def combine_random(self, num_combos=1):
        """
        Create combinations of random variables from the parameter domain.

        Parameters
        ----------
        num_combos : int, optional
            Number of combinations. The default is 1.

        Returns
        -------
        x_combos : list
            List of list of combinations.

        Examples
        --------
        >>> ex_ps = ParameterSample(expd, seed=1)
        >>> ex_ps.combine_random(1)
        [[2.0, 9.504636963259353]]
        """
        ranges = self.paramdomain.variables.values()
        x_combos = combine_random(ranges, seed=self.seed, num_combos=num_combos)
        return x_combos


def combine_random(ranges, seed=None, num_combos=1):
    """
    Create random lists from the given ranges.

    Parameters
    ----------
    ranges : list
        List of potential values (sets) or limits (tuples).
    seed : seed, optional
        Seed to sample with. The default is 0.
    num_combos : int, optional
        Number of combinations to sample. The default is 1.

    Returns
    -------
    x_combos : list
        List of potential combinations.

    Examples
    --------
    >>> x_comb = combine_random([{1,2,3}, (0, 1)])
    >>> x_comb[0][0] in {1, 2, 3}
    True
    >>> 0 <= x_comb[0][1] <= 1
    True
    """
    rng = np.random.default_rng(seed)
    x_combos = []
    for i in range(num_combos):
        x = []
        for ran in ranges:
            if type(ran) is set:
                x.append(rng.choice([*ran]))
            elif type(ran) is tuple:
                x.append(rng.uniform(ran[0], ran[-1]))
            else:
                raise Exception("invalid range: " + str(ran))
        x_combos.append(x)
    return x_combos


class ParameterResultSample(ParameterSample):
    """
    Class for sampling parameters from Results.

    Parameters
    ----------
    vars : tuple
        Variables to sample from the Result.
    res_to_sample : Result
        Subset of input result associated with the variables/groups to sample.

    Examples
    --------
    >>> from fmdtools.define.container.parameter import ExampleParameter
    >>> expd = ParameterDomain(ExampleParameter)
    >>> expd.add_variables("y")
    >>> from fmdtools.analyze.result import Result
    >>> res = Result({'scen1.a.y': 1.0, 'scen2.a.y': 2.0})
    >>> prs = ParameterResultSample(res, 'a.y', paramdomain=expd)
    >>> prs.get_param_ins(rep=0)
    [1.0]
    >>> prs.get_param_ins(rep=1)
    [2.0]

    # prs.add_res_scenario can be used to add a given scenario from the result.
    >>> prs.add_res_scenario(rep=0)
    >>> prs.scenarios()
    [ParameterScenario(sequence={}, times=(), p={'y': 1.0}, r={}, sp={}, prob=1.0, inputparams={'comp_group': 'default', 'rep': 0}, rangeid='', name='res_0')]

    # prs.add_res_reps can be used to add replicate scenarios from the results.
    >>> prs.add_res_reps(reps='all')
    >>> prs
    ParameterSample of scenarios:
     - res_0
     - res_1
     - res_2
    """

    def __init__(self, res, *varnames, comp_groups={}, **kwargs):
        """
        Initialize the ParameterResultSample.

        Parameters
        ----------
        res : Result
            Result to sample from (usually, with multiple scenarios).
        *varnames : str
            Names of values to sample (e.g., 's.x', 's.y') in the order of the
            parameters in the input ParameterDomain
        comp_groups : dict, optional
            Groups to put the scenarios in (to sample differently). The default is {}.
        **kwargs : kwargs
            Input arguments to ParameterSample (e.g., the ParameterDomain).
        """
        ParameterSample.__init__(self, **kwargs)
        self.vars = varnames
        self.res_to_sample = res.get_comp_groups(*varnames, **comp_groups)

    def _get_repname(self, comp_group, rep):
        """
        Get the name of the nth replicate scenario in the comparison group.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group.
        rep : str or int
            If str, the name of the replicate. If int, n, where n is the nth replicate
            to choose.

        Returns
        -------
        rep : str
            Name of the replicate.
        """
        if type(rep) is int:
            return [*self.res_to_sample[comp_group].nest(1).keys()][rep]
        elif type(rep) is str:
            return rep
        else:
            raise Exception("Invalid option for rep: "+str(rep))

    def _get_param(self, comp_group, rep, var, t=None):
        """
        Get the value of the parameter variable in the given group, replicate and time.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        rep : str
            Name of the replicate scenario to get the value from.
        var : str
            Name of the variable to get.
        t : int/None, optional
            If a History, the time to get. The default is None.

        Returns
        -------
        param : value
            Variable value (to be used as the parameter).
        """
        if t is None:
            return self.res_to_sample[comp_group].get(rep).get(var)
        elif is_numeric(t):
            return self.res_to_sample[comp_group].get(rep).get(var)[t]
        else:
            return Exception("Invalid option for t: "+str(t))

    def _get_reps(self, comp_group, reps):
        """
        Get the replicates to sample in the comparison group.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group.
        reps : int/str/list
            If int, the number of replicates to sample. If a list, the names of the
            replicate scenarios to sample. If 'all', all replicates in the group.

        Returns
        -------
        reps_to_sample : list
            List of names of replicates to sample.
        """
        reps_possible = [*self.res_to_sample[comp_group].nest().keys()]
        if reps == 'all':
            reps_to_sample = reps_possible
        elif type(reps) is int:
            reps_to_sample = reps_possible[:reps]
        elif type(reps) is list:
            reps_to_sample = reps
        else:
            raise Exception("Invalid option for reps: "+str(reps))
        return reps_to_sample

    def get_param_ins(self, comp_group='default', rep=1, t=None):
        """
        Get the parameter values at the given Result scenario.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        rep : str
            Name of the replicate scenario to get the value from.
        t : int/None, optional
            If a History, the time to get. The default is None.

        Returns
        -------
        param_list : list
            List of variable values (self.vars) for the given group and replicate.
        """
        param_list = []
        rep = self._get_repname(comp_group, rep)
        for var in self.vars:
            param_list.append(self._get_param(comp_group, rep, var, t))
        return param_list

    def add_res_scenario(self, comp_group='default', rep=1, t=None,
                         name='res', **kwargs):
        """
        Add a scenario to the ParameterResultSample from the Result.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        rep : str
            Name of the replicate scenario to get the value from.
        t : int/None, optional
            If a History, the time to get. The default is None.
        name : str, optional
            Name prefix for the scenario. The default is 'res'.
        **kwargs : kwargs
            Keyword arguments to ParameterSample.add_variable_scenario.
        """
        param_list = self.get_param_ins(comp_group, rep, t)
        if is_numeric(t):
            inputparams = {'comp_group': comp_group, 'rep': rep, 't': t}
        else:
            inputparams = {'comp_group': comp_group, 'rep': rep}
        self.add_variable_scenario(*param_list, name=name, inputparams=inputparams,
                                   **kwargs)

    def add_res_reps(self, comp_group='default', reps='all', name='res', ind_seeds=True,
                     **kwargs):
        """
        Add multiple scenario replicates from the Restul to the ParameterResultSample.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        reps : int/str/list
            If int, the number of replicates to sample. If a list, the names of the
            replicate scenarios to sample. If 'all', all replicates in the group.
        name : str, optional
            Name prefix for the scenarios. The default is 'res'.
        ind_seeds : bool
            Whether to give replicates independent seeds. The default is True.
        **kwargs : kwargs
            Keyword arguments to ParameterScenario.add_res_scenario.
        """
        reps_to_sample = self._get_reps(comp_group, reps)
        if ind_seeds:
            seeds = self.seedsequence.generate_state(len(reps_to_sample))
        for i, rep in enumerate(reps_to_sample):
            if ind_seeds:
                self.add_res_scenario(comp_group, rep, name=name, seed=seeds[i],
                                      **kwargs)
            else:
                self.add_res_scenario(comp_group, rep, name=name, **kwargs)


class ParameterHistSample(ParameterResultSample):
    """
    Class for sampling parameters from Histories. Extends ParameterResultSample.

    Examples
    --------
    >>> from fmdtools.define.container.parameter import ExampleParameter
    >>> expd = ParameterDomain(ExampleParameter)
    >>> expd.add_variables("y")
    >>> from fmdtools.analyze.history import History
    >>> hist = History({'scen1.a.y': [1.0, 2.0, 3.0], 'scen1.time':[0,2,3], 'scen2.a.y': [10.0, 20.0, 30.0], 'scen2.time':[0,2,3]})
    >>> phs = ParameterHistSample(hist, 'a.y', paramdomain=expd)
    >>> phs.get_param_ins(rep=0, t=0)
    [1.0]
    >>> phs.get_param_ins(rep=1, t=1)
    [20.0]

    # Add scenarios to the sample
    >>> phs.add_hist_times('default', 'scen1')
    >>> [s.p for s in phs.scenarios()]
    [{'y': 1.0}, {'y': 2.0}, {'y': 3.0}]
    """

    def add_hist_scenario(self, comp_group='default', rep=1, t=1, name='hist',
                          **kwargs):
        """Add a scenario from a history (extends add_res_scenario)."""
        self.add_res_scenario(comp_group, rep, t, name, **kwargs)

    def _get_times(self, comp_group, rep, ts):
        """
        Get the available times to sample in a given comparison group and replicate.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        rep : str
            Name of the replicate scenario to get the value from.
        ts : int/list/tuple/'all'
            If 'all', returns all times. If int, returns a range of times with ts
            resolution. If a list (of ints), uses as ts_to_sample. IF a tuple, returns
            a range with the given input arguments.

        Returns
        -------
        ts_to_sample : list
            List of times to sample.
        """
        res = self.res_to_sample[comp_group].get(rep)
        if ts == 'all':
            ts_to_sample = [i for i, e in enumerate(res.time)]
        elif type(ts) is int:
            ts_to_sample = [*np.arange(0, len(res.time), ts)]
        elif type(ts) is list:
            ts_to_sample = ts
        elif type(ts) is tuple:
            ts_to_sample = [*np.arange(*ts)]
        else:
            raise Exception("Invalid option for ts: "+str(ts))
        return ts_to_sample

    def add_hist_times(self, comp_group, rep, ts='all', **kwargs):
        """
        Add scenarios in the scenario comp_group and replicate from the given times.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        rep : str
            Name of the replicate scenario to get the value from.
        ts : int/list/tuple/'all'
            If 'all', returns all times. If int, returns a range of times with ts
            resolution. If a list (of ints), uses as ts_to_sample. IF a tuple, returns
            a range with the given input arguments.
        **kwargs : kwargs
            Keyword arguments to .add_hist_scenario
        """
        rep = self._get_repname(comp_group, rep)
        ts_to_sample = self._get_times(comp_group, rep, ts)
        for t in ts_to_sample:
            self.add_hist_scenario(comp_group, rep, t, **kwargs)

    def add_hist_reps(self, comp_group, reps='all', ts='all', ind_seeds=True, **kwargs):
        """
        Add scenarios in the scenario comp_group over given replicates and times.

        Parameters
        ----------
        comp_group : str
            Name of the comparison group to get the value from.
        reps : int/str/list
            If int, the number of replicates to sample. If a list, the names of the
            replicate scenarios to sample. If 'all', all replicates in the group.
        ts : int/list/tuple/'all'
            If 'all', returns all times. If int, returns a range of times with ts
            resolution. If a list (of ints), uses as ts_to_sample. IF a tuple, returns
            a range with the given input arguments.
        ind_seeds : bool
            Whether to give replicates independent seeds. The default is True.
        **kwargs : kwargs
            Keyword arguments to .add_hist_scenario
        """
        reps_to_sample = self._get_reps(comp_group, reps)
        if ind_seeds:
            seeds = self.seedsequence.generate_state(len(reps_to_sample))
        for i, rep in enumerate(reps_to_sample):
            if ind_seeds:
                self.add_hist_times(comp_group, rep, ts, seed=seeds[i], **kwargs)
            else:
                self.add_hist_times(comp_group, rep, ts, **kwargs)

    def add_hist_groups(self, comp_groups=['default'], reps='all', ts='all', **kwargs):
        """
        Add scenarios in the comp_groups over given replicates and times.

        Parameters
        ----------
        comp_groups : list
            Names of the comparison group to get the values from.
        reps : int/str/list
            If int, the number of replicates to sample. If a list, the names of the
            replicate scenarios to sample. If 'all', all replicates in the group.
        ts : int/list/tuple/'all'
            If 'all', returns all times. If int, returns a range of times with ts
            resolution. If a list (of ints), uses as ts_to_sample. IF a tuple, returns
            a range with the given input arguments.
        **kwargs : kwargs
            Keyword arguments to .add_hist_scenario
        """
        for comp_group in comp_groups:
            self.add_hist_reps(comp_group, reps=reps, ts=ts, **kwargs)


def combine_orthogonal(defaults, ranges):
    """
    Create an orthogonal arrays (lists) over the given ranges.

    Parameters
    ----------
    defaults : tuple
        Default variables
    ranges : list
        List of lists of non-default values for each value

    Returns
    -------
    samples : list
        List of lists of orthogonal x.

    Examples
    --------
    >>> combine_orthogonal([1,1], [[0,2], [3,4]])
    [[0, 1], [2, 1], [1, 3], [1, 4]]
    """
    samples = []
    for i, ran in enumerate(ranges):
        for val in ran:
            samp = [*defaults]
            samp[i] = val
            samples.append(samp)
    return samples


exp_ps = ParameterSample(expd, seed=1)
exp_ps.add_variable_scenario(1, 2)
exp_ps.add_variable_ranges(replicates=10)


if __name__ == "__main__":
    from examples.multirotor.drone_mdl_rural import Drone
    mdl = Drone()
    fd = FaultDomain(mdl)
    fd.add_fault("affect_dof", "rf_propwarp")
    # fd.add_faults(("affect_dof", "rf_propwarp"), ("affect_dof", "lf_propwarp"))
    # fd.add_all_modes("propwarp")

    fs = FaultSample(fd, phasemap=PhaseMap({"on": [0, 2], "off": [3, 5]}))
    fs.add_single_fault_scenario(("affect_dof", "rf_propwarp"), 5)
    fs.add_fault_times([1,2,3])
    fs.get_scen_groups("function")
    fs.get_scen_groups("phase")

    s = SampleApproach(mdl)
    s.add_faultdomain("all_faults", "all")
    s.add_faultsample("start_times", "fault_times", "all_faults", [1,3,4])
    s.get_scen_groups("phase")

    import doctest
    doctest.testmod(verbose=True)
