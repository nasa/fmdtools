#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions and Classes to enable optimization and search of model states and parameters.

Classes:

- :class:`ProblemArchitecture`:  Enables the representation of combined joint
  optimization problems
- :class:`DynamicInterface`:  Creates an interface for model simulations for dynamic
  optimization of a single sim
- :class:`SimpleProblem`: Creates an interface for simple optimazation problem without
  any model constructs
- :class:`DisturbanceProblem`: Enables optimizing disturbances that occur at a set time
- :class:`SingleFaultScenarioProblem`: Enables optimizing the time of a given fault
  scenario
- :class:`ResponseCoords`: Enables visualiation of 2-d search space.

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

from fmdtools.define.base import t_key, unpack_x, filter_kwargs
from fmdtools.define.object.base import init_obj
from fmdtools.define.object.coords import BaseCoords
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.scenario import Sequence, SingleFaultScenario, Scenario, ParameterScenario
from fmdtools.sim.sample import FaultDomain
from fmdtools.sim.propagate import Simulation, get_sim_call_kwargs, exec_sim
import fmdtools.sim.propagate as propagate
from fmdtools.analyze.common import setup_plot
from fmdtools.analyze.history import History, init_dicthist

import numpy as np
import networkx as nx
import time
import inspect
from recordclass import dataobject


class BaseObjCon(dataobject):
    """
    Base class for objectives and constraints.

    Fields
    ------
    name : str
        Name of the objective/constraint
    value : float
        Value of the objective/constraint
    """

    name: str = ''
    value: np.float64 = np.nan


class Objective(BaseObjCon):
    """
    Base class for objectives which derive from simulation results.

    Fields
    ------
    negative : bool
        Whether the objective is the negative of the value.
    """

    negative: bool = False

    def obj_from_value(self, value):
        """Get the (+ or 0) objective corresponding to value give self.negative."""
        if self.negative:
            value = - value
        else:
            value = value
        return value

    def update(self, value):
        """Update with given value."""
        self.value = self.obj_from_value(value)


class Constraint(Objective):
    """
    Base class for constraints which derive from results.

    ...

    Fields
    ------
    threshold : float
        Threshold for the constraint. Default is 0.0
    comparator : str
        Whether the constraint is 'greater', 'less', 'equal', or 'notequal'.
    or_equal : bool
        Whether comparator includes equality (e.g., greater or equalto) or not.
    satisfied : bool
        Whether the constraint is satisfied.
    """

    threshold: np.float64 = np.float64(0.0)
    comparator: str = 'greater'
    or_equal: bool = True
    satisfied: bool = True

    def con_from_value(self, value):
        """
        Get the constraint given the value of its variable given threshold.

        By default, constraints follow the form:
            g(x) = threshold - value > 0.0 for 'greater' constraints or
            g(x) = value - threshold > 0.0 for 'less' constraints.

        Parameters
        ----------
        value : float
            Variable value corresponding to the constraint

        Returns
        -------
        con : float
            Constraint function at value.
        """
        if self.comparator == 'greater':
            value = self.threshold - value
        elif self.comparator == 'less':
            value = value - self.threshold
        elif self.comparator == 'equal':
            value = abs(value - self.threshold)
        elif self.comparator == 'notequal':
            # 1 if true, else zero
            value = 1*(value != self.threshold)
        else:
            raise Exception("Invalid comparator: "+self.comparator)
        return self.obj_from_value(value)

    def update(self, value):
        """Update with given value."""
        self.value = self.con_from_value(value)
        if not self.negative:
            if self.or_equal:
                self.satisfied = self.value <= 0.0
            else:
                self.satisfied = self.value < 0.0
        else:
            if self.or_equal:
                self.satisfied = self.value >= 0.0
            else:
                self.satisfied = self.value > 0.0


class BaseProblem(object):
    """
    Base optimization problem.

    Attributes
    ----------
    variables : dict
        Variables being optimized.
    objectives : dict
        Objectives returned.
    constraints : dict
        Constraints returned.
    """

    def __init__(self, name='', **kwargs):
        self.name = name
        self.variables = {}
        self.objectives = {}
        self.constraints = {}
        self.iter_hist = History({"time": [],
                                  "iter": [],
                                  "variables": History(),
                                  "objectives": History(),
                                  "constraints": History()})
        self.init_problem(**kwargs)

    def add_variables(self, *variables):
        """
        Add variables to the problem.

        Parameters
        ----------
        *variables : str
            Names of variables to call in the problem.
        """
        self.variables.update({v: np.nan for v in variables})
        for v in variables:
            self.iter_hist['variables'][v] = []

    def init_problem(self, **kwargs):
        raise Exception(self.__class__.__name__+".init_problem not implemented.")

    def name_repr(self):
        """Single-line name representation."""
        return self.__class__.__name__

    def prob_repr(self):
        """Representation of the problem variables, objectives, constraints."""
        rep_str = ""
        var_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.variables.items()])
        if self.variables:
            rep_str += "\n"+"VARIABLES\n" + var_str
        obj_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v.value)
                                      for k, v in self.objectives.items()])
        if self.objectives:
            rep_str += "\n" + "OBJECTIVES\n" + obj_str
        con_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v.value)
                                      for k, v in self.constraints.items()])
        if self.constraints:
            rep_str += "\n" + "CONSTRAINTS\n" + con_str
        return rep_str

    def add_objective(self, name, varname, objclass=Objective, **kwargs):
        """Add an objective to the Problem."""
        self.objectives[name] = objclass(varname, **kwargs)
        self.add_objective_callable(name)
        self.iter_hist["objectives"][name] = []

    def add_objective_callable(self, name):
        """Add callable objective function with name."""
        def newobj(*x):
            return self.call_objective(*x, objective=name)
        setattr(self, name, newobj)

    def add_constraint(self, name, varname, conclass=Constraint, **kwargs):
        """Add a constraint to the Problem."""
        self.constraints[name] = conclass(varname, **kwargs)
        self.add_constraint_callable(name)
        self.iter_hist["constraints"][name] = []

    def add_constraint_callable(self, name):
        """Add callable constraint function with name name."""
        def newcon(*x):
            return self.call_constraint(*x, constraint=name)
        setattr(self, name, newcon)

    def __repr__(self):
        return self.name_repr()+" with:"+self.prob_repr()

    def current_x(self):
        """Get the current variable value x."""
        return [v for v in self.variables.values()]

    def new_x(self, *x):
        """Check if a given x is the same as the current value of x."""
        return not self.current_x() == list(x)

    def get_objectives(self):
        """Get all current objective values."""
        return [v.value for v in self.objectives.values()]

    def get_constraints(self):
        """Get all current constraint values."""
        return [v.value for v in self.constraints.values()]

    def call_outputs(self, *x, force_update=False):
        """
        Get all outputs at the given value of x.

        Parameters
        ----------
        *x : values
            Variable values

        Returns
        -------
        objectives : list
            values of the objectives
        constraints : list
            values of the constraints
        """
        if self.new_x(*x) or force_update:
            self.update_objectives(*x)
        return self.get_objectives(), self.get_constraints()

    def update_variables(self, *x):
        """Update variables at x."""
        x = unpack_x(*x)
        for i, v in enumerate(self.variables):
            self.variables[v] = x[i]

    def call_objective(self, *x, objective=''):
        """Call a given objective at x."""
        if self.new_x(*x):
            self.update_objectives(*x)
        return self.objectives[objective].value

    def call_constraint(self, *x, constraint=''):
        """Call a given constraint at x."""
        if self.new_x(*x):
            self.update_objectives(*x)
        return self.constraints[constraint].value

    def log_time(self):
        """Log the time/iteration of the optimization."""
        if not hasattr(self, 't_start'):
            self.t_start = time.time()
            self.iter = 0
        else:
            self.iter += 1
        self.iter_hist.time.append(time.time()-self.t_start)
        self.iter_hist.iter.append(self.iter)

    def log_hist(self):
        """Log the history for objectives, constraints, time, etc."""
        self.log_time()
        self.iter_hist.objectives.log(self.objectives, 1)
        self.iter_hist.constraints.log(self.constraints, 1)
        self.iter_hist.variables.log(self.variables, 1)

    def get_opt_hist(self, *objectives, time='time'):
        """
        Get the history of minimum/maximum objective values over time to plot.

        Parameters
        ----------
        *objectives : str
            Objectives to get (if none provided, returns all).
        time : str, optional
            Value to use for time indices (time or iter). The default is 'time'.

        Returns
        -------
        opt_hist : History
            History of minimum/maximum objective values over time.

        Examples
        --------
        >>> class ExpSimp(SimpleProblem):
        ...     def init_problem(self, **kwargs):
        ...         self.add_objective("a", "a", negative=True)
        ...         self.add_objective("b", "b")
        >>> exp = ExpSimp()
        >>> exp.iter_hist = History({'time': [0,1,2], 'objectives.a': [10, 11, 12], 'objectives.b': [8, 10, 7]})
        >>> h = exp.get_opt_hist()
        >>> h.a
        [np.int64(10), np.int64(11), np.int64(12)]
        >>> h.b
        [np.int64(8), np.int64(8), np.int64(7)]
        """
        if objectives:
            hist = self.iter_hist.objectives.get_values(*objectives)
        else:
            hist = self.iter_hist.objectives.copy()
        for val, arr in hist.items():
            if getattr(self.objectives[val], 'negative', False):
                hist[val] = [np.nanmax(arr[:i]) for i in range(1, len(arr)+1)]
            else:
                hist[val] = [np.nanmin(arr[:i]) for i in range(1, len(arr)+1)]
        hist[time] = self.iter_hist.get(time)
        return hist

    def plot_opt_hist(self, *objectives, time='time', **kwargs):
        """
        Plot the optimal value(s) of the search over time.

        Parameters
        ----------
        *objectives : str
            Objectives to get (if none provided, returns all).
        time : str, optional
            Value to use for time indices (time or iter). The default is 'time'.
        **kwargs : kwargs
            Keyword arguments to History.plot_line

        Returns
        -------
        fig, ax: matplotlib figure/axes
            Figures/axes returned from Hist.plot_line.
        """
        hist = self.get_opt_hist(*objectives, time=time)
        if not objectives:
            objectives = [k for k in hist.keys() if k not in ('time', 'iter')]
        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = time
        return hist.plot_line(*objectives, **kwargs)

    def get_default_x(self, *x):
        """Get x for the current iteration (previous x if x not provided)."""
        if not x:
            return tuple([*self.variables.values()])
        else:
            return x

    def time_execution(self, *x, reps=1):
        """
        Time the amount of time it takes to update the problem.

        Parameters
        ----------
        *x : vars
            Variable values to execute at, optional. If not provided, uses the current
            values.
        reps : int, optional
            Number of replicates to average/loop over. The default is 1.

        Returns
        -------
        t_ave : float
            Avarage time over all replicates
        """
        x = self.get_default_x(*x)
        starttime = time.time()
        for rep in range(reps):
            self.update_objectives(*x)
        t_ave = (time.time() - starttime) / reps
        return t_ave


class SimpleProblem(BaseProblem):
    """
    Simple optimization problem (without any given model constructs).

    Attributes
    ----------
    callables : dict
        dict of callables for objectives/constraints

    Examples
    --------
    >>> class ExampleSimpleProblem(SimpleProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_variables("x0", "x1")
    ...         f1 = lambda x0, x1: x0 + x1
    ...         self.add_objective("f1", f1)
    ...         g1 = lambda x0, x1: x0 - x1
    ...         self.add_constraint("g1", g1, threshold=3.0, comparator="less")
    >>> ex_sp = ExampleSimpleProblem()
    >>> ex_sp.f1(1, 1)
    np.float64(2.0)
    >>> ex_sp.g1(1, 1)
    np.float64(-3.0)
    """

    def __init__(self, **kwargs):
        self.callables = {}
        super().__init__(**kwargs)

    def update_objectives(self, *x):
        """Update objectives/constraints by calling callables."""
        self.update_variables(*x)
        for objname, obj in {**self.objectives, **self.constraints}.items():
            obj.update(np.float64(self.callables[objname](*x)))
        self.log_hist()

    def add_objective(self, name, call, **kwargs):
        """
        Add an objective to the problem.

        Parameters
        ----------
        name : str
            Name for the objective.
        call : callable
            Function to call for the objective in terms of the variables.
        **kwargs : kwargs
            kwargs to Objective.
        """
        self.callables[name] = call
        super().add_objective(name, name, **kwargs)

    def add_constraint(self, name, call, **kwargs):
        """
        Add an constraint to the problem.

        Parameters
        ----------
        name : str
            Name for the objective.
        call : callable
            Function to call for the objective in terms of the variables.
        **kwargs : kwargs
            kwargs to Constraint
        """
        self.callables[name] = call
        super().add_constraint(name, name, **kwargs)


class ExampleSimpleProblem(SimpleProblem):
    """Example simple Problem for docs."""

    def init_problem(self, **kwargs):
        """Initialize the problem."""
        self.add_variables("x0", "x1")
        self.add_objective("f1", lambda x0, x1: x0 + x1)
        self.add_constraint("g1", lambda x0, x1: x0 - x1, threshold=3.0, comparator="less")


class ResultObjective(Objective):
    """
    Base class of objectives which derive from Results.

    Fields
    ------
    time : float
        Time the objective is called at. If None, time will be the end of the sim.
    method : callable
        Method to tabulate for the objective. Default is np.sum.

    """

    time: np.float64 = None
    method: callable = np.sum

    def get_result_value(self, res):
        """
        Get the value corresponding to the objective from the result.

        Parameters
        ----------
        res : Result
            Result containing the metric desired.

        Returns
        -------
        val : value
            Value corresponding to the result.

        Examples
        --------
        >>> from fmdtools.analyze.result import Result
        >>> obj = ResultObjective("a.b", time=1.0)
        >>> res = Result({'t1p0.a.b': 10.0, 't2p0.a.b': 13.0})
        >>> obj.get_result_value(res)
        np.float64(10.0)

        >>> obj = ResultObjective("a.b", time=1.0, method=np.sum)
        >>> res = Result({'scen1.t1p0.a.b': 10.0, 'scen2.t1p0.a.b': 12.0})
        >>> obj.get_result_value(res)
        np.float64(22.0)
        """
        if not self.time:
            val = res.get_metric(self.name, method=self.method)
        else:
            t = t_key(float(self.time))
            val = res.get_metric(t+"."+self.name, method=self.method)
        return val

    def update(self, res):
        """Update the value of the objective given the result."""
        value = self.get_result_value(res)
        self.value = self.obj_from_value(value)


class ResultConstraint(ResultObjective):
    """
    Base class for constraints which derive from results.

    Fields
    ------
    threshold : float
        Threshold for the constraint. Default is 0.0
    comparator : str
        Whether the constraint is 'greater', 'less', 'equal', or 'notequal'.
    or_equal : bool
        Whether comparator includes equality (e.g., greater or equalto) or not.
    satisfied : bool
        Whether the constraint is satisfied.
    """

    threshold: np.float64 = np.float64(0.0)
    comparator: str = 'greater'
    or_equal: bool = True
    satisfied: bool = True

    def update(self, res):
        """Update the value of the constraint given the result."""
        value = self.get_result_value(res)
        self.value = self.con_from_value(value)
        if not self.negative:
            if self.or_equal:
                self.satisfied = self.value <= 0.0
            else:
                self.satisfied = self.value < 0.0
        else:
            if self.or_equal:
                self.satisfied = self.value >= 0.0
            else:
                self.satisfied = self.value > 0.0

    def con_from_value(self, value):
        """
        Call con_from_value from Constraint for the ResultConstraint.

        Examples
        --------
        >>> con = ResultConstraint("a", threshold=10.0, comparator='greater')
        >>> con.con_from_value(11.0)
        -1.0

        >>> con2 = ResultConstraint("a", threshold=10.0, comparator='less')
        >>> con2.con_from_value(11.0)
        1.0
        """
        return Constraint.con_from_value(self, value)


class HistoryObjective(ResultObjective):
    """
    Same as ResultObjective but marks that the result is a history.

    Examples
    --------
    >>> obj = HistoryObjective("a.b", method=np.max)
    >>> hist = History({'t1p0.a.b': [5.0, 10.0], 't2p0.a.b': [1.0, 13.0]})
    >>> obj.get_result_value(hist)
    np.float64(13.0)
    """


class HistoryConstraint(ResultConstraint):
    """Same as ResultConstraint but with different name."""


class BaseSimProblem(BaseProblem):
    """
    Base optimization problem for optimizing over simulations.

    Attributes
    ----------
    prop_method : callable
        Method in propagate to call.
    keep_ec : bool, optional
        Whether to get/keep the endresult. Default is False.
    """

    def add_sim(self, mdl, prop_method, *args, keep_ec=False, **kwargs):
        """
        Add the base simulation to the Problem.

        Parameters
        ----------
        mdl : Block/Architecture class or object
            Model to simulate.
        prop_method : str
            Name of propagate method to simulate the model with.
        *args : args
            Arguments to the propagate method.
        keep_ec : bool, optional
            Whether to keep the classify dict. The default is False.
        **kwargs : kwargs
            Keyword arguments to the propagate method.
        """
        if 'track' in kwargs:
            track = kwargs['track']
        else:
            track = mdl.track
        self.mdl = init_obj(objclass=mdl, track=track)
        if type(prop_method) is str:
            self.prop_method = getattr(propagate, prop_method)
        elif callable(prop_method):
            self.prop_method = prop_method
        else:
            raise Exception("Invalid prop_method "+str(prop_method))

        self.keep_ec = keep_ec
        self.args = args
        self.kwargs = kwargs

    def add_result_objective(self, name, varname, **kwargs):
        """
        Add an objective corresponding to a possible to_return.

        Associates a callable to the problem with name 'name' which may be called to
        evaluate the objective at a value of x.

        Parameters
        ----------
        name : str
            Name to give the objective
        varname : str
            Name of the variable to get for the variable.
        **kwargs : kwargs
            Arguments to ResultObjective
        """
        self.add_objective(name, varname, objclass=ResultObjective, **kwargs)

    def add_result_constraint(self, name, varname, **kwargs):
        """
        Add an objective corresponding to a possible to_return.

        Associates a callable to the problem with name 'name' which may be called to
        evaluate the constraint at a value of x.

        Parameters
        ----------
        name : str
            Name to give the constraint.
        varname : str
            Name of the variable to get for the constraint.
        **kwargs : kwargs
            Arguments to ResultConstraint
        """
        self.add_constraint(name, varname, conclass=ResultConstraint, **kwargs)

    def add_history_objective(self, name, varname, **kwargs):
        """Add an objective corresponding to a given history attribute."""
        self.add_objective(name, varname, objclass=HistoryObjective, **kwargs)

    def add_history_constraint(self, name, varname, **kwargs):
        """Add an objective corresponding to a given history attribute."""
        self.add_constraint(name, varname, conclass=HistoryConstraint, **kwargs)

    def get_end_time(self):
        """
        Get the end_time for the simulation that minimizes simulation time.

        Used so that simulations only run until the last objective is called, rather
        than the full set of potential timesteps.

        Returns
        -------
        end_time : float
            Simulation time to simulate to.
        """
        last_time = self.mdl.sp.end_time
        all_times = [a.time if a.time else last_time
                     for a in {**self.objectives, **self.constraints}.values()]
        end_time = max(all_times)
        return end_time

    def obj_con_des_res(self):
        """
        Get the to_return argument for the problem given objectives/constraints.

        Returns
        -------
        des_res : dict
            to_return argument to prop_method.
        """
        if self.keep_ec:
            des_res = {'end': ['classify']}
        else:
            des_res = {}
        for n in {**self.objectives, **self.constraints}.values():
            if n.time:
                t = n.time
            else:
                t = 'end'
            if t in des_res:
                des_res[t].append(n.name)
            else:
                des_res[t] = [n.name]
        return des_res

    def update_objectives(self, *x):
        """Update objectives/constraints by simulating the model at x."""
        self.update_variables(*x)
        self.res, self.hist = self.sim_mdl(*self.current_x())
        for obj in {**self.objectives, **self.constraints}.values():
            if isinstance(obj, HistoryObjective) or isinstance(obj, HistoryConstraint):
                obj.update(self.hist)
            elif isinstance(obj, ResultObjective):
                obj.update(self.res)
        self.log_hist()

    def close_pool(self):
        """Close pool if instantiated in the Problem."""
        propagate.close_pool(self.kwargs.get('pool', None))


class ParameterSimProblem(BaseSimProblem):
    """
    Optimization problem defining the optimization of model parameters over simulations.

    Examples
    --------
    >>> from fmdtools.sim.sample import expd
    >>> from fmdtools.define.block.function import ExampleFunction

    below, we show basic setup of a parameter problem where objectives get values
    from the sim at particular times.

    >>> class ExampleParameterSimProblem(ParameterSimProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_sim(ExampleFunction(), "nominal")
    ...         self.add_parameterdomain(expd)
    ...         self.add_result_objective("f1", "s.x", time=5)
    ...         self.add_result_objective("f2", "s.y", time=5)
    ...         self.add_result_constraint("g1", "s.x", time=10, threshold=10, comparator="greater")

    >>> exprob = ExampleParameterSimProblem()
    >>> exprob
    ExampleParameterSimProblem with:
    VARIABLES
     -y                                                             nan
     -x                                                             nan
    OBJECTIVES
     -f1                                                            nan
     -f2                                                            nan
    CONSTRAINTS
     -g1                                                            nan

    once this is set up, you can use the objectives/constraints as callables, like so:

    >>> exprob.f1(1, 0)
    np.float64(0.0)
    >>> exprob.f1(1, 1)
    np.float64(5.0)
    >>> exprob.f1(1, 2)
    np.float64(10.0)
    >>> exprob.f2(1, 2)
    np.float64(0.0)
    >>> exprob.g1(1, 2)
    np.float64(-10.0)

    below, we use the classify dict as an objective instead of the variable:

    >>> class ExampleParameterSimProblemObj(ParameterSimProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_sim(ExampleFunction(), "nominal")
    ...         self.add_parameterdomain(expd)
    ...         self.add_result_objective("f1", "classify.xy")

    >>> ex_obj_prob = ExampleParameterSimProblemObj()
    >>> ex_obj_prob.f1(1, 1)
    np.float64(100.0)
    >>> ex_obj_prob.f1(1, 2)
    np.float64(200.0)

    finally, note that this class can work with a variety of methods:

    >>> class ExampleParameterSimProblemShort(ParameterSimProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_sim(ExampleFunction(),"one_fault", "examplefunction", "short", 3, include_nominal=False)
    ...         self.add_parameterdomain(expd)
    ...         self.add_result_objective("f1", "s.x", time=3)
    ...         self.add_result_objective("f2", "s.y", time=5)

    >>> exprobshort = ExampleParameterSimProblemShort()
    >>> exprobshort.f1(1, 1)
    np.float64(2.0)
    >>> exprobshort.f2(1, 1)
    np.float64(3.0)
    """

    def add_parameterdomain(self, parameterdomain):
        """
        Add a ParameterDomain to the optimization problem.

        Parameters
        ----------
        parameterdomain : ParameterDomain
            Parameter Domain to optimize over.
        """
        self.parameterdomain = parameterdomain
        self.add_variables(*parameterdomain.variables)

    def sim_mdl(self, *x):
        """
        Simulate the model at the given variable value.

        Parameters
        ----------
        *x : args
            Variable inputs for parameterdomain.

        Returns
        -------
        res : Result
            result for the sim.
        hist : History
            history for the sim.
        """
        p = self.parameterdomain(*x)
        end_time = self.get_end_time()
        mdl_kwargs = {'p': p, 'sp': {'end_time': end_time}}
        to_return = self.obj_con_des_res()
        all_res = self.prop_method(self.mdl.new(**mdl_kwargs),
                                   *self.args,
                                   protect=False,
                                   to_return=to_return,
                                   showprogress=False,
                                   **self.kwargs)
        return all_res[0].flatten(), all_res[1].flatten()


class ExampleParameterSimProblem(ParameterSimProblem):
    """Example ParameterSimProblem class."""

    def init_problem(self, **kwargs):
        self.add_sim(ExampleFunction(), "nominal")
        self.add_parameterdomain(expd)
        self.add_result_objective("f1", "s.x", time=5)
        self.add_result_objective("f2", "s.y", time=5)
        self.add_result_constraint("g1", "s.x", time=10, threshold=10, comparator="greater")


class ScenarioProblem(BaseSimProblem):
    """
    Base class for optimizing scenario parameters.

    ScenarioProblem simulates in a staged fashion, first initializing the base_sim
    (usually, the nominal scenario) which it copies models from
    """

    def __init__(self, **kwargs):
        self.base_sim = None
        super().__init__(**kwargs)

    def add_sim(self, mdl, **kwargs):
        """Add a simulation - always use Simulation."""
        super().add_sim(mdl, exec_sim, **filter_kwargs(exec_sim, **kwargs))

    def prep_sim(self):
        """Prepare simulation by simulating it until the start of the scenario."""
        sim_kwar = {**filter_kwargs(Simulation, **self.kwargs),
                    'scen': ParameterScenario(sp={'end_time': self.get_end_time()}),
                    'to_return': self.obj_con_des_res(),
                    'mdl': self.mdl,
                    'ctimes': [self.get_start_time()]}
        self.base_sim = Simulation(**sim_kwar)
        self.base_sim(**get_sim_call_kwargs(self.base_sim, **self.kwargs))

    def sim_mdl(self, *x):
        """
        Simulate the model at the given variable value.

        Parameters
        ----------
        *x : args
            Variable inputs for parameterdomain.

        Returns
        -------
        res : Result
            result for the sim.
        hist : History
            history for the sim.
        """
        if not self.base_sim:
            self.prep_sim()

        kwargs = {**self.kwargs,
                  'scen': self.gen_scenario(*x),
                  'mdl': [*self.base_sim.get_models().values()][0],
                  'to_return': self.obj_con_des_res(),
                  'protect': False,
                  'copy': True}
        sim_kwar = {'nomhist': self.base_sim.history.copy(),
                    'nomresult': self.base_sim.result.copy()}
        return self.prop_method((kwargs, sim_kwar))


class SingleFaultScenarioProblem(ScenarioProblem):
    """
    Class for optimizing the time of a given fault scenario.

    Attributes
    ----------
    faultdomain : FaultDomain
        FaultDomain containing the fault
    phasemap : PhaseMap
        PhaseMap for fault sampling
    t_start : float
        Minimum start time for the simulation and lower bound on scenario time.
        Default is 0.0.

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> class ExScenProb(SingleFaultScenarioProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_sim(ExampleFunction(), ("examplefunction", "short"))
    ...         self.add_result_objective("f1", "s.y", time=5)


    objective value should be 1.0 (init value) + 3 * time_with_fault
    >>> ex_scenprob = ExScenProb()
    >>> ex_scenprob.f1(5.0)
    np.float64(3.0)
    >>> ex_scenprob.f1(4.0)
    np.float64(6.0)
    """

    def name_repr(self):
        """Get name of the class and faults."""
        faulttup = [*self.faultdomain.faults.keys()][0]
        return "SingleScenarioProblem("+faulttup[0]+", "+faulttup[1]+")"

    def add_sim(self, mdl, faulttup, phasemap=None, sim_start=0.0, **kwargs):
        """
        Initialize the SingleFaultScenarioProblem with a given fault to optimize.

        Parameters
        ----------
        mdl : Model
            Model to simulate/optimize.
        faulttup : tuple
            (fxn, fault) defining the fault.
        phasemap : PhaseMap, optional
            PhaseMap for fault sampling. The default is None.
        sim_start : float, optional
            Minimum start time for the simulation and lower bound on scenario time..
            Default is 0.0.
        **kwargs : kwargs
            Keyword arguments to prop_one_scen (e.g., track, etc.).
        """
        faultdomain = FaultDomain(mdl)
        faultdomain.add_fault(*faulttup)
        self.faultdomain = faultdomain
        self.phasemap = phasemap
        self.sim_start = sim_start
        self.add_variables("time")
        super().add_sim(mdl, **kwargs)

    def get_start_time(self):
        """Get the scenario start time to copy the model at."""
        return self.sim_start

    def gen_scenario(self, x):
        """
        Generate the scenario to simulate in the model.

        Parameters
        ----------
        x : float
            Fault scenario time.

        Returns
        -------
        scen : SingleFaultScenario
            SingleFaultScenario to simulate.
        """
        starttime = self.get_start_time()
        end_time = self.get_end_time()
        if not starttime <= x <= end_time:
            raise Exception("time out of range: "+str((starttime, end_time)))
        fault = [*self.faultdomain.faults][0]
        scen = SingleFaultScenario.from_fault(fault, time=x, mdl=self.mdl,
                                              phasemap=self.phasemap,
                                              starttime=self.get_start_time())
        return scen


class ExScenProb(SingleFaultScenarioProblem):
    """Example Scenario problem for docs and testing."""

    def init_problem(self, **kwargs):
        self.add_sim(ExampleFunction(), ("examplefunction", "short"))
        self.add_result_objective("f1", "s.y", time=5)


class DisturbanceProblem(ScenarioProblem):
    """
    Class for optimizing disturbances that occur at a set time.

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> class ExampleDisturbanceProblem(DisturbanceProblem):
    ...     def init_problem(self, **kwargs):
    ...         self.add_sim(ExampleFunction(), 3, "s.y")
    ...         self.add_result_objective("f1", "s.y", time=5)

    >>> ex_dp = ExampleDisturbanceProblem()
    >>> ex_dp.f1(5.0)
    np.float64(5.0)
    >>> ex_dp.f1(4.0)
    np.float64(4.0)
    """

    def add_sim(self, mdl, time, *disturbances, **kwargs):
        """
        Initialize the DisturbanceProblem sim.

        Parameters
        ----------
        mdl : Simulable
            Model to optimize.
        time : float
            Time to inject the disturbances at.
        *disturbances : str
            Names of variables to perturb at time t (which become the variables)
        **kwargs : TYPE
            DESCRIPTION.
        """
        self.time = time
        self.add_variables(*disturbances)
        super().add_sim(mdl, **kwargs)

    def get_start_time(self):
        """Get the scenario start time to copy the model at."""
        return self.time

    def gen_scenario(self, *x):
        """
        Generate the scenario to simulate in the model.

        Parameters
        ----------
        x : float
            Fault scenario time.

        Returns
        -------
        scen : SingleFaultScenario
            SingleFaultScenario to simulate.
        """
        dist = {self.time: {v: x[i] for i, v in enumerate(self.variables)}}
        faultseq = self.kwargs.get('faultseq', None)
        if faultseq:
            t = np.min([*dist.keys(), *faultseq.keys()])
            seq = Sequence(disturbances=dist, faultseq=faultseq)
        else:
            t = self.time
            seq = Sequence(disturbances=dist)
        scen = Scenario(sequence=seq,
                        name='disturbance',
                        time=t)
        return scen


class ExampleDisturbanceProblem(DisturbanceProblem):
    """Example DisturbanceProblem for testing and docs."""

    def init_problem(self, **kwargs):
        self.add_sim(ExampleFunction(), 3, "s.y")
        self.add_result_objective("f1", "s.y", time=5)


class BaseConnector(dataobject):
    """
    Base class for connectors.

    Connectectors are used in ProblemArchitectures to link the outputs of one problem
    as the inputs to another.

    Fields
    ------
    name : name to give the Connector
    """

    name: str = ''


class VariableConnector(BaseConnector):
    """
    Class connecting variables in one problem to use as variables in another problem.

    Fields
    ------
    keys : tuple
        Names of the variables.
    values : np.array
        Array of values for the variables.
    """

    keys: tuple = ()
    values: np.array = np.array([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.values.size == 0:
            self.values = np.array([np.nan for k in self.keys])

    def update(self, valuedict):
        """
        Update the value of the connector given a dict of values.

        Parameters
        ----------
        valuedict : dict
            dict with structure {k: value}, where k corresponds to a key in self.keys.
        """
        for i, k in enumerate(self.keys):
            self.values[i] = valuedict[k]

    def update_values(self, *x):
        """
        Update values of the connector given input variable x.

        Parameters
        ----------
        *x : iterable
            Variable values to update (must be in order of self.keys).
        """
        x = unpack_x(*x)
        for i, x_i in enumerate(x):
            self.values[i] = x_i

    def get(self, key, defaultval=None):
        """Get a value for a particular key."""
        return self.values[self.keys.index(key)]

    def items(self):
        return {k: self.values[i] for i, k in enumerate(self.keys)}.items()

    def in_(self, k):
        return k in self.keys


class ObjectiveConnector(VariableConnector):
    """Class for linking objectives. Same as VariableConnector but used differently."""

    def update(self, valuedict):
        for i, k in enumerate(self.keys):
            self.values[i] = valuedict[k].value


class ConstraintConnector(ObjectiveConnector):
    """Class for linking constraints. Same as VariableConnector but used differently."""


def obj_name(probname, objname):
    """Create architecture-level objective name for problem."""
    return probname + "_" + objname


class ProblemArchitecture(BaseProblem):
    """
    Class enabling the representation of combined joint optimization problems.

    Combined optimization problems involve multiple variables and objectives which
    interact (e.g., Integrated Resilience Optimization, Two-Stage Optimization, etc.)

    Note that ProblemArchitectures are (presently) limited in the sense that they assume
    problems are sequentially linked in the order of instantiation. While this works
    well for nested problems, there are some limitations when workign with parallel
    problems which we hope to resolve in future work.

    Attributes
    ----------
    connectors : dict
        Dictionary of Connector (variables, models, etc) added using .add_connector
    problems : dict
        Dictionary of optimization problems added using .add_problem
    problem_graph : nx.DiGraph
        Graph structure containing information about how each problem is connected.
    var_mapping : dict
        Dict mapping inputs to each problem to problem variables.

    Examples
    --------
    Below we connect three example problems in a single architecture, linking the vars
    x0 and x1 from ex_sp to be inputs to the scenario simulation (time) as well as the
    disturbance simulation variable (s.y).

    >>> class ExampleProblemArchitecture(ProblemArchitecture):
    ...     def init_problem(self, **kwargs):
    ...         self.add_connector_variable("x0", "x0")
    ...         self.add_connector_variable("x1", "x1")
    ...         self.add_problem("ex_sp", ExampleSimpleProblem, outputs={"x0": ["x0"], "x1": ["x1"]})
    ...         self.add_problem("ex_scenprob", ExScenProb, inputs={"x0": ["time"]})
    ...         self.add_problem("ex_dp", ExampleDisturbanceProblem, inputs={"x1": ["s.y"]})

    >>> ex_pa = ExampleProblemArchitecture()
    >>> ex_pa
    ExampleProblemArchitecture with:
    CONNECTORS
     -x0                                                          [nan]
     -x1                                                          [nan]
    PROBLEMS
     -ex_sp({'ex_sp_xloc': ['x0', 'x1']}) -> ['x0', 'x1']
     -ex_scenprob({'x0': ['time']}) -> []
     -ex_dp({'x1': ['s.y']}) -> []
    VARIABLES
     -ex_sp_xloc                                              [nan nan]
    OBJECTIVES
     -ex_sp_f1                                                      nan
     -ex_scenprob_f1                                                nan
     -ex_dp_f1                                                      nan
    CONSTRAINTS
     -ex_sp_g1                                                      nan

    Setting up this problem gives us callables for each problem which we can use to
    call each objective in each problem in terms of its local variables:

    >>> ex_pa.ex_sp_f1(1, 1)
    np.float64(2.0)
    >>> ex_pa.ex_scenprob_f1()
    np.float64(15.0)
    >>> ex_pa.ex_dp_f1()
    np.float64(1.0)

    We can also call these in terms of the full set of variables:

    >>> ex_pa.ex_scenprob_f1_full(2, 2)
    np.float64(12.0)
    >>> ex_pa.ex_dp_f1_full(3, 3)
    np.float64(3.0)
    """

    def __init__(self, **kwargs):
        self.connectors = {}
        self.problems = {}
        self.problem_graph = nx.DiGraph()
        self.var_mapping = {}
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def prob_repr(self):
        repstr = ""
        constr = " -" + "\n -".join(['{:<45}{:>20}'.format(k, str(var.values))
                                     for k, var in self.connectors.items()])
        if constr:
            repstr += "\nCONNECTORS\n" + constr
        probstr = " -" + "\n -".join([pn+"("+str(self.find_inputs(pn))+")"
                                      + " -> " + str(self.find_outputs(pn))
                                      for pn in self.problems])
        if probstr:
            repstr += "\nPROBLEMS\n" + probstr
        var_str = " -" + "\n -".join(['{:<45}{:>20}'.format(k, str(var.values))
                                      for k, var in self.variables.items()])
        if self.variables:
            repstr += "\n"+"VARIABLES\n" + var_str
        obj_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.objectives.items()])
        if self.objectives:
            repstr += "\n"+"OBJECTIVES\n" + obj_str
        con_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.constraints.items()])
        if self.constraints:
            repstr += "\n"+"CONSTRAINTS\n" + con_str
        return repstr

    def add_connector(self, name, *args, conclass=VariableConnector):
        """
        Add a connector linking variables between problems.

        Parameters
        ----------
        name : str
            Name for the connector
        *args : strs
            arguments to conclass
        conclass : Connector
            Class to instantiate.
        """
        self.connectors[name] = conclass(name, *args)
        self.problem_graph.add_node(name, label=name)

    def add_connector_variable(self, name, *varnames):
        """
        Add a connector linking variables between problems.

        Parameters
        ----------
        name : str
            Name for the connector
        *varnames : strs
            Names of the variable values (used in each problem) to link as a part of the
            connector.
        """
        self.add_connector(name, varnames, conclass=VariableConnector)

    def add_connector_objective(self, name, *objnames):
        """
        Add a connector linking an objective to an input variable in another problem.

        Parameters
        ----------
        name : str
            Name for the connector
        *objnames : strs
            Names of the objective values to link as a part of the connector.
        """
        self.add_connector(name, objnames, conclass=ObjectiveConnector)

    def add_connector_constraint(self, name, *connames):
        """
        Add a connector linking a constraint to an input variable in another problem.

        Parameters
        ----------
        name : str
            Name for the connector
        *varnames : strs
            Names of the constraint values to link as a part of the connector.
        """
        self.add_connector(name, connames, conclass=ConstraintConnector)


    def add_problem(self, name, problem, inputs={}, outputs={}):
        """
        Add a problem to the ProblemArchitecture.

        Parameters
        ----------
        name : str
            Name for the problem.
        problem : BaseProblem/ScenProblem/SimpleProblem...
            Problem object to add to the architecture.
        inputs : dict, optional
            List of input connector names (by name) and their corresponding problem
            variables. The default is [].
        outputs : dict, optional
            List of output connector names (by name) and their corresponding problem
            variables/objectives/constraints. The default is [].
        """
        if self.problems:
            upstream_problem = [*self.problems][-1]
            self.problem_graph.add_edge(upstream_problem, name,
                                        label="next")
            problem.consistent = False
        else:
            problem.consistent = True
        self.problems[name] = init_obj(name, problem)
        self.problem_graph.add_node(name, order=len(self.problems))

        for con in inputs:
            self.problem_graph.add_edge(con, name, label="input", var=inputs[con])
        for con in outputs:
            self.problem_graph.add_edge(name, con, label="output", var=outputs[con])
        xloc_vars = self.find_xloc_vars(name)
        if xloc_vars:
            xloc_name = name+"_xloc"
            self.variables[xloc_name] = VariableConnector(xloc_name, xloc_vars)
            self.problem_graph.add_node(xloc_name, label="xloc")
            self.problem_graph.add_edge(xloc_name, name, label="input", var=xloc_vars)
        self.create_var_mapping(name)
        self.add_objective_callables(name)
        self.add_constraint_callables(name)
        self.update_problem_objectives(name)
        self.update_problem_constraints(name)
        self.iter_hist.variables = init_dicthist(self.variables, None)

    def create_var_mapping(self, probname):
        """Create a dict mapping problem variables to input/connector variables."""
        var_mapping = dict()
        vars_to_match = [*self.problems[probname].variables]
        inputdict = self.find_inputs(probname)
        inputconnectors = self.get_inputs(probname)
        for inputname, inputvars in inputdict.items():
            for i, inputvar in enumerate(inputvars):
                var_mapping[inputvar] = (inputname, inputconnectors[inputname].keys[i])
                vars_to_match.remove(inputvar)
        if vars_to_match:
            raise Exception("Dangling variables: "+str(vars_to_match))
        self.var_mapping[probname] = var_mapping

    def update_downstream_consistency(self, probname):
        """Mark downstream problems as inconsistent (when current problem updated)."""
        probs = self.get_downstream_probs(probname)
        for prob in probs:
            self.problems[prob].consistent = False

    def find_inconsistent_upstream(self, probname):
        """Check that all upstream problems are consistent with current probname."""
        probs = self.get_upstream_probs(probname)
        inconsistent_probs = []
        for prob in probs:
            if not self.problems[prob].consistent:
                inconsistent_probs.append(prob)
        return inconsistent_probs

    def add_objective_callables(self, probname):
        """Add callable objective functions from problem."""
        for objname in self.problems[probname].objectives:
            self.add_objective_callable(probname, objname)

    def add_objective_callable(self, probname, objname):
        """Add callable objective function from problem probname."""
        def newobj(*x):
            return self.call_objective(probname, objname, *x)

        def new_full_obj(*x):
            return self.call_full_objective(probname, objname, *x)
        aname = obj_name(probname, objname)
        setattr(self, aname, newobj)
        setattr(self, aname+"_full", new_full_obj)
        self.iter_hist.objectives[aname] = []

    def add_constraint_callables(self, probname):
        """Add callable constraint function from problem probname."""
        for conname in self.problems[probname].constraints:
            self.add_constraint_callable(probname, conname)

    def add_constraint_callable(self, probname, conname):
        """Add callable constraint function from problem probname."""
        def newcon(*x):
            return self.call_constraint(probname, conname, *x)

        def new_full_con(*x):
            return self.call_full_constraint(probname, conname, *x)
        aname = obj_name(probname, conname)
        setattr(self, aname, newcon)
        setattr(self, aname+"_full", new_full_con)
        self.iter_hist.constraints[aname] = []

    def update_problem_objectives(self, probname):
        """Update architecture-level objectives from problem."""
        for objname, obj in self.problems[probname].objectives.items():
            aname = obj_name(probname, objname)
            if self.problems[probname].consistent:
                self.objectives[aname] = obj.value
            else:
                self.objectives[aname] = np.nan

    def update_problem_constraints(self, probname):
        """Update architecture-level constraints from problem."""
        for objname, obj in self.problems[probname].constraints.items():
            aname = obj_name(probname, objname)
            if self.problems[probname].consistent:
                self.constraints[aname] = obj.value
            else:
                self.constraints[aname] = np.nan

    def find_input_vars(self, probname):
        """Find variables for a problem that are in an input connector."""
        return [var for con in self.find_inputs(probname).values() for var in con]

    def find_xloc_vars(self, probname):
        """Find variables for a problem that aren't in an input connector."""
        return [x for x in self.problems[probname].variables
                if x not in self.find_input_vars(probname)]

    def call_full_objective(self, probname, objname, *x_full):
        """Call objective of a problem over full set of variables *x_full."""
        self.update_full_problem(*x_full, probname=probname)
        return self.problems[probname].objectives[objname].value

    def call_full_constraint(self, probname, conname, *x_full):
        """Call objective of a problem over full set of variables *x_full."""
        self.update_full_problem(*x_full, probname=probname)
        return self.problems[probname].constraints[conname].value

    def call_objective(self, probname, objname, *x_loc):
        """Call objective of a problem over partial its local variables *x_loc."""
        self.update_problem(probname, *x_loc)
        return self.problems[probname].objectives[objname].value

    def call_constraint(self, probname, conname, *x_loc):
        """Call constraint of a problem over partial its local variables *x_loc."""
        self.update_problem(probname, *x_loc)
        return self.problems[probname].constraints[conname].value

    def update_full_problem(self, *x_full, probname='', force_update=False):
        """
        Update the variables for the entire problem (or, problems up to probname).

        Parameters
        ----------
        *x_full : float
            Variable values for all local variables in the problem architecture up to
            the probname.
        probname : str, optional
            If provided, the problems will be updated up to the given problem.
            The default is ''.

        Examples
        --------
        >>> ex_pa = ExampleProblemArchitecture()
        >>> ex_pa.update_full_problem(1, 2)
        >>> ex_pa
        ExampleProblemArchitecture with:
        CONNECTORS
         -x0                                                           [1.]
         -x1                                                           [2.]
        PROBLEMS
         -ex_sp({'ex_sp_xloc': ['x0', 'x1']}) -> ['x0', 'x1']
         -ex_scenprob({'x0': ['time']}) -> []
         -ex_dp({'x1': ['s.y']}) -> []
        VARIABLES
         -ex_sp_xloc                                                [1. 2.]
        OBJECTIVES
         -ex_sp_f1                                                   3.0000
         -ex_scenprob_f1                                            15.0000
         -ex_dp_f1                                                   2.0000
        CONSTRAINTS
         -ex_sp_g1                                                  -4.0000

        >>> ex_pa.problems['ex_dp']
        ExampleDisturbanceProblem with:
        VARIABLES
         -s.y                                                        2.0000
        OBJECTIVES
         -f1                                                         2.0000
        """
        if not probname:
            probname = [*self.problems][-1]
        probs_to_call = [*self.get_upstream_probs(probname), probname]
        x_to_split = [*x_full]
        for problem in probs_to_call:
            loc_var = problem + "_xloc"
            if loc_var in self.variables:
                x_loc = [x_to_split.pop(0) for k in self.variables[loc_var].keys]
            else:
                x_loc = []
            self.update_problem(problem, *x_loc, force_update=force_update)

    def get_default_x(self, *x):
        return tuple([v for gv in self.variables.values() for v in gv.values])

    def update_objectives(self, *x):
        """Take place of update_objectives in base problem."""
        self.update_full_problem(*x, force_update=True)

    def update_problem(self, probname, *x, force_update=False):
        """
        Update a given problem with new values for inputs (and non-input variables).

        Additionally updates output connectors.

        Parameters
        ----------
        probname : str
            Name of the problem to update.
        *x : float
            Input variables to update (aside from inputs).

        Examples
        --------
        >>> ex_pa = ExampleProblemArchitecture()
        >>> ex_pa.update_problem("ex_sp", 1, 2)
        >>> ex_pa.problems["ex_sp"]
        ExampleSimpleProblem with:
        VARIABLES
         -x0                                                         1.0000
         -x1                                                         2.0000
        OBJECTIVES
         -f1                                                         3.0000
        CONSTRAINTS
         -g1                                                        -4.0000

        This update should further update connectors:

         >>> ex_pa.get_outputs("ex_sp")
         {'x0': VariableConnector(name='x0', keys=('x0',), values=array([1.])), 'x1': VariableConnector(name='x1', keys=('x1',), values=array([2.]))}

        Which should then propagate to downstream sims:

        >>> ex_pa.update_problem("ex_scenprob")
        >>> ex_pa.problems["ex_scenprob"]
        SingleScenarioProblem(examplefunction, short) with:
        VARIABLES
         -time                                                       1.0000
        OBJECTIVES
         -f1                                                        15.0000
        """
        inconsistent_upstream = self.find_inconsistent_upstream(probname)
        for upstream_prob in inconsistent_upstream:
            self.update_problem(upstream_prob, force_update=force_update)
        x_inputs = self.get_inputs_as_x(probname, *x)
        self.problems[probname].call_outputs(*x_inputs, force_update=force_update)
        self.problems[probname].consistent = True
        self.update_problem_objectives(probname)
        self.update_problem_constraints(probname)
        self.update_problem_outputs(probname)
        self.update_downstream_consistency(probname)
        self.log_hist()

    def update_problem_outputs(self, probname):
        """
        Update the output connectors from a problem.

        Parameters
        ----------
        probname : str
            Name of the problem.
        """
        outputs = self.get_outputs(probname)
        for output, connector in outputs.items():
            # note: must be put in order of inheritance
            if isinstance(connector, ConstraintConnector):
                connector.update(self.problems[probname].constraints)
            elif isinstance(connector, ObjectiveConnector):
                connector.update(self.problems[probname].objectives)
            elif isinstance(connector, VariableConnector):
                connector.update(self.problems[probname].variables)
            else:
                raise Exception("Invalid connector: "+connector)

    def find_inputs(self, probname):
        """
        List input connectors for a given problem.

        Parameters
        ----------
        probname : str
            Name of the problem.

        Returns
        -------
        inputs : list
            List of names of connectors used as inputs.
        """
        return {e[0]: self.problem_graph.edges[e]['var']
                for e in self.problem_graph.in_edges(probname)
                if self.problem_graph.edges[e]['label'] == 'input'}

    def find_outputs(self, probname):
        """
        List output connectors for a given problem.

        Parameters
        ----------
        probname : str
            Name of the problem.

        Returns
        -------
        outputs : list
            List of names of connectors used as outputs.
        """
        return [e[1] for e in self.problem_graph.out_edges(probname)
                if self.problem_graph.edges[e]['label'] == 'output']

    def get_inputs_as_x(self, probname, *x):
        """
        Get variable values for inputs.

        Parameters
        ----------
        probname : str
            Name of the problem..

        Returns
        -------
        inputs : list
            Input connectors and their values.
        """
        if x:
            self.variables[probname+"_xloc"].update_values(*x)
        vars_to_match = [*self.problems[probname].variables]
        inputs = self.get_inputs(probname)
        x_input = []
        for var in vars_to_match:
            inputname, inputkey = self.var_mapping[probname][var]
            x_input.append(inputs[inputname].get(inputkey))
        return x_input

    def get_inputs(self, probname):
        """Return a dict of input connectors for problem probname."""
        return {c: self.connectors[c] if c in self.connectors else self.variables[c]
                for c in self.find_inputs(probname)}

    def get_outputs(self, probname):
        """Return a dict of output connectors for problem probname."""
        return {c: self.connectors[c] for c in self.find_outputs(probname)}

    def get_downstream_probs(self, probname):
        """Return a list of all problems to be executed after the problem probname."""
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[ind+1:]

    def get_upstream_probs(self, probname):
        """Return a list of all problems to be executed before the problem probname."""
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[:ind]

    def show_sequence(self):
        """
        Show a visualization of the problem architecture.

        Returns
        -------
        fig : mpl.figure
            Figure object.
        ax : mpl.axis
            Axis object.
        """
        fig, ax = setup_plot()
        pos = nx.planar_layout(self.problem_graph, dim=2)
        nx.draw(self.problem_graph, pos=pos)
        orders = nx.get_node_attributes(self.problem_graph, "order")
        labels = {node: str(orders[node]) + ": " + node
                  if node in orders else self.problem_graph.nodes[node]['label']
                  for node in self.problem_graph}
        nx.draw_networkx_labels(self.problem_graph, pos, labels=labels)
        edge_labels = nx.get_edge_attributes(self.problem_graph, "label")
        edge_vars = nx.get_edge_attributes(self.problem_graph, "var")
        edge_labels = {e: lab+": "+str(edge_vars[e]) if e in edge_vars else lab
                       for e, lab in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.problem_graph, pos, edge_labels=edge_labels)
        return fig, ax


class ExampleProblemArchitecture(ProblemArchitecture):
    """Example ProblemProblemArchitecture for docs and testing."""

    def init_problem(self, **kwargs):
        self.add_connector_variable("x0", "x0")
        self.add_connector_variable("x1", "x1")
        self.add_problem("ex_sp", ExampleSimpleProblem, outputs={"x0": ["x0"], "x1": ["x1"]})
        self.add_problem("ex_scenprob", ExScenProb, inputs={"x0": ["time"]})
        self.add_problem("ex_dp", ExampleDisturbanceProblem, inputs={"x1": ["s.y"]})


class DynamicInterface():
    """
    Interface for dynamic search of model states (e.g., AST).

    ...

    Attributes
    ----------
    t : float
        time
    t_max : float
        max time
    t_ind : int
        time index in log
    to_return : list
        variables to get from the model at each time-step
    hist : History
        mdlhist for simulation
    """

    def __init__(self, mdl, mdl_kwargs={}, t_max=False, track="all",
                 run_stochastic="track_pdf", to_return=[], use_end_condition=None):
        """
        Initialize the problem.

        Parameters
        ----------
        mdl : Model
            Model defining the simulation.
        mdl_kwargs : dict, optional
            Parameters to run the model at. The default is {}.
        t_max : float, optional
            Maximum simulation time. The default is False.
        track : str/dict, optional
            Properties of the model to track over time. The default is "all".
        run_stochastic : bool/str, optional
            Whether to run stochastic behaviors (True/False) and/or
            return pdf ("track_pdf"). The default is "track_pdf".
        to_return : list, optional
            List of desired results to return at each update. The default is [].
        use_end_condition : bool, optional
            Whether to use model end-condition. The default is None.
        """
        self.t = 0.0
        self.t_ind = 0
        if not t_max:
            self.t_max = mdl.sp.end_time
        else:
            self.t_max = t_max
        if isinstance(to_return, str):
            self.to_return = [to_return]
        else:
            self.to_return = to_return
        self.mdl = mdl.new(**mdl_kwargs, track=track, sp={'end_time': self.t_max})
        self.hist = self.mdl.h
        self.run_stochastic = run_stochastic
        self.use_end_condition = use_end_condition

    def update(self, seed={}, faults={}, disturbances={}):
        """
        Update the model states at the simulation time and iterates time.

        Parameters
        ----------
        seed : seed, optional
            Seed for the simulation. The default is {}.
        faults : dict, optional
            faults to inject in the model, with structure {fxn:[faults]}.
            The default is {}.
        disturbances : dict, optional
            Variables to change in the model, with structure {fxn.var:value}.
            The default is {}.

        Returns
        -------
        returns : dict
            dictionary of returns with values corresponding to to_return
        """
        if seed:
            self.mdl.update_seed(seed)
        self.mdl(time=self.t, fxnfaults=faults, disturbances=disturbances,
                 run_stochastic=self.run_stochastic)

        returns = {}
        for result in self.to_return:
            returns[result] = self.mdl.get_vars(result)
        if self.run_stochastic == "track_pdf":
            returns['pdf'] = self.mdl.return_probdens()

        self.t += self.mdl.sp.dt
        self.t_ind += 1
        if returns:
            return returns

    def check_sim_end(self, external_condition=False):
        """
        Check the model end-condition (and sim time) and clips the simulation log.

        Parameters
        ----------
        external_condition : bool, optional
            External end-condition to trigger simulation end. The default is False.

        Returns
        -------
        end : bool
            Whether the simulation is finished
        """
        if self.t >= self.t_max:
            end = True
        elif external_condition:
            end = True
        else:
            end = propagate.check_end_condition(self.mdl,
                                                self.use_end_condition, self.t)
        if end:
            self.hist.cut(self.t_ind)
            self.mdl.h.cut(self.t_ind)
        return end


class ResponseCoords(BaseCoords):
    __slots__ = ('__dict__',)

    """
    Class for sampling functions and displaying them on maps.

    Examples
    --------
    >>> rm = ResponseCoords(lambda a, b: a*b, p={'x_size': 3, 'y_size': 3, 'blocksize': 2})
    >>> rm.fval
    array([[ 0.,  0.,  0.],
           [ 0.,  4.,  8.],
           [ 0.,  8., 16.]])
    >>> rm2 = ResponseCoords(lambda a, b: (a*b, a+b), returns={'fval': (float, 0.0), 'fval2': (float, 0.0)}, p={'x_size': 3, 'y_size': 3, 'blocksize': 2})
    >>> rm2.fval2
    array([[0., 2., 4.],
           [2., 4., 6.],
           [4., 6., 8.]])
    """

    def __init__(self, function, *args, returns={'fval': (float, 0.0)}, **kwargs):
        func_params = inspect.signature(function).parameters
        if len(func_params) > 2:
            raise Exception("Too many function parameters: "+str(func_params))
        super().__init__(*args, **kwargs)
        for ret, arg in returns.items():
            setattr(self, ret, self.p.create_grid(*arg))
        self.init_properties(function, returns=[*returns], **kwargs)

    def init_properties(self, function, returns=['fval'], **kwargs):
        """
        Initialize ResponseCoords to given function.

        Parameters
        ----------
        function : callable
            Function to sample from over map/grid.
        returns : list
            Name(s) of returns from function in order.
        """
        for pt in self.pts:
            f_ret = function(*pt)
            if len(returns) == 1:
                self.set(*pt, returns[0], f_ret)
            else:
                for i, ret in enumerate(returns):
                    self.set(*pt, ret, f_ret[i])


if __name__ == "__main__":

    from fmdtools.sim.sample import expd
    from fmdtools.define.block.function import ExampleFunction
    exprob = ExampleParameterSimProblem()
    exprob.f1(1, 1)

    exprob.f2(1, 1)
    exf = ExampleFunction()
    res, hist = propagate.one_fault(exf, 'examplefunction', 'short', 2,
                                    to_return = {3: ['s.y'], 5: ['s.y']},
                                    mdl_kwargs={'p': expd(1, 0)})
    res, hist = propagate.one_fault(exf, 'examplefunction', 'short', 2,
                                    to_return = {3: ['s.y'], 5: ['s.y']},
                                    mdl_kwargs={'p': expd(1, 1), 'sp': {'end_time': 5}},
                                    staged=True)
    import doctest
    doctest.testmod(verbose=True)
    # ex_pa.show_sequence()
