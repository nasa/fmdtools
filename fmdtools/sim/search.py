# -*- coding: utf-8 -*-
"""
Description: Functions and Classes to enable optimization and search of fault model states and parameters.

Classes:
    - :class:`ProblemInterface`:  Creates an interface for model simulations for optimization methods
    - :class:`DynamicInterface`:  Creates an interface for model simulations for dynamic optimization of a single sim
"""
import numpy as np
from recordclass import dataobject
import fmdtools.sim.propagate as propagate
from fmdtools.define.common import t_key


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
    value: float = np.inf


class ResultObjective(BaseObjCon):
    """
    Base class of objectives which derive from Results.

    Fields
    ------
    time : float
        Time the objective is called at. If None, time will be the end of the sim.
    metric : callable
        Metric to tabulate for the objective. Default is np.sum.
    negative : bool
        Whether the objective is the negative of the value.
    """

    time: float = None
    metric: callable = np.sum
    negative: bool = False

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
        10.0

        >>> obj = ResultObjective("a.b", time=1.0, metric=np.sum)
        >>> res = Result({'scen1.t1p0.a.b': 10.0, 'scen2.t1p0.a.b': 12.0})
        >>> obj.get_result_value(res)
        22.0
        """
        if not self.time:
            val = res.get_metric(self.name, metric=self.metric)
        else:
            t = t_key(float(self.time))
            val = res.get_metric(t+"."+self.name, metric=self.metric)
        return val

    def obj_from_value(self, value):
        """Get the (+ or 0) objective corresponding to value give self.negative."""
        if self.negative:
            value = - value
        else:
            value = value
        return value

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
        Theshold for the constraint. Default is 0.0
    comparator : str
        Whether the constraint is 'greater' or 'less'.
    """

    threshold: float = 0.0
    comparator: str = 'greater'

    def con_from_value(self, value):
        """
        Get the constraint given the value of its variable given threshold.

        By default, constraints follow the form:
            g(x) = threshold - value > 0.0 for 'greater' constraints or
            g(x) = value - theshold > 0.0 for 'less' constraints.

        Parameters
        ----------
        value : float
            Variable value corresponding to the constraint

        Returns
        -------
        con : float
            Constraint function at value.

        Examples
        --------
        >>> con = ResultConstraint("a", threshold=10.0, comparator='greater')
        >>> con.con_from_value(11.0)
        -1.0

        >>> con2 = ResultConstraint("a", threshold=10.0, comparator='less')
        >>> con2.con_from_value(11.0)
        1.0
        """
        if self.comparator == 'greater':
            value = self.threshold - value
        elif self.comparator == 'less':
            value = value - self.threshold
        else:
            raise Exception("Invalid comparator: "+self.comparator)
        return self.obj_from_value(value)

    def update(self, res):
        """Update the value of the constraint given the result."""
        value = self.get_result_value(res)
        self.value = self.con_from_value(value)


class ParameterProblem(object):
    """
    Optimization problem defining the optimization of model parameters over simulations.

    Attributes
    ----------
    variables : dict
        Variables being optimized.
    objectives : dict
        Objectives returned.
    constraints : dict
        Constraints returned.

    Examples
    --------
    >>> from fmdtools.sim.sample import expd
    >>> from fmdtools.define.block import ExampleFxnBlock

    # below, we show basic setup of a parameter problem where objectives get values
    # from the sim at particular times.
    >>> exprob = ParameterProblem(ExampleFxnBlock(), expd, "nominal")
    >>> exprob.add_result_objective("f1", "s.x", time=5)
    >>> exprob.add_result_objective("f2", "s.y", time=5)
    >>> exprob.add_result_constraint("g1", "s.x", time=10, threshold=10, comparator='greater')
    >>> exprob
    ParameterProblem with:
    VARIABLES
     -y                                                          0.0000
     -x                                                          0.0000
    OBJECTIVES
     -s.x:                                                          inf
     -s.y:                                                          inf
    CONSTRAINTS
     -s.x:                                                          inf

    # once this is set up, you can use the objectives/constraints as callables, like so:
    >>> exprob.f1(1, 0)
    0.0
    >>> exprob.f1(1, 1)
    5.0
    >>> exprob.f1(1, 2)
    10.0
    >>> exprob.f2(1, 2)
    0.0
    >>> exprob.g1(1, 2)
    -10.0

    # below, we use the endclass as an objective instead of the variable:
    >>> exprob = ParameterProblem(ExampleFxnBlock(), expd, "nominal")
    >>> exprob.add_result_objective("f1", "endclass.xy")
    >>> exprob.f1(1, 1)
    100.0
    >>> exprob.f1(1, 2)
    200.0

    # finally, note that this class can work with a variety of methods:
    >>> exprob = ParameterProblem(ExampleFxnBlock("ex"), expd, "one_fault", "ex", "short", 2)
    >>> exprob.add_result_objective("f1", "s.y", time=3)
    >>> exprob.add_result_objective("f2", "s.y", time=5)
    >>> exprob.f1(1, 1)
    2.0
    >>> exprob.f2(1, 1)
    4.0
    """

    def __init__(self, mdl, parameterdomain, prop_method, *args, **kwargs):
        """
        Define the Parameter problem model, domain, and simulation.

        Parameters
        ----------
        mdl : Simulable
            Model to simulate.
        parameterdomain : ParameterDomain
            ParameterDomain defining variables to optimize over
        prop_method : str/callable
            Name of function to call in fmdtools.sim.propagate
        *args : args
            Arguments to prop_method.
        **kwargs : kwargs
            Keyword arguments to prop_method.
        """
        self.mdl = mdl
        self.parameterdomain = parameterdomain
        if type(prop_method) == str:
            self.prop_method = getattr(propagate, prop_method)
        elif callable(prop_method):
            self.prop_method = prop_method
        else:
            raise Exception("Invalid prop_method "+str(prop_method))

        self.args = args
        self.kwargs = kwargs

        self.variables = {v: 0.0 for v in self.parameterdomain.variables}
        self.objectives = {}
        self.constraints = {}

    def __repr__(self):
        rep_str = "ParameterProblem with:"
        var_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.variables.items()])
        rep_str += "\n"+"VARIABLES\n" + var_str
        obj_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(v.name+":", v.value)
                                      for v in self.objectives.values()])
        if self.objectives:
            rep_str += "\n" + "OBJECTIVES\n" + obj_str
        con_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(v.name+":", v.value)
                                      for v in self.constraints.values()])
        if self.constraints:
            rep_str += "\n" + "CONSTRAINTS\n" + con_str
        return rep_str

    def add_result_objective(self, name, varname, objclass=ResultObjective, **kwargs):
        """
        Add an objective corresponding to a possible desired_result.

        Associates a callable to the problem with name 'name' which may be called to
        evaluate the objective at a value of x.

        Parameters
        ----------
        name : str
            Name to give the objective
        varname : str
            Name of the variable to get for the variable.
        objclass : class, optional
            Class inheritying ResultObjective. The default is ResultObjective.
        **kwargs : kwargs
            Arguments to ResultObjective
        """
        self.objectives[name] = objclass(varname, **kwargs)

        def newobj(*x):
            return self.call_objective(*x, objective=name)
        setattr(self, name, newobj)

    def add_result_constraint(self, name, varname, conclass=ResultConstraint, **kwargs):
        """
        Add an objective corresponding to a possible desired_result.

        Associates a callable to the problem with name 'name' which may be called to
        evaluate the constraint at a value of x.

        Parameters
        ----------
        name : str
            Name to give the constraint.
        varname : str
            Name of the variable to get for the constraint.
        conclass : class, optional
            Class inheritying ResultConstraint. The default is ResultConstraint.
        **kwargs : kwargs
            Arguments to ResultConstraint
        """
        self.constraints[name] = conclass(varname, **kwargs)

        def newcon(*x):
            return self.call_constraint(*x, constraint=name)
        setattr(self, name, newcon)

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
        last_time = self.mdl.sp.times[-1]
        all_times = [a.time if a.time else last_time
                     for a in {**self.objectives, **self.constraints}.values()]
        end_time = max(all_times)
        return end_time

    def obj_con_des_res(self):
        """
        Get the desired_result argument for the problem given objectives/constraints.

        Returns
        -------
        des_res : dict
            desired_result argument to prop_method.
        """
        des_res = {}
        for n in {**self.objectives, **self.constraints}.values():
            if n.time:
                t = n.time
            else:
                t = 'endclass'
            if t in des_res:
                des_res[t].append(n.name)
            else:
                des_res[t] = [n.name]
        return des_res

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
        mdl_kwargs = {'p': p, 'sp': {'times': (0.0, end_time)}}
        desired_result = self.obj_con_des_res()
        res, hist = self.prop_method(self.mdl, *self.args,
                                     mdl_kwargs=mdl_kwargs,
                                     desired_result=desired_result,
                                     **self.kwargs)
        return res, hist

    def current_x(self):
        """Get the current variable value x."""
        return [v for v in self.variables.values()]

    def new_x(self, *x):
        """Check if a given x is the same as the current value of x."""
        return not self.current_x() == list(x)

    def update_objectives(self, *x):
        """Update objectives/constraints by simulating the model at x."""
        for i, v in enumerate(self.variables):
            self.variables[v] = x[i]
        res, hist = self.sim_mdl(*x)
        res = res.flatten()
        for obj in {**self.objectives, **self.constraints}.values():
            obj.update(res)

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

    def get_objectives(self):
        """Get all current objective values."""
        return [v.value for v in self.objectives.values()]

    def get_constraints(self):
        """Get all current constraint values."""
        return [v.value for v in self.constraints.values()]

    def call_outputs(self, *x):
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
        if self.new_x(*x):
            self.update_objectives(*x)
        return self.get_objectives(), self.get_constraints()



class DynamicInterface():
    """ 
    Interface for dynamic search of model states (e.g., AST)
    
    Attributes:
        t : float
            time
        t_max : float
            max time
        t_ind : int
            time index in log
        desired_result : list
            variables to get from the model at each time-step
        hist : History
            mdlhist for simulation
    """
    def __init__(self, mdl, mdl_kwargs={}, t_max=False, track="all", run_stochastic="track_pdf", desired_result=[], use_end_condition=None):
        """
        Initializing the problem

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
            Whether to run stochastic behaviors (True/False) and/or return pdf "track_pdf". The default is "track_pdf".
        desired_result : list, optional
            List of desired results to return at each update. The default is [].
        use_end_condition : bool, optional
            Whether to use model end-condition. The default is None.
        """
        self.t=0.0
        self.t_ind=0
        if not t_max:   self.t_max=mdl.sp.times[-1]
        else:           self.t_max = t_max
        if type(desired_result)==str:   self.desired_result=[desired_result]
        else:                           self.desired_result = desired_result
        self.mdl = mdl.new_with_params(**mdl_kwargs)
        timerange= np.arange(self.t, self.t_max+2*mdl.sp.dt, mdl.sp.dt)
        self.hist = mdl.create_hist(timerange, track)
        if 'time' not in self.hist: 
            self.hist.init_att('time', timerange[0], timerange=timerange, track='all', dtype=float)
        self.run_stochastic=run_stochastic
        self.use_end_condition = use_end_condition
    def update(self, seed={}, faults={}, disturbances={}):
        """
        Updates the model states at the simulation time and iterates time

        Parameters
        ----------
        seed : seed, optional
            Seed for the simulation. The default is {}.
        faults : dict, optional
            faults to inject in the model, with structure {fxn:[faults]}. The default is {}.
        disturbances : dict, optional
            Variables to change in the model, with structure {fxn.var:value}. The default is {}.

        Returns
        -------
        returns : dict
            dictionary of returns with values corresponding to desired_result
        """
        if seed: self.mdl.update_seed(seed)
        self.mdl.propagate(self.t, fxnfaults=faults, disturbances=disturbances, run_stochastic=self.run_stochastic)
        self.hist.log(self.mdl, self.t_ind, time=self.t)
        
        returns = {}
        for result in self.desired_result:      returns[result] = self.mdl.get_vars(result)
        if self.run_stochastic=="track_pdf":    returns['pdf'] = self.mdl.return_probdens()

        self.t += self.mdl.sp.dt
        self.t_ind +=1
        if returns: return returns
    def check_sim_end(self, external_condition=False):
        """
        Checks the model end-condition (and sim time) and clips the simulation log if necessary
        
        Parameters
        ----------
        external_condition : bool, optional
            External end-condition to trigger simulation end. The default is False.
        
        Returns
        ----------
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
            propagate.cut_mdlhist(self.log, self.t_ind)
        return end

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)