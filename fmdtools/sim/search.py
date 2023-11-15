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
from fmdtools.sim.scenario import Sequence, SingleFaultScenario, Scenario
from fmdtools.sim.sample import FaultDomain


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


class Objective(BaseObjCon):
    """
    
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
        """
        if self.comparator == 'greater':
            value = self.threshold - value
        elif self.comparator == 'less':
            value = value - self.threshold
        else:
            raise Exception("Invalid comparator: "+self.comparator)
        return self.obj_from_value(value)

    def update(self, value):
        """Update with given value."""
        self.value = self.con_from_value(value)


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

    def __init__(self):
        self.variables = {}
        self.objectives = {}
        self.constraints = {}

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

    def add_objective_callable(self, name):
        """Add callable objective function with name name."""
        def newobj(*x):
            return self.call_objective(*x, objective=name)
        setattr(self, name, newobj)

    def add_constraint(self, name, varname, conclass=Constraint, **kwargs):
        """Add a constraint to the Problem."""
        self.constraints[name] = conclass(varname, **kwargs)
        self.add_constraint_callable(name)

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

    def update_variables(self, *x):
        """Update variables at x."""
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


class SimpleProblem(BaseProblem):
    """
    Simple optimization problem (without any given model constructs).

    Attributes
    ----------
    callables : dict
        dict of callables for objectives/constraints

    Examples
    --------
    >>> sp = SimpleProblem("x0", "x1")
    >>> f1 = lambda x0, x1: x0 + x1
    >>> sp.add_objective("f1", f1)
    >>> g1 = lambda x0, x1: x0 - x1
    >>> sp.add_constraint("g1", g1, threshold=3.0, comparator="less")

    >>> sp.f1(1, 1)
    2
    >>> sp.g1(1, 1)
    -3.0
    """

    def __init__(self, *variables):
        super().__init__()
        self.variables = {v: np.NaN for v in variables}
        self.callables = {}

    def update_objectives(self, *x):
        """Update objectives/constraints by calling callables."""
        self.update_variables(*x)
        for objname, obj in {**self.objectives, **self.constraints}.items():
            obj.update(self.callables[objname](*x))

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


class ResultObjective(Objective):
    """
    Base class of objectives which derive from Results.

    Fields
    ------
    time : float
        Time the objective is called at. If None, time will be the end of the sim.
    metric : callable
        Metric to tabulate for the objective. Default is np.sum.

    """

    time: float = None
    metric: callable = np.sum

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

    def update(self, res):
        """Update the value of the constraint given the result."""
        value = self.get_result_value(res)
        self.value = self.con_from_value(value)

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


class BaseSimProblem(BaseProblem):
    """
    Base optimization problem for optimizing over simulations.

    Attributes
    ----------
    prop_method : callable
        Method in propagate to call.
    """

    def __init__(self, mdl, prop_method, *args, **kwargs):
        self.mdl = mdl
        if type(prop_method) == str:
            self.prop_method = getattr(propagate, prop_method)
        elif callable(prop_method):
            self.prop_method = prop_method
        else:
            raise Exception("Invalid prop_method "+str(prop_method))

        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def add_result_objective(self, name, varname, **kwargs):
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
        **kwargs : kwargs
            Arguments to ResultObjective
        """
        self.add_objective(name, varname, objclass=ResultObjective, **kwargs)

    def add_result_constraint(self, name, varname, **kwargs):
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
        **kwargs : kwargs
            Arguments to ResultConstraint
        """
        self.add_constraint(name, varname, conclass=ResultConstraint, **kwargs)

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

    def update_objectives(self, *x):
        """Update objectives/constraints by simulating the model at x."""
        self.update_variables(*x)
        res, hist = self.sim_mdl(*x)
        res = res.flatten()
        for obj in {**self.objectives, **self.constraints}.values():
            obj.update(res)


class ParameterSimProblem(BaseSimProblem):
    """
    Optimization problem defining the optimization of model parameters over simulations.

    Examples
    --------
    >>> from fmdtools.sim.sample import expd
    >>> from fmdtools.define.block import ExampleFxnBlock

    # below, we show basic setup of a parameter problem where objectives get values
    # from the sim at particular times.
    >>> exprob = ParameterSimProblem(ExampleFxnBlock(), expd, "nominal")
    >>> exprob.add_result_objective("f1", "s.x", time=5)
    >>> exprob.add_result_objective("f2", "s.y", time=5)
    >>> exprob.add_result_constraint("g1", "s.x", time=10, threshold=10, comparator='greater')
    >>> exprob
    ParameterSimProblem with:
    VARIABLES
     -y                                                             nan
     -x                                                             nan
    OBJECTIVES
     -f1                                                            inf
     -f2                                                            inf
    CONSTRAINTS
     -g1                                                            inf

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
    >>> exprob = ParameterSimProblem(ExampleFxnBlock(), expd, "nominal")
    >>> exprob.add_result_objective("f1", "endclass.xy")
    >>> exprob.f1(1, 1)
    100.0
    >>> exprob.f1(1, 2)
    200.0

    # finally, note that this class can work with a variety of methods:
    >>> exprob = ParameterSimProblem(ExampleFxnBlock("ex"), expd, "one_fault", "ex", "short", 2)
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
        super().__init__(mdl, prop_method, *args, **kwargs)
        self.parameterdomain = parameterdomain
        self.variables = {v: np.NaN for v in self.parameterdomain.variables}

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
        return res.flatten(), hist.flatten()


class ScenarioProblem(BaseSimProblem):
    """
    Base class for optimizing scenario parameters.

    Attributes
    ----------
    prepped_sims : dict
        Dict of outputs from propagate.nom_helper. Used for staged execution of
        scenarios (where the model is copied instead of re-simulated).
    """

    def __init__(self, mdl, faultdomain=None, phasemap=None, **kwargs):
        super().__init__(mdl, "prop_one_scen", **kwargs)
        self.prepped_sims = {}

    def prep_sim(self):
        """Prepare simulation by simulating it until the start of the scenario."""
        end_time = self.get_end_time()
        mdl_kwargs = {'sp': {'times': (0.0, end_time)}}
        run_kwarg = propagate.pack_run_kwargs(**self.kwargs, mdl_kwargs=mdl_kwargs)
        desired_result = self.obj_con_des_res()
        sim_kwarg = propagate.pack_sim_kwargs(**self.kwargs,
                                              desired_result=desired_result,
                                              staged=True)
        n_outs = propagate.nom_helper(self.mdl, [self.get_start_time()],
                                      **{**sim_kwarg, 'use_end_condition': False},
                                      **run_kwarg)
        self.prepped_sims = {"result": n_outs[0],
                             "hist": n_outs[1],
                             "scen": n_outs[2],
                             "mdls": n_outs[3],
                             "t_end_nom": n_outs[4]}

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
        if not self.prepped_sims:
            self.prep_sim()

        scen = self.gen_scenario(*x)

        mdl = propagate.copy_staged([*self.prepped_sims['mdls'].values()][0])
        desired_result = self.obj_con_des_res()
        sim_kwarg = propagate.pack_sim_kwargs(**self.kwargs,
                                              desired_result=desired_result,
                                              staged=True)
        res, hist, _, t_end = self.prop_method(mdl,
                                               scen,
                                               nomhist=self.prepped_sims['hist'],
                                               nomresult=self.prepped_sims['result'],
                                               **sim_kwarg)
        return res.flatten(), hist.flatten()


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
        Minimum start time for the simulation and lower bound on scenario time..
        Default is 0.0.

    Examples
    --------
    >>> from fmdtools.define.block import ExampleFxnBlock
    >>> sp = SingleFaultScenarioProblem(ExampleFxnBlock(), ("examplefxnblock", "short"))
    >>> sp.add_result_objective("f1", "s.y", time=5)

    # objective value should be 1.0 (init value) + 3 * time_with_fault
    >>> sp.f1(5.0)
    4.0
    >>> sp.f1(4.0)
    7.0
    """

    def name_repr(self):
        """Get name of the class and faults."""
        faulttup = [*self.faultdomain.faults.keys()][0]
        return "SingleScenarioProblem("+faulttup[0]+", "+faulttup[1]+")"

    def __init__(self, mdl, faulttup, phasemap=None, t_start=0.0, **kwargs):
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
        t_start : float, optional
            Minimum start time for the simulation and lower bound on scenario time..
            Default is 0.0.
        **kwargs : kwargs
            Keyword arguments to prop_one_scen (e.g., track, etc.).
        """
        faultdomain = FaultDomain(mdl)
        faultdomain.add_fault(*faulttup)
        self.faultdomain = faultdomain
        self.phasemap = phasemap
        self.t_start = t_start
        super().__init__(mdl, **kwargs)
        self.variables = {"time": np.nan}

    def get_start_time(self):
        """Get the scenario start time to copy the model at."""
        return self.t_start

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
        starttime=self.get_start_time()
        end_time = self.get_end_time()
        if not starttime <= x <= end_time:
            raise Exception("time out of range: "+str((starttime, end_time)))
        fault = [*self.faultdomain.faults][0]
        scen = SingleFaultScenario.from_fault(fault, time=x, mdl=self.mdl,
                                              phasemap=self.phasemap,
                                              starttime=self.get_start_time())
        return scen


class DisturbanceProblem(ScenarioProblem):
    """Class for optimizing disturbances that occur at a set time."""

    def __init__(self, mdl, time, *disturbances, **kwargs):
        """
        Initialize the DisturbanceProblem.

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

        Examples
        --------
        >>> from fmdtools.define.block import ExampleFxnBlock
        >>> dp = DisturbanceProblem(ExampleFxnBlock(), 3, "s.y")
        >>> dp.add_result_objective("f1", "s.y", time=5)

        # objective value should the same as the input value
        >>> dp.f1(5.0)
        5.0
        >>> dp.f1(4.0)
        4.0
        """
        super().__init__(mdl, **kwargs)
        self.variables = {d: np.nan for d in disturbances}
        self.time = time

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
        seq = Sequence(disturbances=dist)
        scen = Scenario(sequence=seq,
                        name='disturbance',
                        time=self.time)
        return scen


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