# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:38:43 2023

@author: dhulse
"""
from fmdtools.define.parameter import Parameter
from fmdtools.define.common import nest_dict, get_var


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
    >>> class ExParam(Parameter):
    ...    x: float = 1.0
    ...    y: float = 10.0
    ...    z: float = 0.0
    ...    x_lim = (0, 10)
    ...    y_set = (1.0, 2.0, 3.0, 4.0)

    We can then define the following domain, which by default gets set constraints:
    >>> expd = ParameterDomain(ExParam)
    >>> expd.add_variables("y", "x")
    >>> expd.add_constants(z=20)
    >>> expd
    ParameterDomain with:
     - variables: {'y': {1.0, 2.0, 3.0, 4.0}, 'x': (0, 10)}
     - constants: {'z': 20}
     - parameter_initializer: <class '__main__.ExParam'>

    This ParameterDomain then becomes a callable with the given variables:
    >>> expd(1, 2)
    ExParam(x=2.0, y=1.0, z=20.0)

    And we can separately check the set contraints for the variables:
    >>> expd.get_set_constraints(1, 2)
    (False, False)
    >>> expd.get_set_constraints(0, 20)
    (True, True)

    This can also work with nested parameters:
    >>> class ExNestedParam(Parameter):
    ...    ex_param: ExParam = ExParam()
    ...    k: float = 20.0

    >>> expd1 = ParameterDomain(ExNestedParam)
    >>> expd1.add_variables("ex_param.x", "ex_param.y", "k")
    >>> expd1(1,2, 3)
    ExNestedParam(ex_param=ExParam(x=1.0, y=2.0, z=0.0), k=3.0)
    """

    def __init__(self, parameter_init):
        self.parameter_init = parameter_init
        self.variables = {}
        self.constants = {}

    def add_variable(self, variable, var_set=(),  var_lim=()):
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

    def add_variables(self, *variables, sets={}, lims={}):
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
        """
        for variable in variables:
            self.add_variable(variable,
                              var_set=sets.get(variable, ()),
                              var_lim=lims.get(variable, ()))

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
                if type(const) == tuple:
                    set_const = not (const[0] <= arg <= const[1])
                elif type(const) == set:
                    set_const = not (arg in const)
            set_constraints.append(set_const)
        return tuple(set_constraints)

    def get_param_kwargs(self, *x):
        """Get kwargs for the parameter at the given value of x."""
        return {**self.constants, **x_to_kwargs(self.variables, *x)}

    def __call__(self, *x):
        """Generate the parameter at a given value of the variables."""
        kwargs = self.get_param_kwargs(*x)

        if issubclass(self.parameter_init, Parameter):
            kwargs['check_type'] = False
            kwargs['check_pickle'] = False
            kwargs['check_lim'] = False
            kwargs['set_type'] = True

        return self.parameter_init(**kwargs)

    def __repr__(self):
        rep = ("ParameterDomain with:" +
               "\n - variables: " + str(self.variables) +
               "\n - constants: " + str(self.constants) +
               "\n - parameter_initializer: " + str(self.parameter_init))
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


def x_to_kwargs(variables, *x):
    """Convert x over the defined variables into a set of kwargs."""
    var_args = {}
    for i, variable in enumerate(variables):
        if i < len(x):
            var_args[variable] = x[i]
        else:
            raise Exception("x of invalid length: "+str(len(x)))
    return nest_dict(var_args)


    


class ExParam(Parameter):
    """Example parameter for testing."""

    x: float = 1.0
    y: float = 10.0
    z: float = 0.0
    x_lim = (0, 10)
    y_set = (1.0, 2.0, 3.0, 4.0)

# example paramdomain/sample for testing
expd = ParameterDomain(ExParam)
expd.add_variables("y", "x")
expd.add_constants(z=20)
expd

class ExNestedParam(Parameter):
    ex_param: ExParam = ExParam()
    k: float = 20.0


from fmdtools.sim.scenario import NominalScenario as ParameterScenario
from fmdtools.sim.sample import BaseSample
import numpy as np
import itertools

class ParameterSample(BaseSample):
    """
    Class for sampling parameters and other immutable model attributes.

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

    def __init__(self, paramdomain, seed=None, sp={}):
        self.seed = seed
        self.seedsequence = np.random.SeedSequence(seed)
        self.sp = sp
        self.paramdomain = paramdomain
        self._scenarios = []

    def __repr__(self):
        scens = [s.name for s in self.scenarios()]
        tot = len(scens)
        if tot > 10:
            scens = scens[0:10]+["... (" + str(tot) + " total)"]
        rep = "ParameterSample of scenarios:" + "\n - " + "\n - ".join(scens)
        return rep

    def scenarios(self):
        """Return list of scenarios that make up the ParameterSample."""
        return [*self._scenarios]

    def add_variable_scenario(self, *x, seed=False, sp={}, weight=1.0, name='var'):
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

        Examples
        --------
        >>> ex_ps = ParameterSample(expd)
        >>> ex_ps.add_variable_scenario(1, 2)
        >>> ex_ps
        ParameterSample of scenarios:
         - var_0
        >>> ex_ps.scenarios()[0]
        NominalScenario(sequence={}, times=(), p={'z': 20, 'x': 2, 'y': 1}, r={}, sp={}, prob=1.0, inputparams=(1, 2), rangeid='', name='var_0')
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
        scen = ParameterScenario(p=param_args,
                                 r=r,
                                 sp=sp,
                                 prob=weight,
                                 name=name,
                                 inputparams=x)
        self._scenarios.append(scen)

    def add_variable_replicates(self, x_combos, replicates=1, seed_comb='shared',
                                name='var', weight=1.0):
        """
        Add replicates for the given cominations of variables.

        Parameters
        ----------
        x_combos : list
            List of combinations of variable values. [x_1, x_2...]
        replicates : int, optional
            Number of replicates to add of the variable. The default is 1.
        seed_comb : str, optional
            How to combine seeds ('shared' or 'independent'). 'shared' uses the same
            seed for each value while 'independent' uses different seeds over all
            variable values. The default is 'shared'.
        name : str, optional
            Name prefix for the set of replicates. The default is 'var'.
        weight : float, optional
            Total weight for the replicates (i.e.., total probability to divide between
            samples). The default is 1.0.

        Examples
        --------
        >>> ex_ps = ParameterSample(expd)
        >>> ex_ps.add_variable_replicates([[1,1], [2,2]])
        >>> ex_ps
        ParameterSample of scenarios:
         - rep0_var_0
         - rep0_var_1
        >>> ex_ps.scenarios()[0]
        NominalScenario(sequence={}, times=(), p={'z': 20, 'x': 1, 'y': 1}, r={}, sp={}, prob=0.5, inputparams=(1, 1), rangeid='', name='rep0_var_0')
        >>> ex_ps.scenarios()[1]
        NominalScenario(sequence={}, times=(), p={'z': 20, 'x': 2, 'y': 2}, r={}, sp={}, prob=0.5, inputparams=(2, 2), rangeid='', name='rep0_var_1')
        """
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
            Keyword arguments to the self.combine_'methodname'. The default is {}.
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
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [10.0, 0], [10.0, 1], [10.0, 2], [10.0, 3], [10.0, 4], [10.0, 5], [10.0, 6], [10.0, 7], [10.0, 8], [10.0, 9], [10.0, 10]]
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
            if type(ran) == set:
                x.append(rng.choice([*ran]))
            elif type(ran) == tuple:
                x.append(rng.uniform(ran[0], ran[-1]))
            else:
                raise Exception("invalid range: " + str(ran))
        x_combos.append(x)
    return x_combos
 

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
exp_ps.add_variable_scenario(1,2)
exp_ps.add_variable_ranges(replicates=10)


    

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
