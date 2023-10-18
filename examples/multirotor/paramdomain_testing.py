# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:38:43 2023

@author: dhulse
"""
from fmdtools.define.parameter import Parameter
from fmdtools.define.common import nest_dict


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

    def __call__(self, *x):
        """Generate the parameter at a given value of the variables."""
        kwargs = {**self.constants, **x_to_kwargs(self.variables, *x)}

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
    x: float = 1.0
    y: float = 10.0
    z: float = 0.0
    x_lim = (0, 10)
    y_set = (1.0, 2.0, 3.0, 4.0)

expd = ParameterDomain(ExParam)
expd.add_variables("y", "x")
expd.add_constants(z=20)
# expd(1.0, 2.0)
expd(1, 2)
set_con = expd.get_set_constraints(0,11)

class ExNestedParam(Parameter):
    ex_param: ExParam = ExParam()
    k: float = 20.0

ExNestedParam(ex_param={'x':1.0}, set_type=True)

expd1 = ParameterDomain(ExNestedParam)
expd1.add_variables("ex_param.x", "ex_param.y", "k")
expd1(1,2, 3)

class DoubleNestParam(Parameter):
    ex_nest: ExNestedParam = ExNestedParam()
    v: float=40.0

expd2 = ParameterDomain(DoubleNestParam)
expd2.add_variables("ex_nest.ex_param.x", "v")
expd2(1,2)




if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
