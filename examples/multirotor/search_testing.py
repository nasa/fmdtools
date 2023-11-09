# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""

from fmdtools.sim.search import ProblemInterface
from fmdtools.sim.scenario import Scenario, SingleFaultScenario, JointFaultScenario
from fmdtools.sim.scenario import ParameterScenario
from drone_mdl_rural import Drone, DroneParam, ResPolicy
from fmdtools.sim.sample import FaultDomain, FaultSample
import numpy as np
from recordclass import dataobject



class Variable(dataobject):
    name: str = ''
    value: float = None
    min_value: float = - np.inf
    max_value: float = np.inf


class Constraint(dataobject):
    name: str = ''
    value: float = None


class Objective(dataobject):
    name: str = ''
    value: float = None


class SimProblem(object):
    def __init__(self, mdl, prop_method, *args, **kwargs):
        self.mdl = mdl
        self.prop_method = prop_method
        self.default_args = args
        self.default_kwargs = kwargs

        self.variables = {}
        self.objectives = {}
        self.constraints = {}

    def add_variable(self, name, varclass=Variable, **kwargs):
        self.variables[name] = varclass(name, **kwargs)

    def add_objective(self, name, objclass=Objective, **kwargs):
        self.objectives[name] = objclass(name, **kwargs)

    def add_constraint(self, name, conclass=Constraint, **kwargs):
        self.constraints[name] = conclass(name, **kwargs)


s = SimProblem(Drone(), 'nominal')
s.add_variable('p.')
from fmdtools.sim.sample import ParameterDomain

expd1 = ParameterDomain(DroneParam)
expd1.add_variables("phys_param.bat", "phys_param.linearch")
expd1("series-split", "oct")

# two types of variables:
# parameter variable
# varnames + mapping
# -> creation of a parameterdomain to sample from
# -> mapping tells us whether to sample directly or call mapping first

# scenario variable
# fault or disturbance
# fault variable is the time or type of fault
# disturbance is the time or str of disturbance
# maybe we have a domain for these?
# faultdomain - callable in terms of what?
# disturbancedomain - callable in terms of what?