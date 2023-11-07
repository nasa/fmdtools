# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""

from fmdtools.sim.search import ProblemInterface
from drone_mdl_rural import Drone, DroneParam, ResPolicy
from fmdtools.sim.sample import FaultDomain, FaultSample
import numpy as np

class SimProblem(object):
    def __init__(self, mdl, prop_method, *args, **kwargs):
        self.mdl = mdl
        self.prop_method = prop_method
        self.default_args = args
        self.default_kwargs = kwargs

    def add_param_variable(self, varname, set_const=(), default=None):
        a=1

    def add_fault_variable(self, varname, set_const=(), default=None):
        b=1

    def add_disturbance_variable(self, varname, set_const=(), default=None):
        c=1

    def add_variables(self, vartype, **variables):
        d=1

    def add_result_objective(self, objname, time='end'):
        e=1

    def add_var_objective(self, varname):
        f=1

    def add_external_objective(self, func):
        g=1

    def add_result_constraint(self, objname, time='end'):
        e=1

    def add_var_constraint(self, varname):
        f=1

    def add_external_constraint(self, func):
        g=1