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
import fmdtools.sim.propagate as propagate
from fmdtools.define.common import t_key


class BaseObjCon(dataobject):
    name: str = ''
    value: float = None


class ResultObjective(BaseObjCon):
    time: float = None
    metric: callable = np.sum
    negative: bool = False

    def get_result_value(self, res):
        if not self.time:
            t = 'end'
        else:
            t = t_key(float(self.time))
        met = res.get_metric(t+"."+self.name, metric=self.metric)
        return met

    def obj_from_value(self, value):
        if self.negative:
            value = - value
        else:
            value = value
        return value

    def update(self, res):
        value = self.get_result_value(res)
        self.value = self.obj_from_value(value)



class ResultConstraint(ResultObjective):
    threshold: float = 0.0
    comparator: str = 'greater'

    def con_from_value(self, value):
        if self.comparator == 'greater':
            value = self.threshold - value
        elif self.comparator == 'less':
            value = value - self.threshold
        else:
            raise Exception("Invalid comparator: "+self.comparator)
        return self.obj_from_value(value)

    def update(self, res):
        value = self.get_result_value(res)
        self.value = self.con_from_value(value)


class ParameterProblem(object):
    def __init__(self, mdl, parameterdomain, prop_method, *args, **kwargs):
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
        rep_str += "\n"+"VARIABLES \n" + var_str
        obj_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(v.name+":", v.value)
                                      for v in self.objectives.values()])
        if self.objectives:
            rep_str += "\n" + "OBJECTIVES \n" + obj_str
        con_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(v.name+":", v.value)
                                      for v in self.constraints.values()])
        if self.constraints:
            rep_str += "\n" + "CONSTRAINTS \n" + con_str
        return rep_str

    def add_result_objective(self, name, objclass=ResultObjective, **kwargs):
        self.objectives[name] = objclass(name, **kwargs)

    def add_result_constraint(self, name, conclass=ResultConstraint, **kwargs):
        self.constraints[name] = conclass(name, **kwargs)

    def get_end_time(self):
        last_time = self.mdl.sp.times[-1]
        all_times = [a.time if a.time else last_time
                     for a in {**self.objectives, **self.constraints}.values()]
        end_time = max(all_times)
        return end_time

    def obj_con_des_res(self):
        des_res = {}
        for n in {**self.objectives, **self.constraints}.values():
            if n.time:
                t = n.time
            else:
                t = 'end'
            if t in des_res:
                des_res.append(n.name)
            else:
                des_res[t] = [n.name]
        return des_res

    def sim_mdl(self, *x):
        p = self.parameterdomain(*x)
        end_time = self.get_end_time()
        mdl_kwargs = {'p': p, 'sp': {'times': (0.0, end_time)}}
        desired_result = self.obj_con_des_res()
        res, hist = self.prop_method(self.mdl, *self.args,
                                     mdl_kwargs=mdl_kwargs,
                                     desired_result=desired_result,
                                     **self.kwargs)
        return res, hist

    def update_objectives(self, *x):
        for i, v in enumerate(self.variables):
            self.variables[v] = x[i]
        res, hist = self.sim_mdl(*x)
        res = res.flatten()
        for obj in {**self.objectives, **self.constraints}.values():
            obj.update(res)


from fmdtools.sim.sample import ParameterDomain


bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']

def bat_var_map(x):
    return (bats[x], )

linarchs = ['quad', 'hex', 'oct']

def line_arch_map(x):
    return (linarchs[x], )

expd1 = ParameterDomain(DroneParam)
expd1.add_variable("phys_param.bat", var_map=bat_var_map, var_lim=(0, 3))
expd1.add_variable("phys_param.linearch", var_map=line_arch_map, var_lim=(0, 2))
expd1(0, 0)
expd1.get_set_constraints(0, 0)
expd1.get_set_constraints(4, 1)


s = ParameterProblem(Drone(), expd1, 'nominal', track=None)
s.add_result_objective("dofs.s.z", time=10)
s.sim_mdl(1,1)
s.update_objectives(1,1)

#expd1("series-split", "oct")

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