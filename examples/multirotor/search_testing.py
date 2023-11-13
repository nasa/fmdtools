# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import ParameterProblem
from examples.multirotor.drone_mdl_rural import Drone, DroneParam



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
s.add_result_objective("f1", "dofs.s.z", time=16)
s.add_result_objective("f2", "store_ee.s.soc", time=16)
s.sim_mdl(1,1)
s.update_objectives(1,1)
s.f1(3,0)
s.f2(3,0)

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
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)