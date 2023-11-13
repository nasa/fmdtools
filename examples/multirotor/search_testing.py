# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from fmdtools.sim.sample import ParameterDomain
from fmdtools.sim.search import ParameterProblem
from examples.multirotor.drone_mdl_rural import Drone, DroneParam




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