# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from examples.multirotor.drone_mdl_rural import Drone


from fmdtools.sim.search import BaseSimProblem
from fmdtools.sim import propagate
import numpy as np



# Fault set / sequence generator
def gen_single_fault_times(fd, *x):
    sequences = []
    for i, fault in enumerate(fd.faults):
        seq = Sequence.from_fault(fault, x[i])
        sequences.append(seq)
    return sequences


seqs = gen_single_fault_times(fd1, *[i for i in range(len(fd1.faults))])


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