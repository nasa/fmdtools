# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from fmdtools.define.block import Simulable
from fmdtools.sim.search import SimpleProblem, BaseProblem, ProblemArchitecture
from fmdtools.sim.search import ex_sp, ex_scenprob, ex_dp
from examples.multirotor.test_multirotor import ex_soc_opt, sp, sp2

"""multirotor example cost problem"""

def descost(*x):
    batcostdict = {'monolithic': 0, 'series-split': 300,
                   'parallel-split': 300, 'split-both': 600}
    linecostdict = {'quad': 0, 'hex': 1000, 'oct': 2000}
    return [*batcostdict.values()][int(x[0])]+[*batcostdict.values()][int(x[1])]

def set_con(*x):
    return 0.5 - float(0 <= x[0] <= 3 and 0 <= x[1] <= 2)

sp0 = SimpleProblem("bat", "linearch")
sp0.add_objective("cost", descost)
sp0.add_constraint("set", set_con, comparator="less")
sp0.cost(1,1)

pa = ProblemArchitecture()
pa.add_connector_variable("vars", "bat", "linearch")
pa.add_problem("arch_cost", sp0, outputs={"vars": ["bat", "linearch"]})

pa.add_problem("arch_performance", ex_soc_opt,
               inputs={"vars": ["phys_param.bat", "phys_param.linearch"]})
#pa.add_problem("mechfault_recovery", sp, inputs=["vars"])
#pa.add_problem("charge_resilience", sp2, inputs=["vars"])

pa.show_sequence()

pa.get_downstream_probs("arch_cost")

pa.update_problem_outputs("arch_cost")
pa.get_inputs_as_x("arch_performance")

# output should propagate as input:
pa.update_problem("arch_cost", 2, 2)
pa.update_problem("arch_performance")


# disturbancedomain - callable in terms of what?
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)