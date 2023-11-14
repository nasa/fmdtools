# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from fmdtools.sim.search import SimpleProblem
from examples.multirotor.test_multirotor import ex_soc_opt, sp, sp2

import networkx as nx
import numpy as np


from fmdtools.analyze.common import setup_plot



def descost(*x):
    batcostdict = {'monolithic': 0, 'series-split': 300,
                   'parallel-split': 300, 'split-both': 600}
    linecostdict = {'quad': 0, 'hex': 1000, 'oct': 2000}
    return [*batcostdict.values()][x[0]]+[*batcostdict.values()][x[1]]

def set_con(*x):
    return 0.5 - float(0 <= x[0] <= 3 and 0 <= x[1] <= 2)

sp = SimpleProblem("bat", "arch")
sp.add_objective("cost", descost)
sp.add_constraint("set", set_con, comparator="less")
sp.cost(1,1)


class ProblemArchitecture(object):

    def __init__(self):
        self.problems = {}
        self.problem_graph = nx.DiGraph()

    def add_problem(self, probname, problem, upstream_problems={}):
        self.problems[probname] = problem
        for upprob in upstream_problems:
            self.problem_graph.add_edge(upprob, probname,
                                        label=upstream_problems[upprob])

    def show_sequence(self):
        fig, ax = setup_plot()
        pos = nx.kamada_kawai_layout(self.problem_graph, dim=2)
        nx.draw(self.problem_graph, with_labels=True, pos=pos)
        edge_labels = nx.get_edge_attributes(self.problem_graph, "label")
        nx.draw_networkx_edge_labels(self.problem_graph, pos, edge_labels=edge_labels)
        return fig, ax


pa = ProblemArchitecture()
pa.add_problem("arch_cost", sp)
pa.add_problem("arch_performance", ex_soc_opt, upstream_problems={"arch_cost": "vars"})
pa.add_problem("mechfault_recovery", sp, upstream_problems={"arch_performance": 'mdl'})
pa.add_problem("charge_resilience", sp2, upstream_problems={"arch_performance": 'mdl'})

pa.show_sequence()



# Fault set / sequence generator
# def gen_single_fault_times(fd, *x):
#     sequences = []
#     for i, fault in enumerate(fd.faults):
#         seq = Sequence.from_fault(fault, x[i])
#         sequences.append(seq)
#     return sequences


#seqs = gen_single_fault_times(fd1, *[i for i in range(len(fd1.faults))])


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