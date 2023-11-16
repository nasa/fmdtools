# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:45:28 2023

@author: dhulse
"""
from fmdtools.define.block import Simulable
from fmdtools.sim.search import SimpleProblem, BaseProblem, BaseObjCon
from fmdtools.sim.search import ex_sp, ex_scenprob, ex_dp
from examples.multirotor.test_multirotor import ex_soc_opt, sp, sp2

import networkx as nx
import numpy as np
from recordclass import dataobject


from fmdtools.analyze.common import setup_plot

class BaseConnector(dataobject):
    name: str=''


class VariableConnector(BaseConnector):
    keys: tuple=()
    values: np.array=np.array([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.values:
            self.values = np.array([np.nan for k in self.keys])

    def update(self, valuedict):
        for i, k in enumerate(self.keys):
            self.values[i] = valuedict[k]

    def update_values(self, *x):
        for i, x_i in enumerate(x):
            self.values[i]=x

class ModelConnector(BaseConnector):
    mdl: Simulable

class ObjectiveConnector(VariableConnector):
    a=1

class ConstraintConnector(ObjectiveConnector):
    a=1


"""Problem prototyping"""

class ProblemArchitecture(BaseProblem):

    def __init__(self):
        self.problems = {}
        self.variables = {}
        self.connectors = {}
        self.problem_graph = nx.DiGraph()
        super().__init__()

    def prob_repr(self):
        repstr = ""
        constr = " -" + "\n -".join(['{:<45}{:>20}'.format(k, str(var.values))
                                     for k, var in self.connectors.items()])
        if constr:
            repstr += "\nCONNECTORS\n" + constr
        probstr = " -" + "\n -".join([pn+"("+str(self.find_inputs(pn)+["x"])+")"
                                      + " -> "+ str(self.find_outputs(pn))
                                      for pn in self.problems])
        if probstr:
            repstr += "\nPROBLEMS\n" + probstr
        return repstr

    def add_connector_variable(self, name, *varnames):
        self.connectors[name] = VariableConnector(name, varnames)
        self.problem_graph.add_node(name)

    def add_problem(self, name, problem, inputs=[], outputs=[]):
        if self.problems:
            upstream_problem = [*self.problems][-1]
            self.problem_graph.add_edge(upstream_problem, name,
                                        label = "next")
        self.problems[name] = problem
        self.problem_graph.add_node(name, order = len(self.problems))

        for con in inputs:
            self.problem_graph.add_edge(con, name, label="input")
        for con in outputs:
            self.problem_graph.add_edge(name, con, label="output")

        self.variables.update({name+"."+k: v for k, v in problem.variables.items()})
        self.objectives.update({name+"."+k: v
                                for k, v in problem.objectives.items()})
        self.constraints.update({name+"."+k: v
                                 for k, v in problem.constraints.items()})

    def update_problem(self, probname, *x):
        # TODO: need a way update upstream sims and then update problem
        x_inputs = self.get_inputs_as_x(probname)
        self.problems[probname].update_objectives(*x_inputs, *x)
        self.update_problem_outputs(probname)

    def update_problem_outputs(self, probname):
        outputs = self.get_outputs(probname)
        for output, connector in outputs.items():
            if isinstance(connector, VariableConnector):
                connector.update(self.problems[probname].variables)
            elif isinstance(connector, ObjectiveConnector):
                connector.update(self.problems[probname].objectives)
            elif isinstance(connector, ConstraintConnector):
                connector.update(self.problems[probname].constraints)
            elif isinstance(connector, ModelConnector):
                # TODO: find a way to cache model in ParameterProb
                # TODO: make sure to handle ScenarioProb well
                connector.update(self.problems[probname].mdl)
            else:
                raise Exception("Invalid connector: "+connector)

    def find_inputs(self, probname):
        return [e[0] for e in self.problem_graph.in_edges(probname)
                if self.problem_graph.edges[e]['label']=='input']

    def find_outputs(self, probname):
        return [e[1] for e in self.problem_graph.out_edges(probname)
                if self.problem_graph.edges[e]['label']=='output']

    def get_inputs_as_x(self, probname):
        inputs = self.get_inputs(probname)
        return [vv for v in inputs.values() for vv in v.values]

    def get_inputs(self, probname):
        return {c: self.connectors[c] for c in self.find_inputs(probname)}

    def get_outputs(self, probname):
        return {c: self.connectors[c] for c in self.find_outputs(probname)}

    def get_downstream_probs(self, probname):
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[ind+1:]

    def get_upstream_probs(self, probname):
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[:ind]

    def show_sequence(self):
        fig, ax = setup_plot()
        pos = nx.planar_layout(self.problem_graph, dim=2)
        nx.draw(self.problem_graph, pos=pos)
        orders = nx.get_node_attributes(self.problem_graph, "order")
        names = nx.get_node_attributes(self.problem_graph, "label")
        labels = {node: str(orders[node]) + ": " + node if node in orders else node
                  for node in self.problem_graph}
        nx.draw_networkx_labels(self.problem_graph, pos, labels=labels)
        edge_labels = nx.get_edge_attributes(self.problem_graph, "label")
        nx.draw_networkx_edge_labels(self.problem_graph, pos, edge_labels=edge_labels)
        return fig, ax

ex_sp, ex_scenprob, ex_dp

ex_pa = ProblemArchitecture()
ex_pa.add_connector_variable("x0", "x0")
ex_pa.add_connector_variable("x1", "x1")
ex_pa.add_problem("ex_sp", ex_sp, outputs=["x0", "x1"])
ex_pa.add_problem("ex_scenprob", ex_scenprob, inputs=["x0"])
ex_pa.add_problem("ex_dp", ex_dp, inputs=["x1"])

ex_pa.show_sequence()

ex_pa.update_problem("ex_sp", 1, 2)

# should reflect new variable values:
ex_pa.problems["ex_sp"]

# variables should propagate:
ex_pa.update_problem("ex_scenprob")
ex_pa.problems["ex_scenprob"]

#variables shoudl propagate again:
ex_pa.update_problem("ex_dp")
ex_pa.problems['ex_dp']


"""multirotor example cost problem"""

def descost(*x):
    batcostdict = {'monolithic': 0, 'series-split': 300,
                   'parallel-split': 300, 'split-both': 600}
    linecostdict = {'quad': 0, 'hex': 1000, 'oct': 2000}
    return [*batcostdict.values()][x[0]]+[*batcostdict.values()][x[1]]

def set_con(*x):
    return 0.5 - float(0 <= x[0] <= 3 and 0 <= x[1] <= 2)

sp0 = SimpleProblem("bat", "linearch")
sp0.add_objective("cost", descost)
sp0.add_constraint("set", set_con, comparator="less")
sp0.cost(1,1)

pa = ProblemArchitecture()
pa.add_connector_variable("vars", "bat", "linearch")
pa.add_problem("arch_cost", sp0, outputs=["vars"])

pa.add_problem("arch_performance", ex_soc_opt, inputs=["vars"])
#pa.add_problem("mechfault_recovery", sp, inputs=["vars"])
#pa.add_problem("charge_resilience", sp2, inputs=["vars"])

pa.show_sequence()

pa.get_downstream_probs("arch_cost")

pa.update_problem_outputs("arch_cost")
pa.get_inputs_as_x("arch_performance")

# output should propagate as input:
pa.update_problem("arch_cost", 2, 2)
pa.update_problem("arch_performance")

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