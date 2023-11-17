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
        if self.values.size == 0:
            self.values = np.array([np.nan for k in self.keys])

    def update(self, valuedict):
        for i, k in enumerate(self.keys):
            self.values[i] = valuedict[k]

    def update_values(self, *x):
        for i, x_i in enumerate(x):
            self.values[i]=x_i

    def get(self, key):
        return self.values[self.keys.index(key)]

class ModelConnector(BaseConnector):
    mdl: Simulable

class ObjectiveConnector(VariableConnector):
    a=1

class ConstraintConnector(ObjectiveConnector):
    a=1


def obj_name(probname, objname):
    return probname + "_" + objname


"""Problem prototyping"""

class ProblemArchitecture(BaseProblem):
    """
    Class enabling the representation of combined joint optimization problems.

    Combined optimization problems involve multiple variables and objectives which
    interact (e.g., Integrated Resilience Optimization, Two-Stage Optimization, etc.)

    Attributes
    ----------
    connectors : dict
        Dictionary of Connector (variables, models, etc) added using .add_connector
    problems : dict
        Dictionary of optimization problems added using .add_problem

    Examples
    --------
>>> ex_pa = ProblemArchitecture()
>>> ex_pa.add_connector_variable("x0", "x0")
>>> ex_pa.add_connector_variable("x1", "x1")
>>> ex_pa.add_problem("ex_sp", ex_sp, outputs={"x0": ["x0"], "x1": ["x1"]})
>>> ex_pa.add_problem("ex_scenprob", ex_scenprob, inputs={"x0": ["time"]})
>>> ex_pa.add_problem("ex_dp", ex_dp, inputs={"x1": ["s.y"]})
    >>> ex_pa
    ProblemArchitecture with:
    CONNECTORS
     -x0                                                          [nan]
     -x1                                                          [nan]
    PROBLEMS
     -ex_sp({'ex_sp_xloc': ['x0', 'x1']}) -> ['x0', 'x1']
     -ex_scenprob({'x0': ['time']}) -> []
     -ex_dp({'x1': ['s.y']}) -> []
    VARIABLES
     -ex_sp_xloc                                              [nan nan]
    OBJECTIVES
     -ex_sp_f1                                                      nan
     -ex_scenprob_f1                                                nan
     -ex_dp_f1                                                      nan
    CONSTRAINTS
     -ex_sp_g1                                                      nan
    """

    def __init__(self):
        self.connectors = {}
        self.problems = {}
        self.problem_graph = nx.DiGraph()
        self.var_mapping = {}
        super().__init__()

    def prob_repr(self):
        repstr = ""
        constr = " -" + "\n -".join(['{:<45}{:>20}'.format(k, str(var.values))
                                     for k, var in self.connectors.items()])
        if constr:
            repstr += "\nCONNECTORS\n" + constr
        probstr = " -" + "\n -".join([pn+"("+str(self.find_inputs(pn))+")"
                                      + " -> " + str(self.find_outputs(pn))
                                      for pn in self.problems])
        if probstr:
            repstr += "\nPROBLEMS\n" + probstr
        var_str = " -" + "\n -".join(['{:<45}{:>20}'.format(k, str(var.values))
                                      for k, var in self.variables.items()])
        if self.variables:
            repstr += "\n"+"VARIABLES\n" + var_str
        obj_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.objectives.items()])
        if self.objectives:
            repstr += "\n"+"OBJECTIVES\n" + obj_str
        con_str = " -" + "\n -".join(['{:<45}{:>20.4f}'.format(k, v)
                                      for k, v in self.constraints.items()])
        if self.constraints:
            repstr += "\n"+"CONSTRAINTS\n" + con_str
        return repstr

    def add_connector_variable(self, name, *varnames):
        """
        Add a connector linking variables between problems.

        Parameters
        ----------
        name : str
            Name for the connector
        *varnames : strs
            Names of the variable values (used in each problem) to link as a part of the
            connector.
        """
        self.connectors[name] = VariableConnector(name, varnames)
        self.problem_graph.add_node(name, label=name)

    def add_problem(self, name, problem, inputs={}, outputs={}):
        """
        Add a problem to the ProblemArchitecture.

        Parameters
        ----------
        name : str
            Name for the problem.
        problem : BaseProblem/ScenProblem/SimpleProblem...
            Problem object to add to the architecture.
        inputs : dict, optional
            List of input connector names (by name) and their corresponding problem
            variables. The default is [].
        outputs : dict, optional
            List of output connector names (by name) and their corresponding problem
            variables/objectives/constraints. The default is [].
        """
        if self.problems:
            upstream_problem = [*self.problems][-1]
            self.problem_graph.add_edge(upstream_problem, name,
                                        label="next")
            problem.consistent = False
        else:
            problem.consistent = True
        self.problems[name] = problem
        self.problem_graph.add_node(name, order=len(self.problems))

        for con in inputs:
            self.problem_graph.add_edge(con, name, label="input", var=inputs[con])
        for con in outputs:
            self.problem_graph.add_edge(name, con, label="output", var=outputs[con])
        xloc_vars = self.find_xloc_vars(name)
        if xloc_vars:
            xloc_name = name+"_xloc"
            self.variables[xloc_name] = VariableConnector(xloc_name, xloc_vars)
            self.problem_graph.add_node(xloc_name, label="xloc")
            self.problem_graph.add_edge(xloc_name, name, label="input", var=xloc_vars)
        self.create_var_mapping(name)
        self.add_objective_callables(name)
        self.add_constraint_callables(name)
        self.update_objectives(name)
        self.update_constraints(name)

    def create_var_mapping(self, probname):
        """Create a dict mapping problem variables to input/connector variables."""
        var_mapping = dict()
        vars_to_match = [*self.problems[probname].variables]
        inputdict = self.find_inputs(probname)
        inputconnectors = self.get_inputs(probname)
        for inputname, inputvars in inputdict.items():
            for i, inputvar in enumerate(inputvars):
                var_mapping[inputvar] = (inputname, inputconnectors[inputname].keys[i])
                vars_to_match.remove(inputvar)
        if vars_to_match:
            raise Exception("Dangling variables: "+str(vars_to_match))
        self.var_mapping[probname] = var_mapping

    def update_downstream_consistency(self, probname):
        """Mark downstream problems as inconsistent (when current problem updated)."""
        probs = self.get_downstream_probs(probname)
        for prob in probs:
            self.problems[prob].consistent = False

    def find_inconsistent_upstream(self, probname):
        """Check that all upstream problems are consistent with current probname."""
        probs = self.get_upstream_probs(probname)
        inconsistent_probs = []
        for prob in probs:
            if not self.problems[prob].consistent:
                inconsistent_probs.append(prob)
        return inconsistent_probs

    def add_objective_callables(self, probname):
        """Add callable objective function with name name."""
        for objname in self.problems[probname].objectives:
            def newobj(*x):
                return self.call_objective(probname, objname, *x)

            def new_full_obj(*x):
                return self.call_full_objective(probname, objname, *x)
            aname = obj_name(probname, objname)
            setattr(self, aname, newobj)
            setattr(self, aname+"_full", new_full_obj)

    def add_constraint_callables(self, probname):
        """Add callable constraint function with name name."""
        for conname in self.problems[probname].constraints:
            def newcon(*x):
                return self.call_constraint(probname, conname, *x)

            def new_full_con(*x):
                return self.call_full_constraint(probname, conname, *x)
            aname = obj_name(probname, conname)
            setattr(self, aname, newcon)
            setattr(self, aname+"_full", new_full_con)

    def update_objectives(self, probname):
        """Update architecture-level objectives from problem."""
        for objname, obj in self.problems[probname].objectives.items():
            aname = obj_name(probname, objname)
            if self.problems[probname].consistent:
                self.objectives[aname] = obj.value
            else:
                self.objectives[aname] = np.nan

    def update_constraints(self, probname):
        """Update architecture-level constraints from problem."""
        for objname, obj in self.problems[probname].constraints.items():
            aname = obj_name(probname, objname)
            if self.problems[probname].consistent:
                self.constraints[aname] = obj.value
            else:
                self.constraints[aname] = np.nan

    def find_input_vars(self, probname):
        """Find variables for a problem that are in an input connector."""
        return [var for con in self.find_inputs(probname).values() for var in con]

    def find_xloc_vars(self, probname):
        """Find variables for a problem that aren't in an input connector."""
        return [x for x in self.problems[probname].variables
                if x not in self.find_input_vars(probname)]

    def call_full_objective(self, probname, objname, *x_full):
        """Call objective of a problem over full set of variables *x_full."""
        self.update_full_problem(*x_full, probname=probname)
        return self.problems[probname].objectives[objname].value

    def call_full_constraint(self, probname, conname, *x_full):
        """Call objective of a problem over full set of variables *x_full."""
        self.update_full_problem(*x_full, probname=probname)
        return self.problems[probname].constraints[conname].value

    def call_objective(self, probname, objname, *x_loc):
        """Call objective of a problem over partial its local variables *x_loc."""
        self.update_problem(probname, *x_loc)
        return self.problems[probname].objectives[objname].value

    def call_constraint(self, probname, conname, *x_loc):
        """Call constraint of a problem over partial its local variables *x_loc."""
        self.update_problem(probname, *x_loc)
        return self.problems[probname].constraints[conname].value

    def update_full_problem(self, *x_full, probname=''):
        """
        Update the variables for the entire problem (or, problems up to probname).

        Parameters
        ----------
        *x_full : float
            Variable values for all local variables in the problem architecture up to
            the probname.
        probname : str, optional
            If provided, the problems will be updated up to the given problem.
            The default is ''.

        Examples
        --------
        >>> ex_pa.update_full_problem(1, 2)
        >>> ex_pa
        ProblemArchitecture with:
        CONNECTORS
         -x0                                                           [1.]
         -x1                                                           [2.]
        PROBLEMS
         -ex_sp({'ex_sp_xloc': ['x0', 'x1']}) -> ['x0', 'x1']
         -ex_scenprob({'x0': ['time']}) -> []
         -ex_dp({'x1': ['s.y']}) -> []
        VARIABLES
         -ex_sp_xloc                                                [1. 2.]
        OBJECTIVES
         -ex_sp_f1                                                   3.0000
         -ex_scenprob_f1                                            16.0000
         -ex_dp_f1                                                   2.0000
        CONSTRAINTS
         -ex_sp_g1                                                  -4.0000

        >>> ex_pa.problems['ex_dp']
        DisturbanceProblem with:
        VARIABLES
         -s.y                                                        2.0000
        OBJECTIVES
         -f1                                                         2.0000
        """
        if not probname:
            probname = [*self.problems][-1]
        probs_to_call = [*self.get_upstream_probs(probname), probname]
        x_to_split = [*x_full]
        for problem in probs_to_call:
            loc_var = problem + "_xloc"
            if loc_var in self.variables:
                x_loc = [x_to_split.pop(0) for k in self.variables[loc_var].keys]
            else:
                x_loc = []
            self.update_problem(problem, *x_loc)

    def update_problem(self, probname, *x):
        """
        Update a given problem with new values for inputs (and non-input variables).

        Additionally updates output connectors.

        Parameters
        ----------
        probname : str
            Name of the problem to update.
        *x : float
            Input variables to update (aside from inputs).

        Examples
        --------
        >>> ex_pa.update_problem("ex_sp", 1, 2)
        >>> ex_pa.problems["ex_sp"]
        SimpleProblem with:
        VARIABLES
         -x0                                                         1.0000
         -x1                                                         2.0000
        OBJECTIVES
         -f1                                                         3.0000
        CONSTRAINTS
         -g1                                                        -4.0000

        This update should further update connectors:
         >>> ex_pa.get_outputs("ex_sp")
         {'x0': VariableConnector(name='x0', keys=('x0',), values=array([1.])), 'x1': VariableConnector(name='x1', keys=('x1',), values=array([2.]))}

        Which should then propagate to downstream sims:
        >>> ex_pa.update_problem("ex_scenprob")
        >>> ex_pa.problems["ex_scenprob"]
        SingleScenarioProblem(examplefxnblock, short) with:
        VARIABLES
         -time                                                       1.0000
        OBJECTIVES
         -f1                                                        16.0000
        """
        inconsistent_upstream = self.find_inconsistent_upstream(probname)
        for upstream_prob in inconsistent_upstream:
            self.update_problem(upstream_prob)
        x_inputs = self.get_inputs_as_x(probname, *x)
        self.problems[probname].update_objectives(*x_inputs)
        self.problems[probname].consistent = True
        self.update_objectives(probname)
        self.update_constraints(probname)
        self.update_problem_outputs(probname)
        self.update_downstream_consistency(probname)

    def update_problem_outputs(self, probname):
        """
        Update the output connectors from a problem.

        Parameters
        ----------
        probname : str
            Name of the problem.
        """
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
        """
        List input connectors for a given problem.

        Parameters
        ----------
        probname : str
            Name of the problem.

        Returns
        -------
        inputs : list
            List of names of connectors used as inputs.
        """
        return {e[0]: self.problem_graph.edges[e]['var']
                for e in self.problem_graph.in_edges(probname)
                if self.problem_graph.edges[e]['label'] == 'input'}

    def find_outputs(self, probname):
        """
        List output connectors for a given problem.

        Parameters
        ----------
        probname : str
            Name of the problem.

        Returns
        -------
        outputs : list
            List of names of connectors used as outputs.
        """
        return [e[1] for e in self.problem_graph.out_edges(probname)
                if self.problem_graph.edges[e]['label'] == 'output']

    def get_inputs_as_x(self, probname, *x):
        """
        Get variable values for inputs.

        Parameters
        ----------
        probname : str
            Name of the problem..

        Returns
        -------
        inputs : list
            Input connectors and their values.
        """
        if x:
            self.variables[probname+"_xloc"].update_values(*x)
        vars_to_match = [*self.problems[probname].variables]
        inputs = self.get_inputs(probname)
        x_input = []
        for var in vars_to_match:
            inputname, inputkey = self.var_mapping[probname][var]
            x_input.append(inputs[inputname].get(inputkey))
        return x_input

    def get_inputs(self, probname):
        """Return a dict of input connectors for problem probname."""
        return {c: self.connectors[c] if c in self.connectors else self.variables[c]
                for c in self.find_inputs(probname)}

    def get_outputs(self, probname):
        """Return a dict of output connectors for problem probname."""
        return {c: self.connectors[c] for c in self.find_outputs(probname)}

    def get_downstream_probs(self, probname):
        """Return a list of all problems to be executed after the problem probname."""
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[ind+1:]

    def get_upstream_probs(self, probname):
        """Return a list of all problems to be executed before the problem probname."""
        probs = [*self.problems]
        ind = probs.index(probname)
        return probs[:ind]

    def show_sequence(self):
        """
        Show a visualization of the problem architecture.

        Returns
        -------
        fig : mpl.figure
            Figure object.
        ax : mpl.axis
            Axis object.
        """
        fig, ax = setup_plot()
        pos = nx.planar_layout(self.problem_graph, dim=2)
        nx.draw(self.problem_graph, pos=pos)
        orders = nx.get_node_attributes(self.problem_graph, "order")
        labels = {node: str(orders[node]) + ": " + node
                  if node in orders else self.problem_graph.nodes[node]['label']
                  for node in self.problem_graph}
        nx.draw_networkx_labels(self.problem_graph, pos, labels=labels)
        edge_labels = nx.get_edge_attributes(self.problem_graph, "label")
        edge_vars = nx.get_edge_attributes(self.problem_graph, "var")
        edge_labels = {e: lab+": "+str(edge_vars[e]) if e in edge_vars else lab
                       for e, lab in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.problem_graph, pos, edge_labels=edge_labels)
        return fig, ax

ex_sp, ex_scenprob, ex_dp

ex_pa = ProblemArchitecture()
ex_pa.add_connector_variable("x0", "x0")
ex_pa.add_connector_variable("x1", "x1")
ex_pa.add_problem("ex_sp", ex_sp, outputs={"x0": ["x0"], "x1": ["x1"]})
ex_pa.add_problem("ex_scenprob", ex_scenprob, inputs={"x0": ["time"]})
ex_pa.add_problem("ex_dp", ex_dp, inputs={"x1": ["s.y"]})

ex_pa.show_sequence()

#ex_pa.update_problem("ex_sp", 1, 2)

#ex_pa.update_full_problem(1, 1)


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