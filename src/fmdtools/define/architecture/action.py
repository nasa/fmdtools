#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines classes for representing action architectures.

Defines classes:
- :class:`ActionArchitecture` class for representing action architectures.
- :class:`ActionArchitectureGraph`: Shows a visualization of the internal Action
  Sequence Graph of the Function Block, with Sequences as edges, with Flows (circular)
  and Actions (square) as nodes.
- :class:`ActionArchitectureActGraph`: Variant of ActionArchitectureGraph where only the
  sequence between actions is shown.
- :class:`ActionArchitectureFlowGraph`: Variant of ActionArchitectureGraph where only
  the flow relationships between actions is shown.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.container.mode import Mode
from fmdtools.define.architecture.base import Architecture, ArchitectureGraph
from fmdtools.define.block.action import ExampleAction
from fmdtools.define.flow.base import ExampleFlow
from fmdtools.analyze.history import History
from fmdtools.analyze.graph.model import ModelGraph

import networkx as nx

def set_aa_nx_types(aa, g):
    """
    Label networkx graph structure.

    Adds type attributes corresponding to the ActionArchitecture.

    Parameters
    ----------
    aa : ActionArchitecture
        Action Sequence Graph object to represent
    g : nx.Graph
        Graph to label
    """
    for n in g.nodes():
        if n in aa.action_graph.nodes():
            g.nodes[n]['nodetype'] = 'Action'
        elif n in aa.flow_graph.nodes():
            g.nodes[n]['nodetype'] = 'Flow'
    for e in g.edges():
        if e in aa.action_graph.edges():
            g.edges[e]['edgetype'] = 'activation'
        elif e in aa.flow_graph.edges():
            g.edges[e]['edgetype'] = 'flow'
    return g


class ActionArchitectureGraph(ArchitectureGraph):
    """
    Create a visual representation of an Action Architecture.

    Represents:
        - Sequence as edges
        - Flows as (circular) Nodes
        - Actions as (square) Nodes

    Examples
    --------
    >>> aag = ActionArchitectureGraph(ExampleActionArchitecture())
    >>> aag.g.nodes()
    NodeView(('act_1', 'exf', 'act_2'))
    >>> aag.g.edges()
    OutEdgeView([('act_1', 'exf'), ('act_1', 'act_2'), ('act_2', 'exf')])
    """

    def nx_from_obj(self, aa, **kwargs):
        """Create Graph for ActionArchitecture."""
        return set_aa_nx_types(aa, nx.compose(aa.flow_graph, aa.action_graph))

    def set_nx_states(self, aa, **kwargs):
        """
        Attach state and fault information to the underlying graph.

        Parameters
        ----------
        aa : ActionArchitecture
            Underlying action sequence graph object to get states from
        """
        ArchitectureGraph.set_nx_states(self, aa, **kwargs)
        for g in self.g.nodes():
            self.g.nodes[g]['active'] = g in aa.active_actions

    def set_edge_labels(self, title='edgetype', title2='', subtext='name',
                        **edge_label_styles):
        """
        Set / define the edge labels.

        Parameters
        ----------
        title : str, optional
            property to get for title text. The default is 'label'.
        title2 : str, optional
            property to get for title text after the colon. The default is ''.
        subtext : str, optional
            property to get for the subtext. The default is ''.
        **edge_label_styles : dict
            edgeStyle arguments to overwrite.
        """
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)

    def set_node_styles(self, active={}, **node_styles):
        """
        Set self.node_styles and self.edge_groups given the provided node styles.

        Parameters
        ----------
        **node_styles : dict, optional
            Dictionary of tags, labels, and style kwargs for the nodes that
            overwrite the default.
            Has structure {tag:{label:kwargs}}, where kwargs are the keyword arguments
            to nx.draw_networkx_nodes. The default is {"label":{}}.
        """
        super().set_node_styles(active=active, **node_styles)

    def draw_graphviz(self, layout="twopi", overlap='voronoi', **kwargs):
        """Call Graph.draw_graphviz."""
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)

    def draw_from(self, time, history=History(), **kwargs):
        fault_act_hist = history._prep_faulty().get_values("a.active_actions")
        activities = fault_act_hist.get_slice(time)
        activity = {i for v in activities.values() for i in v}
        for n in self.g.nodes():
            if n in activity:
                self.g.nodes[n]['active'] = True
            else:
                self.g.nodes[n]['active'] = False
        return ModelGraph.draw_from(self, time, history=history, **kwargs)


class ActionArchitectureActGraph(ActionArchitectureGraph):
    """
    ActionArchitectureGraph where only the sequence between actions is shown.

    Examples
    --------
    >>> aag = ActionArchitectureActGraph(ExampleActionArchitecture())
    >>> aag.g.nodes()
    NodeView(('act_1', 'act_2'))
    >>> aag.g.edges()
    OutEdgeView([('act_1', 'act_2')])
    >>> aag.g.edges[('act_1', 'act_2')]
    {'name': 'act_1_done', 'act_1_done': 'name', 'arrow': True, 'edgetype': 'activation'}
    """

    def nx_from_obj(self, aa, **kwargs):
        """Create Graph for ActionArchitecture Actions."""
        return set_aa_nx_types(aa, aa.action_graph.copy())


class ActionArchitectureFlowGraph(ActionArchitectureGraph):
    """
    ActionArchitectureGraph that only shows flow relationships between actions.

    Examples
    --------
    >>> aag = ActionArchitectureFlowGraph(ExampleActionArchitecture())
    >>> aag.g.nodes()
    NodeView(('act_1', 'exf', 'act_2'))
    >>> aag.g.edges()
    OutEdgeView([('act_1', 'exf'), ('act_2', 'exf')])
    """

    def nx_from_obj(self, aa, **kwargs):
        """Create Graph for ActionArchitecture flows."""
        return set_aa_nx_types(aa, aa.flow_graph.copy())


class ActionArchitecture(Architecture):
    """
    Construct the Action Sequence Graph with the given parameters.

    Parameters
    ----------
    initial_action : str/list
        Initial action to set as active. Default is 'auto'
            - 'auto' finds the starting node of the graph and uses it
            - 'ActionName' sets the given action as the first active action
            - providing a list of actions will set them all to active
            (if multi-state rep is used)
    state_rep : 'finite-state'/'multi-state'
        How the states of the system are represented. Default is 'finite-state'
            - 'finite-state' means only one action in the system can be active at once (i.e., a finite state machine)
            - 'multi-state' means multiple actions can be performed at once
    max_action_prop : 'until_false'/'manual'/int
        How actions progress. Default is 'until_false'
            - 'until_false' means actions are simulated until all outgoing conditions are false
            - providing an integer places a limit on the number of actions that can be
            performed per timestep
    per_timestep : bool
        Defines whether the action sequence graph is reset to the initial state each
        time-step (True) or stays in the current action (False). Default is False.

    Examples
    --------
    >>> exaa = ExampleActionArchitecture()
    >>> exaa
    exampleactionarchitecture ExampleActionArchitecture
    - t=Time(time=-0.1, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    FLOWS:
    - exf=ExampleFlow(s=(x=1.0, y=1.0))
    ACTS:
    - act_1=ExampleAction()
    - act_2=ExampleAction()
    CONDS:
    - act_1_done=<method act_1.indicate_done()>
    >>> exaa()
    >>> exaa.active_actions
    {'act_1'}
    >>> exaa()
    >>> exaa
    exampleactionarchitecture ExampleActionArchitecture
    - t=Time(time=2.0, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    FLOWS:
    - exf=ExampleFlow(s=(x=4.0, y=1.0))
    ACTS:
    - act_1=ExampleAction()
    - act_2=ExampleAction()
    CONDS:
    - act_1_done=<method act_1.indicate_done()>
    >>> exaa.active_actions
    {'act_2'}
    """

    __slots__ = ['acts', 'conds', 'action_graph', 'flow_graph', 'active_actions', 'm']
    initial_action = "auto"
    state_rep = "finite-state"
    max_action_prop = "until_false"
    per_timestep = False
    default_track = ('acts', 'flows', 'active_actions', 'i')
    flexible_roles = ['flow', 'act', 'cond']
    roletypes = ['container']
    rolename = 'aa'
    container_m = Mode

    def __init__(self, **kwargs):
        self.action_graph = nx.DiGraph()
        self.flow_graph = nx.DiGraph()
        self.active_actions = set()
        Architecture.__init__(self, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return ActionArchitecture

    def is_static(self):
        """Determine if static based on containment of static actions."""
        any_static_actions = any([obj.is_static() for obj in self.acts.values()])
        return super().is_static() or any_static_actions

    def is_dynamic(self):
        """Determine if dynamic based on containment of dynamic actions."""
        any_dynamic_actions = any([obj.is_dynamic() for obj in self.acts.values()])
        return super().is_dynamic() or any_dynamic_actions

    def copy(self, **kwargs):
        cop = super().copy(**kwargs)
        cop.active_actions = {*self.active_actions}
        return cop

    def reset(self):
        super().reset()
        self.set_initial_active_action()

    def set_initial_active_action(self):
        if self.initial_action == 'auto':
            initial_action = [act for act, in_degree in self.action_graph.in_degree
                              if in_degree == 0]
            if not initial_action:
                raise Exception("Cannot set initial action--no starting node")
        elif isinstance(self.initial_action, str):
            initial_action = [self.initial_action]
        self.set_active_actions(initial_action)

    def build(self, construct_graph=False, **kwargs):
        """Build the action graph."""
        super().build(construct_graph=construct_graph, **kwargs)
        self.set_initial_active_action()
        if self.state_rep == 'finite-state' and len(self.active_actions) > 1:
            raise Exception("Cannot have more than one initial action with" +
                            " finite-state representation")

    def add_act(self, name, actclass, *flownames, **fkwargs):
        """
        Associate an Action with the architecture. Called after add_flow.

        Parameters
        ----------
        name : str
            Internal Name for the Action
        actclass : Action
            Action class to instantiate
        *flownames : flow
            Flows (optional) which connect the actions
        duration:
            Duration of the action. Default is 0.0
        **kwargs : any
            kwargs to instantiate the Action with.
        """
        self.add_sim('acts', name, actclass, *flownames, **fkwargs)

        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        flows = {fl: self.flows[fl] for fl in flownames}
        for flow in flows:
            self.flow_graph.add_node(flow, bipartite=1)
            self.flow_graph.add_edge(name, flow)

    def cond_pass(self): # noqa
        return True

    def add_cond(self, start_action, end_action, name='auto', condition='pass'):
        """
        Associate a Condition with the ActionArchitecture.

        Conditions specify when to precede from one action to the next.

        Parameters
        ----------
        start_action : str
            Action where the condition is checked
        end_action : str
            Action that the condition leads to.
        name : str
            Name for the condition.
            Defaults to numbered conditions if none are provided.
        condition : method
            Method in the class to use as a condition.
            Defaults to self.condition_pass if none are provided.
        """
        if name == 'auto':
            name = str(len(self.conds)+1)
        if condition == 'pass':
            condition = self.cond_pass
        self.conds[name] = condition
        self.action_graph.add_edge(start_action,
                                   end_action,
                                   **{'name': name, name: 'name', 'arrow': True})

    def set_active_actions(self, actions):
        """Set given action(s) as active."""
        if isinstance(actions, str):
            if actions in self.acts:
                actions = [actions]
            else:
                raise Exception("initial_action=" + actions +
                                " not in self.acts: "+str(self.acts))
        if isinstance(actions, list):
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.acts)):
                raise Exception("Initial actions not associated with model: " +
                                str(self.active_actions.difference(self.acts)))
        else:
            raise Exception("Invalid option for initial_action.")

    def prop_dynamic(self):
        """Propagate dynamic behavior through the ActionArchitecture graph.

        If self.per_timestep is set to True, this will also reset the active actions
        each timestep to ensure the graph is reset to the initial active action.
        """
        if self.per_timestep:
            self.set_active_actions(self.initial_action)
            for action in self.active_actions:
                self.acts[action].t.t_loc = 0.0
        self.prop_graph('dynamic')

    def prop_static(self):
        """
        Propagate static behavior through the ActionArchitecture graph.

        Parameters
        ----------
        time : float
            Model time.
        """
        self.prop_graph('static')

    def inc_sim_time(self, **kwargs):
        """Increment action simulation times to current."""
        super().inc_sim_time(**{**kwargs, 'time': self.t.time})

    def prop_graph(self, proptype):
        """
        Propagate behavior through the ActionArchitecture graph.

        Parameters
        ----------
        proptype : str
            Type of propagation to perform (static or dynamic). If proptype="static",
            the static_behavior methods are called and local time is not incremented. If
            proptype="dynamic", the dynamic behavior methods are run and local time is
            incremented.
        """
        active_actions = self.active_actions
        num_prop = 0
        while active_actions:
            new_active_actions = set(active_actions)
            for action in active_actions:
                act = self.acts[action]
                act.t.update_time(self.t.time)
                act(time=self.t.time, proptype=proptype, inc_at="")
                action_cond_edges = self.action_graph.out_edges(action, data=True)
                for act_in, act_out, atts in action_cond_edges:
                    try:
                        cond = self.conds[atts['name']]()
                    except TypeError as e:
                        raise TypeError("Poorly specified condition " +
                                        str(atts['name'])+": ") from e
                    if cond:
                        if act.t.complete():
                            act.t.t_loc = 0.0
                            new_active_actions.add(act_out)
                            new_active_actions.discard(act_in)
                        elif proptype in ['dynamic', 'both']:
                            act.t.t_loc += self.acts[action].t.dt
                    else:
                        act.t.t_loc = 0.0

            if len(new_active_actions) > 1 and self.state_rep == 'finite-state':
                raise Exception("Multiple active actions in a finite-state " +
                                "representation: "+str(new_active_actions))
            num_prop += 1
            if new_active_actions == set(active_actions):
                break
            else:
                active_actions = new_active_actions
            if num_prop > 10000:
                raise Exception("Undesired looping in Function ASG for: "+self.name)
        self.active_actions = active_actions

    def as_modelgraph(self, gtype=ActionArchitectureGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


class ExampleActionArchitecture(ActionArchitecture):
    """Example ActionArchitecture for testing and documentation."""

    def init_architecture(self, **kwargs):
        self.add_flow("exf", ExampleFlow)
        self.add_act("act_1", ExampleAction, "exf", p={'x': 2.0})
        self.add_act("act_2", ExampleAction, "exf", p={'x': 4.0})
        self.add_cond("act_1", "act_2", "act_1_done", self.acts['act_1'].indicate_done)


if __name__ == "__main__":
    exaa = ExampleActionArchitecture()
    aag = ActionArchitectureGraph(exaa)
    import doctest
    doctest.testmod(verbose=True)
