#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines classes for representing functional architectures.

Defines classes:

- :class:`FunctionArchitecture` class to represent functional architecture.
- :class:`FunctionArchitectureGraph`: Graphs Model of functions and flow for display
  where both functions and flows are nodes.
- :class:`FunctionArchitectureFlowGraph`: Graphs Model of flows for display, where flows
  are set as nodes and connections (via functions) are edges.
- :class:`FunctionArchitectureCompGraph`: Graphs Model of functions, and flows, with
  component containment relationships shown for functions.
- :class:`FunctionArchitectureFxnGraph`: Graphs representation of the functions of the
  model, where functions are nodes and flows are edges
- :class:`FunctionArchitectureTypeGraph`: Graph representation of model Classes, showing
  the containment relationship between function classes and flow classes in the model.

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
from fmdtools.define.block.function import ExampleFlow
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter

import numpy as np
import networkx as nx
import sys


class FunctionArchitectureGraph(ArchitectureGraph):
    """
    Graph of FunctionArchitecture, where both functions and flows are nodes.

    If get_states option is used on instantiation, a `states` dict is associated
    with the edges/nodes which can then be used to visualize function/flow attributes.

    Examples
    --------
    >>> efa = FunctionArchitectureGraph(ExFxnArch())
    >>> efa.g.nodes()
    NodeView(('exfxnarch.flows.exf', 'exfxnarch.fxns.ex_fxn', 'exfxnarch.fxns.ex_fxn2'))
    >>> efa.g.edges()
    OutEdgeView([('exfxnarch.fxns.ex_fxn', 'exfxnarch.flows.exf'), ('exfxnarch.fxns.ex_fxn2', 'exfxnarch.flows.exf')])
    """

    def set_fxn_nodestates(self, mdl, with_root=True):
        """
        Attach attributes to Graph corresponding to function states.

        Parameters
        ----------
        mdl: Model
            Model to represent
        """
        for fxnname, fxn in mdl.fxns.items():
            fxn.set_node_attrs(self.g, with_root=with_root)

    def set_flow_nodestates(self, mdl, with_root=True):
        """
        Attach attributes to Graph notes corresponding to flow states.

        Parameters
        ----------
        mdl: Model
            Model to represent
        """
        for flowname, flow in mdl.flows.items():
            flow.set_node_attrs(self.g, with_root=with_root)

    def get_multi_edges(self, graph, subedges):
        """
        Attach functions/flows (subedges arg) to edges.

        Parameters
        ----------
        graph: networkx graph
            Graph of model to represent
        subedges : list
            nodes from the full graph which will become edges in the subgraph
            (e.g., individual flows)

        Returns
        -------
        flows : dict
                Dictionary of edges with keys representing each sub-attribute of the
                edge (e.g., flows)
        """
        flows = {}
        multgraph = nx.projected_graph(graph, subedges, multigraph=True)
        g = nx.projected_graph(graph, subedges)
        for edge in g.edges:
            midedges = list(multgraph.subgraph(edge).edges)
            flows[edge] = [midedge[2] for midedge in midedges]
        return flows

    def get_staticnodes(self, mdl):
        """Get static node information for set_exec_order."""
        staticsims = [mdl.fxns[sf].get_full_name() for sf in mdl.staticsims]
        staticflows = list(set([n for node in staticsims if node in self.g
                                for n in self.g.neighbors(node)]))
        staticnodes = staticsims + staticflows
        static_node_dict = {n: n in staticnodes for n in self.g.nodes()}
        return static_node_dict

    def get_dynamicnodes(self, mdl):
        """Get dynamic node information for set_exec_order."""
        dynamicnodes = [mdl.fxns[sf].get_full_name() for sf in mdl.dynamicsims]
        orders = {n: str(i) for i, n in enumerate(dynamicnodes)}
        dynamic_node_dict = {n: n in orders for n in self.g.nodes()}
        return dynamic_node_dict, orders, dynamicnodes

    def set_exec_order(self, mdl, static={}, dynamic={}, next_edges={},
                       label_order=True, label_tstep=True):
        """
        Overlay FunctionArchitectureGraph execution order data on graph structure.

        Parameters
        ----------
        mdl : Model
            Model to plot the execution order of.
        static : dict/False, optional
            kwargs to overwrite the default style for functions/flows in the static
            execution step.
            If False, static functions are not differentiated. The default is {}.
        dynamic : dict/False, optional
            kwargs to overwrite the default style for functions/flows in the dynamic
            execution step.
            If False, dynamic functions are not differentiated. The default is {}.
        next_edges : dict
            kwargs to overwrite the default style for edges indicating the flow order.
            If False, these edges are not added. the default is {}.
        label_order : bool, optional
            Whether to label execution order (with a number on each node).
            The default is True.
        label_tstep : bool, optional
            Whether to label each timestep (with a number in the subtitle).
            The default is True.
        """
        node_style_kwargs = {}
        if not (isinstance(static, bool) and not static):
            static_node_dict = self.get_staticnodes(mdl)
            nx.set_node_attributes(self.g, static_node_dict, name='static')
            node_style_kwargs['static'] = static

        if not (isinstance(dynamic, bool) and not dynamic):
            dynamic_node_dict, orders, dynamicnodes = self.get_dynamicnodes(mdl)
            nx.set_node_attributes(self.g, dynamic_node_dict, name='dynamic')
            node_style_kwargs['dynamic'] = dynamic

        if not (isinstance(next_edges, bool) and not next_edges):
            next_edges_dict = [(dynamicnodes[n], dynamicnodes[n+1])
                               for n in range(len(dynamicnodes)-1)
                               if (dynamicnodes[n] in self.g.nodes
                               and dynamicnodes[n+1] in self.g.nodes)]
            self.g.add_edges_from(next_edges_dict, edgetype='activation')
            self.set_edge_styles(edgetype={'activation': next_edges})

        if label_order:
            orders.update({n: "" for n in self.g.nodes() if n not in orders})
            nx.set_node_attributes(self.g, orders, name='order')
            title2 = 'order'
        else:
            title2 = ''

        if label_tstep:
            tsteps = {n: str(mdl.fxns[n].t.dt) if n in mdl.fxns else ""
                      for n in self.g.nodes}
            nx.set_node_attributes(self.g, tsteps, name='tstep')
            subtext = 'tstep'
        else:
            subtext = ''

        self.set_node_styles(**node_style_kwargs)
        self.set_node_labels(title='shortname', title2=title2, subtext=subtext)

    def draw_graphviz(self, layout="twopi", overlap='voronoi', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)

    def gen_func_arch_graph(self, mdl):
        """Generate function architecture graph."""
        g0 = mdl.create_graph(with_methods=False, with_root=False).to_undirected()
        bip = {f.get_full_name(): 1 for f in mdl.flows.values()}
        bip.update({f.get_full_name(): 0 for f in mdl.fxns.values()})
        nx.set_node_attributes(g0, bip, name='bipartite')
        return g0


class FunctionArchitectureFlowGraph(FunctionArchitectureGraph):
    """
    Create a Graph of FunctionArchitecture flows.

    In this Graph, flows are set as nodes and connecting functions are edges.

    Examples
    --------
    >>> efa = FunctionArchitectureFlowGraph(ExFxnArch())
    >>> efa.g.nodes()
    NodeView(('exfxnarch.flows.exf',))
    >>> efa.g.nodes['exfxnarch.flows.exf']
    {'nodetype': 'Flow', 'classname': 'ExampleFlow', 'bipartite': 1, 's': ExampleState(x=0.0, y=0.0), 'indicators': []}
    """

    def nx_from_obj(self, mdl, **kwargs):
        g0 = self.gen_func_arch_graph(mdl)
        flows = [f.get_full_name() for f in mdl.flows.values()]
        g = nx.projected_graph(g0, flows)
        nodetypes = {f.get_full_name(): f.get_typename() for f in mdl.flows.values()}
        nx.set_node_attributes(g, nodetypes, name='nodetype')
        fxns = self.get_multi_edges(g0, flows)
        edgelabels = {e: str(fl) for e, fl in fxns.items()}
        nx.set_edge_attributes(g, edgelabels, name='functions')
        nx.set_edge_attributes(g, {e: "functions" for e in g.edges()}, name='edgetype')
        return g

    def set_nx_states(self, mdl, **kwargs):
        self.set_flow_nodestates(mdl)

    def set_edge_labels(self, title='edgetype', title2='', subtext='functions',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)


class FunctionArchitectureFxnGraph(FunctionArchitectureGraph):
    """
    Create a graph representation of the functions of the model.

    In this graph, functions are nodes and flows are edges.

    Examples
    --------
    >>> efa = FunctionArchitectureFxnGraph(ExFxnArch())
    >>> efa.g.nodes()
    NodeView(('exfxnarch.fxns.ex_fxn', 'exfxnarch.fxns.ex_fxn2'))
    >>> efa.g.edges()
    EdgeView([('exfxnarch.fxns.ex_fxn', 'exfxnarch.fxns.ex_fxn2')])
    >>> efa.g.edges[('exfxnarch.fxns.ex_fxn', 'exfxnarch.fxns.ex_fxn2')]
    {'flows': "['exfxnarch.flows.exf']", 'edgetype': 'flows', 'exfxnarch.flows.exf': {'s': ExampleState(x=0.0, y=0.0), 'indicators': []}}
    """

    def nx_from_obj(self, mdl):
        g0 = self.gen_func_arch_graph(mdl)
        fxns = [f.get_full_name() for f in mdl.fxns.values()]
        g = nx.projected_graph(g0, fxns)
        nodetypes = {f.get_full_name(): f.get_typename() for f in mdl.fxns.values()}
        nx.set_node_attributes(g, nodetypes, name='nodetype')
        flows = self.get_multi_edges(g0, fxns)
        edgelabels = {e: str(fl) for e, fl in flows.items()}
        nx.set_edge_attributes(g, edgelabels, name='flows')
        nx.set_edge_attributes(g, {e: "flows" for e in g.edges()}, name='edgetype')
        return g

    def set_nx_states(self, mdl):
        self.set_flow_edgestates(mdl)
        self.set_fxn_nodestates(mdl)

    def set_flow_edgestates(self, mdl):
        edgevals = {}
        g0 = self.gen_func_arch_graph(mdl)
        fxns = [f.get_full_name() for f in mdl.fxns.values()]
        flowiter = {f.get_full_name(): f for f in mdl.flows.values()}
        allflows = self.get_multi_edges(g0, fxns)
        for edge, flows in allflows.items():
            flowdict = {}
            for flow in flows:
                flowdict[flow] = flowiter[flow].get_node_attrs()
            edgevals[edge] = flowdict
        nx.set_edge_attributes(self.g, edgevals)

    def set_degraded(self, other):
        super().set_degraded(other)
        g = self.g
        nomg = other.g
        for edge in g.edges:
            degraded = False
            for flow in list(g.edges[edge].keys()):
                if g.edges[edge][flow] != nomg.edges[edge][flow]:
                    degraded = True
            g.edges[edge]['degraded'] = degraded

    def set_edge_labels(self, title='edgetype', title2='', subtext='flows',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)

    def draw_from(self, *args, rem_ind=0, **kwargs):
        return FunctionArchitectureGraph.draw_from(self, *args,
                                                   rem_ind=rem_ind, **kwargs)


class FunctionArchitectureTypeGraph(FunctionArchitectureGraph):
    """
    Creates a graph representation of FunctionArchitecture Classes.

    Shows the containment relationship between function classes and flow classes in the
    model.

    Examples
    --------
    >>> efa = FunctionArchitectureTypeGraph(ExFxnArch())
    >>> efa.g.nodes()
    NodeView(('ExFxnArch', 'ExampleFunction', 'ExampleFlow'))
    >>> efa.g.edges()
    OutEdgeView([('ExFxnArch', 'ExampleFunction'), ('ExampleFunction', 'ExampleFlow')])
    """

    def nx_from_obj(self, mdl, withflows=True, **kwargs):
        """
        Return graph with just the type relationships of the function/flow classes.

        Parameters
        ----------
        mdl: Model
            Model to represent

        withflows : bool, optional
            Whether to include flows, default is True

        Returns
        -------
        g : nx.DiGraph
            networkx directed graph of the type relationships
        """
        g = nx.DiGraph()
        modelname = type(mdl).__name__
        g.add_node(modelname, level=1, nodetype="architecture")
        g.add_nodes_from(mdl.fxnclasses(), level=2, nodetype="function")
        function_connections = [(modelname, fname) for fname in mdl.fxnclasses()]
        g.add_edges_from(function_connections, edgetype="containment")
        if withflows:
            fts = [(ft, {'nodetype': nt}) for ft, nt in mdl.flowtypes().items()]
            g.add_nodes_from(fts, level=3)
            fxnclass_flowtype = mdl.flowtypes_for_fxnclasses()
            flow_edges = [(fxn, flow) for fxn, flows in fxnclass_flowtype.items()
                          for flow in flows]
            g.add_edges_from(flow_edges, edgetype="flow")
        return g

    def set_nx_states(self, mdl):
        for flowtype in mdl.flowtypes():
            mutes = {}
            indicators = {}
            for flow in mdl.flows_of_type(flowtype):
                mutes[flow] = mdl.flows[flow].get_roles_as_dict('container',
                                                                with_immutable=False)
                indicators[flow] = mdl.flows[flow].return_true_indicators()
            self.g.nodes[flowtype]['mutables'] = mutes
            self.g.nodes[flowtype]['indicators'] = indicators

        for fxnclass in mdl.fxnclasses():
            mutes = {}
            indicators = {}
            for fxn in mdl.fxns_of_class(fxnclass):
                mutes[fxn] = mdl.fxns[fxn].get_roles_as_dict('container',
                                                             with_immutable=False)
                indicators[fxn] = mdl.fxns[fxn].return_true_indicators()
            self.g.nodes[fxnclass]['mutables'] = mutes
            self.g.nodes[fxnclass]['indicators'] = indicators

    def set_degraded(self, nomg):
        g = self.g
        rg = self.g.copy()
        for node in g.nodes:
            if g.nodes[node]['level'] == 2:
                n_faults = {fxn for fxn, m in g.nodes[node]['mutables'].get('m', {}).items()
                            if m not in [['nom'], []]}
                faulty = any(n_faults)
                rg.nodes[node]['faulty'] = faulty
            if g.nodes[node]['level'] >= 2:
                degraded = (g.nodes[node]['mutables'] != nomg.nodes[node]['mutables'] or
                            any([v for v in g.nodes[node]['indicators']]))
                rg.nodes[node]['degraded'] = degraded
        self.g = rg

    def set_pos(self, auto=True, **pos):
        if auto:
            self.pos = nx.multipartite_layout(self.g, 'level')
        super().set_pos(auto=False, **pos)

    def draw_graphviz(self, layout="dot", ranksep='2.0', **kwargs):
        return super().draw_graphviz(layout=layout, ranksep=ranksep, **kwargs)

    def set_exec_order(self, *args, **kwargs):
        raise Exception("Cannot specify exec_order for FunctionArchitectureTypeGraph")


class FunctionArchitecture(Architecture):
    """
    Class representing a functional architecture.

    Functional architectures enable the execution of multiple Function objects
    interacting with each other over time. The FunctionArchitecture uses an
    object-oriented, undirected graph-based model representation to enable the
    arbitrary propagation of flow states through the functions of the system.
    As opposed to a *procedural* *directed* graph-based model representation,
    in which each function has an ``input'' and ``output'', the undirected
    graph approach enables one to propagate behaviors in multiple directions.
    For example, in a pump system, closing a valve can be modelled to not just
    reduce flow in the downstream pipe, but also increase pressure in upstream
    pipes.

    Flexible Roles
    --------------
    flows : dict
        dictionary of flows objects in the model indexed by name
    fxns : dict
        dictionary of functions in the model indexed by name

    Examples
    --------
    >>> class ExFxnArch(FunctionArchitecture):
    ...     container_p = ExampleParameter
    ...     def init_architecture(self, **kwargs):
    ...         self.add_flow("exf", ExampleFlow, s={'x': 0.0, 'y': 0.0})
    ...         self.add_fxn("ex_fxn", ExampleFunction, "exf", p=self.p)
    ...         self.add_fxn("ex_fxn2", ExampleFunction, "exf", p=self.p)

    >>> exfa = ExFxnArch(name="exfa")
    >>> exfa
    exfa ExFxnArch
    - t=Time(time=-0.1, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    FLOWS:
    - exf=ExampleFlow(s=(x=0.0, y=0.0))
    FXNS:
    - ex_fxn=ExampleFunction(s=(x=0.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))
    - ex_fxn2=ExampleFunction(s=(x=0.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))

    This type of functional architecture only has dynamic functions:

    >>> exfa.dynamicsims
    OrderedSet(['ex_fxn', 'ex_fxn2'])
    >>> exfa.staticsims
    OrderedSet()

    This can in turn be simulated using FunctionArchitecture's built-in .propagate
    method. Note how the flow exf accumulates both ex_fxn and ex_fxn2 as reflected in
    their behavior methods:

    >>> exfa()
    >>> exfa
    exfa ExFxnArch
    - t=Time(time=1.0, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    FLOWS:
    - exf=ExampleFlow(s=(x=2.0, y=0.0))
    FXNS:
    - ex_fxn=ExampleFunction(s=(x=1.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))
    - ex_fxn2=ExampleFunction(s=(x=1.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))

    >>> exfa()
    >>> exfa
    exfa ExFxnArch
    - t=Time(time=2.0, timers={})
    - m=Mode(mode='nominal', faults=set(), sub_faults=False)
    FLOWS:
    - exf=ExampleFlow(s=(x=6.0, y=0.0))
    FXNS:
    - ex_fxn=ExampleFunction(s=(x=2.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))
    - ex_fxn2=ExampleFunction(s=(x=2.0, y=0.0), m=(mode='standby', faults=set(), sub_faults=False))
    """

    __slots__ = ['fxns']
    default_track = ('fxns', 'flows', 'i')
    default_name = 'model'
    flexible_roles = ['flow', 'fxn']
    roletypes = ['container']
    rolename = 'fa'
    container_m = Mode

    def base_type(self):
        """Return fmdtools type of the model class."""
        return FunctionArchitecture

    def add_fxn(self, name, fclass, *flownames, **fkwargs):
        """
        Instantiate a given function in the model.

        Parameters
        ----------
        name : str
            Name to give the function.
        fclass : Class
            Class to instantiate the function as. If no class has been developed,
            the user can send the block.GenericFxn class.
        flownames : list
            List of flows to associate with the function.
        args_f : dict.
            Other parameters to send to the __init__ method of the function class
        fkwargs : dict
            Parameters to send to __init__ method of the Function superclass
        """
        self.add_sim('fxns', name, fclass, *flownames, **fkwargs)

    def build(self, construct_graph=True, require_connections=True, **kwargs):
        """Build the function architecture - connections should be enforced."""
        super().build(construct_graph=construct_graph,
                      require_connections=require_connections, **kwargs)

    def fxns_of_class(self, ftype):
        """Return dict of funcitons corresponding to the given class name ftype."""
        return {fxn: obj for fxn, obj in self.fxns.items()
                if obj.__class__.__name__ == ftype}

    def fxnclasses(self):
        """Return the set of class names used in the model."""
        return {obj.__class__.__name__ for fxn, obj in self.fxns.items()}

    def flowtypes_for_fxnclasses(self):
        """Return the flows required by each function class in the model (as a dict)."""
        class_relationship = dict()
        for fxn, obj in self.fxns.items():
            if class_relationship.get(obj.__class__.__name__, False):
                class_relationship[obj.__class__.__name__].update(obj.get_flowtypes())
            else:
                class_relationship[obj.__class__.__name__] = set(obj.get_flowtypes())
        return class_relationship

    def calc_repaircost(self, additional_cost=0, default_cost=0, max_cost=np.inf):
        """
        Calculate the repair cost of the fault modes in the model.

        Uses given mode cost information for each function mode (in fxn.m).

        Parameters
        ----------
        additional_cost : int/float
            Additional cost to add if there are faults in the model. Default is 0.
        default_cost : int/float
            Cost to use for each fault mode if no fault cost information given
            in assoc_faultmodes/ Default is 0.
        max_cost : int/float
            Maximum cost of repair (e.g. cost of replacement). Default is np.inf

        Returns
        -------
        repair_cost : float
            Cost of repairing the fault modes in the given model
        """
        fmodes = self.return_faultmodes()
        modecost = sum([mode['cost'] if mode['cost'] > 0.0 else default_cost
                        for mode in fmodes.values()])
        repair_cost = np.min([modecost, max_cost])
        return repair_cost

    def get_memory(self):
        """
        Return the approximate memory usage of the model.

        Includes profile of fxn/flow memory usage.
        """
        mem_profile = {"params": 0}
        mem = 0
        if hasattr(self, 'p'):
            mem_profile['params'] += sys.getsizeof(self.p)
        mem_profile['params'] += sys.getsizeof(self.sp)
        mem_profile['params'] += sys.getsizeof(self.track)
        for fxnname, fxn in self.fxns.items():
            mem_profile[fxnname], _ = fxn.get_memory()
        for flowname, flow in self.flows.items():
            mem_profile[flowname], _ = flow.get_memory()
        mem = np.sum([i for i in mem_profile.values()])
        return mem, mem_profile

    def reset(self):
        """Reset the model to the initial state (with no faults, etc)."""
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
        super().reset()

    def as_modelgraph(self, gtype=FunctionArchitectureGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


class ExFxnArch(FunctionArchitecture):
    """Example FunctionArchitecture for testing and documentation."""

    container_p = ExampleParameter

    def init_architecture(self, **kwargs):
        self.add_flow("exf", ExampleFlow, s={'x': 0.0, 'y': 0.0})
        self.add_fxn("ex_fxn", ExampleFunction, "exf", p=self.p)
        self.add_fxn("ex_fxn2", ExampleFunction, "exf", p=self.p)

    def classify(self, **kwargs):
        return {'flowval': self.flows['exf'].s.x}

if __name__ == "__main__":
    efa = ExFxnArch()
    efa(1.0)
    effa = FunctionArchitectureFxnGraph(ExFxnArch())
    efla = FunctionArchitectureFlowGraph(ExFxnArch())
    FunctionArchitectureTypeGraph(ExFxnArch())
    import doctest
    doctest.testmod(verbose=True)
