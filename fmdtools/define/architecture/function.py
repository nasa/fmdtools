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
from fmdtools.define.base import set_var
from fmdtools.define.architecture.base import Architecture, ArchitectureGraph
from fmdtools.define.block.function import ExampleFlow
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter

import numpy as np
from ordered_set import OrderedSet
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
        time: float
            Time to execute indicators at. Default is 0.0
        """
        for fxnname, fxn in mdl.fxns.items():
            fxn.set_node_attrs(self.g, time=self.time, with_root=with_root)

    def set_flow_nodestates(self, mdl, with_root=True):
        """
        Attach attributes to Graph notes corresponding to flow states.

        Parameters
        ----------
        mdl: Model
            Model to represent
        """
        for flowname, flow in mdl.flows.items():
            flow.set_node_attrs(self.g, time=self.time, with_root=with_root)

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
        staticfxns = [mdl.fxns[sf].get_full_name() for sf in mdl.staticfxns]
        staticflows = list(set([n for node in staticfxns if node in self.g
                                for n in self.g.neighbors(node)]))
        staticnodes = staticfxns + staticflows
        static_node_dict = {n: n in staticnodes for n in self.g.nodes()}
        return static_node_dict

    def get_dynamicnodes(self, mdl):
        """Get dynamic node information for set_exec_order."""
        dynamicnodes = [mdl.fxns[sf].get_full_name() for sf in mdl.dynamicfxns]
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
                indicators[flow] = mdl.flows[flow].return_true_indicators(self.time)
            self.g.nodes[flowtype]['mutables'] = mutes
            self.g.nodes[flowtype]['indicators'] = indicators

        for fxnclass in mdl.fxnclasses():
            mutes = {}
            indicators = {}
            for fxn in mdl.fxns_of_class(fxnclass):
                mutes[fxn] = mdl.fxns[fxn].get_roles_as_dict('container',
                                                             with_immutable=False)
                indicators[fxn] = mdl.fxns[fxn].return_true_indicators(self.time)
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

    Special Attributes
    ------------------
    functionorder : OrderedSet
        Keeps track of function dynamic execution order
    staticfxns : OrderedSet
        Keeps track of which functions run in static execution step
    dynamicfxns : Orderedset
        Keeps track of which functions run in dynamic execution step
    staticflows : list
        Flows to keep track of in static execution step
    graph : networkx graph
        multigraph view of functions and flows

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
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=1.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=1.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=0.0, y=0.0)

    This type of functional architecture only has dynamic functions:

    >>> exfa.dynamicfxns
    OrderedSet(['ex_fxn', 'ex_fxn2'])
    >>> exfa.staticfxns
    OrderedSet()

    This can in turn be simulated using FunctionArchitecture's built-in .propagate
    method. Note how the flow exf accumulates both ex_fxn and ex_fxn2 as reflected in
    their behavior methods:

    >>> exfa.propagate(1.0)
    >>> exfa
    exfa ExFxnArch
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=2.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=2.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=4.0, y=0.0)

    >>> exfa.propagate(2.0)
    >>> exfa
    exfa ExFxnArch
    FUNCTIONS:
    ex_fxn ExampleFunction
    - ExampleState(x=3.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    ex_fxn2 ExampleFunction
    - ExampleState(x=3.0, y=1.0)
    - ExampleMode(mode=standby, faults=set())
    FLOWS:
    exf ExampleFlow flow: ExampleState(x=10.0, y=0.0)
    """

    __slots__ = ['fxns', 'functionorder', '_fxnflows', '_flowstates',
                 'graph', 'staticfxns', 'dynamicfxns', 'staticflows']
    default_track = ('fxns', 'flows', 'i')
    default_name = 'model'
    flexible_roles = ['flows', 'fxns']
    roletypes = ['container', 'flow', 'fxn']
    rolename = 'fa'
    container_m = Mode

    def __init__(self, h={}, **kwargs):
        self.functionorder = OrderedSet()
        self._fxnflows = []
        self._flowstates = {}
        Architecture.__init__(self, h=h, **kwargs)

    def __repr__(self):
        fxnlist = ['\n' + fxn.__repr__() for fxn in self.fxns.values()]
        fxnlist = [fstr[:115] + '...' if len(fstr) > 120 else fstr for fstr in fxnlist]
        if len(fxnlist) > 15:
            fxnlist = fxnlist[:15]+["...("+str(len(fxnlist))+' total)\n']
        fxnstr = ''.join(fxnlist)
        flowlist = ['\n' + flow.__repr__() for flow in self.flows.values()]
        flowlist = [fstr[:115]+'...\n'if len(fstr) > 120 else fstr for fstr in flowlist]
        if len(flowlist) > 15:
            flowlist = flowlist[:15]+["...("+str(len(flowlist))+' total)\n']
        flowstr = ''.join(flowlist)
        repstr = (self.name + " " + self.__class__.__name__ +
                  '\n' + 'FUNCTIONS:' + fxnstr + '\nFLOWS:' + flowstr)
        return repstr

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'fxns', faults)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return FunctionArchitecture

    def is_static(self):
        """Determine if static based on containment of static functions."""
        return any(self.staticfxns)

    def is_dynamic(self):
        """Determine if dynamic based on containment of dynamic functions."""
        return any(self.dynamicfxns)

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
        for flowname in flownames:
            self._fxnflows.append((name, flowname))
        self.functionorder.update([name])

    def set_functionorder(self, functionorder):
        """
        Manually set the order of functions to be executed.

        (otherwise it will be executed based on the sequence of add_fxn calls)
        """
        if not self.functionorder.difference(functionorder):
            self.functionorder = OrderedSet(functionorder)
        else:
            raise Exception("Invalid list: "+str(functionorder) +
                            " should have elements: "+str(self.functionorder))

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

    def build(self, require_connections=True, **kwargs):
        """Build the model graph after the functions have been added."""
        super().build(**kwargs)
        self.staticfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items()
                                      if fxn.is_static()])
        self.dynamicfxns = OrderedSet([fxnname for fxnname, fxn in self.fxns.items()
                                       if fxn.is_dynamic()])
        self.construct_graph(require_connections=require_connections)
        self.staticflows = [flow for flow in self.flows
                            if any([n in self.staticfxns
                                    for n in self.graph.neighbors(flow)])]

    def construct_graph(self, require_connections=True):
        """Create .graph nx.graph representation of the model."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.fxns, bipartite=0)
        self.graph.add_nodes_from(self.flows, bipartite=1)
        self.graph.add_edges_from(self._fxnflows)

        # check to see that all functions/flows are connected
        dangling_nodes = [e for e in nx.isolates(self.graph)]
        if dangling_nodes and require_connections:
            raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))

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
        repmodes, modeprops = self.return_faultmodes()
        modecost = sum([c['cost'] if c['cost'] > 0.0 else default_cost
                        for m in modeprops.values() for c in m.values()])
        repair_cost = np.min([modecost, max_cost])
        return repair_cost

    def return_faultmodes(self):
        """
        Return faultmodes present in the model.

        Returns
        -------
        modes : dict
            Fault modes present in the model indexed by function name
        modeprops : dict
            Fault mode properties (defined in the function definition).
            Has structure {fxn:mode:properties}.
        """
        modes, modeprops = {}, {}
        for fxnname, fxn in self.fxns.items():
            ms, mps = fxn.return_faultmodes()
            if ms:
                modeprops[fxnname] = mps
                modes[fxnname] = ms
        return modes, modeprops

    def get_memory(self):
        """
        Return the approximate memory usage of the model.

        Includes profile of fxn/flow memory usage.
        """
        mem_profile = {}
        mem = 0
        if hasattr(self, 'p'):
            mem_profile['params'] = sys.getsizeof(self.p)
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

    def return_probdens(self):
        """Return the probability density of the model distributions."""
        probdens = 1.0
        for fxn in self.fxns.values():
            probdens *= fxn.return_probdens()
        return probdens

    def set_vars(self, *args, **kwargs):
        """
        Set variables in the model to set values (useful for optimization, etc.).

        Parameters
        ----------
        varlist : list of lists/tuples
            List of variables to set, with possible structures:
                [['fxnname', 'att1'], ['fxnname2', 'comp1', 'att2'], ['flowname', 'att3']]
                ['fxnname.att1', 'fxnname.comp1.att2', 'flowname.att3']
        varvalues : list
            List of values corresponding to varlist
        kwargs : kwargs
            attribute-value pairs. If provided, must be passed using ** syntax:
            mdl.set_vars(**{'fxnname.varname':value})
        """
        if len(args) > 0:
            varlist = args[0]
            varvalues = args[1]
            if isinstance(varlist, str):
                varlist = [varlist]
            if type(varvalues) in [str, float, int]:
                varvalues = [varvalues]
            if len(varlist) != len(varvalues):
                raise Exception("length of varlist and varvalues do not correspond: "
                                + str(len(varlist)) + ", "+str(len(varvalues)))
        else:
            varlist = []
            varvalues = []
        if kwargs:
            varlist = varlist + [*kwargs.keys()]
            varvalues = varvalues + [*kwargs.values()]
        for i, var in enumerate(varlist):
            if var == 'seed':
                self.update_seed(seed=varvalues[i])
            else:
                if isinstance(var, str):
                    var = var.split(".")
                if var[0] in ['functions', 'fxns']:
                    f = self.fxns[var[1]]
                    var = var[2:]
                elif var[0] == 'flows':
                    f = self.flows[var[1]]
                    var = var[2:]
                elif var[0] in self.fxns:
                    f = self.fxns[var[0]]
                    var = var[1:]
                elif var[0] in self.flows:
                    f = self.flows[var[0]]
                    var = var[1:]
                else:
                    raise Exception(var[0] + " not a function, flow, or seed")
                set_var(f, var, varvalues[i])

    def propagate(self, time, fxnfaults={}, disturbances={}, proptype="both",
                  run_stochastic=False):
        """
        Inject and propagates faults through the graph at one time-step.

        Parameters
        ----------
        time : float
            The current time-step.
        fxnfaults : dict
            Faults to inject during this propagation step.
            With structure {'function':['fault1', 'fault2'...]}
        disturbances : dict
            Variables to change during this propagation step.
            With structure {'function.var1':value}
        proptype : str
            Whether the propagate 'static' or 'dynamic' behaviors, or 'both'. Default
            is 'both'.
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of
            random states over time.
        """
        # Step 0: Update model states with disturbances and faults
        self.set_vars(**disturbances)
        if fxnfaults:
            self.inject_faults(fxnfaults)

        # Step 1: Run Dynamic Propagation Methods in Order Specified
        for fxnname in self.dynamicfxns:
            fxn = self.fxns[fxnname]
            fxn('dynamic', time=time, run_stochastic=run_stochastic)

        # Step 2: Run Static Propagation Methods
        try:
            self.prop_static(time, run_stochastic=run_stochastic)
        except Exception as e:
            raise Exception("Error in static propagation at time t=" + str(time)) from e

    def prop_static(self, time, run_stochastic=False):
        """
        Propagate behaviors through model graph (static propagation step).

        Parameters
        ----------
        time : float
            Current time-step
        run_stochastic : bool
            Whether to run stochastic behaviors or use default values. Default is False.
            Can set as 'track_pdf' to calculate/track the probability densities of
            random states over time.
        """
        # set up history of flows to see if any has changed
        activefxns = self.staticfxns.copy()
        nextfxns = set()
        if not self._flowstates:
            self._flowstates = dict.fromkeys(self.staticflows)
            for flowname in self.staticflows:
                self._flowstates[flowname] = self.flows[flowname].return_mutables()
        n = 0
        while activefxns:
            flows_to_check = {*self.staticflows}
            for fxnname in list(activefxns).copy():
                # Update functions with new values, check to see if new faults or states
                oldmutables = self.fxns[fxnname].return_mutables()
                self.fxns[fxnname]('static', time=time, run_stochastic=run_stochastic)
                if oldmutables != self.fxns[fxnname].return_mutables():
                    nextfxns.update([fxnname])

                # Check what flows now have new values and add connected functions
                # (done for each because of communications potential)
                for flowname in self.fxns[fxnname].flows:
                    if flowname in flows_to_check:
                        try:
                            if self._flowstates[flowname] != self.flows[flowname].return_mutables():
                                nextfxns.update(set([n for n in self.graph.neighbors(flowname)
                                                     if n in self.staticfxns]))
                                flows_to_check.remove(flowname)
                        except ValueError as e:
                            raise Exception("Invalid mutables in flow: "
                                            + flowname) from e
            # check remaining flows that have not been checked already
            for flowname in flows_to_check:
                if self._flowstates[flowname] != self.flows[flowname].return_mutables():
                    nextfxns.update(set([n for n in self.graph.neighbors(flowname)
                                         if n in self.staticfxns]))
            # update flowstates
            for flowname in self.staticflows:
                self._flowstates[flowname] = self.flows[flowname].return_mutables()
            activefxns = nextfxns.copy()
            nextfxns.clear()
            n += 1
            if n > 1000:  # break if this is going for too long
                raise Exception("Undesired looping for functions in static propagation",
                                "at t=" + str(time) + ", these functions remain active:"
                                + str(activefxns))

    def plot_dynamic_run_order(self, rotateticks=False, title="Dynamic Run Order"):
        """
        Plot the run order for the model during the dynamic propagation step.

        The x-direction is the order of each function executed and the y are the
        corresponding flows acted on by the given methods.

        Parameters
        ----------
        rotateticks : Bool, optional
            Whether to rotate the x-ticks (for bigger plots). The default is False.
        title : str, optional
            String to use for the title (if any). The default is "Dynamic Run Order".

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        from matplotlib import pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import AutoMinorLocator
        fxnorder = list(self.dynamicfxns)
        times = [i+0.5 for i in range(len(fxnorder))]
        fxntimes = {f: i for i, f in enumerate(fxnorder)}

        flowtimes = {f: [fxntimes[n] for n in self.graph.neighbors(
            f) if n in self.dynamicfxns] for f in self.flows}

        lengthorder = {k: v for k, v in
                       sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True)
                       if len(v) > 0}
        starttimeorder = {k: v for k, v in sorted(lengthorder.items(),
                                                  key=lambda x: x[1][0], reverse=True)}
        endtimeorder = [k for k, v in sorted(starttimeorder.items(),
                                             key=lambda x: x[1][-1], reverse=True)]
        flowtimedict = {flow: i for i, flow in enumerate(endtimeorder)}

        fig, ax = plt.subplots()

        for flow in flowtimes:
            phaseboxes = [((t, flowtimedict[flow]-0.5),
                           (t, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]-0.5))
                          for t in flowtimes[flow]]
            bars = PolyCollection(phaseboxes)
            ax.add_collection(bars)

        flowtimes = [i+0.5 for i in range(len(self.flows))]
        ax.set_yticks(list(flowtimedict.values()))
        ax.set_yticklabels(list(flowtimedict.keys()))
        ax.set_ylim(-0.5, len(flowtimes)-0.5)
        ax.set_xticks(times)
        ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
        ax.set_xlim(0, len(times))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which='minor', linewidth=2)
        ax.tick_params(axis='x', bottom=False, top=False,
                       labelbottom=False, labeltop=True)
        if title:
            if rotateticks:
                fig.suptitle(title, fontweight='bold', y=1.15)
            else:
                fig.suptitle(title, fontweight='bold')
        return fig, ax

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

if __name__ == "__main__":
    efa = FunctionArchitectureFxnGraph(ExFxnArch())
    efla = FunctionArchitectureFlowGraph(ExFxnArch())
    FunctionArchitectureTypeGraph(ExFxnArch())
    import doctest
    doctest.testmod(verbose=True)
