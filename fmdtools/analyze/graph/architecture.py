"""
Graph analyses for architectures defined in define.architecture.

Main user-facing individual graphing classes:
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
- :class:`ActionArchitectureGraph`: Shows a visualization of the internal Action
  Sequence Graph of the Function Block, with Sequences as edges, with Flows (circular)
  and Actions (square) as nodes.
- :class:`ActionArchitectureActGraph`: Variant of ActionArchitectureGraph where only the
  sequence between actions is shown.
- :class:`ActionArchitectureFlowGraph`: Variant of ActionArchitectureGraph where only
  the flow relationships between actions is shown.
"""


import networkx as nx
from fmdtools.analyze.history import History
from fmdtools.analyze.graph.base import Graph


class FunctionArchitectureGraph(Graph):
    """
    Graph of FunctionArchitecture, where both functions and flows are nodes.

    If get_states option is used on instantiation, a `states` dict is associated
    with the edges/nodes which can then be used to visualize function/flow attributes.
    """

    def __init__(self, mdl, get_states=True, time=0.0, **kwargs):
        """
        Generate the FunctionArchitectureGraph corresponding to a given Model.

        Parameters
        ----------
        mdl : define.Model
            fmdtools model to represent graphically
        get_states : bool, optional
            Whether to copy states to the node/edge 'states' property.
            The default is True.
        time: float
            Time model is run at (to execute indicators at). Default is 0.0
        **kwargs : kwargs
            (placeholder for kwargs)
        """
        self.g = self.nx_from_obj(mdl)
        if get_states:
            self.time = time
            self.set_nx_states(mdl)

    def nx_from_obj(self, mdl):
        """
        Generate the networkx.graph object corresponding to the model.

        Parameters
        ----------
        mdl: Model
            Model to create the graph representation of

        Returns
        -------
        g : networkx.Graph
            networkx.Graph representation of model functions and flows
            (along with their attributes)
        """
        g = mdl.graph.copy()
        labels = {fname: f.get_typename() for fname, f in mdl.fxns.items()}
        labels.update({fname: f.get_typename() for fname, f in mdl.flows.items()})
        nx.set_node_attributes(g, labels, name='label')
        nx.set_edge_attributes(g, 'flow', name='label')
        return g

    def set_nx_states(self, mdl):
        """
        Attach state attributes to Graph corresponding to the states of the model.

        Parameters
        ----------
        mdl: Model
            Model to represent.
        """
        self.set_flow_nodestates(mdl)
        self.set_fxn_nodestates(mdl)

    def set_fxn_nodestates(self, mdl):
        """
        Attach attributes to Graph corresponding to function states.

        Parameters
        ----------
        mdl: Model
            Model to represent
        time: float
            Time to execute indicators at. Default is 0.0
        """
        fxnfaults, fxnstates, indicators = {}, {}, {}
        for fxnname, fxn in mdl.fxns.items():
            fxnstates[fxnname] = self.get_obj_state(mdl.fxns[fxnname])
            fxnfaults[fxnname] = self.get_obj_mode(mdl.fxns[fxnname])
            indicators[fxnname] = fxn.return_true_indicators(self.time)
        nx.set_node_attributes(self.g, fxnstates, 'states')
        nx.set_node_attributes(self.g, fxnfaults, 'faults')
        nx.set_node_attributes(self.g, indicators, 'indicators')

    def set_flow_nodestates(self, mdl):
        """
        Attach attributes to Graph notes corresponding to flow states.

        Parameters
        ----------
        mdl: Model
            Model to represent
        """
        flowstates, indicators = {}, {}
        for flowname, flow in mdl.flows.items():
            flowstates[flowname] = self.get_obj_state(flow)
            indicators[flowname] = flow.return_true_indicators(self.time)
        nx.set_node_attributes(self.g, flowstates, 'states')
        nx.set_node_attributes(self.g, indicators, 'indicators')

    def get_multi_edges(self, mdl, subedges):
        """
        Attach functions/flows (subedges arg) to edges.

        Parameters
        ----------
        mdl: Model
            Model to represent
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
        multgraph = nx.projected_graph(mdl.graph, subedges, multigraph=True)
        g = nx.projected_graph(mdl.graph, subedges)
        for edge in g.edges:
            midedges = list(multgraph.subgraph(edge).edges)
            flows[edge] = [midedge[2] for midedge in midedges]
        return flows

    def get_staticnodes(self, mdl):
        """Get static node information for set_exec_order."""
        staticfxns = list(mdl.staticfxns)
        staticflows = list(set([n for node in mdl.staticfxns
                                for n in mdl.graph.neighbors(node)]))
        staticnodes = staticfxns + staticflows
        static_node_dict = {n: n in staticnodes for n in self.g.nodes()}
        return static_node_dict

    def get_dynamicnodes(self, mdl):
        """Get dynamic node information for set_exec_order."""
        dynamicnodes = list(mdl.dynamicfxns)
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
            self.g.add_edges_from(next_edges_dict, label='activation')
            self.set_edge_styles(label={'activation': next_edges})

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
        self.set_node_labels(title='id', title2=title2, subtext=subtext)

    def draw_graphviz(self, layout="twopi", overlap='voronoi', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)


class FunctionArchitectureFlowGraph(FunctionArchitectureGraph):
    """
    Create a Graph of FunctionArchitecture flows.

    In this Graph, flows are set as nodes and ther connections (via functions) are edges
    """

    def nx_from_obj(self, mdl):
        g = nx.projected_graph(mdl.graph, mdl.flows)
        labels = {fname: f.get_typename() for fname, f in mdl.flows.items()}
        nx.set_node_attributes(g, labels, name='label')
        fxns = self.get_multi_edges(mdl, mdl.flows)
        edgelabels = {e: str(fl) for e, fl in fxns.items()}
        nx.set_edge_attributes(g, edgelabels, name='functions')
        nx.set_edge_attributes(g, {e: "functions" for e in g.edges()}, name='label')
        return g

    def set_nx_states(self, mdl):
        self.set_flow_nodestates(mdl)

    def set_edge_labels(self, title='label', title2='', subtext='functions',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)


class FunctionArchitectureCompGraph(FunctionArchitectureGraph):
    """
    Creates a graph of FunctionArchitecture functions, and flows, and components.

    Shows components as contained within functions.
    """

    def nx_from_obj(self, mdl):
        graph = super().nx_from_obj(mdl)
        for fxnname, fxn in mdl.fxns.items():
            if {**fxn.comps, **fxn.acts}:
                graph.add_nodes_from({**fxn.comps, **fxn.acts},
                                     bipartite=1, label="block")
                graph.add_edges_from([(fxnname, comp)
                                      for comp in {**fxn.comps, **fxn.acts}])
        return graph

    def set_nx_states(self, mdl):
        self.set_flowgraph_states(mdl)
        self.set_compgraph_blockstates(mdl)

    def set_compgraph_blockstates(self, mdl):
        compfaults, compstates, comptypes = {}, {}, {}
        fxnstates, fxnfaults, indicators = {}, {}, {}
        for fxnname, fxn in mdl.fxns.items():
            fxnstates[fxnname] = self.get_obj_state(mdl.fxns[fxnname])
            fxnfaults[fxnname] = self.get_obj_mode(mdl.fxns[fxnname])
            indicators[fxnname] = fxn.return_true_indicators(self.time)
            for mode in fxnfaults[fxnname].copy():
                for compname, comp in {**fxn.acts, **fxn.comps}.items():
                    compstates[compname] = {}
                    comptypes[compname] = True
                    indicators[compname] = comp.return_true_indicators(self.time)
                    if mode in comp.faultfaults:
                        compfaults[compname] = compfaults.get(compname, set())
                        compfaults[compname].update([mode])
                        fxnfaults[fxnname].remove(mode)
                        fxnfaults[fxnname].update(['comp_fault'])
        nx.set_node_attributes(self.g, fxnstates, 'states')
        nx.set_node_attributes(self.g, fxnfaults, 'faults')
        nx.set_node_attributes(self.g, compstates, 'states')
        nx.set_node_attributes(self.g, compfaults, 'faults')
        nx.set_node_attributes(self.g, comptypes, 'iscomponent')
        nx.set_node_attributes(self.g, indicators, 'indicators')


class FunctionArchitectureFxnGraph(FunctionArchitectureGraph):
    """
    Create a graph representation of the functions of the model.

    In this graph, functions are nodes and flows are edges.
    """

    def nx_from_obj(self, mdl):
        g = nx.projected_graph(mdl.graph, mdl.fxns)
        labels = {fname: f.get_typename() for fname, f in mdl.fxns.items()}
        nx.set_node_attributes(g, labels, name='label')
        flows = self.get_multi_edges(mdl, mdl.fxns)
        edgelabels = {e: str(fl) for e, fl in flows.items()}
        nx.set_edge_attributes(g, edgelabels, name='flows')
        nx.set_edge_attributes(g, {e: "flows" for e in g.edges()}, name='label')
        return g

    def set_nx_states(self, mdl):
        self.set_flow_edgestates(mdl)
        self.set_fxn_nodestates(mdl)

    def set_flow_edgestates(self, mdl):
        edgevals = {}
        flows = self.get_multi_edges(mdl, mdl.fxns)
        for edge, flows in flows.items():
            flowdict = {}
            for flow in flows:
                flowdict[flow] = self.get_obj_state(mdl.flows[flow].s)
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

    def set_edge_labels(self, title='label', title2='', subtext='flows',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)


class FunctionArchitectureTypeGraph(FunctionArchitectureGraph):
    """
    Creates a graph representation of FunctionArchitecture Classes.

    Shows the containment relationship between function classes and flow classes in the
    model.
    """

    def nx_from_obj(self, mdl, withflows=True, **kwargs):
        """
        Return graph with type containment relationships of the function/flow classes.

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
        g.add_node(modelname, level=1, label="architecture")
        g.add_nodes_from(mdl.fxnclasses(), level=2, label="block")
        function_connections = [(modelname, fname) for fname in mdl.fxnclasses()]
        g.add_edges_from(function_connections, label="containment")
        if withflows:
            g.add_nodes_from(mdl.flowtypes(), level=3, label="flow")
            fxnclass_flowtype = mdl.flowtypes_for_fxnclasses()
            flow_edges = [(fxn, flow) for fxn, flows in fxnclass_flowtype.items()
                          for flow in flows]
            g.add_edges_from(flow_edges, label="flow")
        return g

    def set_nx_states(self, mdl):
        graph = self.g
        flowstates = {}
        indicators = {}
        for flowtype in mdl.flowtypes():
            flowstates[flowtype] = {}
            indicators[flowtype] = {}
            for flow in mdl.flows_of_type(flowtype):
                flowstates[flowtype][flow] = self.get_obj_state(mdl.flows[flow])
                indicators[flowtype][flow] = mdl.flows[flow].return_true_indicators(self.time)
        nx.set_node_attributes(graph, flowstates, 'states')

        fxnstates = {}
        fxnfaults = {}
        for fxnclass in mdl.fxnclasses():
            fxnstates[fxnclass] = {}
            fxnfaults[fxnclass] = {}
            indicators[fxnclass] = {}
            for fxn in mdl.fxns_of_class(fxnclass):
                fxnstates[fxnclass][fxn] = self.get_obj_state(mdl.fxns[fxn])
                fxnfaults[fxnclass][fxn] = self.get_obj_mode(mdl.fxns[fxn])
                indicators[fxnclass][fxn] = mdl.fxns[fxn].return_true_indicators(self.time)

        nx.set_node_attributes(graph, fxnstates, 'states')
        nx.set_node_attributes(graph, fxnfaults, 'faults')
        nx.set_node_attributes(graph, indicators, 'indicators')

    def set_degraded(self, nomg):
        g = self.g
        rg = self.g.copy()
        for node in g.nodes:
            if g.nodes[node]['level'] == 2:
                n_faults = {fxn for fxn, m in g.nodes[node]['faults'].items()
                            if m not in [['nom'], []]}
                faulty = any(n_faults)
                rg.nodes[node]['faulty'] = faulty
            if g.nodes[node]['level'] >= 2:
                degraded = g.nodes[node]['states'] != nomg.nodes[node]['states']
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


# ActionArchitecture
class ActionArchitectureGraph(Graph):
    """
    Create a visual representation of an Action Architecture.

    Represents:
        - Sequence as edges
        - Flows as (circular) Nodes
        - Actions as (square) Nodes
    """

    def __init__(self, aa, time=0.0, get_states=True):
        self.g = nx.compose(aa.flow_graph, aa.action_graph)
        self.set_nx_labels(aa)
        if get_states:
            self.time = time
            self.set_nx_states(aa)

    def set_nx_labels(self, aa):
        """
        Label the underlying networkx graph structure.

        Adds type attributes corresponding to the ActionArchitecture.

        Parameters
        ----------
        aa : ActionArchitecture
            Action Sequence Graph object to represent
        """
        for n in self.g.nodes():
            if n in aa.action_graph.nodes():
                self.g.nodes[n]['label'] = 'Action'
            elif n in aa.flow_graph.nodes():
                self.g.nodes[n]['label'] = 'Flow'
        for e in self.g.edges():
            if e in aa.action_graph.edges():
                self.g.edges[e]['label'] = 'activation'
            elif e in aa.flow_graph.edges():
                self.g.edges[e]['label'] = 'flow'

    def set_nx_states(self, aa):
        """
        Attach state and fault information to the underlying graph.

        Parameters
        ----------
        aa : ActionArchitecture
            Underlying action sequence graph object to get states from
        """
        for g in self.g.nodes():
            self.g.nodes[g]['active'] = g in aa.active_actions
        states = {}
        faults = {}
        indicators = {}
        for aname, action in aa.acts.items():
            states[aname] = self.get_obj_state(action)
            faults[aname] = self.get_obj_mode(action)
            indicators[aname] = action.return_true_indicators(self.time)
        for fname, flow in aa.flows.items():
            states[fname] = self.get_obj_state(flow)
            indicators[fname] = flow.return_true_indicators(self.time)
        nx.set_node_attributes(self.g, states, 'states')
        nx.set_node_attributes(self.g, faults, 'faults')
        nx.set_node_attributes(self.g, indicators, 'indicators')

    def set_edge_labels(self, title='label', title2='', subtext='name',
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
        return super().draw_from(time, history=history, **kwargs)


class ActionArchitectureActGraph(ActionArchitectureGraph):
    """ActionArchitectureGraph where only the sequence between actions is shown."""

    def __init__(self, aa, get_states=True):
        self.g = aa.action_graph.copy()
        self.set_nx_labels(aa)
        if get_states:
            self.set_nx_states(aa)


class ActionArchitectureFlowGraph(ActionArchitectureGraph):
    """ActionArchitectureGraph that only shows flow relationships between actions."""

    def __init__(self, aa, get_states=True):
        self.g = aa.flow_graph.copy()
        self.set_nx_labels(aa)
        if get_states:
            self.set_nx_states(aa)
