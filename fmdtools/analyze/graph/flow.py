"""
Provides representations and visualizations for flows and their variants.

Main user-facing individual graphing classes:
- :class:`MultiFlowGraph`: Creates a networkx graph corresponding to the MultiFlow.
- :class:`CommsFlowGraph`: Creates a graph representation of the CommsFlow (assuming no
  additional locals).

Private Methods:

- :func:`node_is_tagged`: Returns if node is tagged
- :func:`add_g_nested`: Helper function for MultiFlow.create_multigraph to construct the
  containment tree.
- :func:`get_node_info`: Get info for a given node.
"""
import networkx as nx
from fmdtools.analyze.graph.model import ModelGraph


class MultiFlowGraph(ModelGraph):
    """
    Create networkx graph corresponding to the MultiFlow.

    Parameters
    ----------
    include_glob : bool, optional
        Whether to include the base flow (if used). The default is False.
    send_connections : dict/list, optional
        Tags/edges to create as send/recieve connections between local views of the
        flow without explicit containment relationships.

        With structure {in_tag : out_tag}. The default is {}.
        Or structure [(in_node : out_node)]
    include_states:
        whether to include states in the graph
    get_states:
        whether to attach state information as node attributes
    get_indicators : bool, optional
        Whether to attach indicators as attributs to the graph. The default is False
    time : float
        Time to run the indicator methods at.

    Returns
    -------
    g : nx.DiGraph
        Networkx graph corresponding to the MultiFlow
    """

    def __init__(self, flow, include_glob=False, send_connections={"closest": "base"},
                 connections_as_tags=True, include_states=False, get_states=True,
                 get_indicators=True, time=0.0):
        g = nx.DiGraph()
        if include_glob:
            add_g_nested(g, flow, flow.name, include_states=include_states,
                         get_states=get_states, get_indicators=get_indicators,
                         time=time)
        else:
            for loc in flow.locals:
                local_flow = getattr(flow, loc)
                add_g_nested(g, local_flow, loc, include_states=include_states,
                             get_states=get_states, get_indicators=get_indicators,
                             time=time)
        if isinstance(send_connections, dict):
            send_iter = send_connections.items()
            connections_as_tags = True
        elif isinstance(send_connections, list):
            send_iter = send_connections
            connections_as_tags = False

        for in_tag, out_tag in send_iter:
            for in_node in g.nodes:
                if node_is_tagged(connections_as_tags, in_tag, in_node):
                    for out_node in g.nodes:
                        if ((node_is_tagged(connections_as_tags, out_tag, out_node)
                             and not ((in_node, out_node) in g.edges
                                      or (out_node, in_node) in g.edges))
                             and in_node != out_node):
                            g.add_edge(in_node, out_node, edgetype="connection")
        self.g = g

    def set_resgraph(self, other=False):
        """
        Process results for results graphs (show faults and degradations).

        Parameters
        ----------
        other : Graph, optional
            Graph to compare with (for degradations). The default is False.
        """
        if other:
            self.set_degraded(other)
            self.set_node_styles(degraded={}, faulty={})
        else:
            self.set_degraded(self)
            self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='indicators')

    def draw_graphviz(self, layout="neato", overlap='false', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)


class CommsFlowGraph(MultiFlowGraph):
    """
    Create graph representation of the CommsFlow (assuming no additional locals).

    Parameters
    ----------
    include_glob : bool, optional
        Whether to include the base (root) node. The default is False.
    ports_only : bool, optional
        Whether to only include the explicit port connections betwen flows.
        The default is False
    with_internal: bool, optional
        Whether to include the internal aspect of the commsflow in the commsflow.
    get_indicators : bool, optional
        Whether to attach indicators as attributs to the graph. The default is False
    time : float
        Time to run the indicator methods at.

    Returns
    -------
    g : networkx.DiGraph
        Graph of the commsflow connections.
    """

    def __init__(self, flow, include_glob=False, ports_only=False,
                 get_states=True, get_indicators=True, time=0.0):
        send_connections = []
        for f in flow.fxns:
            int_flow = getattr(flow, f)
            int_ports = int_flow.locals
            out_flow = getattr(flow, f+"_out")
            out_ports = out_flow.locals
            send_connections.append((f, f+"_out"))
            for port in int_ports:
                portname = f+"_"+port
                if port in out_ports:
                    send_connections.append((portname, f+"_out_"+port))
                else:
                    send_connections.append((portname, f+"_out"))
            for f2 in flow.fxns:
                f2_int = getattr(flow, f2)
                if f2 in out_ports:
                    for port in out_ports:
                        portname = f+"_out: "+port
                        if port in f2_int.locals:
                            send_connections.append((portname, f2+": "+port))
                        elif port == f2:
                            send_connections.append((portname, f2))
                else:
                    if f in f2_int.locals:
                        send_connections.append((f+"_out", f2+"_"+f))
                    elif not ports_only:
                        send_connections.append((f+"_out", f2))

        super().__init__(flow, include_glob=include_glob,
                         send_connections=send_connections,
                         get_states=get_states,
                         get_indicators=get_indicators,
                         time=time)


def node_is_tagged(connections_as_tags, tag, node):
    """
    Return if a node is tagged, and thus if a connection should be made.

    If connections_as_tags, checks if either the tag is in the node string, or, if the
    tag is "base", connects with all base nodes (without an underscore)

    Parameters
    ----------
    connections_as_tags : bool
        Whether to treat connections as tags. If False, tagged is only True if the
        node is the tag
    tag : str
        tag to query/check if it is in the node string.
    node : str
        Name of the node.

    Returns
    -------
    tagged : bool
    """
    return ((connections_as_tags and
             (tag in node or (tag == "base" and not ("_" in node)))) or
            tag == node)


def add_g_nested(g, multiflow, base_name, include_states=False,
                 get_states=False, get_indicators=False, time=0.0):
    """
    Create graph for MultiFlow.create_multigraph.

    Iterates recursively through multigraph locals to construct the containment tree.

    Parameters
    ----------
    g : networkx.graph
        Existing graph
    multiflow : MultiFlow
        Multiflow Structure
    base_name : str
        Name at the current level of recursion
    include_states : bool, optional
        Whether to include state attributes in the plot. The default is False.
    get_states : bool, optional
        Whether to attach states as attributes to the graph. The default is False.
    get_indicators : bool, optional
        Whether to attach indicators as attributs to the graph. The default is False
    time : float
        Time to run the indicator methods at.
    """
    kwargs = get_node_info(multiflow, get_states, get_indicators, time)
    g.add_node(base_name, nodetype=multiflow.get_typename(), **kwargs)

    if include_states:
        for state in multiflow.s.__fields__:
            if get_states:
                kwargs = {"states": getattr(multiflow.s, state), "indicators": {}}
            else:
                kwargs = {"states": {}, "indicators": {}}
            g.add_node(base_name+"_"+state, nodetype="state", **kwargs)
            g.add_edge(base_name, base_name+"_"+state, edgetype="containment")
    for loc in multiflow.locals:
        local_flow = getattr(multiflow, loc)
        local_name = base_name+"_"+loc
        kwargs = get_node_info(local_flow, get_states, get_indicators, time)
        g.add_node(local_name, nodetype=local_flow.get_typename(), **kwargs)
        g.add_edge(base_name, local_name, edgetype="containment")
        if local_flow.locals:
            add_g_nested(g, local_flow, local_name,
                         include_states=include_states, get_states=get_states,
                         get_indicators=get_indicators, time=time)
        if include_states:
            for state in local_flow.s.__fields__:
                if get_states:
                    kwargs = {"states": getattr(multiflow.s, state), "indicators": {}}
                else:
                    kwargs = {"states": {}, "indicators": {}}
                g.add_node(local_name+"_"+state, nodetype="state", **kwargs)
                g.add_edge(local_name, local_name+"_"+state, edgetype="containment")


def get_node_info(flow, get_states, get_indicators, time):
    """
    Get the state/indicator information for a given flow.

    Parameters
    ----------
    flow : Flow
        Flow object to get node info from.
    get_states : bool
        Whether to get states for the flow
    get_indicators : bool
        Whether to get indicators for the flow
    time : float
        Time to execute the indicator functions at

    Returns
    -------
    kwargs : kwargs
        keyword arguments to add_node for the given flow.
    """
    kwargs = {"states": {}, "indicators": {}}
    if get_states:
        kwargs.update({"states": flow.return_states()})
    if get_indicators:
        kwargs.update({"indicators": flow.return_true_indicators(time)})
    return kwargs
