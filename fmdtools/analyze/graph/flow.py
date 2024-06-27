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
"""
import networkx as nx
from fmdtools.analyze.graph.model import ModelGraph, remove_base
from fmdtools.analyze.graph.model import add_node, add_sub_nodes, get_obj_name


class MultiFlowGraph(ModelGraph):
    """
    Create networkx graph corresponding to the MultiFlow.

    Parameters
    ----------
    include_glob : bool, optional
        Whether to include the base flow (if used). The default is False.
    roles : list
        What parts of the multiflow to include. Default is ['locals'].
    roles_to_connect : list
        What roles to provide connections for. Default is ['locals'].
    connect_ports_only : Bool
        Whether to only connect port flows. Default is False.
    **kwargs : kwargs
        Keyword arguments to add_sub_nodes.

    Returns
    -------
    g : nx.DiGraph
        Networkx graph corresponding to the MultiFlow
    """

    def nx_from_obj(self, flow, include_glob=False, roles=['locals'],
                    roles_to_connect=['locals'], **kwargs):
        g = nx.DiGraph()
        name = get_obj_name(flow, '')
        add_node(g, flow, rolename=name)
        add_sub_nodes(g, flow, roles=roles, recursive=True, basename=name,
                      roles_to_connect=roles_to_connect, **kwargs)
        if not include_glob:
            remove_base(g, name)
        return g

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

from fmdtools.analyze.graph.model import add_edge

class CommsFlowGraph(MultiFlowGraph):
    """
    Create graph representation of the CommsFlow.

    Returns
    -------
    g : networkx.DiGraph
        Graph of the commsflow connections.
    """

    def nx_from_obj(self, flow, include_glob=False,
                    roles=['locals'], roles_to_connect=[], **kwargs):
        g = MultiFlowGraph.nx_from_obj(self, flow, roles=roles, include_glob=True,
                                       roles_to_connect=roles_to_connect,
                                       **kwargs)
        for f in flow.fxns:
            int_flow = getattr(flow, f)
            int_ports = int_flow.locals
            out_flow = getattr(flow, f+"_out")
            out_ports = out_flow.locals
            # add internal ports going out
            for portname, portobj in int_flow.get_roles_as_dict('locals').items():
                if portname in out_ports:
                    out_port = getattr(out_flow, portname)
                else:
                    out_port = out_flow
                out_name = get_obj_name(out_flow, portname, basename=int_flow.name)
                pname = portobj.get_full_name()
                add_edge(g, portobj, pname, out_port, out_name)
            # add external ports going in
            for f2 in flow.fxns:
                f2_out = getattr(flow, f2+"_out")
                f2_out_ports = f2_out.locals
                if int_flow.name in f2_out_ports:
                    out_port = getattr(f2_out, int_flow.name)
                else:
                    out_port = f2_out
                if f2 in int_ports:
                    in_port = getattr(int_flow, f2)
                else:
                    in_port = int_flow
                in_name = get_obj_name(in_port, in_port.name, basename=int_flow.root)
                out_name = get_obj_name(out_port, out_port.name, basename=int_flow.root)
                add_edge(g, in_port, in_name, out_port, out_name)
        if not include_glob:
            remove_base(g, flow.get_full_name())
        return g
