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
from fmdtools.analyze.graph.model import ModelGraph


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
        Keyword arguments to MultiFlow.create_graph.

    Returns
    -------
    g : nx.DiGraph
        Networkx graph corresponding to the MultiFlow
    """

    def __init__(self, flow, role_nodes=['local'], recursive=True, with_root=False,
                    **kwargs):
        ModelGraph.__init__(self, flow, role_nodes=role_nodes, recursive=recursive,
                            with_root=with_root, **kwargs)


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
    Create graph representation of the CommsFlow.

    Returns
    -------
    g : networkx.DiGraph
        Graph of the commsflow connections.
    """

    def __init__(self, flow, role_nodes=['local'], recursive=True, **kwargs):
        ModelGraph.__init__(self, flow, role_nodes=role_nodes, recursive=recursive,
                            **kwargs)
