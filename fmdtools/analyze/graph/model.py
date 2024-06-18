"""
Module for generating graphs from models.

Provides classes:

- :class:`ModelGraph`: Superclass for graphs generated from modelling constructs.

and methods:

- :func:`graph_factory`: Creates the default Graph for a given object.
"""
from fmdtools.analyze.graph.base import Graph
from fmdtools.analyze.result import Result
from fmdtools.analyze.history import History
from fmdtools.analyze.common import prep_animation_title
from fmdtools.analyze.common import clear_prev_figure
import networkx as nx


class ModelGraph(Graph):
    """
    Superclass for Graphs meant to represent specific model constructs.

    Specifically, ModelGraphs have node/edge properties like "state" and "mode"
    corresponding to an external model simulation.

    To make a ModelGraph for a particular object, one needs to create two methods
    for the class:
        - nx_from_obj, which returns a networkx graph with 'edgetype' and 'nodetype'
        information, and
        - set_nx_states, which attaches state information such as states and modes
        to node/edge attributes.

    Parameters
    ----------
    get_states: bool
        whether to get states for the graph
    **kwargs:
        keyword arguments for self.nx_from_obj
    """

    def __init__(self, mdl, get_states=True, time=0.0, **kwargs):
        """
        Generate the FunctionArchitectureGraph corresponding to a given Model.

        Parameters
        ----------
        mdl : object
            fmdtools object to represent graphically
        get_states : bool, optional
            Whether to copy states to the node/edge 'states' property.
            The default is True.
        time: float
            Time model is run at (to execute indicators at). Default is 0.0
        **kwargs : kwargs
            (placeholder for kwargs)
        """
        Graph.__init__(self, self.nx_from_obj(mdl, **kwargs))
        if get_states:
            self.time = time
            self.set_nx_states(mdl, **kwargs)

    def nx_from_obj(self, mdl):
        """Alias for nx_from_obj, the method used to instantiate the graph."""
        raise Exception("nx_from_obj method not implemented for "
                        + self.__class__.__name__)

    def set_nx_states(self, mdl):
        """Alias for set_nx_states, which is used to map model states to graph attr."""
        raise Exception("set_nx_states method not implemented for "
                        + self.__class__.__name__)

    def set_from(self, time, history=History()):
        """Set ModelGraph faulty/degraded attributes from a given history."""
        faulty = history.get_faulty_hist(*self.g.nodes,
                                         withtotal=False,
                                         withtime=False).get_slice(time)
        fault_nodes = {n: bool(faulty.get(n, 0)) for n in self.g.nodes}
        nx.set_node_attributes(self.g, fault_nodes, 'faulty')

        faults = Result(history.get_faults_hist(*self.g.nodes).get_slice(time))
        faults_nodes = {n: [k for k, v in faults.get(n).items() if v]
                        if fault_nodes.get(n)
                        else [] for n in self.g.nodes}
        nx.set_node_attributes(self.g, faults_nodes, 'faults')

        degraded = history.get_degraded_hist(*self.g.nodes,
                                             withtotal=False,
                                             withtime=False).get_slice(time)
        deg_nodes = {n: bool(degraded.get(n, 0)) for n in self.g.nodes}
        nx.set_node_attributes(self.g, deg_nodes, 'degraded')

        # nx.set_node_attributes(self.g, state_nodes, 'states')
        self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults')

    def draw_from(self, time, history=History(), **kwargs):
        """
        Draws the graph with degraded/fault data at a given time.

        Parameters
        ----------
        time : int
            Time to draw the graph (in the history)
        history : History, optional
            History with nominal and faulty history. The default is History().
        **kwargs : **kwargs
            arguments for Graph.draw

        Returns
        -------
        fig : matplotlib figure
            matplotlib figure to draw
        ax : matplotlib axis
            Ax in the figure
        """
        self.set_from(time, history)
        kwargs = prep_animation_title(time, **kwargs)
        kwargs = clear_prev_figure(**kwargs)
        return self.draw(**kwargs)

    def draw_graphviz_from(self, time, history=History(), **kwargs):
        """
        Draws the graph with degraded/fault data at a given time.

        Parameters
        ----------
        time : int
            Time to draw the graph (in the history)
        history : History, optional
            History with nominal and faulty history. The default is History().
        **kwargs : **kwargs
            arguments for Graph.draw

        Returns
        -------
        dot : graphvis graph
            Graph drawn with attributes at the given time.
        """
        self.set_from(time, history)
        self.draw_graphviz(**kwargs)

    def animate(self, history, times='all', figsize=(6, 4), **kwargs):
        """
        Successively animate a plot using Graph.draw_from.

        Parameters
        ----------
        history : History
            History with faulty and nominal states
        times : list, optional
            List of times to animate over. The default is 'all'
        figsize : tuple, optional
            Size for the figure. The default is (6,4)
        **kwargs : kwargs

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            Animation object with the given frames
        """
        return history.animate(self.draw_from, times=times, figsize=figsize,
                               withlegend=False, **kwargs)

    def set_resgraph(self, other=False):
        """
        Process results for results graphs (show faults and degradations).

        Parameters
        ----------
        other : Graph, optional
            Graph to compare with (for degradations). The default is False.
        """
        if not other:
            other = self
        self.set_degraded(other)
        self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults_and_indicators')

    def set_degraded(self, other):
        """
        Set 'degraded' state in networkx graph.

        Uses difference between states with another Graph object.

        Parameters
        ----------
        other : Graph
            (assumed nominal) Graph to compare to
        """
        g = self.g
        nomg = other.g
        for node in g.nodes:
            degstates = any([g.nodes[node][s] != nomg.nodes[node][s]
                             for s in g.nodes[node]])
            degindicators = (set(g.nodes[node].get('indicators', {}))
                                 != set(nomg.nodes[node].get('indicators', {})))
            g.nodes[node]['degraded'] = degstates or degindicators
            g.nodes[node]['faulty'] = any(g.nodes[node].get('m', {'faults': {}})['faults'])


def set_block_node(g, block, nodename, time=None):
    """Set graph attributes for a given block."""
    roledict = block.get_roles_as_dict('container', with_immutable=False)
    roledict['indicators'] = block.return_true_indicators(time)
    g.nodes[nodename].update(roledict)



def graph_factory(obj, **kwargs):
    """
    Create the default Graph for a given object. Used in fmdtools.sim.get_result.

    Parameters
    ----------
    obj : object
        object corresponding to a specific graph type
    **kwargs : kwargs
        Keyword arguments for the Graph class

    Returns
    -------
    graph : Graph
        Graph of the appropriate (default) class
    """
    from fmdtools.define.architecture.function import FunctionArchitecture
    from fmdtools.define.flow.multiflow import MultiFlow
    from fmdtools.define.flow.commsflow import CommsFlow
    from fmdtools.define.architecture.action import ActionArchitecture

    if isinstance(obj, FunctionArchitecture):
        from fmdtools.analyze.graph.architecture import FunctionArchitectureGraph
        return FunctionArchitectureGraph(obj, **kwargs)
    elif isinstance(obj, CommsFlow):
        from fmdtools.analyze.graph.flow import CommsFlowGraph
        return CommsFlowGraph(obj, **kwargs)
    elif isinstance(obj, MultiFlow):
        from fmdtools.analyze.graph.flow import MultiFlowGraph
        return MultiFlowGraph(obj, **kwargs)
    elif isinstance(obj, ActionArchitecture):
        from fmdtools.analyze.graph.architecture import ActionArchitectureGraph
        return ActionArchitectureGraph(obj, **kwargs)
    else:
        raise Exception("No default graph for class "+obj.__class__.__name__)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
