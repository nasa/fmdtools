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
from fmdtools.define.base import get_code_attrs
import networkx as nx
import inspect


def get_obj_name(obj, role='', basename=''):
    """
    Get the name of an object to be graphed.

    Parameters
    ----------
    obj : object
        Object to be graphed (BaseObject, BaseContainer, or other).
    role : str
        Role the object plays in the larger system. Determines the name of Containers.

    Returns
    -------
    name : str
        Name of the object.
    """
    if hasattr(obj, 'get_full_name'):
        return obj.get_full_name()
    elif inspect.ismethod(obj):
        superobj = obj.__self__
        return get_obj_name(superobj, basename=basename, role=role) + '.' + obj.__name__
    else:
        if not basename or not role:
            raise Exception("No role (" + role + ") or basename (" + basename +
                            ") for object: " + str(obj))
        return basename + "." + role


def add_node(obj, g=None, name='', classname='', nodetype='', get_attrs=False,
             get_source=False, **kwargs):
    """
    Add a node to a graph for a given object.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to add to the graph.
    rolename : str, optional
        role of the graph in the larger system (if not base). The default is 'base'.
    """
    if not g:
        g = nx.DiGraph()
    if hasattr(obj, 'get_full_name'):
        name = obj.get_full_name()
    if not classname:
        classname = obj.__class__.__name__
    if not nodetype:
        if hasattr(obj, 'get_typename'):
            nodetype = obj.get_typename()
        else:
            nodetype = obj.__class__.__name__
    g.add_node(name, nodetype=nodetype, classname=classname)
    if hasattr(obj, 'get_node_attr') and get_attrs is True:
        g.nodes[name].update(obj.get_node_attr(**kwargs))
    elif get_attrs is True:
        g.nodes[name]['obj'] = obj
    if get_source:
        g.nodes[name].update(get_code_attrs(obj))
    return g


def add_edge(g, basename, name, rolename, edgetype):
    """
    Add an edge from a base object to another object.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    baseobj : object
        Object that the object is within.
    basename : str
        Name of the base object.
    roleobj : object
        Object to add.
    rolename : str
        Name of the object in the context of the base object.
    """
    if name not in g.nodes():
        raise Exception("Node not in g.nodes: "+name)
    if basename not in g.nodes():
        raise Exception("Node not in g.nodes: "+basename)
    g.add_edge(basename, name, edgetype=edgetype, role=rolename)


def set_sub_nodes(g, obj, time=None, recursive=False, basename=''):
    """
    Set the object's contained nodes (roles).

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to find the objects in.
    time : float, optional
        Time to evaluate conditions at (if any). The default is None.
    recursive : bool, optional
        Whether to set nodes recursively. The default is False.
    basename : str, optional
        Name of the overall object. The default is ''.
    """
    from fmdtools.define.object.base import BaseObject
    for rolename, roleobj in obj.get_roles_as_dict(flex_prefixes=True).items():
        name = get_obj_name(roleobj, rolename, basename=basename)
        set_node_states(g, roleobj, name, time=time)
        if recursive and isinstance(roleobj, BaseObject):
            set_sub_nodes(g, roleobj, recursive=recursive, basename=name)


def set_node_states(g, obj, name, time=None):
    """
    Attach stateful attributes to the given node.

    Used to determine faults and degradations in scenario visualization.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to get the states from.
    rolename : str
        Name of the object in the larger system (if the node is not a BaseObject).
    time : float, optional
        Time to get the states from. The default is None.
    """
    if name in g.nodes:
        if hasattr(obj, 'set_node_attr'):
            obj.set_node_attr(time=time)
        elif inspect.ismethod(obj):
            g.nodes[name]['condition'] = obj()
        else:
            g.nodes[name]['obj'] = obj


def add_meth_edge(g, obj, rolename="action", edgetype="activation"):
    """
    If the object is a method, attaches an edge to the containing object.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to check.
    objname : str
        Name of the object.
    """
    if inspect.ismethod(obj):
        true_base = obj.__self__
        true_basename = true_base.get_full_name()
        name = get_obj_name(obj, basename=true_basename, role=obj.__name__)
        add_edge(g, true_basename, name, rolename, edgetype)


def remove_base(g, basename):
    """Remove base node to enable flat view of model graph."""
    g.remove_node(basename)
    g.remove_nodes_from([*nx.isolates(g)])


class BaseModelGraph(Graph):
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


class ModelGraph(BaseModelGraph):
    """Represent the entire containment/aggregation hierarchy of a given model."""

    def nx_from_obj(self, mdl, **kwargs):
        """Recursively add nodes from the top level of the model graph."""
        return mdl.create_graph(**kwargs)

    def set_nx_states(self, mdl, **kwargs):
        """Recursively set node states from the top level of the model graph."""
        name = get_obj_name(mdl, '')
        set_node_states(self.g, mdl, name, time=self.time)
        set_sub_nodes(self.g, mdl, time=self.time, recursive=True, basename=name)


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
