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
import inspect


def get_obj_name(obj, role, basename=''):
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
    from fmdtools.define.object.base import BaseObject
    if isinstance(obj, BaseObject):
        return obj.get_full_name()
    elif inspect.ismethod(obj):
        return obj.__self__.get_full_name() + '.' + obj.__name__
    else:
        if not basename or not role:
            raise Exception("No role (" + role + ") or basename (" + basename +
                            ") for object: " + str(obj))
        return basename + "." + role


def get_edge_type(baseobj, obj):
    """
    Get the edge type connecting two objects.

    Parameters
    ----------
    baseobj : object
        Object at the base of the edge (e.g., thing contain-ing or aggregate-ing).
    obj : object
        Object at the end of the edge (e.g., thing being contained).

    Returns
    -------
    edgetype : str
        edgetype label to give the edge (e.g., containment, flow, aggregation, etc).
    """
    from fmdtools.define.flow.base import Flow
    from fmdtools.define.flow.multiflow import MultiFlow
    from fmdtools.define.object.base import BaseObject
    from fmdtools.define.container.base import BaseContainer
    if isinstance(obj, BaseObject):
        if obj.root == baseobj.get_full_name():
            edgetype = 'containment'
        elif isinstance(obj, MultiFlow) and baseobj.glob == obj.glob:
            edgetype = 'connection'
        elif isinstance(obj, Flow):
            edgetype = 'flow'
        else:
            edgetype = 'aggregation'
    elif inspect.ismethod(obj):
        if isinstance(baseobj, dict):
            edgetype = "aggregation"
        if obj.__self__.name == baseobj.name:
            edgetype = 'containment'
        else:
            edgetype = 'aggregation'
    elif isinstance(obj, BaseContainer):
        edgetype = 'containment'
    else:
        raise Exception("Unsupported object: " + str(obj))
    return edgetype


def get_node_type(obj):
    """
    Get the type of an object to attach to a node.

    Parameters
    ----------
    obj : object
        Object to represent as a node.

    Returns
    -------
    nodetype : str
        Node type (e.g., Function, State, Component, etc).
    """
    from fmdtools.define.object.base import BaseObject
    from fmdtools.define.container.base import BaseContainer
    if isinstance(obj, BaseObject) or isinstance(obj, BaseContainer):
        nodetype = obj.get_typename()
    else:
        nodetype = obj.__class__.__name__
    return nodetype


def add_node(g, obj, rolename='base', basename='', get_source=False):
    """
    Add a node to a graph.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to add to the graph.
    rolename : str, optional
        role of the graph in the larger system (if not base). The default is 'base'.
    """
    classname = obj.__class__.__name__
    name = get_obj_name(obj, rolename, basename=basename)
    nodetype = get_node_type(obj)
    g.add_node(name, nodetype=nodetype, classname=classname)
    if get_source:
        set_node_code(g, obj, name)


def set_node_code(g, obj, name):
    """Set code attributes to graph nodes (docs, source, code) for given objects."""
    from fmdtools.define.object.base import BaseObject
    from fmdtools.define.container.base import BaseContainer
    docs = ''
    source = ''
    code = ''
    if isinstance(obj, BaseObject) or isinstance(obj, BaseContainer):
        docs = inspect.getdoc(obj)
        source = inspect.getsource(obj.__class__)
        if isinstance(obj, BaseContainer):
            if obj.__class__ == obj.base_type():
                code = ''
            elif '\n\n    def' in source:
                code = source.split('\n\n    def')[0].split("'''")[-1].split('"""')[-1]
            else:
                code = source.split("'''")[-1].split('"""')[-1]
            code = "\n".join(code.split("\n    "))
    elif inspect.ismethod(obj):
        docs = inspect.getdoc(obj)
        source = inspect.getsource(obj)
        code = source.split("'''")[-1].split('"""')[-1]
        code = "\n".join(code.split("\n        "))
    code = remove_para(code)
    g.nodes[name]['docs'] = docs
    g.nodes[name]['source'] = source
    g.nodes[name]['code'] = code


def remove_para(source):
    """Remove paragraph newlines in a string (e.g., of code)."""
    if source.startswith("\n"):
        return remove_para(source[1:])
    else:
        return source


def get_obj_methods(obj):
    """Get methods from the given object."""
    methods = {at[0]: at[1] for at in inspect.getmembers(obj)
               if at[0] not in dir(obj.base_type()) and inspect.ismethod(at[1])}
    return methods


def get_sub_multiflows(obj):
    """Return a dict of multiflows set as object variables."""
    from fmdtools.define.flow.base import Flow
    if hasattr(obj, 'flows'):
        return {k: getattr(obj, k) for k in dir(obj)
                if isinstance(getattr(obj, k, ''), Flow)
                and k not in obj.flows}
    else:
        return {}


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
    from fmdtools.define.object.base import BaseObject
    if name in g.nodes:
        if isinstance(obj, BaseObject):
            set_block_node(g, obj, name, time=time)
        elif inspect.ismethod(obj):
            g.nodes[name]['condition'] = obj()
        else:
            g.nodes[name]['obj'] = obj


def add_role_node(g, baseobj, basename, roleobj, rolename, get_source=False):
    """
    Attach the sub-object of a base object.

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
    add_node(g, roleobj, rolename=rolename, basename=basename, get_source=get_source)
    add_edge(g, baseobj, basename, roleobj, rolename)
    add_cond_edge(g, roleobj)


def add_cond_edge(g, obj):
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
        add_edge(g, true_base, true_basename, obj, obj.__name__)


def add_edge(g, baseobj, basename, roleobj, rolename):
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
    name = get_obj_name(roleobj, rolename, basename=basename)
    edgetype = get_edge_type(baseobj, roleobj)
    if name not in g.nodes():
        raise Exception("Node not in g.nodes: "+name)
    g.add_edge(basename, name, edgetype=edgetype, role=rolename)


def connect_roles(g, obj, roles_to_connect=[], basename=''):
    """Connect roles at the same level of hierarchy."""
    if roles_to_connect:
        roledict = obj.get_roles_as_dict(*roles_to_connect)
        for rolename, roleobj in roledict.items():
            name = get_obj_name(roleobj, rolename, basename)
            for rolename2, roleobj2 in roledict.items():
                name2 = get_obj_name(roleobj2, rolename2, basename)
                if not ((name2, name) in [*g.edges]) and name2 != name:
                    add_edge(g, roleobj, name, roleobj2, name2)


def add_sub_nodes(g, obj, roles='all', recursive=False, basename='', with_methods=False,
                  get_source=False, with_subflow_edges=True, roles_to_connect=[],
                  connect_ports_only=False):
    """
    Add the objects contained in the given object to the graph.

    Sub-objects are in the objects roles (e.g. added as container_x, flow_y).

    Parameters
    ----------
    g : nx.Graph
        Networkx graph to add to.
    obj : object
        Object to find the objects in.
    """
    from fmdtools.define.object.base import BaseObject
    # determine roles to iterate through
    if roles == 'all':
        roledict = obj.get_roles_as_dict(flex_prefixes=True)
    else:
        roledict = obj.get_roles_as_dict(*roles, flex_prefixes=True)
    # add nodes for each role (recursively)
    for rolename, roleobj in roledict.items():
        name = get_obj_name(roleobj, rolename, basename)
        add_role_node(g, obj, basename, roleobj, rolename, get_source=get_source)
        if recursive and isinstance(roleobj, BaseObject):
            add_sub_nodes(g, roleobj, recursive=recursive, basename=name, roles=roles,
                          with_methods=with_methods, get_source=get_source,
                          roles_to_connect=roles_to_connect)
    # connect roles at same level of containment (if desired)
    connect_roles(g, obj, roles_to_connect=roles_to_connect, basename=basename)
    # attach methods if desired
    if with_methods:
        for methodname, methodobj in get_obj_methods(obj).items():
            name = get_obj_name(methodobj, methodobj, basename=basename)
            add_role_node(g, obj, basename, methodobj, name, get_source=get_source)
    # edges from blocks to their subflows
    if with_subflow_edges:
        for subflowname, subflowobj in get_sub_multiflows(obj).items():
            add_edge(g, obj, basename, subflowobj, subflowname)


def set_block_node(g, block, nodename, time=None):
    """Set graph attributes for a given block."""
    roledict = block.get_roles_as_dict('container', with_immutable=False)
    roledict['indicators'] = block.return_true_indicators(time)
    g.nodes[nodename].update(roledict)


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

    def __init__(self, mdl, get_states=True, get_source=False, time=0.0, **kwargs):
        """
        Generate the FunctionArchitectureGraph corresponding to a given Model.

        Parameters
        ----------
        mdl : object
            fmdtools object to represent graphically
        get_states : bool, optional
            Whether to copy states to the node/edge 'states' property.
            The default is True.
        get_source : bool, optional
            Whether to get the source code/objects from the object. The default is
            False.
        time: float
            Time model is run at (to execute indicators at). Default is 0.0
        **kwargs : kwargs
            (placeholder for kwargs)
        """
        Graph.__init__(self, self.nx_from_obj(mdl, get_source=get_source, **kwargs))
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

    def nx_from_obj(self, mdl, with_methods=False, get_source=False):
        """Recursively add nodes from the top level of the model graph."""
        g = nx.DiGraph()
        name = get_obj_name(mdl, '')
        add_node(g, mdl, rolename=name, get_source=get_source)
        add_sub_nodes(g, mdl, recursive=True, basename=name, with_methods=with_methods,
                      get_source=get_source)
        return g

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
