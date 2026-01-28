#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for generating graphs from models.

Provides classes:

- :class:`ModelGraph`: Superclass for graphs generated from modelling constructs.
- :class:`ExtModelGraph`: ModelGraph class that can be extended w- custom inits.

and methods:

- :func:`add_node`: Add node to a graph for a given object.
- :func:`add_edge`: Add an edge from a base object to another object.
- :func:`set_node_states`: Set node states from an object.
- :func:`add_meth_edge`: Addes edges from methods to their objects.
- :func:`remove_base`: Remove the root note of a graph.
- :func:`graph_factory`: Creates the default Graph for a given object.

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
from fmdtools.define.base import get_code_attrs, get_obj_name, get_inheritance
from fmdtools.analyze.graph.base import Graph
from fmdtools.analyze.graph.label import shorten_name
from fmdtools.analyze.result import Result
from fmdtools.analyze.history import History
from fmdtools.analyze.common import prep_animation_title
from fmdtools.analyze.common import clear_prev_figure

import networkx as nx
import inspect


def add_node(obj, g=None, name='', classname='', nodetype='', get_attrs=False,
             get_source=False, get_states=False, time=None, **kwargs):
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
    if not name:
        name = get_obj_name(obj)
    if not classname:
        classname = obj.__class__.__name__
    if not nodetype:
        if hasattr(obj, 'get_typename') and not inspect.isclass(obj):
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
    if get_states:
        set_node_states(g, obj, name, time=time)
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


def create_inheritance_subgraph(obj, g=None, name='', end_at_fmdtools=True):
    """
    Create a graph of the inheritance of a given object from fmdtools classes.

    Parameters
    ----------
    obj : Object
        Object.
    g : graph, optional
        Networkx graph to add to. The default is None.
    end_at_fmdtools : bool
        Option to end at first fmdtools node while building the subgraph
        Default is True, which stops the subgraph at the first fmdtools class, rather
        than includeing the fmdtools class inheritance.

    Returns
    -------
    g : graph
        Networkx graph of inheritance from classes.

    Example
    -------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> g = create_inheritance_subgraph(ExampleFunction(), end_at_fmdtools=False)
    >>> [*g.nodes]
    ['examplefunction', 'fmdtools.define.block.function.ExampleFunction', 'fmdtools.define.block.function.Function', 'fmdtools.define.block.base.Block', 'fmdtools.define.block.base.Simulable', 'fmdtools.define.object.base.BaseObject']
    >>> [*g.edges]
    [('examplefunction', 'fmdtools.define.block.function.ExampleFunction'), ('fmdtools.define.block.function.ExampleFunction', 'fmdtools.define.block.function.Function'), ('fmdtools.define.block.function.Function', 'fmdtools.define.block.base.Block'), ('fmdtools.define.block.base.Block', 'fmdtools.define.block.base.Simulable'), ('fmdtools.define.block.base.Simulable', 'fmdtools.define.object.base.BaseObject')]
    """
    if not name:
        name = get_obj_name(obj)
    if inspect.isclass(obj):
        nodetype = "class"
        g = add_node(obj, g, name=name, nodetype=nodetype)
    else:
        g = add_node(obj, g, name=name)
    if not (end_at_fmdtools and 'fmdtools.' in name):
        base_classes = get_inheritance(obj)
        for bc in base_classes:
            g = create_inheritance_subgraph(bc, g, end_at_fmdtools=end_at_fmdtools)
            add_edge(g, name, get_obj_name(bc), "base", "inheritance")
    return g


def set_node_states(g, obj, name):
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
        if hasattr(obj, 'set_node_attrs'):
            obj.set_node_attrs(g)
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


class ModelGraph(Graph):
    """
    Superclass for Graphs meant to represent specific model constructs.

    Specifically, ModelGraphs have node/edge properties like "state" and "mode"
    corresponding to an external model simulation.

    Parameters
    ----------
    get_states: bool
        whether to get states for the graph
    **kwargs:
        keyword arguments for self.nx_from_obj

    Examples
    --------
    >>> from fmdtools.define.block.function import ExampleFunction
    >>> mg = ModelGraph(ExampleFunction())
    >>> mg.get_nodes()
    ['examplefunction', 'examplefunction.t', 'examplefunction.sp', 'examplefunction.p', 'examplefunction.s', 'examplefunction.m', 'examplefunction.exampleflow', 'examplefunction.dynamic_behavior']
    """

    def __init__(self, mdl, check_info=True, **kwargs):
        Graph.__init__(self, mdl.create_graph(**kwargs), check_info=check_info)

    def get_nodes(self, rem_ind=0):
        """Get nodes (with shortened names if needed)."""
        return [shorten_name(n, rem_ind) for n in self.g.nodes]

    def set_from(self, time, history=History(), rem_ind=0):
        """Set ModelGraph faulty/degraded attributes from a given history."""
        faulty = history.get_faulty_hist(*self.get_nodes(rem_ind),
                                         withtotal=False,
                                         withtime=False).get_slice(time)
        fault_nodes = {n: bool(faulty.get(shorten_name(n, rem_ind), default=False))
                       for n in self.g.nodes}
        nx.set_node_attributes(self.g, fault_nodes, 'faulty')

        faults = Result(history.get_faults_hist(*self.get_nodes(rem_ind)).get_slice(time))
        faults_nodes = {n:
                        [k for k, v in faults.get(shorten_name(n, rem_ind)).items() if v]
                        if fault_nodes.get(n)
                        else [] for n in self.g.nodes}
        nx.set_node_attributes(self.g, faults_nodes, 'faults')

        degraded = history.get_degraded_hist(*self.get_nodes(rem_ind),
                                             withtotal=False,
                                             withtime=False).get_slice(time)
        deg_nodes = {n: bool(degraded.get(shorten_name(n, rem_ind), default=False))
                     for n in self.g.nodes}
        nx.set_node_attributes(self.g, deg_nodes, 'degraded')

        # nx.set_node_attributes(self.g, state_nodes, 'states')
        self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='faults')

    def draw_from(self, time, history=History(), rem_ind=0, **kwargs):
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
        self.set_from(time, history, rem_ind=rem_ind)
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
            m = g.nodes[node].get('m', {'faults': {}, 'sub_faults': False})
            g.nodes[node]['faulty'] = any(m['faults']) or m['sub_faults']


class ExtModelGraph(ModelGraph):
    """
    Extensible ModelGraph which separates graph generation from state setting.

    To make a ModelGraph for a particular object, one needs to create two methods
    for the class:
        - nx_from_obj, which returns a networkx graph with 'edgetype' and 'nodetype'
        information, and
        - set_nx_states, which attaches state information such as states and modes
        to node/edge attributes.
    """

    def __init__(self, mdl, get_states=True, time=0.0, check_info=False, **kwargs):
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
        Graph.__init__(self, self.nx_from_obj(mdl, **kwargs), check_info=check_info)
        if get_states:
            self.time = time
            self.set_nx_states(mdl, **kwargs)

    def nx_from_obj(self, mdl, **kwargs):
        """Alias for nx_from_obj, the method used to instantiate the graph."""
        raise Exception("nx_from_obj method not implemented for "
                        + self.__class__.__name__)

    def set_nx_states(self, mdl, **kwargs):
        """Alias for set_nx_states, which is used to map model states to graph attr."""
        raise Exception("set_nx_states method not implemented for "
                        + self.__class__.__name__)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
