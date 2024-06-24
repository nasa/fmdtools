# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:11 2024

@author: dhulse
"""

from fmdtools.analyze.graph.model import ModelGraph
from fmdtools.analyze.graph.model import set_block_node
from fmdtools.define.flow.base import Flow
from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.base import BaseContainer
import networkx as nx
import inspect
import re


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
    if isinstance(obj, BaseObject):
        if obj.root == baseobj.name:
            edgetype = 'containment'
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


def add_sub_nodes(g, obj, recursive=False, basename='', with_methods=False,
                  get_source=False):
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
    for rolename, roleobj in obj.get_roles_as_dict(flex_prefixes=True).items():
        name = get_obj_name(roleobj, rolename, basename)
        add_role_node(g, obj, basename, roleobj, rolename, get_source=get_source)
        if recursive and isinstance(roleobj, BaseObject):
            add_sub_nodes(g, roleobj, recursive=recursive, basename=name,
                          with_methods=with_methods, get_source=get_source)
    if with_methods:
        for methodname, methodobj in get_obj_methods(obj).items():
            name = get_obj_name(methodobj, methodobj, basename=basename)
            add_role_node(g, obj, basename, methodobj, name, get_source=get_source)


def get_obj_methods(obj):
    methods = {at[0]: at[1] for at in inspect.getmembers(obj)
               if at[0] not in dir(obj.base_type()) and inspect.ismethod(at[1])}
    return methods


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
    if isinstance(obj, BaseObject):
        set_block_node(g, obj, name, time=time)
    elif inspect.ismethod(obj):
        g.nodes[name]['condition'] = obj()
    else:
        g.nodes[name]['obj'] = obj


def set_node_code(g, obj, name):
    docs = ''
    source = ''
    code = ''
    if isinstance(obj, BaseObject) or isinstance(obj, BaseContainer):
        docs = inspect.getdoc(obj)
        source = inspect.getsource(obj.__class__)
        code = ''
    elif inspect.ismethod(obj):
        docs = inspect.getdoc(obj)
        source = inspect.getsource(obj)
        code = source.split("'''")[-1].split('"""')[-1]
    g.nodes[name]['docs'] = docs
    g.nodes[name]['source'] = source
    g.nodes[name]['code'] = code


class FullModelGraph(ModelGraph):

    def nx_from_obj(self, mdl, with_methods=False, get_source=False):
        g = nx.DiGraph()
        name = get_obj_name(mdl, '')
        add_node(g, mdl, rolename=name, get_source=get_source)
        add_sub_nodes(g, mdl, recursive=True, basename=name, with_methods=with_methods,
                      get_source=get_source)
        return g

    def set_nx_states(self, mdl, **kwargs):
        name = get_obj_name(mdl, '')
        set_node_states(self.g, mdl, name, time=self.time)
        set_sub_nodes(self.g, mdl, time=self.time, recursive=True, basename=name)



class BlockGraph(ModelGraph):

    def nx_from_obj(self, mdl, with_methods=True, get_source=False):
        g = nx.DiGraph()
        add_node(g, mdl, get_source=get_source)
        add_sub_nodes(g, mdl, basename=mdl.get_full_name(), with_methods=with_methods,
                      get_source=get_source)
        return g

    def set_nx_states(self, mdl, **kwargs):
        basename = mdl.get_full_name()
        for role, roleobj in mdl.get_roles_as_dict().items():
            name = get_obj_name(roleobj, role, basename=basename)
            set_node_states(self.g, roleobj, name, time=self.time)

    def set_edge_labels(self, title='edgetype', title2='', subtext='role',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)

    def set_node_labels(self, title='shortname', title2='classname', **node_label_styles):
        super().set_node_labels(title=title, title2=title2, **node_label_styles)


class ArchitectureGraph(BlockGraph):

    def nx_from_obj(self, mdl, flow_edges=True, cond_edges=True, get_source=False):
        g = BlockGraph.nx_from_obj(self, mdl, get_source=get_source)
        for flex_role in mdl.flexible_roles:
            role_objs = mdl.get_flex_role_objs(flex_role)
            for rolename, obj in role_objs.items():
                objname = get_obj_name(obj, rolename, mdl.get_full_name())
                add_cond_edge(g, obj)
                if flow_edges and hasattr(obj, 'flows'):
                    for locflowname, flowobj in obj.get_roles_as_dict('flow').items():
                        add_edge(g, obj, objname, flowobj, locflowname)
        return g


from examples.asg_demo.demo_model import Human, HazardModel
mdl = HazardModel()
fmg = FullModelGraph(mdl, with_methods=True)
fmg.set_edge_labels(title="role")
fmg.draw_graphviz()

from examples.pump.ex_pump import Pump, ImportEE

rg = ArchitectureGraph(Pump(), flow_edges=True)
rg.draw()

rg2 = BlockGraph(Pump().fxns['import_ee'], get_source=True)
rg2.set_node_labels(subtext='code')
rg2.draw()

rg3 = ArchitectureGraph(Human(), flow_edges=True)
rg3.draw()
mdl = Pump()


# looking at hierarchy
