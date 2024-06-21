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


def get_node_shortname(fullname):
    return fullname.split('.')[-1]


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
    if role:
        name = role
    elif isinstance(obj, BaseObject):
        name = obj.name
    elif isinstance(obj, BaseContainer):
        name = obj.__class__.__name__
    elif inspect.ismethod(obj):
        name = obj.__self__.name + '.' + obj.__name__
    elif hasattr(obj, '__name__'):
        name = obj.__name__
    else:
        raise Exception("Unknown name for object" + str(obj))
    if basename:
        name = basename + '.' + name
    return name


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


def add_node(g, obj, rolename='base', basename=''):
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


def add_role_node(g, baseobj, basename, roleobj, rolename):
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
    add_node(g, roleobj, rolename=rolename, basename=basename)
    add_edge(g, baseobj, basename, roleobj, rolename)
    # if a method, check/add edge to containing object
    # if inspect.ismethod(roleobj):
    #     true_base = roleobj.__self__
    #     true_basename = get_obj_name(true_base, 'na')
    #     add_edge(g, true_base, true_basename, roleobj, rolename)


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


def add_sub_nodes(g, obj, recursive=False, basename=''):
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
        if isinstance(roleobj, BaseObject) and roleobj.root != obj.name:
            a=1
            # need a way to get the correct name for the role and add edge
            # add_edge(g, obj, basename, roleobj, rolename)
        else:
            name = get_obj_name(roleobj, rolename, basename)
            add_role_node(g, obj, basename, roleobj, rolename)
            if recursive and isinstance(roleobj, BaseObject):
                add_sub_nodes(g, roleobj, recursive=recursive, basename=name)


def set_sub_nodes(g, obj, time=None, recursive=False, basename=''):
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


class FullModelGraph(ModelGraph):

    def nx_from_obj(self, mdl):
        g = nx.DiGraph()
        name = get_obj_name(mdl, '')
        add_node(g, mdl, rolename=name)
        add_sub_nodes(g, mdl, recursive=True, basename=name)
        return g

    def set_nx_states(self, mdl, **kwargs):
        name = get_obj_name(mdl, '')
        # set_node_states(self.g, mdl, name, time=self.time)
        # set_sub_nodes(self.g, mdl, time=self.time, recursive=True, basename=name)



class RoleGraph(ModelGraph):

    def nx_from_obj(self, mdl):
        g = nx.DiGraph()
        add_node(g, mdl)
        add_sub_nodes(g, mdl)
        return g

    def set_nx_states(self, mdl, **kwargs):
        for role, roleobj in mdl.get_roles_as_dict().items():
            set_node_states(self.g, roleobj, role, time=self.time)


class ArchRoleGraph(RoleGraph):

    def nx_from_obj(self, mdl, flow_edges=True, cond_edges=True):
        g = RoleGraph.nx_from_obj(self, mdl)
        for flex_role in mdl.flexible_roles:
            role_objs = mdl.get_flex_role_objs(flex_role)
            for objname, obj in role_objs.items():
                if cond_edges and inspect.ismethod(obj):
                    baseobj = obj.__self__
                    basename = get_obj_name(baseobj, 'na')
                    add_edge(g, baseobj, basename, obj, objname)
                if flow_edges and hasattr(obj, 'flows'):
                    for locflowname, flowobj in obj.get_roles_as_dict('flow').items():
                        add_edge(g, obj, objname, flowobj, locflowname)
        return g


from examples.asg_demo.demo_model import Human, HazardModel
mdl = HazardModel()
fmg = FullModelGraph(mdl)
fmg.draw_graphviz()

from examples.pump.ex_pump import Pump, ImportEE

rg = ArchRoleGraph(Pump(), flow_edges=True)
rg.set_node_labels(title2='classname', subtext='s')
rg.set_edge_labels(title='role')
rg.draw()

rg2 = RoleGraph(Pump().fxns['import_ee'])
rg2.set_node_labels(title2='classname')
rg2.set_edge_labels(title='role')
rg2.draw()



rg3 = ArchRoleGraph(Human(), flow_edges=True)
rg3.draw()
mdl = Pump()

# looking at hierarchy
g = nx.DiGraph()
add_node(g, mdl)
add_sub_nodes(g, mdl, recursive=True)
from fmdtools.analyze.graph.base import Graph
Graph(g).draw()

g = nx.DiGraph()


Graph(g).draw_graphviz()


