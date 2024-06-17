# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:11 2024

@author: dhulse
"""

from fmdtools.analyze.graph.model import ModelGraph
from fmdtools.define.flow.base import Flow
import networkx as nx


class RoleGraph(ModelGraph):

    def nx_from_obj(self, mdl):
        g = nx.DiGraph()
        basename = mdl.name
        g.add_node(basename, nodetype=mdl.get_typename(),
                   classname=mdl.__class__.__name__)
        for role in mdl.get_roles():
            roleobj = getattr(mdl, role)
            classname = roleobj.__class__.__name__
            if isinstance(roleobj, Flow):
                rolename = roleobj.name
                g.add_node(rolename, nodetype=roleobj.get_typename(),
                           classname=classname)
                g.add_edge(basename, rolename, edgetype="flow", role=rolename)
            else:
                g.add_node(role, nodetype=roleobj.get_typename(), classname=classname)
                g.add_edge(basename, role, edgetype="containment", role=role)
        return g

    def set_nx_states(self, mdl):
        for role in mdl.get_roles():
            roleobj = getattr(mdl, role)


class ArchRoleGraph(RoleGraph):

    def nx_from_obj(self, mdl):
        g = RoleGraph.nx_from_obj(self, mdl)
        for flex_role in mdl.flexible_roles:
            g.add_node(flex_role, nodetype="dict", classname='dict')
            g.add_edge(mdl.name, flex_role, edgetype="containment",
                       role=flex_role)
        return g


from examples.pump.ex_pump import Pump
rg = ArchRoleGraph(Pump())
rg.set_node_labels(title2='classname')
rg.set_edge_labels(title='role')
rg.draw()

rg = RoleGraph(Pump().fxns['import_ee'])
rg.set_node_labels(title2='classname')
rg.set_edge_labels(title='role')
rg.draw()