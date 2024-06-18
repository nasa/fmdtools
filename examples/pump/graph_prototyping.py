# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:11 2024

@author: dhulse
"""

from fmdtools.analyze.graph.model import ModelGraph
from fmdtools.analyze.graph.model import set_block_node
from fmdtools.define.flow.base import Flow
from fmdtools.define.object.base import BaseObject
import networkx as nx
import inspect


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
                g.add_edge(basename, rolename, edgetype="flow", role=role)
            else:
                g.add_node(role, nodetype=roleobj.get_typename(), classname=classname)
                g.add_edge(basename, role, edgetype="containment", role=role)
        return g

    def set_nx_states(self, mdl, **kwargs):
        for role, roleobj in mdl.get_roles_as_dict().items():
            if isinstance(roleobj, Flow):
                set_block_node(self.g, roleobj, roleobj.name, time=self.time)
            else:
                self.g.nodes[role]['obj'] = roleobj


class ArchRoleGraph(RoleGraph):

    def nx_from_obj(self, mdl, with_flow_edges=False):
        g = RoleGraph.nx_from_obj(self, mdl)
        for flex_role in mdl.flexible_roles:
            g.add_node(flex_role, nodetype="dict", classname='dict')
            g.add_edge(mdl.name, flex_role, edgetype="containment",
                       role=flex_role)
            role_objs = mdl.get_flex_role_objs(flex_role)
            for objname, obj in role_objs.items():
                if isinstance(obj, BaseObject):
                    g.add_node(obj.name, nodetype=obj.get_typename(),
                               classname=obj.__class__.__name__)
                    g.add_edge(flex_role, obj.name, edgetype="containment", role=objname)
                elif inspect.ismethod(obj):
                    g.add_node(obj.__name__, nodetype='Condition', classname='method')
                    g.add_edge(flex_role, obj.__name__, edgetype="aggregation", role=objname)
                    g.add_edge(obj.__self__.name, obj.__name__, edgetype='containment',
                               role=obj.__self__.name)
                if with_flow_edges and hasattr(obj, 'flows'):
                    flowdict = obj.get_roles_as_dict('flow')
                    for locflowname, flowobj in flowdict.items():
                        g.add_edge(obj.name, flowobj.name, edgetype="flow",
                                   role=locflowname)
        return g

    def set_nx_states(self, mdl, **kwargs):
        super().set_nx_states(mdl, **kwargs)
        for flex_role in mdl.flexible_roles:
            role_objs = mdl.get_flex_role_objs(flex_role)
            for objname, obj in role_objs.items():
                if isinstance(obj, BaseObject):
                    set_block_node(self.g, obj, obj.name, time=self.time)


from examples.pump.ex_pump import Pump

rg = ArchRoleGraph(Pump(), with_flow_edges=True)
rg.set_node_labels(title2='classname', subtext='s')
rg.set_edge_labels(title='role')
rg.draw()

rg2 = RoleGraph(Pump().fxns['import_ee'])
rg2.set_node_labels(title2='classname', subtext='obj')
rg2.set_edge_labels(title='role')
rg2.draw()

from examples.asg_demo.demo_model import Human

rg3 = ArchRoleGraph(Human(), with_flow_edges=True)
rg3.draw()