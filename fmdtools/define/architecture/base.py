#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines base :class:`Architecture` class used by other architecture classes.

Includes:
- :class:`Architecture` class defining architectures.
- :class:`ArchitectureGraph` class which represents `Architecture` in a ModelGraph.

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

from fmdtools.define.object.base import check_pickleability, BaseObject
from fmdtools.define.flow.base import Flow
from fmdtools.define.block.base import Simulable
from fmdtools.define.object.base import init_obj, get_obj_name
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.graph.model import add_meth_edge, add_edge
from fmdtools.analyze.graph.model import ExtModelGraph, set_node_states

import time


class ArchitectureGraph(ExtModelGraph):
    """Base ModelGraph for Architectures."""

    def nx_from_obj(self, mdl, with_root=False, **kwargs):
        """
        Generate the networkx.graph object corresponding to the model.

        Parameters
        ----------
        mdl: FunctionArchitecture
            Model to create the graph representation of

        Returns
        -------
        g : networkx.Graph
            networkx.Graph representation of model functions and flows
            (along with their attributes)
        """
        return mdl.create_graph(with_root=with_root, **kwargs)

    def draw_from(self, *args, rem_ind=2, **kwargs):
        """Set from a history (removes prefixes so it works at top level)."""
        return super().draw_from(*args, rem_ind=rem_ind, **kwargs)

    def set_nx_states(self, mdl, **kwargs):
        """Set the states of the graph."""
        basename = mdl.get_full_name()
        for role, roleobj in mdl.get_roles_as_dict().items():
            name = get_obj_name(roleobj, role, basename=basename)
            if name in self.g.nodes:
                set_node_states(self.g, roleobj, name, time=self.time)


class Architecture(Simulable):
    """
    Superclass for architectures.

    Architectures are distinguished from Block classes in that they have flexible role
    dictionaries that are populated using add_xxx methods in an overall user-defined
    init_architecture method.

    This method is called for a copy using the as_copy option, which copies passed
    flexible roles.
    """

    __slots__ = ['flows', 'as_copy', 'h', '_init_flexroles', 'm']
    flexible_roles = ['flows']
    roletype = 'arch'

    def __init__(self, *args, as_copy=False, h={}, **kwargs):
        self.as_copy = as_copy
        Simulable.__init__(self, *args, h=h, roletypes=['container'], **kwargs)
        self.init_hist(h=h)
        self._init_flexroles = []
        self.init_flexible_roles(**kwargs)
        self.init_architecture(**kwargs)
        self.build(**kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Architecture

    def check_role(self, roletype, rolename):
        """Check that 'arch_xa' role is used for the arch."""
        if roletype != self.roletype:
            raise Exception("Invalid roletype for Architecture: " + roletype +
                            ", should be: " + self.roletype)
        if rolename != self.rolename:
            raise Exception("invalid roletype for " + str(self.__class__) +
                            ", should be: " + self.rolename)

    def is_static(self):
        """Determine if static (False by default)."""
        return False

    def is_dynamic(self):
        """Determine if dynamic (False by default)."""
        return False

    def init_flexible_roles(self, **kwargs):
        """
        Initialize flexible roles.

        If initializing as a copy, uses a passed copy instead.

        Parameters
        ----------
        **kwargs : kwargs
            Existing roles (if any).
        """
        for role in self.flexible_roles:
            if self.as_copy and role in kwargs:
                setattr(self, role, {**kwargs[role]})
            elif self.as_copy:
                raise Exception("No role argument "+role+" to copy.")
            elif role in kwargs:
                setattr(self, role, {**kwargs[role]})
            else:
                setattr(self, role, dict())

    def add_flex_role_obj(self, flex_role, name, objclass=BaseObject, use_copy=False,
                          **kwargs):
        """
        Add a flexible role object to the architecture.

        Used for add_fxn, add_flow, etc methods in subclasses.
        If called during copying (self.as_copy=True), the object is copied instead
        of instantiated.

        Parameters
        ----------
        flex_role : str
            Name of the role dictionary to initialize the object in.
        name : str
            Name of the object
        objclass : class, optional
            Class to instantiate in the dict. The default is BaseObject.
        as_copy : bool
            Whether to instantiate obj as a copy. The default is fault.
        **kwargs : kwargs
            Non-default kwargs to send to the object class.
        """
        roledict = getattr(self, flex_role)
        if name in roledict:
            objclass = roledict[name]

        if use_copy:
            as_copy = False
        else:
            as_copy = self.as_copy

        track = get_sub_include(name, get_sub_include(flex_role, self.track))
        # sync rands, if present
        if hasattr(self, 'r') and hasattr(objclass, "container_r"):
            kwargs = {**{'r': {"seed": self.r.seed}}, **kwargs}
        obj = init_obj(name=name, objclass=objclass, track=track,
                       as_copy=as_copy, root=self.get_full_name()+"."+flex_role,
                       **kwargs)

        if hasattr(obj, 'h') and obj.h:
            hist = obj.h
        elif isinstance(obj, BaseObject):
            timerange = self.sp.get_histrange()
            hist = obj.create_hist(timerange)
        else:
            hist = False
        if hist:
            self.h[flex_role + '.' + name] = hist
        roledict[name] = obj
        self._init_flexroles.append(name)

    def add_flow(self, name, fclass=Flow, **kwargs):
        """
        Add a flow with given attributes to the model.

        Parameters
        ----------
        name : str
            Unique flow name to give the flow in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        if name in self.flows:
            use_copy = True
        else:
            use_copy = False
        self.add_flex_role_obj('flows', name, objclass=fclass, use_copy=use_copy,
                               **kwargs)

    def add_flows(self, *names, fclass=Flow, **kwargs):
        """
        Add a set of flows with the same type and initial parameters.

        Parameters
        ----------
        flownames : list
            Unique flow names to give the flows in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        for name in names:
            self.add_flow(name, fclass=fclass, **kwargs)

    def add_sim(self, flex_role, name, simclass, *flownames, **kwargs):
        """
        Add a Simulable to the given flex_role.

        Parameters
        ----------
        flex_role : str
            Name of the flexible role to add to.
        name : str
            Name to give the Simulable.
        simclass: Simulable
            Simulable to instantiate.
        flownames : list
            List of flows to associate with the function.
        **kwargs : kwargs
            Flows, dicts for non-default values to p, s, etc.
        """
        flows = self.get_flows(*flownames, all_if_empty=False)
        fkwargs = {**{'sp': self.sp.asdict()}, **kwargs}
        if not self.sp.use_local:
            fkwargs = {**{'t': {'dt': self.sp.dt}}, **fkwargs}

        self.add_flex_role_obj(flex_role, name, objclass=simclass, flows=flows, **fkwargs)

        # add modes to overall mode dict
        sim = self.get_flex_role_objs(flex_role)[name]
        if hasattr(sim, 'm') and hasattr(self, 'm'):
            self.add_obj_modes(sim)

    def init_architecture(self, *args, **kwargs):
        """Use to initialize architecture."""
        return 0

    def build(self, update_seed=True, **kwargs):
        """
        Construct the overall model structure.

        Use in subclasses to build the model after init_architecture is called.

        Parameters
        ----------
        update_seed : bool
            Whether to update the seed
        """
        # remove any dangling objects (flows usually) passed from above but not
        # initialized
        for role in self.flexible_roles:
            roledict = getattr(self, role)
            roledict = {k: v for k, v in roledict.items() if k in self._init_flexroles}

        if update_seed and not self.as_copy:
            self.update_seed()
        if hasattr(self, 'h'):
            self.h = self.h.flatten()

    def get_flows(self, *flownames, all_if_empty=True):
        """Return a list of the model flow objects."""
        if all_if_empty and not flownames:
            flownames = self.flows
        return {flowname: self.flows[flowname] for flowname in flownames}

    def flowtypes(self):
        """Return the set of flow types used in the model."""
        return {obj.__class__.__name__: obj.get_typename()
                for f, obj in self.flows.items()}

    def flows_of_type(self, ftype):
        """Return the set of flows for each flow type."""
        return {flow for flow, obj in self.flows.items()
                if obj.__class__.__name__ == ftype}

    def update_seed(self, seed=[]):
        """
        Update model seed and the seed in all contained roles.

        Must have an associated Rand role.

        Parameters
        ----------
        seed : int, optional
            Seed to use. The default is [].
        """
        if hasattr(self, 'r'):
            super().update_seed(seed)
            role_objs = self.get_flex_role_objs()
            for obj in role_objs.values():
                if hasattr(obj, 'update_seed'):
                    obj.update_seed(self.r.seed)

    def get_rand_states(self, auto_update_only=False):
        """Get dictionary of random states throughout the model objs."""
        rand_states = {}
        role_objs = self.get_flex_role_objs()
        for objname, obj in role_objs.items():
            if hasattr(obj, 'get_rand_states'):
                rand_state = obj.get_rand_states(auto_update_only=auto_update_only)
                if rand_state:
                    rand_states[objname] = rand_state
        return rand_states

    def get_faults(self):
        """Get faults from contained roles."""
        return {obj.name+"_"+f for obj in self.get_flex_role_objs().values()
                if hasattr(obj, 'm') for f in obj.m.faults}

    def reset(self):
        """Reset the architecture and its contained objects."""
        super().reset()
        for obj in self.get_flex_role_objs().values():
            if hasattr(obj, 'reset'):
                obj.reset()

    def add_obj_modes(self, obj):
        """Add modes from an object to faultmodes."""
        modes_to_add = {obj.name+'_'+f: val
                        for f, val in obj.m.faultmodes.items()}
        fmode_intersect = set(modes_to_add).intersection(self.m.sub_modes)
        if any(fmode_intersect):
            raise Exception("Action " + obj.name +
                            " overwrites existing fault modes: "+str(fmode_intersect) +
                            ". Rename the faults")
        self.m.sub_modes.update({obj.name+'_'+modename: obj.name
                                 for modename in obj.m.faultmodes})

    def inject_faults(self, flexible_role, faults):
        """
        Inject faults in the ComponentArchitecture/ASG object obj.

        Parameters
        ----------
        flexible_role: str
            Name of role to inject faults in (e.g., fxns, .acts, comps)
        faults : dict
            Dict of faults to inject {'fxnname': ['faultname']}.
        """
        compdict = getattr(self, flexible_role)
        for fault in faults:
            if hasattr(self, 'm') and self.m.sub_modes and fault in self.m.sub_modes:
                comp = compdict[self.m.sub_modes[fault]]
                comp.inject_faults(fault[len(comp.name)+1:])
            elif fault in compdict:
                comp = compdict[fault]
                comp.inject_faults(faults[fault])

    def copy(self, flows={}, **kwargs):
        """
        Copy the architecture at the current state.

        Parameters
        ----------
        flows : dict
            Dict of flows to use in the copy.

        Returns
        -------
        copy : Architecture
            Copy of the curent architecture.
        """
        cargs = dict(p=getattr(self, 'p', {}),
                     sp=getattr(self, 'sp', {}),
                     track=getattr(self, 'track', {}),
                     h=self.h.copy(),
                     t=kwargs.get('t', self.t.copy()),
                     as_copy=True)
        # send role dicts in to be copied via as_copy param.
        for flex_role in self.flexible_roles:
            cargs[flex_role] = getattr(self, flex_role)
        # if flows provided from above, use those flows. Otherwise copy own.
        if hasattr(self, 'flows'):
            cargs['flows'] = {f: flows[f] if f in flows else obj.copy()
                              for f, obj in self.flows.items()}

        if hasattr(self, 'r'):
            cargs['r'] = self.r.copy()
        cop = self.__class__(**cargs)
        cop.assign_roles('container', self)
        return cop

    def get_all_possible_track(self):
        return super().get_all_possible_track() + self.flexible_roles

    def add_subgraph_edges(self, g, cond_edges=True, flow_edges=True, **kwargs):
        """Add edges connecting the objects (conditions and flows) to a graph."""
        BaseObject.add_subgraph_edges(self, g, **kwargs)
        for flex_role in self.flexible_roles:
            role_objs = self.get_flex_role_objs(flex_role)
            for rolename, obj in role_objs.items():
                objname = get_obj_name(obj, rolename, self.get_full_name())
                add_meth_edge(g, obj, rolename, edgetype="activation")
                if flow_edges and hasattr(obj, 'flows'):
                    for locflowname, flowobj in obj.get_roles_as_dict('flow').items():
                        fname = flowobj.get_full_name()
                        if fname in g.nodes():
                            add_edge(g, objname, fname, locflowname, 'flow')

    def as_modelgraph(self, gtype=ArchitectureGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


def check_model_pickleability(model, try_pick=False):
    """
    Check to see which attributes of a model object will pickle.

    Provides more detail about functions/flows.
    """
    print('FLOWS ')
    for flowname, flow in model.flows.items():
        print(flowname)
        check_pickleability(flow, try_pick=try_pick)
    print('FUNCTIONS ')
    for fxnname, fxn in model.fxns.items():
        print(fxnname)
        check_pickleability(fxn, try_pick=try_pick)
    time.sleep(0.2)
    print('MODEL')
    unpickleable = check_pickleability(model, try_pick=try_pick)
