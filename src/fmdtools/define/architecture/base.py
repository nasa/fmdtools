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

from fmdtools.define.object.base import check_pickleability, BaseObject, get_dict_repr
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.block.base import Simulable
from fmdtools.define.object.base import init_obj, get_obj_name
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.graph.model import add_meth_edge, add_edge
from fmdtools.analyze.graph.model import ExtModelGraph, set_node_states

import time
import networkx as nx
from ordered_set import OrderedSet


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
                set_node_states(self.g, roleobj, name)


class Architecture(Simulable):
    """
    Superclass for architectures.

    Architectures are distinguished from Block classes in that they have flexible role
    dictionaries that are populated using add_xxx methods in an overall user-defined
    init_architecture method.

    This method is called for a copy using the as_copy option, which copies passed
    flexible roles.

    Attributes
    ----------
    simorder : OrderedSet
        Keeps track of simulable dynamic execution order
    staticsims : OrderedSet
        Keeps track of which simulables run in static execution step
    dynamicsims : Orderedset
        Keeps track of which simulables run in dynamic execution step
    staticflows : list
        Flows to keep track of in static execution step
    graph : networkx graph
        multigraph view of sims and flows
    """

    __slots__ = ['flows', 'as_copy', 'h', '_init_flexroles', 'm', 'simorder',
                 '_simflows', 'graph', 'staticsims', 'dynamicsims',
                 'staticflows']
    flexible_roles = ['flow']
    roletype = 'arch'

    def __init__(self, *args, as_copy=False, h={}, **kwargs):
        self.simorder = OrderedSet()
        self._simflows = []
        self.as_copy = as_copy
        Simulable.__init__(self, *args, h=h, roletypes=['container'], **kwargs)
        self.init_hist(h=h)
        self._init_flexroles = []
        self.init_flexible_roles(**kwargs)
        self.init_architecture(**kwargs)
        self.build(**kwargs)
        self.mut_kwargs = {role: kwargs.get(role)
                           for role in self.get_roles('container', with_immutable=False)
                           if role in kwargs}

    def create_repr(self, rolenames=['s', 'm'], sim_rolenames=['s', 'm'],
                    with_classname=True, with_name=True, one_line=False):
        """Show string with sims and flows."""
        repstr = super().create_repr(rolenames=rolenames, with_classname=with_classname,
                                     with_name=with_name, one_line=one_line)
        if not one_line:
            for flex_role in self.flexible_roles:
                roledict = self.get_flex_role_objs(flex_role)
                if roledict:
                    rolestr = get_dict_repr(roledict, one_line=one_line)
                    repstr += "\n"+flex_role.upper()+"S:"+rolestr
        return repstr

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
        """Determine if static based on containment of static sims."""
        return super().is_static() or any(self.staticsims)

    def is_dynamic(self):
        """Determine if dynamic based on containment of dynamic sims."""
        return super().is_dynamic() or any(self.dynamicsims)

    def update_arch_behaviors(self, proptype):
        """
        Update/propagate behavior in the architecture.

        Parameters
        ----------
        proptype : str
            Type of propagation step to update ('dynamic', 'static', or 'both'). If
            'dynamic', this method calls self.prop_dynamic(). If 'static', this method
            calls self.prop_static()
        """
        if proptype in ["dynamic", "both"] and hasattr(self, 'prop_dynamic'):
            self.prop_dynamic()
        if proptype in ["static", "static-once", "both"] and hasattr(self, 'prop_static'):
            self.prop_static()

    def prop_dynamic(self):
        """
        Run dynamic propagation functions.

        Calls the defined dynamaic functions
        (e.g., those with dynamic_behavior() methods) in order the specified by the
        init_architecture method.
        """
        sims = self.get_sims()
        for simname in self.dynamicsims:
            sim = sims[simname]
            sim(time=self.t.time, proptype='dynamic', inc_at="")

    def prop_static(self):
        """
        Propagate behaviors through model graph (static propagation step).

        Runs by maintaining a list of "active" sims to update, starting with all
        sims with a static_behavior() method to call. For each of these sims,
        it updates the static_behavior() method. If new mutables are present, the
        simulable is kept in the list (otherwise it is removed). If its connected flows
        have new mutables, other sims connected to that flow are added to the list.
        This algorithm is run until there are no more "active" sims, which may
        require several executions of each simulable's static_behavior() method to
        propagate behavior through the entire model graph.
        """
        # set up history of flows to see if any has changed
        activesims = self.staticsims.copy()
        nextsims = set()
        # Set the flowstates from the current state of the given flows
        for flowname in self.staticflows:
            self.flows[flowname].set_mutables()
        sims = self.get_sims()
        n = 0
        while activesims:
            flows_to_check = {*self.staticflows}
            for simname in list(activesims).copy():
                sim = sims[simname]
                # Update functions with new values, check to see if new faults or states
                sim.set_mutables(exclude=[*self.staticflows])
                sim(time=self.t.time, proptype='static-once', inc_at="")
                if sim.has_changed(update=True, exclude=[*self.staticflows]):
                    nextsims.update([simname])

                # Check what flows now have new values and add connected functions
                # (done for each because of communications potential)
                for flowname in sim.flows:
                    if flowname in flows_to_check:
                        if self.flows[flowname].has_changed(update=True):
                            nextsims.update(self.get_connected_sims(flowname))
                            flows_to_check.remove(flowname)
            # check and update remaining flows that have not been checked already
            for flowname in flows_to_check:
                if self.flows[flowname].has_changed(update=True):
                    nextsims.update(self.get_connected_sims(flowname))
            activesims = nextsims.copy()
            nextsims.clear()
            n += 1
            if n > 1000:  # break if this is going for too long
                raise Exception("Undesired looping for Simulables in static",
                                "propagation at t=" + str(self.t.time) + ", these",
                                "Simulables remain active: " + str(activesims))

    def get_connected_sims(self, flowname):
        """Get the simulables connected to a given flow."""
        return set([n for n in self.graph.neighbors(flowname) if n in self.staticsims])

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
            rname = role+'s'
            if self.as_copy and rname in kwargs:
                setattr(self, rname, {**kwargs[rname]})
            elif self.as_copy:
                raise Exception("No role argument "+role+" to copy.")
            elif rname in kwargs:
                setattr(self, rname, {**kwargs[rname]})
            else:
                setattr(self, rname, dict())

    def get_flex_role_kwargs(self, objclass, **kwargs):
        """
        Get role keyword arguments for init_obj.

        Ensures that (1) Rands are synced and (2) SimParams are passed down.

        Parameters
        ----------
        objclass : class/obj
            Class or object being instantiated.
        **kwargs : kwargs
            Keyword arguments to self.add_sim.

        Returns
        -------
        **kwargs : kwargs
            Keyword arguments to self.add_flex_role_obj
        """
        kwar = {}
        # ensure global seed
        if hasattr(self, 'r') and hasattr(objclass, "container_r"):
            kwar['r'] = {"seed": self.r.seed}
        # pass simparam from arch
        if hasattr(objclass, "container_sp"):
            kwar['sp'] = self.sp.get_sub_kwargs()
        return {**kwar, **kwargs}

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
        use_copy : bool
            Whether to use existing copy in self.flows. The default is False.
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
        kwargs = self.get_flex_role_kwargs(objclass, **kwargs)
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
        self.add_flex_role_obj(flex_role, name,
                               objclass=simclass, flows=flows, **kwargs)
        for flowname in flownames:
            self._simflows.append((name, flowname))
        self.simorder.update([name])

    def set_simorder(self, simorder):
        """
        Manually set the order of sims to be executed.

        (otherwise it will be executed based on the sequence of add_sim calls)
        """
        if not self.simorder.difference(simorder):
            self.simorder = OrderedSet(simorder)
        else:
            raise Exception("Invalid list: "+str(simorder) +
                            " should have elements: "+str(self.simorder))

    def init_architecture(self, *args, **kwargs):
        """Use to initialize architecture."""
        return 0

    def build(self, update_seed=True, construct_graph=False, require_connections=False,
              **kwargs):
        """
        Construct the overall model structure.

        Use in subclasses to build the model after init_architecture is called.

        Parameters
        ----------
        update_seed : bool
            Whether to update the seed
        construct_graph : bool
            Whether to construct a graph at self.graph using construct_graph().
            Default is True.
        require_connections : bool
            Whether to require that all sims/flows be connected. Default is True.
        """
        # remove any dangling objects (flows usually) passed from above but not
        # initialized
        for role in self.flexible_roles:
            roledict = getattr(self, role+'s')
            roledict = {k: v for k, v in roledict.items() if k in self._init_flexroles}

        if update_seed and not self.as_copy:
            self.update_seed()
        if hasattr(self, 'h'):
            self.h = self.h.flatten()
        self.staticsims = OrderedSet([name for name, sim in self.get_sims().items()
                                      if sim.is_static()])
        self.dynamicsims = OrderedSet([name for name, sim in self.get_sims().items()
                                       if sim.is_dynamic()])
        if construct_graph:
            self.construct_graph(require_connections=require_connections)
            self.staticflows = [flow for flow in self.flows
                                if any([n in self.staticsims
                                        for n in self.graph.neighbors(flow)])]

    def construct_graph(self, require_connections=True):
        """Create .graph nx.graph representation of the model."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.get_sims(), bipartite=0)
        self.graph.add_nodes_from(self.flows, bipartite=1)
        self.graph.add_edges_from(self._simflows)

        # check to see that all functions/flows are connected
        dangling_nodes = [e for e in nx.isolates(self.graph)]
        if dangling_nodes and require_connections:
            raise Exception("Fxns/flows disconnected from model: "+str(dangling_nodes))

    def plot_dynamic_run_order(self, rotateticks=False, title="Dynamic Run Order"):
        """
        Plot the run order for the model during the dynamic propagation step.

        The x-direction is the order of each function executed and the y are the
        corresponding flows acted on by the given methods.

        Parameters
        ----------
        rotateticks : Bool, optional
            Whether to rotate the x-ticks (for bigger plots). The default is False.
        title : str, optional
            String to use for the title (if any). The default is "Dynamic Run Order".

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        from matplotlib import pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import AutoMinorLocator
        if not hasattr(self, 'graph'):
            raise Exception("Unable to plot run order without graph attribute.")
        fxnorder = list(self.dynamicsims)
        times = [i+0.5 for i in range(len(fxnorder))]
        fxntimes = {f: i for i, f in enumerate(fxnorder)}

        flowtimes = {f: [fxntimes[n] for n in self.graph.neighbors(
            f) if n in self.dynamicsims] for f in self.flows}

        lengthorder = {k: v for k, v in
                       sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True)
                       if len(v) > 0}
        starttimeorder = {k: v for k, v in sorted(lengthorder.items(),
                                                  key=lambda x: x[1][0], reverse=True)}
        endtimeorder = [k for k, v in sorted(starttimeorder.items(),
                                             key=lambda x: x[1][-1], reverse=True)]
        flowtimedict = {flow: i for i, flow in enumerate(endtimeorder)}

        fig, ax = plt.subplots()

        for flow in flowtimes:
            phaseboxes = [((t, flowtimedict[flow]-0.5),
                           (t, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]+0.5),
                           (t+1.0, flowtimedict[flow]-0.5))
                          for t in flowtimes[flow]]
            bars = PolyCollection(phaseboxes)
            ax.add_collection(bars)

        flowtimes = [i+0.5 for i in range(len(self.flows))]
        ax.set_yticks(list(flowtimedict.values()))
        ax.set_yticklabels(list(flowtimedict.keys()))
        ax.set_ylim(-0.5, len(flowtimes)-0.5)
        ax.set_xticks(times)
        ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
        ax.set_xlim(0, len(times))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which='minor', linewidth=2)
        ax.tick_params(axis='x', bottom=False, top=False,
                       labelbottom=False, labeltop=True)
        if title:
            if rotateticks:
                fig.suptitle(title, fontweight='bold', y=1.15)
            else:
                fig.suptitle(title, fontweight='bold')
        return fig, ax

    def get_flows(self, *flownames, all_if_empty=True):
        """Return a list of the model flow objects."""
        if hasattr(self, 'flows'):
            if all_if_empty and not flownames:
                flownames = self.flows
            return {flowname: self.flows[flowname] for flowname in flownames}
        else:
            return {}

    def flowtypes(self):
        """Return the set of flow types used in the model."""
        return {obj.__class__.__name__: obj.get_typename()
                for f, obj in self.get_flows().items()}

    def flows_of_type(self, ftype):
        """Return the set of flows for each flow type."""
        return {flow for flow, obj in self.get_flows().items()
                if obj.__class__.__name__ == ftype}

    def get_rand_states(self, auto_update_only=False):
        """Get dictionary of random states throughout the model objs."""
        rand_states = {}
        role_objs = self.sims()
        for objname, obj in role_objs.items():
            if hasattr(obj, 'get_rand_states'):
                rand_state = obj.get_rand_states(auto_update_only=auto_update_only)
                if rand_state:
                    rand_states[objname] = rand_state
        return rand_states

    def reset(self):
        """Reset the architecture and its contained objects."""
        super().reset()
        for obj in self.get_flex_role_objs().values():
            if hasattr(obj, 'reset'):
                obj.reset()

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
        cargs = dict(root=self.root,
                     p=getattr(self, 'p', {}),
                     sp=getattr(self, 'sp', {}),
                     track=getattr(self, 'track', {}),
                     h=self.h.copy(),
                     t=kwargs.get('t', self.t.copy()),
                     as_copy=True)
        # send role dicts in to be copied via as_copy param.
        for flex_role in self.flexible_roles:
            cargs[flex_role+'s'] = getattr(self, flex_role+'s')
        # if flows provided from above, use those flows. Otherwise copy own.
        if hasattr(self, 'flows'):
            cargs['flows'] = self.copy_flows(flows=flows)

        if hasattr(self, 'r'):
            cargs['r'] = self.r.copy()
        cop = self.__class__(**cargs)
        cop.assign_roles('container', self)
        return cop

    def copy_flows(self, flows={}):
        """Copy flows, along with MultiFlow structures if present."""
        globs = {}
        cflows = {}
        for fn, flow in self.flows.items():
            if fn in flows:
                cflows[fn] = flows[fn]
            elif not isinstance(flow, MultiFlow) or flow.glob.name == flow.name:
                cflows[fn] = flow.copy()
            else:
                if flow.glob.name not in globs:
                    globs[flow.glob.name] = flow.glob.copy()
                cflows[fn] = globs[flow.glob.name].get_view(fn)
        return cflows

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

    def as_drawio(self, saveas='', **kwargs):
        """
        Generate DrawIO diagram from the architecture.

        Parameters
        ----------
        saveas : str, optional
            File path to save the DrawIO XML. If empty, returns XML content.
        **kwargs : dict
            Additional arguments passed to the graph creation.

        Returns
        -------
        xml_content: str
            DrawIO XML content.

        Examples
        --------
        >>> from fmdtools.define.architecture.function import ExFxnArch
        >>> arch = ExFxnArch()
        >>> xml_content = arch.as_drawio()  # Get XML content
        >>> xml_content = arch.as_drawio("architecture.drawio")  # Save to file
        """
        graph = self.as_modelgraph(**kwargs)
        return graph.draw_drawio(saveas=saveas, **kwargs)


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


if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=True)

