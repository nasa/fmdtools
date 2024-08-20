#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Representation of flows with multiple contained flows for perception/communication.

Defines:
- :class:`MultiFlow` class which represents multiple flows in one graph.
- :class:`MultiFlowGraph` class which represents `MultiFlow` in a ModelGraph structure.
"""

from fmdtools.define.flow.base import Flow
from fmdtools.analyze.graph.model import ModelGraph


class MultiFlowGraph(ModelGraph):
    """
    Create ModelGraph corresponding to the MultiFlow sturucture.

    Parameters
    ----------
    flow : MultiFlow
        Multiflow object to represent.
    role_nodes : list
        What roles of the multiflow to include. Default is ['locals'].
    recursive : bool
        Whether to construct the graph recursively (with multiple levels). The default
        is True
    with_root :bool
        Whether to include the root node. Default is False.
    **kwargs : kwargs
    """

    def __init__(self, flow, role_nodes=['local'], recursive=True, with_root=False,
                 **kwargs):
        ModelGraph.__init__(self, flow, role_nodes=role_nodes, recursive=recursive,
                            with_root=with_root, **kwargs)


    def set_resgraph(self, other=False):
        """
        Process results for results graphs (show faults and degradations).

        Parameters
        ----------
        other : Graph, optional
            Graph to compare with (for degradations). The default is False.
        """
        if other:
            self.set_degraded(other)
            self.set_node_styles(degraded={}, faulty={})
        else:
            self.set_degraded(self)
            self.set_node_styles(degraded={}, faulty={})
        self.set_node_labels(title='id', subtext='indicators')

    def draw_graphviz(self, layout="neato", overlap='false', **kwargs):
        return super().draw_graphviz(layout=layout, overlap=overlap, **kwargs)

class MultiFlow(Flow):
    """
    MultiFlow class enables represenation of multiple connected copies of the same flow.

    It enables the addition of local flows in an overall flow architecture, which are
    essentially copies of the main flow which live in functions. MultiFlows are
    helpful in cases where each function has a different view of the same external
    flow (e.g., perception, etc). Notably, this class adds the methods:
        - create_local(), which can be used to add a local flow to a function/block
        - get_view(), which can be used to look at other local views of the flow
        - update(), which can be used to update one view of the flow from another

    A MultiFlow can have any number of local views (listed by name in MultiFlow.locals)
    as well as a single global view (which may represent the actual value)
    """

    slots = ['locals', '__dict__']
    check_dict_creation = False
    flexible_roles = ['locals']
    roletypes = ['container', 'local']

    def __init__(self, name='', root='', glob=[], track=['s'], **kwargs):
        if not glob:
            self.glob = self
        else:
            self.glob = glob
            if not root:
                root = self.glob.get_full_name()
        super().__init__(name=name, root=root, track=track, **kwargs)
        self.locals = []

    def __repr__(self):
        rep_str = Flow.__repr__(self)
        for loc in self.locals:
            rep_str = rep_str+"\n   "+self.get_view(loc).__repr__()
        return rep_str

    def base_type(self):
        """Return fmdtools type of the model class."""
        return MultiFlow

    def create_local(self, name, attrs="all", p='global', s='global', track=['s'],
                     **kwargs):
        """
        Create a local view of the Flow.

        Parameters
        ----------
        name : str
            Name for the view (to retrieve at the Flow level)
        attrs : dict/list/str, optional
            Attributes to include in the local copy. The default is "all". Has options:
                str: to use if only using a single attribute of the local flow
                list: list of attributes to use in the local flow
                dict: dict of attributes to use in the local flow and initial values
        p : dict
            Parameters to instantiate the local version with (if params used in flow)
            Default is 'global', which uses the same parameter as the multiflow
        s : dict
            Initial values for the states. Default is 'global', which uses the
            same initial states as the multiflow

        Returns
        -------
        newflow : MultiFlow
            Local view of the MultiFlow with its own individual values
        """

        if hasattr(self, name):
            oldflow = getattr(self, name)
            newflow = oldflow.copy(glob=self)
        else:
            kwar = {}
            if p == 'global' and hasattr(self, 'p'):
                kwar['p'] = self.p
            else:
                kwar['p'] = p
            if s == 'global' and hasattr(self, 's'):
                kwar['s'] = self.s.asdict()
            else:
                kwar['s'] = s
            kwar = {**kwar, **kwargs}
            newflow = self.__class__(name=name, glob=self, track=track, **kwar)
        setattr(self, name, newflow)
        self.locals.append(name)
        if hasattr(self, 'h') and self.h:
            self.create_hist([*self.h.values()][0])
        return newflow

    def get_local_name(self, name):
        """
        Get the name of the view corresponding to the given name.

        Enables "local" or "global" options.
        """
        if name == "local":
            return self.name
        elif name == "global":
            return "glob"
        else:
            return name

    def get_view(self, name):
        """Get the view of the MultiFlow corresponding to the given name."""
        if name == "":
            raise Exception("Must provide view")
        elif name == "local":
            view = self
        elif name == "global":
            view = self.glob
        elif name == "out":
            view = getattr(self.glob, self.name + "_out")
        elif name in getattr(self, 'locals', []):
            view = getattr(self, name)
        else:
            view = getattr(self.glob, name)
        return view

    def update(self, to_update="local", to_get="global", *states):
        """
        Update a view of the MultiFlow to the values of another view.

        Parameters
        ----------
        to_update : str/list, optional
            Name of the view to update. The default is "local". If "all", updates all
            locals (or ports for commsflows).
            If a list is provided, updates the list (in locals)
        to_get : str, optional
            Name of the view to update from. The default is "global".
        *states : str
            States to update (defaults to all states)
        """
        get = self.get_view(to_get)
        if to_update == 'all':
            if hasattr(self, 'fxns'):
                updatelist = [*self.fxns]
            else:
                updatelist = self.locals
        elif isinstance(to_update, str):
            updatelist = [to_update]
        elif isinstance(to_update, list):
            updatelist = to_update
        else:
            raise Exception("Invalid to_update: "+str(to_update))
        for to_up in updatelist:
            up = self.get_view(to_update)
            up.s.assign(get.s, *states, as_copy=True)

    def reset(self):
        super().reset()
        for local in self.locals:
            getattr(self, local).reset()

    def copy(self, name='', glob=[], p={}, s={}, track=['s']):
        if not s and hasattr(self, 's'):
            s = self.s.asdict()
        cop = self.__class__(self.name, glob=glob, p=p, s=s, track=track)
        for loc in self.locals:
            local = getattr(self, loc)
            cop.create_local(local.name, **local.copy_mut_containers())
        return cop

    def create_hist(self, timerange):
        super().create_hist(timerange)
        for localname in self.locals:
            local_flow = getattr(self, localname)
            self.h[localname] = local_flow.create_hist(timerange)
        return self.h

    def find_mutables(self):
        """Find mutables (includes locals)."""
        localflows = [getattr(self, lo) for lo in self.locals]
        return [*super().find_mutables(), *localflows]

    def as_modelgraph(self, gtype=MultiFlowGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)
