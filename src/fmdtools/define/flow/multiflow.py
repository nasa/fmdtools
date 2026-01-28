#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Representation of flows with multiple contained flows for perception/communication.

Defines:
- :class:`MultiFlow` class which represents multiple flows in one graph.
- :class:`MultiFlowGraph` class which represents `MultiFlow` in a ModelGraph structure.
"""

from fmdtools.define.base import get_dict_repr
from fmdtools.define.flow.base import Flow
from fmdtools.analyze.graph.model import ModelGraph
from fmdtools.define.container.state import ExampleState


class MultiFlowGraph(ModelGraph):
    """
    Create ModelGraph corresponding to the MultiFlow sturucture.

    Parameters
    ----------
    flow : MultiFlow
        Multiflow object to represent.
    role_nodes : list
        What roles of the multiflow to include. Default is ['local'].
    recursive : bool
        Whether to construct the graph recursively (with multiple levels). The default
        is True
    with_root :bool
        Whether to include the root node. Default is False.
    **kwargs : kwargs
    """

    def __init__(self, flow, role_nodes=['local'], recursive=True, with_root=False,
                 with_methods=False, **kwargs):
        ModelGraph.__init__(self, flow, role_nodes=role_nodes, recursive=recursive,
                            with_root=with_root, with_methods=with_methods, **kwargs)


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


    Examples
    --------
    >>> exf = ExampleMultiFlow()
    >>> sub_flow = exf.create_local("sub_flow")
    >>> sub_flow
    sub_flow ExampleMultiFlow
    - s=ExampleState(x=1.0, y=1.0)
    >>> sub_flow.s.put(x=0.0, y=0.0)
    >>> exf
    examplemultiflow ExampleMultiFlow
    - s=ExampleState(x=1.0, y=1.0)
    LOCALS:
    - sub_flow=(s=(x=0.0, y=0.0))
    >>> exf.update(to_get="sub_flow")
    >>> exf
    examplemultiflow ExampleMultiFlow
    - s=ExampleState(x=0.0, y=0.0)
    LOCALS:
    - sub_flow=(s=(x=0.0, y=0.0))
    >>> exf.s.put(x=10.0, y=10.0)
    >>> sub_flow.update("x", to_get="global")
    >>> exf
    examplemultiflow ExampleMultiFlow
    - s=ExampleState(x=10.0, y=10.0)
    LOCALS:
    - sub_flow=(s=(x=10.0, y=0.0))
    """

    __slots__ = ['glob', '__dict__']
    check_dict_creation = False
    flexible_roles = ['local']
    roletypes = ['container', 'local']

    def __init__(self, name='', root='', glob=[], track=['s'], **kwargs):
        """Initialize MultiFlow (glob is global MultiFlow)."""
        if not glob:
            self.glob = self
        else:
            self.glob = glob
            if not root:
                root = self.glob.get_full_name()
        super().__init__(name=name, root=root, track=track, **kwargs)
        self.locals = []

    def __repr__(self):
        """Print console string."""
        return super().create_repr(one_line=False) + self.create_local_repr()

    def create_local_repr(self):
        """Create repr string for local flows."""
        rep_str = ""
        if self.locals:
            rep_str += "\nLOCALS:"
            locs = {loc: self.get_view(loc) for loc in self.locals}
            rep_str += get_dict_repr(locs, with_classname=False, one_line=False)
        return rep_str

    def base_type(self):
        """Return fmdtools type of the model class."""
        return MultiFlow

    def create_local(self, name, attrs="all", p='global', s='global', track=['s'],
                     as_copy=False, **kwargs):
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
        as_copy : bool
            Whether to copy the local flow if it already exists.

        Returns
        -------
        newflow : MultiFlow
            Local view of the MultiFlow with its own individual values
        """
        if hasattr(self, name):
            oldflow = getattr(self, name)
            if as_copy:
                newflow = oldflow.copy(glob=self)
            else:
                newflow = oldflow
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
        if name not in self.locals:
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
        """
        Get the view of the MultiFlow corresponding to the given name.

        Parameters
        ----------
        name : str
            Name of the MultiFlow object, or other identifier, such as "global",
            "local" (which returns self), and "out" (outgoing flow for CommsFlows).

        Examples
        --------
        >>> exf = ExampleMultiFlow()
        >>> sub_f = exf.create_local("sub_f")
        >>> sub_f2 = exf.create_local("sub_f2")
        >>> sub_f.get_view("global")
        examplemultiflow ExampleMultiFlow
        - s=ExampleState(x=1.0, y=1.0)
        LOCALS:
        - sub_f=(s=(x=1.0, y=1.0))
        - sub_f2=(s=(x=1.0, y=1.0))
        >>> sub_f.get_view("sub_f2")
        sub_f2 ExampleMultiFlow
        - s=ExampleState(x=1.0, y=1.0)
        """
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
        elif name in self.glob.locals:
            view = getattr(self.glob, name)
        else:
            raise Exception("Inaccessible view name: "+name+". Is it a real flow name?")
        return view

    def update(self, *states, to_update="local", to_get="global"):
        """
        Update a view of the MultiFlow to the values of another view.

        Parameters
        ----------
        *states : str
            States to update (defaults to all states)
        to_update : str/list, optional
            Name of the view to update. The default is "local". If "all", updates all
            locals (or ports for commsflows).
            If a list is provided, updates the list (in locals)
        to_get : str, optional
            Name of the view to update from. The default is "global".
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
        """Reset the flow and sub-flows to their initial states."""
        super().reset()
        for local in self.locals:
            getattr(self, local).reset()

    def copy(self, name='', glob=[], p={}, s={}, track=['s']):
        """Copy the flow and create new sub-flow copies for it."""
        if not s and hasattr(self, 's'):
            s = self.s.asdict()
        if not glob and self.glob.name != self.name:
            raise Exception("Unable to copy "+self.name+" without "+self.glob.name+".")
        cop = self.__class__(self.name, glob=glob, p=p, s=s, track=track, root=self.root)
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        for loc in self.locals:
            local = getattr(self, loc)
            mutes = local.copy_mutes(exclude=['local'])
            cl = cop.create_local(local.name, **mutes, as_copy=True)
            if hasattr(cl, 'h'):
                cop.h[loc] = cl.h
        return cop

    def create_hist(self, timerange, track=None):
        """
        Create history for entire flow structure (with sub-flows).

        Examples
        --------
        >>> exf = ExampleMultiFlow()
        >>> exf1 = exf.create_local("exf1")
        >>> exf2 = exf1.create_local("exf2")
        >>> exf.create_hist([1,2,3])
        s.x:                            array(3)
        s.y:                            array(3)
        exf1: 
        --s.x:                          array(3)
        --s.y:                          array(3)
        --exf2: 
        ----s.x:                        array(3)
        ----s.y:                        array(3)
        >>> exf1.h
        s.x:                            array(3)
        s.y:                            array(3)
        exf2: 
        --s.x:                          array(3)
        --s.y:                          array(3)
        >>> exf2.h
        s.x:                            array(3)
        s.y:                            array(3)
        """
        super().create_hist(timerange, track=track)
        for localname in self.locals:
            local_flow = getattr(self, localname)
            self.h[localname] = local_flow.create_hist(timerange)
        return self.h

    def find_mutables(self, **kwargs):
        """Find mutables (includes locals)."""
        localflows = [getattr(self, lo) for lo in self.locals]
        return [*super().find_mutables(), *localflows]

    def as_modelgraph(self, gtype=MultiFlowGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


class ExampleMultiFlow(MultiFlow):
    """Extension of ExampleFlow to MultiFlow case."""

    __slots__ = ()
    container_s = ExampleState


if __name__ == "__main__":
    ex = ExampleMultiFlow()
    import doctest
    doctest.testmod(verbose=True)
