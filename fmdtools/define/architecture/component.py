# -*- coding: utf-8 -*-
"""
Module for action architectures.

Classes
-------
:class:`ComponentArchitecture`: Architecture defining agglomeration of Components.
"""
from fmdtools.define.architecture.base import Architecture


class ComponentArchitecture(Architecture):
    """Class defining Component Architectures."""

    __slots__ = ['comps', 'faultmodes']
    flexible_roles = ['flows', 'comps']
    rolename = 'ca'

    def __init__(self, **kwargs):
        self.faultmodes = {}
        Architecture.__init__(self, **kwargs)

    def add_comp(self, name, compclass, *flownames, **kwargs):
        """
        Associate an Action with the architecture. Called after add_flow.

        Parameters
        ----------
        name : str
            Internal Name for the Action
        compclass : Component
            Component class to instantiate
        *flownames : flow
            Flows (optional) which connect the actions
        duration:
            Duration of the action. Default is 0.0
        **kwargs : any
            kwargs to instantiate the Action with.
        """
        # same as fxns:
        flows = {fl: self.flows[fl] for fl in flownames}
        fkwargs = {**{'t': {'dt': self.sp.dt}},
                   **{'sp': {'end_time': self.sp.end_time}},
                   **kwargs}
        if hasattr(self, 'r'):
            fkwargs = {**{'r': {"seed": self.r.seed}}, **fkwargs}
        self.add_flex_role_obj('comps', name, objclass=compclass,
                               flows=flows, **fkwargs)

        if hasattr(self.comps[name], 'm'):
            self.add_obj_modes(self.comps[name])

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'comps', faults)
