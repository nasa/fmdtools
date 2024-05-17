# -*- coding: utf-8 -*-
"""Defines :class:`ComponentArchitecture` class to represent component architectures."""
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
        Associate a Component with the architecture. Called after add_flow.

        Parameters
        ----------
        name : str
            Internal Name for the Component
        compclass : Component
            Component class to instantiate
        *flownames : flow
            Flows (optional) which connect the components
        **kwargs : any
            kwargs to instantiate the Component with
        """
        self.add_sim('comps', name, compclass, *flownames, **kwargs)

        if hasattr(self.comps[name], 'm'):
            self.add_obj_modes(self.comps[name])

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'comps', faults)
