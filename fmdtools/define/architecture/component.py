# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:22:12 2023

@author: dhulse
"""
from typing import ClassVar
from recordclass import dataobject, asdict
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist
from fmdtools.define.base import get_true_fields, get_true_field


class ComponentArchitecture(dataobject, mapping=True):
    """Container for holding component architectures."""

    archtype: ClassVar[str] = 'default'
    components: dict = dict()
    faultmodes: dict = dict()
    default_track = ('i', 'components')
    def make_components(self, CompClass, *args, **kwargs): # noqa
        """
        Adds components to the component architecture.

        Parameters
        ----------
        CompClass : Component
            Component to add
        *args : strs
            Names for the components to instantiate in the architecture
        **kwargs : kwargs
            keyword arguments to send CompClass, of form {'name':kwarg}.
            unless all have the same kwargs
        """
        if not self.components:
            self.components = dict()
        if not self.faultmodes:
            self.faultmodes = dict()

        for arg in args:
            if arg in kwargs:
                kwargs_comp = kwargs[arg]
            else:
                kwargs_comp = kwargs
            self.components[arg] = CompClass(arg, **kwargs_comp)
            self.faultmodes.update({self.components[arg].name + '_' + modename: arg
                                    for modename in self.components[arg].m.faultmodes})

    def copy_with_arg(self, **kwargs):
        cop = self.__class__(**kwargs)
        for compname, component in self.components.items():  # TODO: needs to cover all attributes, copy should a part of Block
            cop_comp = cop.components[compname]
            cop_comp.s = cop_comp.container_s(**asdict(component.s))
            cop_comp.m.mirror(component.m)
            cop_comp.t = component.t.copy()
            if hasattr(component, 'h'):
                cop_comp.h = component.h.copy()
        return cop

    def update_seed(self, seed):
        for comp in self.components.values():
            comp.update_seed(seed)

    def get_rand_states(self, auto_update_only=False):
        rand_states={}
        for compname, comp in self.components.items():
            if comp.get_rand_states(auto_update_only=auto_update_only): 
                rand_states[compname] = comp.get_rand_states(auto_update_only=auto_update_only)
        return rand_states

    def get_faults(self):
        return {comp.name+'_'+f for comp in self.components.values() for f in comp.m.faults}

    def reset(self):
        for name, component in self.components.items():
            component.reset()

    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)

    def get_true_fields(self, *args, **kwargs):
        return get_true_fields(self, *args, **kwargs)

    def create_hist(self, timerange, track):
        """
        Creates a history corresponding to ComponentArchitecture attributes.

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.

        Returns
        -------
        h : History
            History corresponding to the ComponentArchitecture
        """
        h = History()
        if track == 'default':
            track = self.default_track
        init_indicator_hist(self, h, timerange, track)

        components_track = get_sub_include('components', track)
        if components_track:
            hc = History()
            for c, comp in self.components.items():
                comp_track = get_sub_include(c, components_track)
                if comp_track: 
                    hc[c] = comp.create_hist(timerange, comp_track)
            h['components'] = hc
        return h

    def return_mutables(self):
        cm = []
        for c in self.components.values():
            cm.extend(c.return_mutables())
        return cm
