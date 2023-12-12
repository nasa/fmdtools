# -*- coding: utf-8 -*-
"""
Description: A module to define flows used to conect functions in a model. Contains:

- :class:`Flow`:        Superclass for flows to be instantiated in a model.
- :class:`MultiFlow`:   Class for flows which enable multiple copies to be instantiated within itself (e.g., for perception)
- :class:`CommsFlow`:   Class for flows which enable communications (e.g., sending/recieving messages) between functions
- :func:`init_flow`:    Flow constructor/factory method.
"""
import sys
from recordclass import asdict, astuple

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.object.base import BaseObject
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist


class Flow(BaseObject):
    """Superclass for flows."""

    __slots__ = ['p', '_args_p', 's', '_args_s', 'h', 'is_copy']
    container_p = Parameter
    container_s = State
    default_track = ('s', 'i')

    def __repr__(self):
        if hasattr(self,'name'):
            return getattr(self, 'name')+' '+self.__class__.__name__+' flow: '+self.s.__repr__()
        else:
            return "Uninitialized Flow"

    def reset(self):
        """Reset the flow to the initial state."""
        self.s = self.container_s(**self._args_s)

    def return_mutables(self):
        return astuple(self.s)

    def status(self):
        """Return a dict with the current states of the flow."""
        return asdict(self.s)

    def get_memory(self):
        """Return the approximate memory usage of the flow."""
        mem = 0
        for state in self.s.__fields__:
            # (*2 to account for initstates)
            mem += 2*sys.getsizeof(getattr(self.s, state))
        return mem

    def copy(self):
        """Return a copy of the flow object (used when copying the model)."""
        cop = self.__class__(self.name, p=self.p.copy(), s=self.s.copy())
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def get_typename(self):
        return "Flow"

    def create_hist(self, timerange, track):
        """
        Create the history for the flow.

        Parameters
        ----------
        timerange : np.array
            Time-range to initialize the array over
        track : dict
            States to track

        Returns
        -------
        h : History
            History to initialize.
        """
        if hasattr(self, 'h'):
            return self.h
        else:
            track = self.get_track(track, all_possible=Flow.default_track)
            if track:
                h = History()
                sh = self.s.create_hist(timerange, get_sub_include('s', track))
                if sh:
                    h['s'] = sh
                init_indicator_hist(self, h, timerange, track)
                self.h = h
                return h
            else:
                return False

def init_flow(flowname, fclass=Flow, p={}, s={}, **kwargs):
    """
    Initialize a flow (factory method).

    Enables one to instantiate different types of flows with given
    states/parameters or  pass an already-constructured flow class.

    Parameters
    ----------
    flowname : str
        Name to give the flow object
    fclass : Flow/MultiFlow/Comms/CustomFlow
        Flow class to instantiate OR already-instanced object to pass
    p : dict
        Parameter values to override from defaults.
    s : dict
        State values to override from defaults.
    **kwargs :dict
        Other specialized roles to overrride
    """
    if not callable(fclass):
        fl = fclass
    else:
        fl = fclass(flowname, p=p, s=s, **kwargs)
    return fl
