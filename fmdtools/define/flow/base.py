# -*- coding: utf-8 -*-
"""
Description: A module to define flows used to conect functions in a model. Contains:

- :class:`Flow`: Superclass for flows to be instantiated in a model.
- :func:`init_flow`: Flow constructor/factory method.
"""
from recordclass import asdict, astuple

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.object.base import BaseObject


class Flow(BaseObject):
    """Superclass for flows."""

    __slots__ = ['p', 's', 'h', 'is_copy']
    container_p = Parameter
    container_s = State
    default_track = ('s', 'i')

    def __repr__(self):
        if hasattr(self, 'name'):
            return (getattr(self, 'name') + ' ' + self.__class__.__name__ +
                    ' flow: ' + self.s.__repr__())
        else:
            return "Uninitialized Flow"

    def check_role(self, rolename):
        """Flows may be given any role name."""
        a = 1

    def reset(self):
        """Reset the flow to the initial state."""
        self.s = self.container_s(**self._args_s)

    def return_mutables(self):
        return astuple(self.s)

    def status(self):
        """Return a dict with the current states of the flow."""
        return asdict(self.s)

    def copy(self):
        """Return a copy of the flow object (used when copying the model)."""
        cop = self.__class__(self.name, p=self.p.copy(), s=self.s.copy())
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def get_typename(self):
        return "Flow"


def init_flow(flowname, fclass=Flow, track='default', **kwargs):
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
    **kwargs :dict
        Other specialized roles to overrride
    """
    if not callable(fclass):
        fl = fclass
        fl.init_track(track)
    else:
        fl = fclass(flowname, track=track, **kwargs)
    return fl


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)