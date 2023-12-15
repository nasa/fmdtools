# -*- coding: utf-8 -*-
"""
Description: A module to define flows used to conect functions in a model. Contains:

- :class:`Flow`: Superclass for flows to be instantiated in a model.
"""
from recordclass import asdict, astuple

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.object.base import BaseObject


class Flow(BaseObject):
    """Superclass for flows."""

    __slots__ = ['p', 's', 'h']
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

    def copy(self, **kwargs):
        """Return a copy of the flow object (used when copying the model)."""
        loc_kwargs = {'p': self.p.copy(), **kwargs, 's': self.s.copy(), 'name': self.name}
        cop = self.__class__(**loc_kwargs)
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def get_typename(self):
        return "Flow"


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)