# -*- coding: utf-8 -*-
"""Module defining :class:`Flow` class used to connect multiple blocks in a model."""
from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.state import ExampleState


class Flow(BaseObject):
    """
    Superclass for flows.

    Flows are used to connect simulable parts of a model (e.g., functions)
    and respresent shared variables or states.

    Examples
    --------
    >>> class ExampleFlow(Flow):
    ...     __slots__ = ()
    ...     container_s = ExampleState

    >>> exf = ExampleFlow('exf', s={'x': 0.0})
    >>> exf
    exf ExampleFlow flow: ExampleState(x=0.0, y=1.0)

    Note that copying creates independent copies of states:

    >>> exf2 = exf.copy()
    >>> exf2
    exf ExampleFlow flow: ExampleState(x=0.0, y=1.0)
    >>> exf2.s.x = 2.0
    >>> exf2.s == exf.s
    False
    """

    __slots__ = ('p', 's', 'h')
    default_track = ('s', 'i')
    check_dict_creation = True

    def __repr__(self):
        if hasattr(self, 'name'):
            return (getattr(self, 'name') + ' ' + self.__class__.__name__ +
                    ' flow: ' + self.s.__repr__())
        else:
            return "Uninitialized Flow"

    def check_role(self, roletype, rolename):
        """Flows may be given any role name."""
        if roletype != 'flow':
            raise Exception("Invalid roletype for flow: " + roletype)

    def reset(self):
        """Reset the flow to the initial state."""
        self.s.reset()

    def copy(self, **kwargs):
        """Return a copy of the flow object (used when copying the model)."""
        loc_kwargs = {}
        if hasattr(self, 'p'):
            loc_kwargs['p'] = self.p.copy()
        loc_kwargs = {**loc_kwargs, **kwargs}
        if hasattr(self, 's'):
            loc_kwargs['s'] = self.s.copy()
        loc_kwargs['name'] = self.name
        cop = self.__class__(**loc_kwargs)
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Flow

    def create_hist(self, timerange):
        self.h = BaseObject.create_hist(self, timerange)
        return self.h


class ExampleFlow(Flow):
    """Example flow for testing."""

    __slots__ = ()
    container_s = ExampleState


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)