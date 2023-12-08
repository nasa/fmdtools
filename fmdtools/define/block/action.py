# -*- coding: utf-8 -*-
"""
Description: A module to define Actions.

- :class:`Action`: Class for defining Actions.

"""
from fmdtools.define.block.base import Block
# Actions/ASGs


class Action(Block):
    """
    Superclass for actions.

    Actions are blocks which have behaviors and live in an ActionArchitecture.
    """

    __slots__ = ('duration',)

    def __init__(self, name, duration=0.0, **kwargs):
        self.duration = duration
        super().__init__(name, **kwargs)

    def __call__(self, time=0, run_stochastic=False, proptype='dynamic', dt=1.0):
        """
        Update the behaviors, faults, times, etc of the action.

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if time > self.t.time:
            self.r.update_stochastic_states()
        if proptype == 'dynamic':
            if self.t.time < time:
                self.behavior(time)
                self.t.t_loc += dt
        else:
            self.behavior(time)
            self.t.t_loc += dt
        self.t.time = time

    def copy(self, *args, **kwargs):
        cop = super().copy(*args, **kwargs)
        cop.duration = self.duration
        return cop

    def behavior(self, time):
        """Placeholder behavior method for actions."""
        a = 0
