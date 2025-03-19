# -*- coding: utf-8 -*-
"""Store and supply electricity function."""

from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State

from aerialdrm.base.aircraft.arch.flows import Force, Electricity

class StoreEEState(State):
    """State of the battery."""
    charge: float = 100.0


class StoreAndSupplyElectricity(Function):
    """Function used to store and supply energy to the drone."""

    __slots__ = ('force', 'electricity')
    flow_force = Force
    flow_electricity = Electricity
    container_s = StoreEEState

    def dynamic_behavior(self, time):
        rate_high = self.electricity.s.mul('current_high', 'voltage_high')
        rate_low = self.electricity.s.mul('current_low', 'voltage_low')
        self.s.inc(charge=rate_high+0.1*rate_low)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)