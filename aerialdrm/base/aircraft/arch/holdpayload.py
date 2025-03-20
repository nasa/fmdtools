# -*- coding: utf-8 -*-
"""HoldPayload function used for force balance."""

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force


class HoldPayloadMode(Mode):
    fault_break = ()


class HoldPayload(Function):
    """Function determining force balance and payload."""

    __slots__ = ('force', 'trajectories')
    flow_force = Force
    flow_trajectories = Trajectories
    container_m = HoldPayloadMode

    def static_behavior(self, time):
        if self.trajectories.s.z > 0.0:
            self.force.s.put(contact_support=0.0)
        elif self.force.s.lift_support > 0.0 or abs(self.trajectories.s.dz) < 15.0:
            self.force.s.put(contact_support=1.0)
        else:
            self.force.s.put(contact_support=10.0)

        if self.force.s.contact_support >= 5.0:
            self.m.add_fault('break')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)