# -*- coding: utf-8 -*-
"""HoldPayload function used for force balance."""

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode

from dronelib.base.arch.flows import Trajectories, Force


class HoldPayloadMode(Mode):
    """Modes for drone structure."""

    fault_break = ()


class HoldPayload(Function):
    """Function determining force balance and payload."""

    __slots__ = ('force', 'trajectories')
    flow_force = Force
    flow_trajectories = Trajectories
    container_m = HoldPayloadMode

    def static_behavior(self):
        next_z = self.trajectories.s.z + self.trajectories.s.dz
        if next_z <= 0.0 and self.trajectories.s.dz < -15.0:
            self.force.s.put(contact_support=10.0)
        elif next_z <= 0.0 or self.trajectories.s.z <= 0.0:
            self.force.s.put(contact_support=1.0)
        else:
            self.force.s.put(contact_support=0.0)

        if self.force.s.contact_support >= 5.0:
            self.m.add_fault('break')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)