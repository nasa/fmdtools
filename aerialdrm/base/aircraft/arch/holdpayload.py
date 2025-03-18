# -*- coding: utf-8 -*-
"""HoldPayload function used for force balance."""

from fmdtools.define.block.function import Function

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force


class HoldPayload(Function):
    """Function determining force balance and payload."""

    __slots__ = ('force', 'trajectories')
    flow_force = Force
    flow_trajectories = Trajectories

    def static_behavior(self, time):
        if self.trajectories.s.location_z > 0.0:
            self.force.s.put(ground_support=0.0)
        elif self.force.s.lift_support > 0.0:
            self.force.s.put(ground_support=1.0)
        else:
            self.force.s.put(ground_support=10.0)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)