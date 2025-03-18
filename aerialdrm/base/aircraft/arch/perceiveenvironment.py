# -*- coding: utf-8 -*-
"""Perceive Environment function used to perceive the position in the environment."""

from fmdtools.define.block.function import Function

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity, Environment

class PerceiveEnvironment(Function):
    """Placeholder for environment perception"""

    __slots__ = ('environment', 'force', 'electricity', 'trajectories', 'des_traj')
    flow_environment = Environment
    flow_force = Force
    flow_electricity = Electricity
    flow_trajectories = Trajectories

    def init_block(self, **kwargs):
        self.des_traj = self.trajectories.create_local("des_traj")

    def dynamic_behavior(self):
        if self.electricity.voltage_low > 0.0:
            # perceive current location, goal, etc
            self.des_traj.s.assign(self.trajectories.s, 'location_x', 'location_y', 'location_z')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
