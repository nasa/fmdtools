# -*- coding: utf-8 -*-
"""Perceive Environment function used to perceive the position in the environment."""

from fmdtools.define.block.function import Function

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity, Environment


class PerceiveEnvironment(Function):
    """Function that percieves the environment."""

    __slots__ = ('environment', 'force', 'electricity', 'trajectories', 'perc_traj')
    flow_environment = Environment
    flow_force = Force
    flow_electricity = Electricity
    flow_trajectories = Trajectories

    def init_block(self, **kwargs):
        """Initialize the block with des_traj sub-flow."""
        self.perc_traj = self.trajectories.create_local("perc_traj")

    def dynamic_behavior(self, time):
        """
        Environmental perception behavior - mirrors trajectory.

        Examples
        --------
        >>> pe = PerceiveEnvironment()
        >>> pe.trajectories.s.x=0.5
        >>> pe.dynamic_behavior(1)
        >>> pe.perc_traj.s.x
        0.5
        """
        if self.electricity.s.voltage_low > 0.0:
            # perceive current location, goal, etc
            self.perc_traj.update()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
