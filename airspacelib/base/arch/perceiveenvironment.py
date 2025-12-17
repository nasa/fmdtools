# -*- coding: utf-8 -*-
"""Perceive Environment function used to perceive the position in the environment."""

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode

from airspacelib.base.arch.flows import Trajectories, Force, Electricity, AircraftEnvironment


class PerceiveEnvironmentMode(Mode):
    """
    Perception Faults
    TODO:
    fault_bias : perceived trajectory off the actual trajectory by a fixed amount
    fault_noisy: perceived trajectory randomly off the actual trajectory
    fault_dimloss: perceived trajectory not updating in a dimension
    """
    fault_break = ()

class PerceiveEnvironment(Function):
    """Function that perceives the environment."""

    __slots__ = ('environment', 'force', 'electricity', 'trajectories', 'perc_traj')
    container_m = PerceiveEnvironmentMode
    flow_environment = AircraftEnvironment
    flow_force = Force
    flow_electricity = Electricity
    flow_trajectories = Trajectories

    def init_block(self, **kwargs):
        """Initialize the block with des_traj sub-flow."""
        self.perc_traj = self.trajectories.create_local("perc_traj")

    def static_behavior(self):
        if self.force.s.contact_support >= 7.0:
            self.m.add_fault('break')

    def dynamic_behavior(self):
        """
        Environmental perception behavior - mirrors trajectory.

        Examples
        --------
        >>> pe = PerceiveEnvironment()
        >>> pe.trajectories.s.x=0.5
        >>> pe.dynamic_behavior()
        >>> pe.perc_traj.s.x
        0.5
        """
        if self.electricity.s.voltage_low > 0.0 and not self.m.any_faults():
            # perceive current location, goal, etc
            self.perc_traj.update("x", "y", "z", "dx", "dy", "dz")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
