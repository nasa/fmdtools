# -*- coding: utf-8 -*-
"""Aviate functionality."""


from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
import numpy as np

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity, Environment


class AviateMode(Mode):
    """Aviate function modes."""

    opermodes = ('flight', 'idle', 'falling')
    mode: str = "idle"
    fault_crash = ()


class Aviate(Function):
    """Function that moves the drone within the environment."""

    __slots__ = ('trajectories', 'force', 'electricity', 'environment')

    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment
    container_m = AviateMode

    def dynamic_behavior(self, time):
        """
        Overall dynamic behavior of the drone (flight, falling, and idle).

        Examples
        --------
        >>> t = Trajectories()
        >>> t.s.put(z=100.0, goal_z=100.0, dz=0.0, dx=10.0)
        >>> des_traj = t.create_local("des_traj")
        >>> av = Aviate(trajectories=t)
        >>> av.dynamic_behavior(1)
        >>> av.trajectories.s.x
        10.0
        >>> av.trajectories.des_traj.s.put(dx=0.0, dy=10.0)
        >>> av.dynamic_behavior(2)
        >>> av.trajectories.s.y
        10.0
        """
        if self.electricity.s.voltage_high > 0.0:
            self.m.set_mode('flight')
        elif self.trajectories.z > 0.0:
            self.m.set_mode('falling')
        else:
            self.m.set_mode("idle")

        if self.m.in_mode('flight'):
            self.flight_behavior()
        elif self.m.in_mode("falling", "crash"):
            self.falling_behavior()
        elif self.m.in_mode("idle"):
            self.idle_behavior()

    def flight_behavior(self):
        """
        Behavior when the drone is flying in the air.

        The trajectory increments to a new location determined by the direction and
        distance of the desired trajectory.
        """
        self.trajectories.s.assign(self.trajectories.des_traj.s, 'dx', 'dy', 'dz')
        self.trajectories.s.increment_position()
        self.electricity.s.current_high = 2.0
        self.force.s.put(lift_support=1.0)

    def falling_behavior(self):
        """
        Behavior when falling.

        The drone falls to the ground, removing all support to the drone.
        """
        self.trajectories.s.dz = -self.trajectories.s.z
        self.trajectories.s.dist = 0.0
        self.trajectories.s.z = 0.0
        self.force.s.put(lift_support=0.0)
        self.m.add_fault("crash")
        self.m.set_mode("idle")

    def idle_behavior(self):
        """Behavior when not moving and grounded."""
        self.trajectories.s.put(dx=0.0, dy=0.0, dz=0.0)
        self.force.s.put(lift_support=0.0)
        self.s.current_high = 0.0

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    av = Aviate()