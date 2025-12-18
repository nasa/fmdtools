# -*- coding: utf-8 -*-
"""
Aviate functionality.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function

from examples.airspacelib.base.arch.flows import Trajectories, Force, Electricity
from examples.airspacelib.base.arch.flows import AircraftEnvironment
import numpy as np

class AviateMode(Mode):
    """Aviate function modes."""

    opermodes = ('flight', 'idle', 'falling')
    mode: str = "idle"
    fault_crash = ()


class Aviate(Function):
    """
    Function that moves the drone within the environment.

    Examples
    --------
    >>> t = Trajectories()
    >>> t.s.put(z=100.0, goal_z=100.0, dz=0.0, dx=10.0)
    >>> des_traj = t.create_local("des_traj")
    >>> av = Aviate(trajectories=t)
    >>> av.static_behavior()
    >>> av.dynamic_behavior()
    >>> av.trajectories.s.x
    10.0
    >>> av.trajectories.des_traj.s.put(dx=0.0, dy=10.0)
    >>> av.static_behavior()
    >>> av.dynamic_behavior()
    >>> av.trajectories.s.y
    10.0
    """

    __slots__ = ('trajectories', 'force', 'electricity', 'environment')

    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = AircraftEnvironment
    container_m = AviateMode

    def static_behavior(self):
        """Overall static behavior of drone (determines modes based on states)."""
        if self.force.s.contact_support >= 10.0:
            self.m.add_fault('crash')

        if self.trajectories.s.z > 0.0:
            if self.electricity.s.voltage_high <= 0.0:
                self.m.set_mode('falling')
            else:
                self.m.set_mode('flight')
        else:
            if self.electricity.s.voltage_high > 0.0:
                if not self.trajectories.des_traj.s.same(dx=0.0, dy=0.0):
                    self.m.add_fault('crash')
                    self.m.set_mode('idle')
                elif not self.trajectories.des_traj.s.dz <= 0.0:
                    self.m.set_mode('flight')
                else:
                    self.m.set_mode('idle')
            else:
                self.m.set_mode('idle')

        if self.m.in_mode("falling", "crash"):
            self.falling_behavior()
        elif self.m.in_mode('flight'):
            self.flight_behavior()
        elif self.m.in_mode("idle"):
            self.idle_behavior()

    def dynamic_behavior(self):
        """Overall dynamic behavior of the drone (flight, falling, and idle)."""
        self.trajectories.s.increment_position()
        self.trajectories.s.limit(z=(0.0, np.inf))

    def flight_behavior(self):
        """
        Behavior when the drone is flying in the air.

        The trajectory increments to a new location determined by the direction and
        distance of the desired trajectory.
        """
        self.trajectories.s.assign(self.trajectories.des_traj.s, 'dx', 'dy', 'dz')
        self.electricity.s.current_high = 1.0
        self.force.s.put(lift_support=1.0)

    def falling_behavior(self):
        """
        Behavior when falling.

        The drone falls to the ground, removing all support to the drone.
        """
        if self.trajectories.s.z > 0.0:
            #self.trajectories.s.assign(self.trajectories.des_traj.s, 'dx', 'dy')
            self.trajectories.s.dz = -self.trajectories.s.z
            self.force.s.put(lift_support=0.0)
        else:
            self.idle_behavior()

    def idle_behavior(self):
        """Behavior when not moving and grounded."""
        self.trajectories.s.put(dx=0.0, dy=0.0, dz=0.0)
        self.force.s.put(lift_support=0.0)
        self.electricity.s.current_high = 0.0

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    av = Aviate()