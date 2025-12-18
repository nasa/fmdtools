# -*- coding: utf-8 -*-
"""
ControlFlight function used to control the aircraft.

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


from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode

from examples.airspacelib.base.arch.flows import Trajectories, Force, Electricity
from examples.airspacelib.base.arch.flows import AircraftEnvironment


class ControlState(State):
    """
    State of ControlFlight Function planning.

    Parameters
    ----------
    flightplan: tuple
        Set of x/y points the drone is to fly through.
    height: float
        Height the drone is to fly at.
    pt: int
        Point within the flight plan that the drone is to fly to.
    """

    flightplan: tuple = ((0.0, 0.0), (25.0, 25.0))
    height: float = 25.0
    pt: int = 0

    def get_goal(self):
        """Get the current goal point (x,y) for the drone."""
        if self.pt < len(self.flightplan):
            return self.flightplan[self.pt]
        else:
            return self.flightplan[-1]

    def inc_goal(self):
        """Increment the current goal."""
        if self.pt < len(self.flightplan):
            self.pt += 1

    def is_start(self):
        """Determine whether the drone is as the start of its flightplan."""
        return self.pt == 0

    def is_end(self):
        """Determine whether the drone is at the end of its flightplan."""
        return self.pt >= len(self.flightplan)-1


class ControlMode(Mode):
    """ControlFlight mode."""

    opermodes = ('ascend', 'descend', 'flight', 'idle', 'pause')
    exclusive = True
    fault_off = ()
    fault_loss = ()
    mode: str = 'idle'


class ControlFlight(Function):
    """
    Flight control function.

    Determines direction and distance to travel based on location and flightplan.

    Examples
    --------
    >>> t = Trajectories()
    >>> perc_traj = t.create_local('perc_traj')
    >>> cf = ControlFlight(trajectories=t)
    >>> cf.m.mode
    'idle'
    >>> cf(0)
    >>> cf(1)
    >>> cf.m.mode
    'ascend'
    >>> cf.trajectories.perc_traj.s.z = cf.s.height
    >>> cf(2)
    >>> cf.m.mode
    'flight'
    >>> cf.des_traj.s.get('dx', 'dy')
    array([7.07106781, 7.07106781])
    >>> cf.trajectories.perc_traj.s.put(x=25.0, y=25.0)
    >>> cf(3)
    >>> cf.m.mode
    'descend'
    >>> cf.trajectories.perc_traj.s.put(z=0.0)
    >>> cf(4)
    >>> cf.m.mode
    'idle'
    """

    __slots__ = ('trajectories', 'perc_traj', 'des_traj', 'force', 'electricity',
                 'environment')
    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = AircraftEnvironment
    container_s = ControlState
    container_m = ControlMode

    def init_block(self, **kwargs):
        """Add the desired trajectory local flow to the Function."""
        self.perc_traj = self.trajectories.create_local("perc_traj")
        self.des_traj = self.trajectories.create_local("des_traj")

    def static_behavior(self):
        """
        Propagate static behaviors for flight control.

        Determines modes and behavior for loss/idling.
        """
        self.set_faultmode()
        self.set_des_traj()

    def set_faultmode(self):
        if self.force.s.contact_support >= 5.0:
            self.m.add_fault('loss')
        if self.electricity.s.voltage_low <= 0.0 and not self.m.in_mode('loss'):
            self.m.set_mode('idle')
        else:
            self.electricity.s.current_low = 1.0
        if self.m.in_mode('idle'):
            self.electricity.s.power_high = False
        else:
            self.electricity.s.power_high = True

    def set_des_traj(self):
        if not self.m.in_mode('idle', 'loss'):
            self.set_goal()
            self.des_traj.s.update_position(maxvel=self.des_traj.p.max_vel)
        elif self.m.in_mode('idle'):
            self.des_traj.s.put(dx=0.0, dy=0.0, dz=0.0)
        else:
            self.des_traj.s.put(dz=-self.trajectories.s.z)

    def dynamic_behavior(self):
        """Propagate overall modal logic for flight control."""
        if not self.m.in_mode('loss'):
            self.trajectories.perc_traj.update('goal_x', 'goal_y', 'goal_z',
                                               to_get='des_traj')
            if self.s.is_start():
                if self.t.time > 0.0:
                    self.takeoff_planning()
            elif self.s.is_end():
                self.landing_planning()
            else:
                self.flight_planning()

    def takeoff_planning(self):
        """Determine flight mode at the start of the flight plan."""
        if self.m.in_mode('idle'):
            self.m.set_mode('ascend')
        elif self.m.in_mode('ascend') and self.trajectories.perc_traj.s.z >= self.s.height:
            self.m.set_mode('flight')
            self.s.inc_goal()

    def landing_planning(self):
        """Determine flight mode at the end of the flight plan."""
        if self.m.in_mode('flight') and self.trajectories.perc_traj.s.at_goal():
            self.m.set_mode('descend')
        elif self.m.in_mode('descend') and self.trajectories.perc_traj.s.z <= 0.0:
            self.m.set_mode('idle')

    def flight_planning(self):
        """Determine flight mode in the middle of the flight plan."""
        if self.trajectories.perc_traj.s.at_goal() and not self.s.is_end() and not self.m.in_mode('pause'):
            self.s.inc_goal()

    def set_goal(self):
        """Set the goal properties of the trajectories flow based on the flight mode."""
        self.des_traj.s.assign(self.trajectories.perc_traj.s, 'x', 'y', 'z')
        if self.m.in_mode('ascend'):
            newgoal = [*self.des_traj.s.get('x', 'y'), self.s.height]
        elif self.m.in_mode('descend'):
            newgoal = [*self.des_traj.s.get('x', 'y'), 0.0]
        elif self.m.in_mode('flight'):
            newgoal = [*self.s.get_goal(), self.s.height]
        elif self.m.in_mode('idle', 'pause'):
            newgoal = self.des_traj.s.get_loc()
        self.des_traj.s.assign(newgoal, 'goal_x', 'goal_y', 'goal_z')


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    cf = ControlFlight()
    cf.dynamic_behavior()
    cf.trajectories.perc_traj.s.z = cf.s.height
    cf.dynamic_behavior()
    cf.trajectories.perc_traj.s.put(x=25.0, y=25.0)
    cf.dynamic_behavior()
    cf.trajectories.perc_traj.s.put(z=0.0)
    cf.dynamic_behavior()
