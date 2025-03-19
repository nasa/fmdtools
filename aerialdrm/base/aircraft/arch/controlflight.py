# -*- coding: utf-8 -*-
"""ControlFlight function used to control the aircraft."""


from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
import numpy as np

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity
from aerialdrm.base.aircraft.arch.flows import Environment


class ControlState(State):
    """State of ControlFlight Function planning."""

    flightplan: tuple = ((0.0, 0.0), (25.0, 25.0))
    height: float = 25.0
    pt: int = 0

    def get_goal(self):
        return self.flightplan[self.pt]

    def inc_goal(self):
        self.pt+=1

    def is_start(self):
        return self.pt == 0

    def is_end(self):
        return self.pt >= len(self.flightplan)-1


class ControlMode(Mode):
    """ControlFlight mode."""

    opermodes = ('ascend', 'descend', 'flight', 'idle')
    fault_off = ()
    mode: str = 'idle'


class ControlFlight(Function):
    """
    Flight control function.

    Determines direction and distance to travel based on location and flightplan.
    """

    __slots__ = ('trajectories', 'perc_traj', 'des_traj', 'force', 'electricity', 'environment')
    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment
    container_s = ControlState
    container_m = ControlMode

    def init_block(self, **kwargs):
        """Add the desired trajectory local flow to the Function."""
        self.des_traj = self.trajectories.create_local("des_traj")

    def static_behavior(self, time):
        """Propagate static behaviors for flight control."""
        if self.electricity.s.voltage_low <= 0.0:
            self.m.set_mode('idle')
        else:
            self.electricity.s.current_low = 1.0

    def dynamic_behavior(self, time):
        """
        Propagate overall modal logic for flight control.

        Examples
        ----------
        >>> t = Trajectories()
        >>> perc_traj = t.create_local('perc_traj')
        >>> cf = ControlFlight(trajectories=t)
        >>> cf.m.mode
        'idle'
        >>> cf.dynamic_behavior(1)
        >>> cf.m.mode
        'ascend'
        >>> cf.trajectories.perc_traj.s.z = cf.s.height
        >>> cf.dynamic_behavior(2)
        >>> cf.m.mode
        'flight'
        >>> cf.des_traj.s.get('dx', 'dy')
        array([7.07106781, 7.07106781])
        >>> cf.trajectories.perc_traj.s.put(x=25.0, y=25.0)
        >>> cf.dynamic_behavior(3)
        >>> cf.m.mode
        'descend'
        >>> cf.trajectories.perc_traj.s.put(z=0.0)
        >>> cf.dynamic_behavior(4)
        >>> cf.m.mode
        'idle'
        """
        self.trajectories.perc_traj.update('goal_x', 'goal_y', 'goal_z',
                                           to_get='des_traj')
        if self.s.is_start() and self.t.time>0.0:
            self.takeoff_planning()
        elif self.s.is_end():
            self.landing_planning()
        else:
            self.flight_planning()

        if self.m.in_mode('ascend'):
            self.ascend_behavior()
        elif self.m.in_mode('descend'):
            self.descend_behavior()
        elif self.m.in_mode('flight'):
            self.flight_behavior()
        elif self.m.in_mode('idle'):
            self.idle_behavior()

    def takeoff_planning(self):
        """Determine flight mode at the start of the flight plan."""
        if self.m.in_mode('idle'):
            self.m.set_mode('ascend')
            self.set_goal()
        elif self.m.in_mode('ascend') and self.trajectories.perc_traj.s.z >= self.s.height:
            self.m.set_mode('flight')
            self.s.inc_goal()
            self.set_goal()

    def landing_planning(self):
        """Determine flight mode at the end of the flight plan."""
        if self.m.in_mode('flight') and self.trajectories.perc_traj.s.at_goal():
            self.m.set_mode('descend')
            self.set_goal()
        elif self.m.in_mode('descend') and self.trajectories.perc_traj.s.z <= 0.0:
            self.m.set_mode('idle')
            self.set_goal()

    def flight_planning(self):
        """Determine flight mode in the middle of the flight plan."""
        if self.trajectories.perc_traj.s.at_goal():
            self.s.inc_goal()
            self.set_goal()

    def set_goal(self):
        """Set the goal properties of the trajectories flow based on the flight mode."""
        self.des_traj.s.assign(self.trajectories.perc_traj.s, 'x', 'y', 'z')
        self.des_traj.s.assign(self.s.get_goal(), 'goal_x', 'goal_y')
        if self.m.in_mode('ascend'):
            self.des_traj.s.goal_z = self.s.height
        elif self.m.in_mode('descend'):
            self.des_traj.s.goal_z = 0.0
        elif self.m.in_mode('flight', 'idle'):
            self.des_traj.s.goal_z = self.des_traj.s.goal_z

    def descend_behavior(self):
        """Set dist and direction of the aircraft when descending."""
        self.des_traj.s.update_position(maxvel=0.0,
                                        max_zvel=self.trajectories.perc_traj.s.z)

    def ascend_behavior(self):
        """Set dist and direction of aircraft when ascending."""
        self.des_traj.s.update_position(maxvel=0.0, max_zvel=self.des_traj.p.max_vel)

    def flight_behavior(self):
        """Set dist and direction of aircraft when in normal flight."""
        # assign direction and distance to go
        self.des_traj.s.update_position(maxvel=self.des_traj.p.max_vel, max_zvel=0.0)

    def idle_behavior(self):
        """Set dist and direction of aircraft when idling."""
        self.des_traj.s.update_position(maxvel=0.0)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    t = Trajectories()
    t.create_local('perc_traj')
    cf = ControlFlight(trajectories=t)
    cf.dynamic_behavior(1)
    cf.trajectories.perc_traj.s.z = cf.s.height
    cf.dynamic_behavior(2)
    cf.trajectories.perc_traj.s.put(x=25.0, y=25.0)
    cf.dynamic_behavior(3)
    cf.trajectories.perc_traj.s.put(z=0.0)
    cf.dynamic_behavior(4)

