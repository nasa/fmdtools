# -*- coding: utf-8 -*-
"""ControlFlight function used to control the aircraft."""


from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode

from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity
from aerialdrm.base.aircraft.arch.flows import Environment


class ControlState(State):
    """State of ControlFlight Function planning."""

    flightplan: tuple = ((0, 0), (25, 25))
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

    __slots__ = ('trajectories', 'des_traj', 'force', 'electricity', 'environment')
    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment
    container_s = ControlState
    container_m = ControlMode

    def init_block(self, **kwargs):
        self.des_traj = self.trajectories.create_local("des_traj")

    def dynamic_behavior(self, time):
        if self.electricity.s.voltage_low <= 0.0:
            self.m.set_mode('idle')
        else:
            self.electricity.s.current_low = 1.0

        if self.s.is_start():
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
        if self.m.in_mode('idle'):
            self.m.set_mode('ascend')
            self.set_goal()
        elif self.m.in_mode('ascend') and self.des_traj.s.location_z >= self.s.height:
            self.m.set_mode('flight')
            self.s.inc_goal()
            self.set_goal()

    def landing_planning(self):
        if self.m.in_mode('flight') and self.des_traj.s.at_goal():
            self.m.get_mode('descend')

    def flight_planning(self):
        if self.des_traj.s.at_goal():
            self.s.inc_goal()
            self.set_goal()

    def set_goal(self):
        self.des_traj.s.assign(self.s.get_goal(), 'goal_x', 'goal_y')

    def descend_behavior(self):
        self.des_traj.s.dist = self.des_traj.s.location_z

    def ascend_behavior(self):
        self.des_traj.s.dist = min(self.des_traj.s.goal_z, self.des_traj.p.max_vel)

    def flight_behavior(self):
        # assign direction and distance to go
        self.des_traj.s.direction = self.des_traj.s.find_direction()
        self.des_traj.s.dist = self.des_traj.s.calc_dist_to_travel(self.des_traj.p.max_vel)
        self.des_traj.s.dist_z = 0.0

    def idle_behavior(self):
        a=1


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
