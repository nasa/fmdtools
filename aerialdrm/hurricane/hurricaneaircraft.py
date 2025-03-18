# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:10:02 2025

@author: dhulse
"""

from fmdtools.define.block.function import Function
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode

from aerialdrm.base.aircraft.state import AircraftState
import numpy as np


class ForceState(State):
    weight: float = 1.0
    contact_support: float = 0.0
    lift_support: float = 1.0


class Force(Flow):
    __slots__ = ()
    container_s = ForceState


class ElectricityState(State):
    voltage_high: float = 1.0
    current_high: float = 1.0
    voltage_low: float = 1.0
    current_low: float = 1.0


class Electricity(Flow):
    __slots__ = ()
    container_s = ElectricityState


class Environment(Flow):
    __slots__ = ()
    "placeholder for environment TBD"


class AicraftControlParameter(Parameter):
    max_vel: float = 10.0


class AircraftControlState(AircraftState):
    dist: float = 0.0 # dist to travel/travelled
    dist_z: float = 0.0 # dist in y/z
    goal_z: float = 0.0
    location_z: float = 0.0

    def at_goal(self):
        return self.same(self.gett("goal_x", "goal_y", "goal_z"),
                         "location_x", "location_y", "location_z")

    def set_new_loc(self):
        dist_x, dist_y = self.direction*self.dist
        self.inc(location_x=dist_x, location_y=dist_y, location_z=self.dist_z)

class Trajectories(MultiFlow):
    __slots__ = ()
    container_s = AircraftControlState
    container_p = AicraftControlParameter


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





class ControlState(State):
    flightplan: tuple =  ((0, 0), (25, 25))
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
    opermodes = ('ascend', 'descend', 'flight', 'idle')
    fault_off = ()
    mode: str = 'idle'


class ControlFlight(Function):
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


class AviateMode(Mode):
    opermodes = ('flight', 'idle', 'falling')
    mode: str = "idle"
    fault_crash = ()


class Aviate(Function):
    __slots__ = ('trajectories', 'force', 'electricity', 'environment')

    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment
    container_m = AviateMode

    def dynamic_behavior(self, time):
        if self.electricity.s.voltage_high > 0.0:
            self.m.set_mode('flight')
        elif self.trajectories.location_z > 0.0:
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
        self.trajectories.s.assign(self.trajectories.des_traj.s, 'dist', 'dist_z', 'direction')
        self.trajectories.s.set_new_loc()
        self.electricity.s.current_high = (self.trajectories.s.dist + abs(self.trajectories.s.dist_z))/12
        self.force.s.put(lift_support=1.0)

    def falling_behavior(self):
        self.trajectories.s.dist_z =  - self.trajectories.s.z
        self.trajectories.s.dist = 0.0
        self.trajectories.s.z = 0.0
        self.force.s.put(lift_support=0.0)
        self.m.add_fault("crash")
        self.m.set_mode("idle")

    def idle_behavior(self):
        self.trajectories.s.put(dist=0.0, dist_z=0.0, direction=np.array([0,0]))
        self.force.s.put(lift_support=0.0)
        self.s.current_high = 0.0


class HoldPayload(Function):
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


class StoreEEState(State):
    charge: float = 100.0


class StoreAndSupplyElectricity(Function):
    __slots__ = ('force', 'electricity')
    flow_force = Force
    flow_electricity = Electricity

    def dynamic_behavior(self, time):
        rate_high = self.electricity.s.mul('current_high', 'voltage_high')
        rate_low = self.electricity.s.mul('current_low', 'voltage_low')
        self.s.inc(charge=rate_high+0.1*rate_low)



from fmdtools.define.architecture.function import FunctionArchitecture

class DroneArchitecture(FunctionArchitecture):
    __slots__ = ()
    def init_architecture(self, **kwargs):
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories)
        self.add_flow('environment', Environment)

        self.add_fxn('control_flight', ControlFlight,
                     'trajectories', 'force', 'electricity', 'environment')
        self.add_fxn('aviate', Aviate,
                     'trajectories', 'force', 'electricity', 'environment')
        self.add_fxn('store_and_supply_ee', StoreAndSupplyElectricity,
                     'force', 'electricity')
        self.add_fxn('perceive_environment', PerceiveEnvironment,
                     'environment', 'force', 'electricity', 'trajectories')



if __name__ == "__main__":
    t = Trajectories()
    cf = ControlFlight(trajectories=t)
    av = Aviate(trajectories=t)


    cf.dynamic_behavior(1)
    cf.des_traj.s
    av.dynamic_behavior(1)
    av.trajectories.s

    cf.dynamic_behavior(2)
    cf.des_traj.s
    av.dynamic_behavior(2)
    av.trajectories.s

    da = DroneArchitecture()