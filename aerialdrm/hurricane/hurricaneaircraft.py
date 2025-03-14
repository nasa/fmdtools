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

from aerialdrm.base.aircraft.state import AircraftState


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


class PerceiveEnvironment(Function):
    """Placeholder for environment perception"""
    __slots__ = ('environment', 'force', 'electricity')
    flow_environment = Environment
    flow_force = Force
    flow_electricity = Electricity


class AircraftControlState(AircraftState):
    dist: float = 0.0 # dist to travel/travelled

class AicraftControlParameter(Parameter):
    max_vel: float = 10.0


class Trajectories(MultiFlow):
    __slots__ = ()
    container_s = AircraftControlState
    container_p = AicraftControlParameter


class ControlFlight(Function):
    __slots__ = ('trajectories', 'des_traj', 'force', 'electricity', 'environment')

    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment

    def init_block(self, **kwargs):
        self.des_traj = self.trajectories.create_local("des_traj")

    def dynamic_behavior(self, time):
        # perceive current location, goal, etc
        self.des_traj.s.assign(self.trajectories.s, 'location_x', 'location_y')
        # assign direction and distance to go
        self.des_traj.s.direction = self.des_traj.s.find_direction()
        self.des_traj.s.dist = self.des_traj.s.calc_dist_to_travel(self.des_traj.p.max_vel)


class Aviate(Function):
    __slots__ = ('trajectories', 'force', 'electricity', 'environment')

    flow_trajectories = Trajectories
    flow_force = Force
    flow_electricity = Electricity
    flow_environment = Environment

    def dynamic_behavior(self, time):
        self.trajectories.s.assign(self.trajectories.des_traj.s, 'dist', 'direction')
        newloc = self.trajectories.s.set_new_loc()


class HoldPayload(Function):
    __slots__ = ('force', 'trajectories')
    flow_force = Force
    flow_trajectories = Trajectories

    def static_behavior(self, time):
        self.force.s.put(wight=1.0, lift_support=1.0)


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
                     'environment', 'force', 'electricity')



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