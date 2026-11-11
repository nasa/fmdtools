# -*- coding: utf-8 -*-
"""
Model of the aircraft control operations.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np


from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.mode import Mode

from model_common import ControlsFlow, AircraftDOFs, AircraftTrajectory
from model_environment import AirportEnvironment


class ControlParameter(Parameter):
    x: float = 1865527.46
    y: float = 596499.47
    z: float = 1000.0*0.3048
    goal_x: float= 1863436.83
    goal_y: float = 601624.24
    goal_z: float = 0.0


class ControlMode(Mode):
    opermodes = ('decend', 'land', 'taxi', 'abort')
    fault_false_abort = ()
    mode: str = 'descend'


class ControlFlight(Function):
    container_p = ControlParameter
    container_m = ControlMode
    flow_controls = ControlsFlow
    flow_aircraft_dofs = AircraftDOFs
    flow_aircraft_trajectory = AircraftTrajectory
    flow_environment = AirportEnvironment
    default_t = {'local_dt': 0.01}

    def init_block(self, **kwargs):
        self.aircraft_trajectory.s.put(**self.p)

    def dynamic_loading_before(self):
        self.aircraft_dofs.s.inc(xe=-10, ye=10)

    def convert_dofs_to_traj(self):
        """Convert local NED frame of reference to global frame."""
        init_angle = np.atan((self.p.goal_x-self.p.x)/(self.p.goal_y-self.p.y)) 
        xe, ye, ze = self.aircraft_dofs.s.get("xe", "ye", "ze")
        x = ye*np.cos(init_angle) + xe*np.sin(init_angle) + self.p.x
        y = - ye*np.sin(init_angle) + xe*np.cos(init_angle) + self.p.y
        z = -ze + self.p.z
        self.aircraft_trajectory.s.put(x=x, y=y, z=z)
        self.aircraft_trajectory.s.update_dist_to_travel(maxvel=self.aircraft_dofs.s.u)

    def dynamic_behavior(self):
        # convert NED to x=E, y=N, z=Up
        self.convert_dofs_to_traj()
        if self.m.has_fault("false_abort"):
            self.m.mode="abort"
        if self.m.mode == 'descend':
            self.descend_to_runway()
        elif self.m.mode == 'land':
            self.land()
        elif self.m.mode == 'taxi':
            pass
        elif self.m.mode == 'abort':
            self.abort_descent()

    def descend_to_runway(self):
        """Controls descent profile toward runway threshold (goal_x, goal_y)."""
        controls = self.controls.s
        # Compute horizontal distance to runway threshold
        self.aircraft_trajectory.s.distance = self.aircraft_trajectory.s.calc_dist("x", "y")

        # Desired descent rate: proportional to height, capped
        # 1000 ft up 3 miles out, 300-500 feet 1 mile out
        if self.aircraft_trajectory.s.distance > 1609*3:
            self.aircraft_trajectory.s.goal_z = 305.0
        elif self.aircraft_trajectory.s.distance > 1609:
            self.aircraft_trajectory.s.goal_z = 122.0
        elif self.aircraft_trajectory.s.distance > 100:
            self.aircraft_trajectory.s.goal_z = 10.0
        else:
            self.aircraft_trajectory.s.goal_z = 0.0
        self.aircraft_trajectory.s.update_dist_to_travel(maxvel=100.0)

        if self.aircraft_trajectory.s.distance > 50:
            if self.aircraft_trajectory.s.dz < - 5.0:
                controls.set_descent(angle=self.aircraft_dofs.s.theta, throttle=0.3)
            elif self.aircraft_trajectory.s.dz > -2.0:
                controls.set_ascent(angle=self.aircraft_dofs.s.theta, throttle=0.3)
            else:
                controls.set_steady_flight(angle=self.aircraft_dofs.s.theta, throttle=0.3)
            # Optionally steer toward runway centerline
            # Heading control omitted for brevity
        else:
            # If near ground, transition to 'land'
            self.m.mode = 'land'

    def land(self):
        """Set controls for touchdown and rollout."""
        controls = self.controls.s
        #controls.set_descent(angle=self.aircraft_dofs.s.theta, throttle=0.3)
        controls.throttle = 0.0  # Idle throttle
        controls.elevator = 0.0004  # Nose-up for gentle touchdown
        controls.flaps = 1.0
        controls.brakes = 1.0
        # Optionally, add braking logic or switch to 'taxi' after stop

    def abort_descent(self):
        controls = self.controls.s
        if self.aircraft_trajectory.s.dz < 100.0:
            controls.set_ascent(angle=self.aircraft_dofs.s.theta, throttle=0.8)
        else:
            controls.set_steady_flight(angle=self.aircraft_dofs.s.theta, throttle=0.5)


if __name__ == "__main__":
    cf = ControlFlight(track='all', sp={'with_loadings': True})
    cf.h.plot_line('aircraft_trajectory.s.x', 'aircraft_trajectory.s.y')
    cf(time='end')
    fig, ax = cf.environment.ga.show_infrastructure()
    cf.h.plot_trajectories('aircraft_trajectory.s.x', 'aircraft_trajectory.s.y', fig=fig, ax=ax)