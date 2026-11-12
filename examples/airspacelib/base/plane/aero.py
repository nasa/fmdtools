# -*- coding: utf-8 -*-
"""
Common aircraft aerodynamic state/parameter constructs.

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

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State

class AeroParameter(Parameter):
    # Aircraft physical constants (generic small commercial aviation example)
    mass: float = 65000.0         # kg (varies; 737 typical)
    g: float = 9.81    # m/s^2
    Ixx: float = 500000.0         # kg*m^2 (approximate)
    Iyy: float = 1.0e6            # kg*m^2
    Izz: float = 1.4e6            # kg*m^2
    S: float = 124.6              # m^2
    b: float = 35.8               # m
    cbar: float = 3.45            # m
    rho: float = 1.225            # kg/m^3 (sea level)
    C_L0: float = 0.4
    C_La: float = 5.7             # per radian
    C_m0: float = 0.04
    C_ma: float = -1.0            # per radian
    C_mq: float = -100.0          # very strong pitch damping
    C_mede: float = -0.8          # less effective elevator
    Cl_da: float = 0.04           # roll from aileron (much less sensitive)
    Cn_da: float = -0.005         # yaw from aileron
    Cl_beta: float = -0.12
    Cn_beta: float = 0.065


class AeroState(State):
    u: float = 70.0 #x - airspeed m/s
    u_dot: float = 0.0
    v: float = 0.0 # y - left/right
    v_dot: float = 0.0
    w: float = 0.0 # z - up/down
    w_dot: float = 0.0
    p: float = 0.0 # roll
    p_dot: float = 0.0
    q: float = 0.0 # pitch
    q_dot: float = 0.0
    r: float = 0.0 # yaw
    r_dot: float = 0.0
    phi: float = 0.0 # roll angle
    phi_dot: float = 0.0
    theta: float = 0.0 # pitch
    theta_dot: float = 0.0
    psi: float = 0.0 # yaw
    psi_dot: float = 0.0
    xe: float = 0.0
    xe_dot: float = 0.0
    ye: float = 0.0
    ye_dot: float = 0.0
    ze: float = 0.0
    ze_dot: float = 0.0

    def state_fields(self):
        return [s for s in self.__fields__ if "_dot" not in s]

    def dot_fields(self):
        return [s for s in self.__fields__ if "_dot" in s]

    def inc_state(self, dt):
        state_dot = {k: self[k+'_dot']*dt for k in self.state_fields()}
        self.inc(**state_dot)


class AircraftDOFs(Flow):
    container_s = AeroState