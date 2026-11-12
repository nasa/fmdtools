# -*- coding: utf-8 -*-
"""
Model of an aircraft's flight functionality.

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

from fmdtools_examples.airspacelib.base.plane.control import ControlsFlow
from fmdtools_examples.airspacelib.base.common.position import Trajectory3
from fmdtools_examples.airspacelib.base.plane.aero import AircraftDOFs, AeroParameter

from fmdtools_examples.airspacelib.aircraft_landing.model_environment import Environment


class Aviate(Function):
    container_p = AeroParameter
    flow_controls = ControlsFlow
    flow_aircraft_dofs = AircraftDOFs
    flow_aircraft_trajectory = Trajectory3
    flow_environment = Environment
    default_sp = {'with_loadings': True, 'end_time': 100, 'dt': 1}
    default_t = {'dt': 0.01}

    def dynamic_loading_before(self):
        if 0.0 <= self.t.time <= 30.0:
            self.controls.s.elevator = 0.029 # neutral trim
            self.controls.s.aileron = 0.000
        elif 30.0 < self.t.time < 33.0:
            self.controls.s.elevator = 0.029+0.01
            self.controls.s.aileron =  0.01
        #elif 33.0 < self.t.time < 36.0:
        #    self.controls.s.elevator = -0.01
        #    self.controls.s.aileron =  -0.02
        elif 36.0 < self.t.time < 50.0:
            self.controls.s.elevator = 0.029+0.01
            self.controls.s.aileron =  0.0
        else:
            self.controls.s.elevator = 0.029 # neutral trim
            self.controls.s.aileron = - 0.00


    def dynamic_behavior(self):
        s = self.aircraft_dofs.s
        p = self.p

        # Airspeed and angles
        V = np.sqrt(s.u**2 + s.v**2 + s.w**2)
        alpha = np.arctan2(s.w, s.u) if V > 0 else 0.0
        qbar = 0.5 * p.rho * V**2

        # Forces (Body axes: X, Y, Z)
        C_L0 = p.C_L0 + 0.6 * self.controls.s.flaps
        CD_flap = 0.5 * self.controls.s.flaps
        Fx_thrust = self.controls.s.throttle * 120000.0  # N
        Lift = qbar * p.S * (C_L0 + p.C_La * alpha)
        Drag = qbar * p.S * (0.05 + CD_flap)

        Fx = Fx_thrust - Drag * np.cos(alpha) + Lift * np.sin(alpha)
        Fy = qbar * p.S * (-0.4 * (s.v/V if V > 0 else 0))
        Fz = -Lift * np.cos(alpha) - Drag * np.sin(alpha)

        # Gravity components in body frame
        gx = -p.g * np.sin(s.theta)
        gy = p.g * np.cos(s.theta) * np.sin(s.phi)
        gz = p.g * np.cos(s.theta) * np.cos(s.phi)

        # Moments
        two_v = (2 * V if V > 0 else 1)
        beta = np.arcsin(s.v / V) if V > 0 else 0.0
        L_mom = qbar * p.S * p.b * (p.Cl_beta * beta - 0.5 * (s.p * p.b / two_v) + p.Cl_da * self.controls.s.aileron)
        N_mom = qbar * p.S * p.b * (p.Cn_beta * beta - 0.6 * (s.r * p.b / two_v) + p.Cn_da * self.controls.s.aileron)
        M_mom = qbar * p.S * p.cbar * (p.C_m0 + p.C_ma * alpha + p.C_mq * (s.q * p.cbar / two_v) + p.C_mede * self.controls.s.elevator)


        # Calculate s_dot
        # Accelerations
        brake=0.0
        if self.aircraft_trajectory.s.z < 1.0:
            brake = self.controls.s.brakes * bool(max(s.u, 0.0)) * p.mass*0.0001

        s.u_dot = (Fx / p.mass) + gx - (s.q * s.w - s.r * s.v) - brake
        s.v_dot = (Fy / p.mass) + gy - (s.r * s.u - s.p * s.w)
        s.w_dot = (Fz / p.mass) + gz - (s.p * s.v - s.q * s.u)

        s.p_dot = L_mom / p.Ixx
        s.q_dot = M_mom / p.Iyy
        s.r_dot = N_mom / p.Izz
        
        # Kinematics
        s.phi_dot = s.p + s.q * np.sin(s.phi) * np.tan(s.theta) + s.r * np.cos(s.phi) * np.tan(s.theta)
        s.theta_dot = s.q * np.cos(s.phi) - s.r * np.sin(s.phi)
        s.psi_dot = (s.q * np.sin(s.phi) + s.r * np.cos(s.phi)) / np.cos(s.theta) if np.cos(s.theta) != 0 else 0.0
        
        # Navigation (NED Frame)
        s.xe_dot = np.cos(s.theta)*np.cos(s.psi)*s.u + (np.sin(s.phi)*np.sin(s.theta)*np.cos(s.psi) - np.cos(s.phi)*np.sin(s.psi))*s.v + (np.cos(s.phi)*np.sin(s.theta)*np.cos(s.psi) + np.sin(s.phi)*np.sin(s.psi))*s.w
        s.ye_dot = np.cos(s.theta)*np.sin(s.psi)*s.u + (np.sin(s.phi)*np.sin(s.theta)*np.sin(s.psi) + np.cos(s.phi)*np.cos(s.psi))*s.v + (np.cos(s.phi)*np.sin(s.theta)*np.sin(s.psi) - np.sin(s.phi)*np.cos(s.psi))*s.w
        s.ze_dot = -np.sin(s.theta)*s.u + np.sin(s.phi)*np.cos(s.theta)*s.v + np.cos(s.phi)*np.cos(s.theta)*s.w
        if self.aircraft_trajectory.s.z < 1.0:
            s.ze_dot = 0.0

        s.inc_state(self.t.dt)



if __name__ == "__main__":

    ff = Aviate(track='all')
    ff(time='end')
    ff.h.plot_trajectories("aircraft_dofs.s.xe", "aircraft_dofs.s.ye", "aircraft_dofs.s.ze")
    ff.h.plot_trajectories("aircraft_dofs.s.xe", "aircraft_dofs.s.ye")
    ff.h.plot_line("aircraft_dofs.s.ze")
    ff.h.plot_line("aircraft_dofs.s.phi")