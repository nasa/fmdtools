# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:10:02 2025

@author: dhulse
"""

from aerialdrm.base.aircraft.aircraftarchitecture import Trajectories, ControlFlight, Aviate, AircraftArchitecture


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

    da = AircraftArchitecture()