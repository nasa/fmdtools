# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:16:40 2025

@author: cwang29

File for all relevant planning logic, project overview.

Time = 0: MUST Create a flightplan. How? Current implementation: static behavior?


replan_mission(), in hurricaneaircraft.py:
    Previous rudimentary implementation:
        IF BATTERY LOW:
            If either start or end distance within some arbitrary bound, fly to it.
            Else, land in the closest suitable region. 

NEED: A* IMPLEMENTATION. 
    In hurricaneflightpath.py: perhaps battery_low_astar?

NEEDS CLEANUP: a_star, a_star_worldcoords.

        
        
        
        
"""

