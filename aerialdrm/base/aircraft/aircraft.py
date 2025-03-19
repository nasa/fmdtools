# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:20:28 2025

@author: dhulse
"""
from fmdtools.define.block.function import Function

from aerialdrm.base.aircraft.state import AircraftState
from aerialdrm.base.aircraft.parameter import AircraftParameter


class BaseAircraft(Function):
    __slots__=()
    container_s = AircraftState
    container_p = AircraftParameter

    def indicate_at_goal(self):
        return self.s.at_goal()

    def indicate_in_range(self):
        return self.s.in_range()

    def fly_to_goal(self):
        if not self.indicate_at_goal():
            dist = self.s.calc_dist()
            if self.s.in_range():
                # if in range, clip to goal location
                self.s.x = self.s.goal_x
                self.s.y = self.s.goal_y
                self.s.inc(fuel_status=-dist/self.p.max_range)
            else:
                # otherwise, move in a straight line
                self.s.update_position(vel=self.p.max_speed)
                self.s.inc(fuel_status=-self.p.max_speed/self.p.max_range)
                direction = self.s.find_direction()


if __name__ == "__main__":
    ba = BaseAircraft()