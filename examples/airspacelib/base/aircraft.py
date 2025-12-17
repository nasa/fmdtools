# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:20:28 2025

@author: dhulse
"""
from fmdtools.define.block.function import Function

from examples.airspacelib.base.state import AircraftState
from examples.airspacelib.base.parameter import AircraftParameter


class BaseAircraft(Function):
    """Base Aircraft function to be used by other models."""

    __slots__ = ()
    container_s = AircraftState
    container_p = AircraftParameter

    def indicate_at_goal(self):
        """Indicate whether the aircraft is at its goal location."""
        return self.s.at_goal()

    def indicate_in_range(self):
        """Indicate whether the aircraft is in range of its goal location."""
        return self.s.in_range()

    def fly_to_goal(self):
        """Fly to a pre-determined goal."""
        if not self.indicate_at_goal():
            self.s.update_position(maxvel=self.p.max_speed)
            dist = self.s.calc_dist_to_travel()
            self.s.inc(fuel_status=-dist/self.p.max_range)


if __name__ == "__main__":
    ba = BaseAircraft()
