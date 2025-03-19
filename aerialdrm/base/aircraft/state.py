# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:04:43 2025

@author: dhulse
"""
import numpy as np
from fmdtools.define.container.state import State


class AircraftPosition(State):
    x: float = 0.0
    goal_x: float = 10.0
    dx: float = 0.0
    y: float = 0.0
    goal_y: float = 10.0
    dy: float = 0.0

    def find_direction(self):
        dist = self.calc_dist()
        if dist > 0.0:
            return self.calc_vector_dist()/self.calc_dist()
        else:
            return np.array([0.0, 0.0])

    def calc_vector_dist(self):
        return np.array([self.goal_x-self.x, self.goal_y-self.y])

    def calc_dist(self):
        vector_dist = self.calc_vector_dist()
        return np.sqrt(vector_dist[0]**2+vector_dist[1]**2)

    def at_goal(self):
        return self.same(self.get("goal_x", "goal_y"), "x", "y")

    def in_range(self, dist_range=10.0):
        return self.calc_dist() <= dist_range

    def calc_dist_to_travel(self, dist_range=10.0):
        return np.min([dist_range, self.calc_dist()])

    def update_dist_to_travel(self, maxvel=10.0):
        if self.in_range():
            vel = self.calc_dist()
        else:
            vel = maxvel
        dx, dy = vel * self.find_direction()
        self.put(dx=dx, dy=dy)

    def update_position(self, maxvel=10.0):
        self.update_dist_to_travel(maxvel=maxvel)
        self.increment_position()

    def increment_position(self):
        self.inc(x=self.dx, y=self.dy)


class AircraftPosition3(AircraftPosition):
    """State of Trajectories flow."""

    z: float = 0.0
    goal_z: float = 0.0
    dz: float = 0.0  # dist in y/z

    def at_goal(self):
        """Determine whether the aircraft is at its goal location."""
        return self.same(self.get("goal_x", "goal_y", "goal_z"),
                         "x", "y", "z")

    def update_position(self, maxvel=10.0, max_zvel=0.0):
        """Update the (3d) aircraft state."""
        zdist = self.goal_z - self.z
        if abs(max_zvel) > abs(zdist):
            zvel = zdist
        else:
            zvel = np.sign(zdist)*max_zvel
        self.put(dz=zvel)
        super().update_position(maxvel)

    def increment_position(self):
        self.inc(x=self.dx, y=self.dy, z=self.dz)


class AircraftState(AircraftPosition):
    fuel_status: float = 100  # starting fuel at 100%




if __name__ == "__main__":
    s = AircraftState()
    s.find_direction()