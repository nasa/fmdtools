# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:04:43 2025

@author: dhulse
"""
import numpy as np
from fmdtools.define.container.state import State


class AircraftState(State):
    fuel_status: float = 100  # starting fuel at 100%
    goal_x: float = 10.0
    goal_y: float = 10.0
    location_x: float = 0.0
    location_y: float = 0.0
    direction: np.array = np.array([0, 0])

    def find_direction(self):
        return self.calc_vector_dist()/self.calc_dist()

    def calc_vector_dist(self):
        return np.array([self.goal_x-self.location_x, self.goal_y-self.location_y])

    def calc_dist(self):
        vector_dist = self.calc_vector_dist()
        return np.sqrt(vector_dist[0]**2+vector_dist[1]**2)

    def at_goal(self):
        return self.same(self.gett("goal_x", "goal_y"), "location_x", "location_y")

    def in_range(self, dist_range=10.0):
        return (abs(self.goal_x - self.location_x) <= dist_range and
                abs(self.goal_y - self.location_y) <= dist_range)

    def calc_dist_to_travel(self, dist_range=10.0):
        return np.min([dist_range, self.calc_dist()])

    def set_new_loc(self):
        dist_x, dist_y = self.direction*self.dist
        return self.inc(location_x=dist_x, location_y=dist_y)


if __name__ == "__main__":
    s = AircraftState()
    s.find_direction()