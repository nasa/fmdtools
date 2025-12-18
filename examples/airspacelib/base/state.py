# -*- coding: utf-8 -*-
"""
Base Aircraft States.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
import numpy as np
from fmdtools.define.container.state import State


class AircraftPosition(State):
    """
    State of an aircraft's position (modelled in 2d).

    Parameters
    ----------
    x/y : float
        Position in the x/y dimension
    goal_x/goal_y : float
        Desired position in the x/y dimension
    dx/dy : float
        Velocity in the x/y dimension.
    """

    x: float = 0.0
    goal_x: float = 10.0
    dx: float = 0.0
    y: float = 0.0
    goal_y: float = 10.0
    dy: float = 0.0
    dims = ('x', 'y')

    def get_goal(self):
        """Get the goal attributes (goal_x, goal_y) of the AircraftPosition."""
        return self.get(*['goal_'+d for d in self.dims])

    def get_loc(self):
        """Get the location (x, y) attributes of the AircraftPosition."""
        return self.get(*self.dims)

    def get_vel(self):
        """Get the velocity attributes (dx, dy) of the AircraftPosition."""
        return self.get(*['d'+d for d in self.dims])

    def find_direction(self):
        """
        Find the (unit vector) direction from the x,y to goal_x, goal_y.

        Returns
        -------
        direction : array
            unit vector direction to goal_x, goaly.

        Examples
        --------
        >>> ap = AircraftPosition()
        >>> ap.find_direction()
        array([0.70710678, 0.70710678])
        >>> ap.goal_x = 0.0
        >>> ap.find_direction()
        array([0., 1.])
        """
        dist = self.calc_dist()
        if dist > 0.0:
            return self.calc_vector_dist()/self.calc_dist()
        else:
            return np.zeros(len(self.dims))

    def calc_vector_dist(self):
        """Calculate the vector distance from x,y to goal_x, goal_y."""
        return self.get_goal() - self.get_loc()

    def calc_dist(self):
        """Calculate the scalar distance from x,y to goal_x, goal_y."""
        vector_dist = self.calc_vector_dist()
        return np.sqrt(sum(vector_dist**2))

    def at_goal(self):
        """
        Determine if the aircraft is at its goal location.

        Examples
        --------
        >>> ap = AircraftPosition()
        >>> ap.at_goal()
        False
        >>> ap.assign(ap.get_goal(), 'x', 'y')
        >>> ap.at_goal()
        True
        """
        return all(self.get_goal() == self.get_loc())

    def in_range(self, dist_range=10.0):
        """Determine if the aircraft is in the range of its goal location."""
        return self.calc_dist() <= dist_range

    def calc_dist_to_travel(self, dist_range=10.0):
        """Determine the distance to the goal location (under max dist dist_range)."""
        return np.min([dist_range, self.calc_dist()])

    def update_dist_to_travel(self, maxvel=10.0):
        """
        Update dx, dy to reflect goal location (travelling at max velocity).

        Parameters
        ----------
        maxvel : float, optional
            Maximum velocity to travel at to goal_x, goal_y. The default is 10.0.

        Examples
        --------
        >>> ap = AircraftPosition(goal_x=10.0, goal_y=0.0)
        >>> ap.update_dist_to_travel(maxvel=8.0)
        >>> ap
        AircraftPosition(x=0.0, goal_x=10.0, dx=8.0, y=0.0, goal_y=0.0, dy=0.0)
        >>> ap.update_dist_to_travel(maxvel=15.0)
        >>> ap
        AircraftPosition(x=0.0, goal_x=10.0, dx=10.0, y=0.0, goal_y=0.0, dy=0.0)
        """
        if self.in_range(dist_range=maxvel):
            vel = self.calc_dist()
        else:
            vel = maxvel
        vels = vel * self.find_direction()
        self.put(**{'d'+dim: vels[i] for i, dim in enumerate(self.dims)})

    def update_position(self, maxvel=10.0):
        """
        Update x, y to reflect progress towards the goal location at maxvel.

        Parameters
        ----------
        maxvel : float, optional
            Maximum velocity to travel at to goal_x, goal_y. The default is 10.0.

        Examples
        --------
        >>> ap = AircraftPosition()
        >>> ap.update_position(maxvel=10.0)
        >>> ap
        AircraftPosition(x=7.071067811865475, goal_x=10.0, dx=7.071067811865475, y=7.071067811865475, goal_y=10.0, dy=7.071067811865475)
        >>> ap.update_position(maxvel=10.0)
        >>> ap
        AircraftPosition(x=10.0, goal_x=10.0, dx=2.9289321881345254, y=10.0, goal_y=10.0, dy=2.9289321881345254)
        >>> ap.update_position()
        >>> ap
        AircraftPosition(x=10.0, goal_x=10.0, dx=0.0, y=10.0, goal_y=10.0, dy=0.0)
        """
        self.update_dist_to_travel(maxvel=maxvel)
        self.increment_position()

    def increment_position(self):
        """Increment position (x,y) by (dx, dy)."""
        self.inc(**{dim: getattr(self, 'd'+dim) for dim in self.dims})


class AircraftPosition3(AircraftPosition):
    """3d State of Trajectories flow."""

    z: float = 0.0
    goal_z: float = 0.0
    dz: float = 0.0  # dist in y/z
    dims = ('x', 'y', 'z')


class AircraftState(AircraftPosition):
    fuel_status: float = 100  # starting fuel at 100%


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    ap = AircraftPosition()

    s = AircraftState()
    s.find_direction()
