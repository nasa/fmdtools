# -*- coding: utf-8 -*-
"""Flows used in the model."""

from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.container.parameter import Parameter


from aerialdrm.base.aircraft.state import AircraftState


class ForceState(State):
    """State of Force Flow."""

    weight: float = 1.0
    contact_support: float = 0.0
    lift_support: float = 1.0


class Force(Flow):
    """Flow of force through the aircraft functions."""

    __slots__ = ()
    container_s = ForceState


class ElectricityState(State):
    """State of electricity - assumes high and low volate lines."""

    voltage_high: float = 1.0
    current_high: float = 1.0
    voltage_low: float = 1.0
    current_low: float = 1.0


class Electricity(Flow):
    """Flow of electricity through the aircraft functions."""

    __slots__ = ()
    container_s = ElectricityState


class Environment(Flow):
    "Placeholder for environment flow TBD."

    __slots__ = ()


class AicraftControlParameter(Parameter):
    """Parameter determining aircraft control."""

    max_vel: float = 10.0


class AircraftControlState(AircraftState):
    """State of Trajectories flow."""

    dist: float = 0.0  # dist to travel/travelled
    dist_z: float = 0.0  # dist in y/z
    goal_z: float = 0.0
    location_z: float = 0.0

    def at_goal(self):
        """Determine whether the aircraft is at its goal location."""
        return self.same(self.gett("goal_x", "goal_y", "goal_z"),
                         "location_x", "location_y", "location_z")

    def set_new_loc(self):
        """Set the aircraft location given its direction and distance."""
        dist_x, dist_y = self.direction*self.dist
        self.inc(location_x=dist_x, location_y=dist_y, location_z=self.dist_z)


class Trajectories(MultiFlow):
    """Degrees of freedom of the aircraft."""

    __slots__ = ()
    container_s = AircraftControlState
    container_p = AicraftControlParameter


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

