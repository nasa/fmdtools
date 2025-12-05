# -*- coding: utf-8 -*-
"""Flows used in the model."""

from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow
from fmdtools.define.environment import Environment
from fmdtools.define.container.parameter import Parameter


from dronelib.base.state import AircraftPosition3


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

    charge: float = 100.0
    voltage_high: float = 1.0
    current_high: float = 1.0
    power_high: bool = False
    voltage_low: float = 1.0
    current_low: float = 1.0
    power_low: bool = True



class Electricity(Flow):
    """Flow of electricity through the aircraft functions."""

    __slots__ = ()
    container_s = ElectricityState


class AircraftEnvironment(Environment):
    "Placeholder for environment flow TBD."


class AicraftControlParameter(Parameter):
    """Parameter determining aircraft control."""

    max_vel: float = 10.0


class Trajectories(MultiFlow):
    """Degrees of freedom of the aircraft."""

    __slots__ = ()
    container_s = AircraftPosition3
    container_p = AicraftControlParameter


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

