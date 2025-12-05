# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:08:38 2025

@author: dhulse
"""

from fmdtools.define.container.parameter import Parameter


class AircraftParameter(Parameter, readonly=True):
    number: int = 1
    max_range: float = 1287.0  # in km
    max_speed: float = 6.6  # in km/min
    base: int = 0
    resupply_time: float = 10.0  # 10 minute resupply time


if __name__ == "__main__":

    p = AircraftParameter()