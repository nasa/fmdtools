# -*- coding: utf-8 -*-
"""
Location for common attributes (States, Flows, etc.)
"""

from fmdtools.define.state import State
from fmdtools.define.flow import MultiFlow, CommsFlow

class LocationState(State):
    x: float=0.0
    y: float=0.0
    z: float=0.0

class Location(MultiFlow):
    _init_s = LocationState

class MessageState(State):
    status:     str="continue"
    job:        str="standby"
    request:    str="none"

class Message(CommsFlow):
    _init_s = MessageState
    