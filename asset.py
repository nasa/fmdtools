# -*- coding: utf-8 -*-
"""
Module for assets (e.g. aircraft, etc.)
"""

from fmdtools.define.block import FxnBlock 
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State

from common import Location, Message

class AssetParameter(Parameter):
    speed:  float = 10.0

class AssetState(State):
    dx:     float=0.0
    dy:     float=0.0
    dz:     float=0.0


class Asset(FxnBlock):
    _init_location = Location
    _init_message = Message 
    _init_p = AssetParameter
    _init_s = AssetState
    def dynamic_behavior(self, time):
        dist = self.p.speed*self.t.dt
        self.location.s.inc(x=dist, y=dist)
        
if __name__=="__main__":
    a = Asset()
    
