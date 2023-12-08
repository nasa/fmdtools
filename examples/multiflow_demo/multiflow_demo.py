# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:44:15 2023

@author: dhulse
"""
from fmdtools.define.container.state import State
class LocationState(State):
    x: float=0.0
    y: float=0.0

from fmdtools.define.flow import CommsFlow, MultiFlow
class Communications(CommsFlow):
    container_s = LocationState
class Location(MultiFlow):
    container_s = LocationState

from fmdtools.define.container.parameter import Parameter
class MoveParam(Parameter):
    x_up: float=0.0
    y_up: float=0.0

from fmdtools.define.block.function import Function
class Mover(Function):
    container_p = MoveParam
    _init_communications = Communications
    _init_location = Location 
    def __init__(self, name='mover', flows={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
        self.internal_info = self.communications.create_comms(name)
        self.loc = self.location.create_local(name)
    def dynamic_behavior(self, time):
        #move
        self.loc.s.inc(x=self.p.x_up, y=self.p.y_up)
        # the inbox should be cleared each timestep to allow new messages
        self.internal_info.clear_inbox()
    def behavior(self, time):
        #recieve messages
        self.internal_info.receive()
        #communicate
        if self.p.x_up!=0.0:  
            self.internal_info.s.x=self.loc.s.x
            self.internal_info.send("all", "local", "x")
        elif self.p.y_up!=0.0:   
            self.internal_info.s.y=self.loc.s.y
            self.internal_info.send("all", "local", "y")   
    def find_classification(self, scen, fxnhist):
        return {"last_x": self.loc.s.x, "min_x": fxnhist.faulty.location.get(self.name).x}

class Coordinator(Function):
    _init_communications = Communications
    def __init__(self, name='coordinator', flows={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
        self.coord_view= self.communications.create_comms(name, ports=["mover_1", "mover_2"])
    def dynamic_behavior(self, time):
        self.coord_view.clear_inbox()
    def behavior(self, time):
        self.coord_view.receive()
        self.coord_view.update("local", "mover_1", "y")
        self.coord_view.update("local", "mover_2", "x")

from fmdtools.define.architecture.function import FunctionArchitecture
class TestModel(FunctionArchitecture):
    default_sp = dict(times=(0,10))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_flow("communications", Communications)
        self.add_flow("location",       Location)
        self.add_fxn("mover_1",     Mover, "communications", "location", p = {"x_up":1.0})
        self.add_fxn("mover_2",     Mover, "communications", "location", p = {"y_up":1.0})

        self.add_fxn("coordinator", Coordinator, "communications")

        self.build()

if __name__=='__main__':
    mdl = TestModel()
    mdl.flows["communications"].mover_1.s.x=25
    mdl.flows["communications"].mover_1.send("mover_2")
    from fmdtools.sim import propagate
    
    endres, mdlhist = propagate.nominal(mdl, desired_result='graph.flows.communications')
    
    endres.graph.flows.communications.draw()