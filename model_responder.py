from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.container.parameter import Parameter
from model_environment import BeachEnvironment
import numpy as np


class ResponderState(State):
    location: tuple = (12,12)
    on_rescue: bool = False

class ResponderMode(Mode):
    opermodes = ("standby", "rescue")
    mode: str = "standby"


class Responder(Function):
    container_s = ResponderState
    container_m = ResponderMode
    flow_environment = BeachEnvironment
    def dynamic_behavior(self):
        t=10
        if(t == 10 and not self.m.mode == "rescue"):
            self.m.mode = "rescue"
            return self.m.mode
        if(self.environment.c.check_vicinity(self.s.location) == (0,0)):
            self.m.mode = "standby"
            return self.m.mode

    

responder = Responder()
print(responder.dynamic_behavior())