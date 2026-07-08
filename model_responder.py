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
    path_index: int = 0
    path: list = []

class ResponderMode(Mode):
    opermodes = ("standby", "rescue")
    mode: str = "standby"


class Responder(Function):
    container_s = ResponderState
    container_m = ResponderMode
    flow_environment = BeachEnvironment

    def create_rescue_path(self, speed):
        victim_loc = getattr(self.flow_environment.c, "person_location", self.environment.c.p.rescue_locations[0])
        dx = victim_loc[0] - self.s.location[0]
        dy = victim_loc[1] - self.s.location[1]
        dist = np.sqrt(dx**2 + dy**2)

        steps = int(dist / speed)    # number of 1-second steps at "speed" m/s
        self.s.path_index = steps

        # unit direction vector
        ux = dx / dist
        uy = dy / dist

        path = []
        for i in range(steps):
            point = (self.s.location[0] + ux * speed * i,
                    self.s.location[1] + uy * speed * i)
            path.append(point)

        path.append(victim_loc)   # end at victim

        return path #return list of points of path

    def set_rescue_goal(self):
        """Determine path to follow to rescue person."""
        self.m.set_mode("rescue")
        self.s.path = self.create_rescue_path()


    def follow_rescue_path(self):
        if self.s.path_index < len(self.s.path):
            self.s.location = self.s.path[self.s.path_index]
            self.s.path_index += 1
        else:
            self.m.set_mode("standby")  # arrived, go back to standby
        

    def dynamic_behavior(self):
        if self.m.in_mode("standby"):
            if self.environment.c.s.rescue:
                self.set_rescue_goal()
        if self.m.in_mode("rescue"):
            self.follow_rescue_path()
    

responder = Responder()
print(responder.create_rescue_path(4))