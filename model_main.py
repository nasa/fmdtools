from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.container.parameter import Parameter

class DroneState(State):
    location: tuple = (0,0)

class Drone(Function):
    container_s = DroneState
    def dynamic_behavior():
        print('hi')