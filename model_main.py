from model_environment import BeachEnvironment
from model_responder import Responder
from drone import BeachAircraft

from fmdtools.define.architecture.function import FunctionArchitecture
import fmdtools.sim.propagate as prop
from model_environment import BeachBehavior

import inspect

# creating function architecture for beach
class Beach(FunctionArchitecture):
    def init_architecture(self, **kwargs):
        self.add_flow("environment", BeachEnvironment)
        #self.add_fxn("responder", Responder, "environment")
        self.add_fxn("swimmers", BeachBehavior, "environment")
        self.add_fxn("beach_aircraft", BeachAircraft, "environment")



