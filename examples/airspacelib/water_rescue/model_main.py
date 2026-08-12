"""
Top-level model for the water-rescue simulation.

Wires together the shared beach environment with the three functions:
the lifeguard responder, the swimmer behavior, and the patrol drone.
"""
from model_environment import BeachEnvironment, BeachBehavior
from model_responder import Responder
from drone import BeachAircraft
from fmdtools.define.architecture.function import FunctionArchitecture


class Beach(FunctionArchitecture):
    """Function architecture connecting all functions through the environment flow."""

    def init_architecture(self, **kwargs):
        self.add_flow("environment", BeachEnvironment)
        self.add_fxn("responder", Responder, "environment")
        self.add_fxn("swimmers", BeachBehavior, "environment")
        self.add_fxn("beach_aircraft", BeachAircraft, "environment")
