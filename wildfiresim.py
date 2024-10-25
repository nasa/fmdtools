#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:33:54 2024

@author: smbaye and dhulse
"""

from fmdtools.define.architecture.function import FunctionArchitecture
from fireenvironment import FireEnvironment
from fireenvironment import FirePropagation
from aircraft import Aircraft


class WildfireSim(FunctionArchitecture):
    __slots__=()
    default_sp={}
    """
    flows: environment, supplies
    functions: fire_propagation, aircraft, bases
    """
    def init_architecture(self, **kwargs):

        # self.add_flow("supplies")
        self.add_flow("fireenvironment", FireEnvironment)

        # self.add_fxn("fire_propagation", GenericFxn, "environment", "supplies")
        # self.add_fxn("aircraft", GenericFxn, "environment", "supplies")
        # self.add_fxn("bases", GenericFxn, "supplies", "environment")
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")
        self.add_fxn("aircraft", Aircraft, "fireenvironment")



if __name__ == "__main__":
    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    import fmdtools.sim.propagate as prop
    mdl = WildfireSim()

    mdl_graph = FunctionArchitectureGraph(mdl)
    mdl_graph.draw()

    res, hist = prop.nominal(mdl, protect=False)
    hist.flows.fireenvironment.c.burned
