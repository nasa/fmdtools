#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:33:54 2024

@author: smbaye and dhulse
"""

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.parameter import Parameter
from fireenvironment import FireEnvironment, FirePropagation, FireMapParam
from aircraft import Aircraft


class WildFireSimParameter(Parameter):
    firemapparam: FireMapParam = FireMapParam()


class WildfireSim(FunctionArchitecture):
    __slots__ = ()
    container_p = WildFireSimParameter
    default_sp = {'end_time': 200}
    """
    flows: environment, supplies
    functions: fire_propagation, aircraft, bases
    """
    def init_architecture(self, **kwargs):

        # self.add_flow("supplies")
        self.add_flow("fireenvironment", FireEnvironment,
                      c={"p": self.p.firemapparam})

        # self.add_fxn("fire_propagation", GenericFxn, "environment", "supplies")
        # self.add_fxn("aircraft", GenericFxn, "environment", "supplies")
        # self.add_fxn("bases", GenericFxn, "supplies", "environment")
        bases = [i for i in range(len(self.p.firemapparam.base_locations))]
        for base in bases:
            self.add_fxn("aircraft_"+str(base), Aircraft, "fireenvironment",
                         p={'base': base})
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")



if __name__ == "__main__":
    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    import fmdtools.sim.propagate as prop
    mdl = WildfireSim(p = {'firemapparam': {'num_strikes': 4,
                                            'base_locations': ((10,10), (40,40))}})

    mdl_graph = FunctionArchitectureGraph(mdl)
    mdl_graph.draw()

    res, hist = prop.nominal(mdl, protect=False)
    hist.flows.fireenvironment.c.burning
    fig, ax = hist.plot_line('fxns.aircraft_0.s.fuel_status',
                             'fxns.aircraft_0.s.location_x',
                             'fxns.aircraft_0.s.location_y',
                             'fxns.aircraft_0.m.mode')

    properties={'burning': {"color": "red", "as_bool": True},
                "base": {"color": "grey"},
                "to_burn": {"color": "yellow", "as_bool": True, "alpha": 0.5},
                "extinguished": {"color": "blue", "alpha": 0.5}}

    fig, ax = mdl.flows['fireenvironment'].c.show_from(8, hist.flows.fireenvironment.c,
                                                    properties=properties)
    hist.plot_trajectory('s.location_x', 's.location_y', fig=fig, ax=ax, )

    fig, ax = mdl.flows['fireenvironment'].c.show_from(45, hist.flows.fireenvironment.c,
                                                    properties=properties)
    hist.plot_trajectory('s.location_x', 's.location_y', fig=fig, ax=ax)

    ani = mdl.flows['fireenvironment'].c.animate(hist.flows.fireenvironment.c,
                                                 properties=properties)
