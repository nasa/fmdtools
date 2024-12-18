#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:33:54 2024

@author: smbaye and dhulse
"""

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.rand import Rand
from fmdtools.sim.sample import ParameterSample
from fireenvironment import FireEnvironment, FirePropagation, FireMapParam
from fireenvironment import sim_properties, double_size_p
from aircraft import Aircraft


class WildFireSimParameter(Parameter):
    """Parameters defining the wildfire map and response."""

    firemapparam: FireMapParam = FireMapParam()

    def from_base_loc(x, y, num_strikes):
        return WildFireSimParameter(firemapparam={**double_size_p,
                                                  'base_locations': ((x, y), ),
                                                  'num_strikes': num_strikes})


class WildfireSim(FunctionArchitecture):
    """Simulation of wildfire propagation and response."""

    __slots__ = ()
    container_p = WildFireSimParameter
    container_r = Rand
    default_sp = {'end_time': 400, "end_condition": "indicate_complete"}

    def init_architecture(self, **kwargs):

        # self.add_flow("supplies")
        self.add_flow("fireenvironment", FireEnvironment,
                      c={"p": self.p.firemapparam})

        bases = [i for i in range(len(self.p.firemapparam.base_locations))]
        for base in bases:
            self.add_fxn("aircraft_"+str(base), Aircraft, "fireenvironment",
                         p={'base': base})
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")

    def find_classification(self, scen, hist):
        return {'perc_burned': self.flows['fireenvironment'].c.calc_perc_burned(),
                'burn_pts': self.flows['fireenvironment'].c.get_all_burned()}

    def indicate_complete(self, *t):
        """Returns true when fire is contained."""
        return self.flows['fireenvironment'].c.indicate_contained()


def_p = {'firemapparam': {**double_size_p,
                          "base_locations": ((42.0, 20.0), (20.0, 20.0)),
                          "num_strikes": 6}}


def create_scen_sample(seed=10, replicates=10):
    ps = ParameterSample(seed=seed)
    ps.add_variable_replicates([], replicates=replicates, seed_comb='independent')
    return ps



if __name__ == "__main__":
    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    import fmdtools.sim.propagate as prop

    mdl = WildfireSim(p=def_p,
                      r={'seed': 100})

    mdl_graph = FunctionArchitectureGraph(mdl)
    mdl_graph.draw()

    res, hist = prop.nominal(mdl)
    hist.flows.fireenvironment.c.burning
    fig, ax = hist.plot_line('fxns.aircraft_0.s.fuel_status',
                             'fxns.aircraft_0.s.location_x',
                             'fxns.aircraft_0.s.location_y',
                             'fxns.aircraft_0.m.mode',
                             'fxns.firepropagation.s.leading_edge_length',
                             'fxns.firepropagation.s.perc_burned')


    fig, ax = mdl.flows['fireenvironment'].c.show_from(8, hist.flows.fireenvironment.c,
                                                       properties=sim_properties,
                                                       xlabel="x (km)", ylabel="y (km)")
    hist.plot_trajectories('s.location_x', 's.location_y', fig=fig, ax=ax)

    fig, ax = mdl.flows['fireenvironment'].c.show_from(45, hist.flows.fireenvironment.c,
                                                       properties=sim_properties)
    hist.plot_trajectories('s.location_x', 's.location_y', fig=fig, ax=ax)

    ani = mdl.flows['fireenvironment'].c.animate(hist.flows.fireenvironment.c,
                                                 properties=sim_properties)

    # light_mdl = WildfireSim(p=def_p) #  track=None)

    # res, hist = prop.parameter_sample(light_mdl, create_scen_sample())