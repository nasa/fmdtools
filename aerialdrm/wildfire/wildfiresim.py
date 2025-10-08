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
from fmdtools.sim.search import ParameterSimProblem
from fmdtools.sim.sample import ParameterDomain

from aerialdrm.wildfire.fireenvironment import FireEnvironment, FirePropagation, FireMapParam
from aerialdrm.wildfire.fireenvironment import sim_properties, double_size_p
from aerialdrm.wildfire.fireaircraft import FireAircraft
import numpy as np


class WildFireSimParameter(Parameter):
    """Parameters defining the wildfire map and response."""

    firemapparam: FireMapParam = FireMapParam()

    @classmethod
    def from_base_loc(cls, x, y, p=double_size_p):
        fmp = {**p.get('firemapparam', {}), 'base_locations': ((x, y), )}
        return WildFireSimParameter(firemapparam=fmp)


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
            self.add_fxn("aircraft_"+str(base), FireAircraft, "fireenvironment",
                         p={'base': base})
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")

    def classify(self, **kwargs):
        return {'perc_burned': self.flows['fireenvironment'].c.calc_perc_burned(),
                'burn_pts': self.flows['fireenvironment'].c.get_all_burned()}

    def indicate_complete(self):
        """Returns true when fire is contained."""
        return self.flows['fireenvironment'].c.indicate_contained()


def_p = {'firemapparam': {**double_size_p,
                          "base_locations": ((42.0, 20.0), (20.0, 20.0)),
                          "num_strikes": 6}}


def create_scen_sample(seed=10, replicates=10):
    ps = ParameterSample(seed=seed)
    ps.add_variable_replicates([], replicates=replicates, seed_comb='independent')
    return ps


class BasePlacementProblem(ParameterSimProblem):
    """Optimization problem for picking best base location(s)."""

    def init_problem(self, p=def_p, track=None, seed=10, replicates=10,
                     **kwargs):
        """
        Initializes base optimization problem.

        Parameters
        ----------
        p : dict, optional
            Non-default model parameters. The default is def_p.
        num_strikes : int, optional
            Number of strikes to optimize over. The default is 4.
        track : list/dict, optional
            Track argument for model instantiation. The default is None.
        seed : int, optional
            Random seed for generating strike locations. The default is 10.
        replicates : int, optional
            Number of strikes to optimize over. The default is 10.
        **kwargs : kwargs
            kwargs to propagate.parameter_sample (e.g., pool).
        """
        # create model
        light_mdl = WildfireSim(p=p, track=track)
        # create parameter domain of base locations
        pd = ParameterDomain(WildFireSimParameter.from_base_loc)
        pd.add_variable('x', var_lim=(0, 45))
        pd.add_variable('y', var_lim=(0, 45))
        pd.add_constant('p', p)
        self.add_parameterdomain(pd)
        # create sample of strike locations
        ps = create_scen_sample(seed=seed, replicates=replicates)
        # sim optimizes over strike samples
        self.add_sim(light_mdl, "parameter_sample", ps, keep_ec=True, **kwargs)
        self.add_result_objective('perc_burned', 'fxns.firepropagation.s.perc_burned',
                                  method=np.mean)


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
                             'fxns.aircraft_0.s.x',
                             'fxns.aircraft_0.s.y',
                             'fxns.aircraft_0.m.mode',
                             'fxns.firepropagation.s.leading_edge_length',
                             'fxns.firepropagation.s.perc_burned')


    fig, ax = mdl.flows['fireenvironment'].c.show_from(8, hist.flows.fireenvironment.c,
                                                       properties=sim_properties,
                                                       xlabel="x (km)", ylabel="y (km)")
    hist.plot_trajectories('s.x', 's.y', fig=fig, ax=ax)

    fig, ax = mdl.flows['fireenvironment'].c.show_from(45, hist.flows.fireenvironment.c,
                                                       properties=sim_properties)
    hist.plot_trajectories('s.x', 's.y', fig=fig, ax=ax)

    ani = mdl.flows['fireenvironment'].c.animate(hist.flows.fireenvironment.c,
                                                 properties=sim_properties)

    # light_mdl = WildfireSim(p=def_p) #  track=None)

    # res, hist = prop.parameter_sample(light_mdl, create_scen_sample())

    # psp = BasePlacementProblem()
    # psp.perc_burned(10, 10)