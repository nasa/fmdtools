#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated wildfire response simulation.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from examples.airspacelib.wildfireresponse.environment import FireEnvironment, FirePropagation, FireMapParam
from examples.airspacelib.wildfireresponse.environment import sim_properties, double_size_p
from examples.airspacelib.wildfireresponse.aircraft import FireAircraft

from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.rand import Rand
from fmdtools.sim.sample import ParameterSample
from fmdtools.sim.search import ParameterSimProblem
from fmdtools.sim.sample import ParameterDomain
from fmdtools.analyze.common import consolidate_legend, add_title_xylabs
from fmdtools.analyze.common import prep_animation_title, clear_prev_figure
from fmdtools.define.architecture.function import FunctionArchitectureGraph
import fmdtools.sim.propagate as prop

import numpy as np


class WildFireSimParameter(Parameter):
    """Parameters defining the wildfire map and response."""

    firemapparam: FireMapParam = FireMapParam()

    @classmethod
    def from_base_loc(cls, x, y, p=double_size_p):
        """Create parameter from base location x,y."""
        fmp = {**p.get('firemapparam', {}), 'base_locations': ((x, y), )}
        return WildFireSimParameter(firemapparam=fmp)


class WildfireSim(FunctionArchitecture):
    """Simulation of wildfire propagation and response."""

    container_p = WildFireSimParameter
    container_r = Rand
    default_sp = {'end_time': 400, "end_condition": "indicate_complete"}

    def init_architecture(self, **kwargs):
        """Initialize architecture with aircraft at bases."""
        # self.add_flow("supplies")
        self.add_flow("fireenvironment", FireEnvironment,
                      c={"p": self.p.firemapparam})

        bases = [i for i in range(len(self.p.firemapparam.base_locations))]
        for base in bases:
            self.add_fxn("aircraft_"+str(base), FireAircraft, "fireenvironment",
                         p={'base': base})
        self.add_fxn("firepropagation", FirePropagation, "fireenvironment")

    def classify(self, **kwargs):
        """Calculate percent burned for a given simulation."""
        return {'perc_burned': self.flows['fireenvironment'].c.calc_perc_burned(),
                'burn_pts': self.flows['fireenvironment'].c.get_all_burned()}

    def indicate_complete(self):
        """Returns true when fire is contained."""
        return self.flows['fireenvironment'].c.indicate_contained()


"""Default overall simulation parameter."""
def_p = {'firemapparam': {**double_size_p,
                          "base_locations": ((42.0, 20.0), (20.0, 20.0)),
                          "num_strikes": 6}}


def plot_combined_response_from(time, history={}, mdl=None, fig=None, ax=None,
                                legend_kwargs={}, title='', **kwargs):
    """
    Plot the overall progression of the simulation up to a given time.

    Parameters
    ----------
    time : int
        Time-step to simulate to.
    history : History, optional
        History of model states. The default is {}.
    mdl : WildfireSim, optional
        Wildfire Simulation model. The default is None.
    fig : matplotlib figure, optional
        Figure to attach to. The default is None.
    ax : matplotlib axis, optional
        Axis to attach to. The default is None.
    legend_kwargs : dict, optional
        kwargs for legend (removes legend if False). The default is {}.
    title : str, optional
        Title for the plot. The default is ''.
    **kwargs : kwargs
        kwargs to add_title_xylabs.

    Returns
    -------
    fig : mpl.figure
        Figure with response shown.
    ax : mpl.axis
        Axis of figure.
    """
    kw = prep_animation_title(time, title=title)
    title = kw['title']
    if fig:
        kw = clear_prev_figure(fig=fig, ax=ax)
        fig = kw.get('fig', None)
        ax = kw.get('ax', None)
    fig, ax = mdl.flows['fireenvironment'].c.show_from(time,
                                                       history.flows.fireenvironment.c,
                                                       properties = sim_properties,
                                                       legend_kwargs=legend_kwargs,
                                                       fig=fig, ax=ax)
    nhist = history.cut(time, newcopy=True)
    legend = legend_kwargs is not False
    for fxnname in nhist.fxns.nest(1):
        if 'aircraft_' in fxnname:
            fig, ax = nhist.plot_trajectories(fxnname+'.s.x', fxnname+'.s.y',
                                              fig=fig, ax=ax, time_groups=['nominal'],
                                              linestyle='--', color='purple', lw=1,
                                              legend=legend, label='flightpath')
            fh = nhist.fxns.get(fxnname)
            slc = fh.get_slice(time)
            ax.scatter(slc.s.x, slc.s.y, marker="^", label=fxnname)

    add_title_xylabs(ax, title=title, aspect='equal',
                     xlabel = "x (km)", ylabel = "y (km)",
                     **kwargs)
    if legend:
        consolidate_legend(ax, **legend_kwargs)
    return fig, ax

def create_scen_sample(seed=10, replicates=10):
    """Create sample of scenarios for a given seed and replicates."""
    ps = ParameterSample(seed=seed)
    ps.add_variable_replicates([], replicates=replicates, seed_comb='independent')
    return ps


class BasePlacementProblem(ParameterSimProblem):
    """Optimization problem for picking best base location(s)."""

    def init_problem(self, p=def_p, track=None, seed=10, replicates=10,
                     **kwargs):
        """
        Initialize base optimization problem.

        Parameters
        ----------
        p : dict, optional
            Non-default model parameters. The default is def_p.
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