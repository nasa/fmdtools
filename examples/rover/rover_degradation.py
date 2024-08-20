#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model of rover degradation.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.rand import Rand
from fmdtools.define.block.function import Function
from fmdtools.sim.sample import ParameterSample
from fmdtools.sim import propagate as prop

import numpy as np


class DriveDegradationStates(State):
    """
    State defining the degradation of drive functionality.

    Fields
    ------
    wear : float
        How much wear is in the system (0-1). Default is 0.0.
    corrosion : float
        Amount of corrosion present. Default is 0.0.
    friction : float
        Amount of friction present, slowing down the system. Default is 0.0.
    drift : float
        Drift in the system, causing unintentional turning. Default is 0.0.
    """

    wear: float = 0.0
    corrosion: float = 0.0
    friction: float = 0.0
    drift: float = 0.0


class DriveRandStates(State):
    """
    Random degradation states for the drive function.

    Fields
    ------
    corrode_rate : float
        Rate of chemical corrosion of the mechanical system. Default is 0.01. Updates
        according to a pareto law with parameter 50.
    wear_rate : float
        Rate of wear of the mechanical system. Default is 0.02. Updates according
        to a pareto law with parameter 25.
    yaw_load : float
        Yaw imbalance one way or another. Default is 0.01 but updates according to a
        uniform distribution between -0.1 (left yaw) and 0.1 (right yaw).
    """

    corrode_rate: float = 0.01
    corrode_rate_update = ("pareto", (50,))
    wear_rate: float = 0.02
    wear_rate_update = ("pareto", (25,))
    yaw_load: float = 0.01
    yaw_load_update = ("uniform", (-0.1, 0.1))


class DriveRand(Rand):
    """Rand defining random states for Drive degradation."""

    s: DriveRandStates = DriveRandStates()


class DriveDegradation(Function):
    """Function defining the stochastic degradation of the Drive function."""

    __slots__ = ()
    container_s = DriveDegradationStates
    container_r = DriveRand
    default_sp = dict(end_time=100)

    def dynamic_behavior(self, time):
        self.s.inc(corrosion=self.r.s.corrode_rate, wear=self.r.s.wear_rate)
        opp_drift = (np.sign(self.s.drift) == np.sign(self.r.s.yaw_load))
        self.s.inc(drift=self.r.s.yaw_load / 1000 + opp_drift * self.r.s.yaw_load)
        self.s.friction = np.sqrt(self.s.corrosion**2 + self.s.wear**2)
        self.s.limit(drift=(-1, 1), corrosion=(0, 1), wear=(0, 1))


class PSFDegradationShortStates(State):
    """
    State defining the short-term human performance shaping factor degradation.

    Fields
    ------
    fatigue : float
        Operator fatigue over the course of a day. Default is 0.0.
    stress : float
        Operator stress over the course of a day. Default is 0.0.
    """

    fatigue: float = 0.0
    stress: float = 0.0
    experience: float = 0.0


class PSFDegShortRandStates(State):
    """
    State defining the stochastic states of human performance shaping factors.

    Fields
    ------
    fatigue_param : float
        Operator starting fatigue. Default is 1.0. Updates according to gamma
        distribution with parameters 1 and 1.9.
    """

    fatigue_param: float = 1.0
    fatigue_param_update = ("gamma", (1, 1.9))


class PSFDegShortRand(Rand):
    """Rand defining randomness of human performance shaping factor degradation."""

    s: PSFDegShortRandStates = PSFDegShortRandStates()


class PSFShortParams(Parameter, readonly=True):
    """
    Parameter defining the input/starting parameters of human PSFs.

    Fields
    ------
    experience : float
        Operator experience. Default is 1.0.
    stress_param : float
        Operator base stress. Default is 0.0.
    fatigue_param : float
        Operator base fatigue increment. Default is 0.0
    """

    experience: float = 1.0
    stress_param: float = 0.0
    fatigue_param: float = 1.0


class PSFDegradationShort(Function):
    """Function defining short-term operator performance shaping factor degradation."""

    __slots__ = ()
    container_s = PSFDegradationShortStates
    container_r = PSFDegShortRand
    container_p = PSFShortParams
    default_sp = dict(end_time=12)

    def init_block(self, **kwargs):
        """Initialize parameter-defined base states."""
        self.s.stress = self.p.stress_param
        self.s.experience = self.p.experience
        self.r.s.fatigue_param = self.p.fatigue_param

    def dynamic_behavior(self, time):
        self.s.inc(fatigue=self.r.s.fatigue_param)

        if self.s.stress < 100:
            s_inc = (1 + (1 / self.p.experience)) ** self.t.time
            self.s.stress = int(self.s.stress + s_inc)
        self.s.limit(fatigue=(0, 10), stress=(0, 100))


class LongParams(Parameter, readonly=True):
    """
    Parameter defining long-range PSF degradation.

    Fields
    ------
    experience_param : float
        Operator experience. Default is 9.0.
    training_frequency : float
        Operator training frequency. Default is 8.0.
    experience_scale_max : float
        Maximum operator experience. Default is 10.0.
    """

    experience_param: float = 9.0
    training_frequency: float = 8.0
    experience_scale_max: float = 10.0


class PSFDegradationLongStates(State):
    """
    State defining long-range PSF degradation.

    Fields
    ------
    experience : float
        Base/starting experience of the operator. Default is 0.0.
    """

    experience: float = 0.0


class PSFDegradationLong(Function):
    """Long-term degradation of operator behavior."""

    __slots__ = ()
    container_s = PSFDegradationLongStates
    container_p = LongParams
    default_sp = dict(end_time=100)

    def dynamic_behavior(self, time):
        # normalize time between -1 and 1 to enable sigmoid usage
        norm_time = (((self.t.time - 0) * 2) / (self.sp.end_time - self.sp.start_time)
                     - 1)
        norm_exp = ((self.p.experience_scale_max - self.p.experience_param) /
                    self.p.experience_scale_max)
        exp_den = 1 + (norm_exp * np.exp(-1 * self.p.training_frequency * norm_time))
        self.s.experience = self.p.experience_scale_max / exp_den


if __name__ == "__main__":
    # nominal
    deg_mdl = DriveDegradation('DriveDeg')
    endresults, mdlhist = prop.nominal(deg_mdl)
    mdlhist.plot_line("s.wear")
    # stochastic
    deg_mdl = DriveDegradation('DriveDeg')
    endresults, mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
    mdlhist.plot_line("s.friction")


    ps = ParameterSample()
    ps.add_variable_replicates([], replicates=100, seed_comb='independent')
    endclasses_deg, mdlhists_deg = prop.parameter_sample(deg_mdl, ps, run_stochastic=True)
    fig, ax = mdlhists_deg.plot_line('s.wear', 's.corrosion', 's.friction', 's.drift',
                                 'r.s.corrode_rate', 'r.s.wear_rate', 'r.s.yaw_load',
                                 title="", xlabel='lifecycle time (months)')

    # individual slice
    mdlhists_deg.plot_metric_dist([1, 10, 20],
                             {'s': ['wear', 'corrosion', 'friction', 'drift']})

    # question -- how do we sample this:
    #   - all replicates?
    #   - random sample of them?
    #   - what about times?
    #   - what if we get a complementary sample of times and etc?
    #   - if states in one replicate are the same as a different at the next, can we only sample one?

    # behave_nomapp = ParameterSample()
    # behave_nomapp.add_variable_ranges(
    #     gen_sample_params,
    #     "behave_nomapp",
    #     mdlhists,
    #     histname,
    #     t=(1, 100, 10),
    #     scen=(1, 100, 5),
    # )

    # mdl = Rover()
    # behave_endclasses, behave_mdlhists = prop.nominal_approach(mdl, behave_nomapp)
    # f = plt.figure()
    # f = plot_trajectories(behave_mdlhists)
    # an.plot.nominal_vals_2d(behave_nomapp, behave_endclasses, "inputparams.t", "inputparams.scen",
    #                         nom_func=lambda x: x == 'nominal',
    #                         metric='classification')

    # comp_groups = {
    #     "group_1": [*behave_endclasses.nest().keys()][:100],
    #     "group_2": [*behave_endclasses.nest().keys()][100:],
    # }


    # an.plot.metric_dist(behave_endclasses, 'line_dist', 'end_dist',
    #                     comp_groups=comp_groups, alpha=0.5, bins=10,
    #                     metric_bins={'x':20})

    # an.plot.metric_dist_from(behave_mdlhists, [0, 10, 20],
    #                          {'flows': {'ground': {'s': ['x', 'y', 'linex', 'ang']}}},
    #                          alpha=0.5, bins=10)

    # an.plot.metric_dist_from(behave_mdlhists, 30,
    #                          {'flows': {'ground': {'s':['x', 'y', 'linex', 'ang']}}},
    #                          comp_groups=comp_groups, alpha=0.5, bins=10)


    # # human PSF degradation code starts here

    # fxn_deg_long = PSFDegradationLong("PSFDeg")
    # fxn_deg_short = PSFDegradationShort("PSFDeg")
    # endresults, mdlhist_hum_long = prop.nominal(fxn_deg_long)
    # fig, ax = an.plot.hist(mdlhist_hum_long, "s.experience")

    # # endresults, mdlhist_hum_short = prop.nominal(fxn_deg_short)
    # # fig,ax = an.plot.hist(mdlhist_hum_short, {'s':['fatigue', 'stress']})

    # nomapp_hum_long = NominalApproach()
    # experience_param = np.random.default_rng(seed=101).gamma(1, 1.9, 101)
    # experience_param = list(experience_param)
    # nomapp_hum_long.add_param_ranges(
    #     gen_long_degPSF_param, "nomapp_hum_long", experience_param, scen=(0, 25, 1)
    # )

    # endclasses, mdlhists_hum_long = prop.nominal_approach(
    #     fxn_deg_long, nomapp_hum_long, run_stochastic=True
    # )
    # histname_long = "nomapp_hum_long"

    # stress_param = np.random.default_rng(seed=101).gamma(2, 1.9, 101)
    # stress_param = list(stress_param)
    # nomapp_short_long = NominalApproach()
    # histname_short = "nomapp"
    # nomapp_short_long.add_param_ranges(
    #     gen_short_degPSF_param,
    #     histname_short,
    #     mdlhists_hum_long,
    #     stress_param,
    #     histname_long,
    #     scen=(0, 25, 1),
    #     t=(1, 15, 4),
    # )
    # nomapp_short_long.update_factor_seeds(histname_short, "scen")

    # endclasses, mdlhists_hum_short_long = prop.nominal_approach(
    #     fxn_deg_short, nomapp_short_long, run_stochastic=True
    # )

    # behave_nomapp_hum = NominalApproach()
    # behave_nomapp_hum.add_param_ranges(
    #     gen_human_params_combined,
    #     "behave_nomapp_hum",
    #     mdlhists_hum_short_long,
    #     nomapp_short_long,
    #     t_stress=(1, 11, 2),
    #     t_exp=(1, 15, 4),
    #     scen=(1, 25, 1),
    #     linetype="sine",
    # )
