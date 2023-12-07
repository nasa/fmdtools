# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:53:00 2023

@author: dhulse
"""
from fmdtools.define.role.parameter import Parameter
from fmdtools.define.role.state import State
from fmdtools.define.role.rand import Rand
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model
from fmdtools.sim.sample import ParameterApproach
import numpy as np
from fmdtools.sim import propagate as prop
import fmdtools.analyze as an
import matplotlib.pyplot as plt
from rover_model import Rover, DegParam



    
class DriveDegradationStates(State):
    wear: float = 0.0
    corrosion: float = 0.0
    friction: float = 0.0
    drift: float = 0.0


class DriveRandStates(State):
    corrode_rate: float = 0.01
    corrode_rate_update = ("pareto", (50,))
    wear_rate: float = 0.02
    wear_rate_update = ("pareto", (25,))
    yaw_load: float = 0.01
    yaw_load_update = ("uniform", (-0.1, 0.1))


class DriveRand(Rand):
    s: DriveRandStates = DriveRandStates()


class DriveDegradation(FxnBlock):
    _init_s = DriveDegradationStates
    _init_r = DriveRand
    default_sp = dict(times=(0, 100))

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def dynamic_behavior(self, time):
        self.s.inc(corrosion=self.r.s.corrode_rate, wear=self.r.s.wear_rate)
        self.s.inc(
            drift=self.r.s.yaw_load / 1000
            + (np.sign(self.s.drift) == np.sign(self.r.s.yaw_load)) * self.r.s.yaw_load
        )
        self.s.friction = np.sqrt(self.s.corrosion**2 + self.s.wear**2)
        self.s.limit(drift=(-1, 1), corrosion=(0, 1), wear=(0, 1))


class PSFDegradationShortStates(State):
    fatigue: float = 0.0
    stress: float = 0.0


class PSFDegShortRandStates(State):
    fatigue_param: float = 2.0
    fatigue_param_update = ("gamma", (2, 1.9))


class PSFDegShortRand(Rand):
    s: PSFDegShortRandStates = PSFDegShortRandStates()


class PSFShortParams(Parameter, readonly=True):
    experience: float = 1.0
    stress_param: float = 0.0
    fatigue_param: float = 0.0
    stoch_fatigue: bool = False


class PSFDegradationShort(FxnBlock):
    _init_s = PSFDegradationShortStates
    _init_r = PSFDegShortRand
    _init_p = PSFShortParams
    default_sp = dict(times=(0, 100))

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.s.stress = self.p.stress_param

    def dynamic_behavior(self, time):
        if self.p.stoch_fatigue:
            self.s.fatigue = int(self.r.s.fatigue_param)
        else:
            self.s.fatigue = int(self.p.fatigue_param)
        if self.s.stress < 100:
            self.s.stress = int(
                self.s.stress + (1 + (1 / self.p.experience)) ** self.t.time
            )
        self.s.limit(fatigue=(0, 10), stress=(0, 100))


class LongParams(Parameter, readonly=True):
    experience_param: float = 9.0
    training_frequency: float = 8.0
    experience_scale_max: float = 10.0


class PSFDegradationLongStates(State):
    experience: float = 0.0


class PSFDegradationLong(FxnBlock):
    _init_s = PSFDegradationLongStates
    _init_p = LongParams
    default_sp = dict(times=(0, 100))

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # self.s.experience= self.p.experience_param

    def dynamic_behavior(self, time):
        # normalize time between -1 and 1 to enable sigmoid usage
        norm_time = ((self.t.time - 0) * 2) / (self.sp.times[1] - self.sp.times[0]) - 1
        norm_experience_param = (
            self.p.experience_scale_max - self.p.experience_param
        ) / self.p.experience_scale_max
        self.s.experience = self.p.experience_scale_max / (
            1
            + (
                norm_experience_param
                * np.exp(-1 * self.p.training_frequency * norm_time)
            )
        )



def get_params_from(mdlhist, t=1):
    friction = mdlhist.s.friction[t]
    drift = mdlhist.s.drift[t]
    return {"friction": friction, "drift": drift}


def get_paramdist_from(mdlhists, t):
    friction = []
    drift = []
    for rep in mdlhists:
        fdict = get_params_from(mdlhists[rep], t)
        friction.append(fdict["friction"])
        drift.append(fdict["drift"])
    return {"friction": friction, "drift": drift}


def sample_params(mdlhists, histname, t=1, scen=1):
    mdlhists = mdlhists.nest(1)
    mdlhist = mdlhists[histname + "_" + str(scen)]
    return get_params_from(mdlhist, t)


def gen_sample_params(mdlhists, histname, t=1, scen=1):
    degparams = sample_params(mdlhists, histname, t=t, scen=scen)
    return { 'drive_modes': {"mode_args": "degradation"}, "degradation": DegParam(**degparams)}


def gen_long_degPSF_param(experience_param, scen=1):
    params = {"experience_param": experience_param[scen]}
    return params


def gen_short_degPSF_param(mdlhists, stress_param, histname, t=1, scen=1):
    mdlhists = mdlhists.nest(1)
    mdlhist = mdlhists[histname + "_" + str(scen + 1)]
    params = {"experience": get_longhuman_params_from(mdlhist, t)}
    params.update({"stress_param": stress_param[scen]})
    return params


def get_longhuman_params_from(mdlhist, t):
    experience = mdlhist.s.experience[t]
    return experience


def gen_human_params_combined(
    mdlhists_stress,
    app_stress,
    t_exp=1,
    t_stress=1,
    scen=1,
    linetype="sine",
):
    scen_groups = app_stress.get_param_scens("inputparams.t", "inputparams.scen")
    scen = [scenario[0] for scenario in scen_groups if scenario[1] == [t_exp, scen]]
    if len(scen) > 1:
        raise Exception("multiple scenarios for the given time and scen")
    scen = [*scen][0]
    mdlhists_stress = mdlhists_stress.nest(1)
    mdlhist = mdlhists_stress[scen]
    human_params = get_human_params_from(mdlhist, t_stress)
    return dict(linetype=linetype, **human_params)


def get_human_params_from(mdlhist, t=1):
    fatigue = mdlhist.s.fatigue[t]
    stress = mdlhist.s.stress[t]
    return {"fatigue": fatigue, "stress": stress}


# def gen_sample_params(mdlhists, long_deg_params, t=1, scen=1, linetype='sine'):
#     degparams={}
#     degparams.update(sample_human_params(mdlhists, t=t, scen=scen))
#     degparams.update({'friction': long_deg_params[scen][0], 'drift': long_deg_params[scen][1], 'experience': long_deg_params[scen][2]})
#     return dict(linetype=linetype, **degparams)


def sample_human_params(mdlhists, t=1, scen=1):
    mdlhist = [*mdlhists.values()][scen]
    return get_human_params_from(mdlhist, t)


def gen_sample_params_human(mdlhists, t=1, scen=1, linetype="sine"):
    degparams = sample_human_params(mdlhists, t=t, scen=scen)
    return dict(linetype=linetype, **degparams)


def gen_sample_params_comp(mdlhists, t=1, scen=1, linetype="sine"):
    degparams = sample_params(mdlhists, t=t, scen=scen)
    return dict(linetype=linetype, **degparams)


def gen_sample_params_combined(
    mdlhists_comp,
    mdlhists_stress,
    app_stress,
    stress_id="nomapp",
    t_comp=1,
    t_exp=1,
    t_stress=1,
    scen=1,
    linetype="sine",
):
    scen_groups = app_stress.get_param_scens(stress_id, "t", "scen")
    stress_scen = scen_groups[t_exp, scen]
    if len(stress_scen) > 1:
        raise Exception("multiple scenarios for the given time and scen")
    stress_scen = [*stress_scen][0]
    mdlhist_stress = mdlhists_stress[stress_scen]
    degparams = get_human_params_from(mdlhist_stress, t_stress)
    degparams.update(sample_params(mdlhists_comp, t=t_comp, scen=scen))
    return dict(linetype=linetype, **degparams)


def gen_long_deg_param_list(mdlhists, mdlhists_hum, t_total, total_scen):
    params = []
    for s in range(total_scen):
        for t in range(t_total):
            mdlhist = [*mdlhists.values()][s]
            temp = get_params_from(mdlhist, t + 1)
            mdlhist_hum = [*mdlhists_hum.values()][s]
            temp = temp + (get_longhuman_params_from(mdlhist_hum, t + 1),)
            params.append(temp)
    return params


if __name__ == "__main__":
    # nominal
    deg_mdl = DriveDegradation('DriveDeg')
    endresults, mdlhist = prop.nominal(deg_mdl)
    mdlhist.plot_line("s.wear")
    # stochastic
    deg_mdl = DriveDegradation('DriveDeg')
    endresults, mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
    mdlhist.plot_line("s.friction")

    # stochastic over replicates
    nomapp = NominalApproach()
    histname = "test"
    nomapp.add_seed_replicates(histname, 100)
    endclasses, mdlhists = prop.nominal_approach(
        deg_mdl, nomapp, run_stochastic=True, desired_result="endclass"
    )
    an.plot.hist(
        mdlhists,
        {"s": ["wear", "corrosion", "friction", "drift"]},
        aggregation="mean_std",
    )

    # individual slice
    an.plot.metric_dist_from(mdlhists, [1, 10, 20],
                             {'s': ['wear', 'corrosion', 'friction', 'drift']})

    # question -- how do we sample this:
    #   - all replicates?
    #   - random sample of them?
    #   - what about times?
    #   - what if we get a complementary sample of times and etc?
    #   - if states in one replicate are the same as a different at the next, can we only sample one?

    behave_nomapp = NominalApproach()
    behave_nomapp.add_param_ranges(
        gen_sample_params,
        "behave_nomapp",
        mdlhists,
        histname,
        t=(1, 100, 10),
        scen=(1, 100, 5),
    )

    mdl = Rover()
    behave_endclasses, behave_mdlhists = prop.nominal_approach(mdl, behave_nomapp)
    f = plt.figure()
    f = plot_trajectories(behave_mdlhists)
    an.plot.nominal_vals_2d(behave_nomapp, behave_endclasses, "inputparams.t", "inputparams.scen",
                            nom_func=lambda x: x == 'nominal',
                            metric='classification')

    comp_groups = {
        "group_1": [*behave_endclasses.nest().keys()][:100],
        "group_2": [*behave_endclasses.nest().keys()][100:],
    }


    an.plot.metric_dist(behave_endclasses, 'line_dist', 'end_dist',
                        comp_groups=comp_groups, alpha=0.5, bins=10,
                        metric_bins={'x':20})

    an.plot.metric_dist_from(behave_mdlhists, [0, 10, 20],
                             {'flows': {'ground': {'s': ['x', 'y', 'linex', 'ang']}}},
                             alpha=0.5, bins=10)

    an.plot.metric_dist_from(behave_mdlhists, 30,
                             {'flows': {'ground': {'s':['x', 'y', 'linex', 'ang']}}},
                             comp_groups=comp_groups, alpha=0.5, bins=10)


    # human PSF degradation code starts here

    fxn_deg_long = PSFDegradationLong("PSFDeg")
    fxn_deg_short = PSFDegradationShort("PSFDeg")
    endresults, mdlhist_hum_long = prop.nominal(fxn_deg_long)
    fig, ax = an.plot.hist(mdlhist_hum_long, "s.experience")

    # endresults, mdlhist_hum_short = prop.nominal(fxn_deg_short)
    # fig,ax = an.plot.hist(mdlhist_hum_short, {'s':['fatigue', 'stress']})

    nomapp_hum_long = NominalApproach()
    experience_param = np.random.default_rng(seed=101).gamma(1, 1.9, 101)
    experience_param = list(experience_param)
    nomapp_hum_long.add_param_ranges(
        gen_long_degPSF_param, "nomapp_hum_long", experience_param, scen=(0, 25, 1)
    )

    endclasses, mdlhists_hum_long = prop.nominal_approach(
        fxn_deg_long, nomapp_hum_long, run_stochastic=True
    )
    histname_long = "nomapp_hum_long"

    stress_param = np.random.default_rng(seed=101).gamma(2, 1.9, 101)
    stress_param = list(stress_param)
    nomapp_short_long = NominalApproach()
    histname_short = "nomapp"
    nomapp_short_long.add_param_ranges(
        gen_short_degPSF_param,
        histname_short,
        mdlhists_hum_long,
        stress_param,
        histname_long,
        scen=(0, 25, 1),
        t=(1, 15, 4),
    )
    nomapp_short_long.update_factor_seeds(histname_short, "scen")

    endclasses, mdlhists_hum_short_long = prop.nominal_approach(
        fxn_deg_short, nomapp_short_long, run_stochastic=True
    )

    behave_nomapp_hum = NominalApproach()
    behave_nomapp_hum.add_param_ranges(
        gen_human_params_combined,
        "behave_nomapp_hum",
        mdlhists_hum_short_long,
        nomapp_short_long,
        t_stress=(1, 11, 2),
        t_exp=(1, 15, 4),
        scen=(1, 25, 1),
        linetype="sine",
    )
