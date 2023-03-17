# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:53:00 2023

@author: dhulse
"""
from fmdtools.define.common import Parameter, State, Rand
from fmdtools.define.block import FxnBlock
from fmdtools.define.model import Model, ModelParam
from fmdtools.define.approach import NominalApproach
import numpy as np
from fmdtools.sim import propagate as prop
import fmdtools.analyze as rd
import matplotlib.pyplot as plt
from rover_model import Rover, plot_trajectories, DegParam


class DriveDegradationStates(State):
    wear:       float = 0.0
    corrosion:  float=0.0
    friction:   float=0.0
    drift:      float=0.0
class DriveRandStates(State):
    corrode_rate:   float = 0.01
    corrode_rate_update = ('pareto', (50,))
    wear_rate:      float = 0.02
    wear_rate_update = ('pareto', (25,))
    yaw_load:       float = 0.01
    yaw_load_update = ('uniform', (-0.1, 0.1))
class DriveRand(Rand):
    s: DriveRandStates = DriveRandStates()

class DriveDegradation(FxnBlock):
    _init_s=DriveDegradationStates
    _init_r = DriveRand
    def __init__(self,name,flows, params={}, **kwargs):
        super().__init__(name, flows, **kwargs)
        #self.assoc_rand_state('corrode_rate', 0.01, auto_update = ['pareto', (50,)])
        #self.assoc_rand_state('wear_rate', 0.02, auto_update = ['pareto', (25,)])
        #self.assoc_rand_state('yaw_load', 0.01, auto_update = ['uniform', (-0.1, 0.1)])
    def dynamic_behavior(self, time):
        self.s.inc(corrosion=self.r.s.corrode_rate, wear=self.r.s.wear_rate)
        self.s.inc(drift = self.r.s.yaw_load/1000 + (np.sign(self.s.drift)==np.sign(self.r.s.yaw_load))*self.r.s.yaw_load)
        self.s.friction = np.sqrt(self.s.corrosion**2+self.s.wear**2)
        self.s.limit(drift=(-1,1), corrosion=(0,1), wear=(0,1))
class RoverDegradation(Model):
    def __init__(self, params=Parameter(), modelparams=ModelParam(times=(0,100), seed=102), valparams={}):
        super().__init__(params, modelparams, valparams)
        self.add_fxn("Drive", DriveDegradation)
        self.build_model(require_connections=False)

def get_params_from(mdlhist, t=1):
    friction = mdlhist['functions']['Drive']['friction'][t]
    drift = mdlhist['functions']['Drive']['drift'][t]
    return {'friction':friction, 'drift':drift}
def get_paramdist_from(mdlhists, t):
    friction=[]
    drift=[]
    for rep in mdlhists:
        fdict = get_params_from(mdlhists[rep], t)
        friction.append(fdict['friction'])
        drift.append(fdict['drift'])
    return {'friction':friction, 'drift':drift}

def sample_params(mdlhists, t=1, scen=1):
    mdlhist = [*mdlhists.values()][scen]
    return get_params_from(mdlhist, t)

def gen_sample_params(mdlhists, t=1, scen=1):
    degparams = sample_params(mdlhists, t=t, scen=scen)
    return {'linetype':'turn', 'degradation':DegParam(**degparams)}

if __name__=="__main__":
    #nominal
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl)
    rd.plot.mdlhists(mdlhist)
    #stochastic
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
    rd.plot.mdlhists(mdlhist)
    
    #stochastic over replicates
    nomapp = NominalApproach()
    nomapp.add_seed_replicates('test', 100)
    endclasses, mdlhists = prop.nominal_approach(deg_mdl, nomapp, run_stochastic=True, desired_result='endclass')
    rd.plot.mdlhists(mdlhists, fxnflowvals={'Drive':['wear', 'corrosion', 'friction', 'drift']}, aggregation='mean_std')
    
    #individual slice
    rd.plot.metric_dist_from(mdlhists, [1,10,20], fxnflowvals={'Drive':['wear', 'corrosion', 'friction', 'drift']})
    
    
    #question -- how do we sample this:
    #   - all replicates?
    #   - random sample of them?
    #   - what about times?
    #   - what if we get a complementary sample of times and etc?
    #   - if states in one replicate are the same as a different at the next, can we only sample one?

    behave_nomapp = NominalApproach()
    behave_nomapp.add_param_ranges(gen_sample_params, 'behave_nomapp', mdlhists, t=(1,100, 10), scen = (1,100,5))

    mdl=Rover()
    behave_endclasses, behave_mdlhists = prop.nominal_approach(mdl, behave_nomapp)
    f = plt.figure()
    f = plot_trajectories(behave_mdlhists)
    rd.plot.nominal_vals_2d(behave_nomapp, behave_endclasses, 't', 'scen')

    comp_groups = {'group_1': [*behave_endclasses][:100],'group_2': [*behave_endclasses][100:]}

    rd.plot.metric_dist(behave_endclasses, metrics=['line_dist', 'end_dist', 'x', 'y'], comp_groups=comp_groups, alpha=0.5, bins=10, metric_bins={'x':20})

    rd.plot.metric_dist_from(behave_mdlhists, times= [0, 10, 20], fxnflowvals = {'ground':['x', 'y', 'linex', 'ang']}, alpha=0.5, bins=10)

    rd.plot.metric_dist_from(behave_mdlhists, times= 30, fxnflowvals = {'ground':['x', 'y', 'linex', 'ang']}, comp_groups=comp_groups, alpha=0.5, bins=10)
    