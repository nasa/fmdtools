# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:53:00 2023

@author: dhulse
"""
from fmdtools.modeldef.common import Parameter, State, Rand
from fmdtools.modeldef.block import FxnBlock
from fmdtools.modeldef.model import Model, ModelParam
import numpy as np
from fmdtools.faultsim import propagate as prop
import fmdtools.resultdisp as rd

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
        self.add_fxn("Drive", [], fclass= DriveDegradation)
        self.build_model(require_connections=False)
        
if __name__=="__main__":
    #nominal
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl)
    rd.plot.mdlhists(mdlhist)
    #stochastic
    deg_mdl = RoverDegradation()
    endresults,  mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
    rd.plot.mdlhists(mdlhist)
    
    