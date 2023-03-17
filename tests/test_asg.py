# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:47:04 2023

@author: dhulse
"""
import unittest

from fmdtools.define.common import State
from fmdtools.define.block import FxnBlock, Action, Mode, ASG
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow
import fmdtools.analyze as an
import fmdtools.sim.propagate as prop

class OutcomeStates(State):
    num_perceptions: int=0
    num_actions:     int=0
class Outcome(Flow):
    _init_s = OutcomeStates
    
class HazardState(State):
    present:    bool=False
    percieved:  bool=False
    mitigated:  bool=False
class Hazard(Flow):
    _init_s = HazardState


class ActionMode(Mode):
    faultparams=('failed','unable')
    exclusive=True
class Perceive(Action):
    _init_m = ActionMode
    _init_hazard = Hazard
    _init_outcome = Outcome
    def behavior(self,time):
        if not self.m.in_mode('failed', 'unable'): 
            self.hazard.s.percieved = self.hazard.s.present
            self.outcome.s.num_perceptions+=self.hazard.s.percieved
        else:
            self.hazard.s.percieved = False
            self.m.remove_fault('failed', 'nom')
    def percieved(self):
        return self.hazard.s.percieved
class Act(Action):
    _init_m = ActionMode
    _init_hazard = Hazard
    _init_outcome = Outcome
    def behavior(self,time):
        if not self.m.in_mode('failed', 'unable'): 
            self.outcome.s.num_actions+=1
            self.hazard.s.mitigated=True
        elif self.m.in_mode('failed'): 
            self.hazard.s.mitigated=False
            self.m.remove_fault('failed', 'nom')
        else: self.hazard.s.mitigated=False
    def acted(self):
        return not self.m.in_mode('failed')
class Done(Action):
    _init_hazard = Hazard
    def behavior(self,time):
        if not self.hazard.s.present: self.hazard.s.mitigated=False
    def ready(self):
        return not self.hazard.s.present

class Human(ASG):
    initial_action="perceive"
    _init_hazard = Hazard    # flows from external fxn/model should be defined as a part of the class definition                
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_flow("outcome",    fclass=Outcome) #flows can be added in the ASG for custom flow architectures.  
        
        self.add_act("perceive",    Perceive,   "outcome", "hazard")
        self.add_act("act",         Act,        "outcome", "hazard")
        self.add_act("done",        Done,       "hazard")
        
        self.add_cond("perceive",   "act",      "percieved",    self.actions['perceive'].percieved)
        self.add_cond("act",        "done",     "acted",        self.actions['act'].acted)
        self.add_cond("done",       "perceive", "ready",        self.actions['done'].ready)
        self.build()
        
h = Human()

p = Perceive("a", {"outcome":Outcome("outcome")})

class DetectHazard(FxnBlock):
    _init_a =           Human
    _init_hazard=       Hazard


#ex_fxn = DetectHazard('detect_hazard', [])

#ex_fxn.set_timestep(local_tstep=1.0)

#ex_fxn.a.flows['hazard']

#fig = ex_fxn.a.show()

#ex_fxn.a.flows['hazard'].s.present=True
#ex_fxn.updatefxn('dynamic', time= 1)
#fig = ex_fxn.a.show()
#ex_fxn.a.flows['hazard'].s.present=False
#ex_fxn.updatefxn('dynamic', time= 2)
#fig = ex_fxn.a.show()


class ProduceHazard(FxnBlock):
    _init_hazard = Hazard
    def dynamic_behavior(self,time):
        if not time%4: self.hazard.s.present=True
        else:          self.hazard.s.present=False
class PassStates(State):
    hazards_mitigated:  int=0
    hazards_propagated: int=0
class PassHazard(FxnBlock):
    _init_s = PassStates
    _init_hazard = Hazard
    def dynamic_behavior(self,time):
        if self.hazard.s.present and self.hazard.s.mitigated:       self.s.hazards_mitigated+=1
        elif self.hazard.s.present and not self.hazard.s.mitigated: self.s.hazards_propagated+=1

from fmdtools.define.common import Parameter
from fmdtools.define.model import ModelParam
class HazardModel(Model):
    def __init__(self, params=Parameter(), modelparams=ModelParam(times=(0,60), dt=1.0), valparams={}):
        super().__init__(params,modelparams,valparams)
        
        self.add_flow("hazard", Hazard)
        
        self.add_fxn("produce_hazard", ProduceHazard, 'hazard')
        self.add_fxn("detect_hazard",  DetectHazard,  'hazard')
        self.add_fxn("pass_hazard",    PassHazard,    'hazard')
        self.build_model()

mdl = HazardModel()
#endstate,  mdlhist = prop.nominal(mdl)

resgraph_fault, mdlhist_fault = prop.one_fault(mdl, 'detect_hazard','perceive_failed', time=4, desired_result='bipartite')

reshist, diff, summary = an.process.hist(mdlhist_fault)

reshist['functions']['detect_hazard']['perceive']['faults'][4]
fig = an.graph.result_from(mdl.fxns['detect_hazard'], reshist, 4, gtype='combined')

