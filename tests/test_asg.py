# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:47:04 2023

@author: dhulse
"""
import unittest

from fmdtools.modeldef.common import State
from fmdtools.modeldef.block import FxnBlock, Action, Mode, ASG
from fmdtools.modeldef.model import Model
from fmdtools.modeldef.flow import Flow
import fmdtools.resultdisp as rd
import fmdtools.faultsim.propagate as prop

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
    faultparams=('failed',)
class Perceive(Action):
    _init_m = ActionMode
    def __init__(self, name, *flows):
        super().__init__(name, flows)
    def behavior(self,time):
        if not self.m.in_mode('failed'): 
            self.Hazard.s.percieved = self.Hazard.s.present
            self.Outcome.s.num_perceptions+=self.Hazard.s.percieved
        else:
            self.Hazard.s.percieved = False
            self.remove_fault('failed', 'nom')
    def percieved(self):
        return self.Hazard.s.percieved
class Act(Action):
    _init_m = ActionMode
    def __init__(self, name, *flows):
        super().__init__(name,flows)
    def behavior(self,time):
        if not self.m.in_mode('failed'): 
            self.Outcome.s.num_actions+=1
            self.Hazard.s.mitigated=True
        elif self.m.in_mode('failed'): 
            self.Hazard.s.mitigated=False
            self.remove_fault('failed', 'nom')
        else: self.Hazard.s.mitigated=False
    def acted(self):
        return not self.m.in_mode('failed')
class Done(Action):
    def __init__(self, name, *flows):
        super().__init__(name,flows)
    def behavior(self,time):
        if not self.Hazard.s.present: self.Hazard.s.mitigated=False
    def ready(self):
        return not self.Hazard.s.present

class Human(ASG):
    initial_action="Perceive"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_flow("Outcome", fclass=Outcome)
        self.add_flow("Hazard", fclass=Hazard)
        
        self.add_act("Perceive", Perceive, "Outcome", "Hazard")
        self.add_act("Act", Act, "Outcome", "Hazard")
        self.add_act("Done", Done, "Outcome", "Hazard")
        
        self.add_cond("Perceive","Act", "Percieved",    self.actions['Perceive'].percieved)
        self.add_cond("Act","Done", "Acted",            self.actions['Act'].acted)
        self.add_cond("Done", "Perceive", "Ready",      self.actions['Done'].ready)
        self.build()
        
h = Human()

p = Perceive("a", Outcome("Outcome"))

class DetectHazard(FxnBlock):
    _init_a = Human
    def __init__(self,name, flows, *args, **kwargs):
        super().__init__(name, flows, *args, **kwargs)


ex_fxn = DetectHazard('DetectHazard', [])
ex_fxn.set_timestep(local_tstep=1.0)

ex_fxn.a.flows['Hazard']

fig = ex_fxn.a.show()

ex_fxn.a.flows['Hazard'].s.present=True
ex_fxn.updatefxn('dynamic', time= 1)
fig = ex_fxn.a.show()
ex_fxn.a.flows['Hazard'].s.present=False
ex_fxn.updatefxn('dynamic', time= 2)
fig = ex_fxn.a.show()



