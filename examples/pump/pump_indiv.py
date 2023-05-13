# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:59:41 2023

@author: dhulse
"""

from ex_pump import MoveWat
from fmdtools.define.common import set_var
from fmdtools.sim.propagate import nominal, one_fault
from fmdtools.analyze import plot


class MoveWatStatic(MoveWat):
    def static_loading(self, time):
        # Signal Inputs
        if time<5:      self.sig_in.s.power=0.0
        elif time<50:   self.sig_in.s.power=1.0
        else:           self.sig_in.s.power=0.0
        # EE Inputs
        self.ee_in.voltage=500.0
        # Water Input
        self.wat_out.s.level=1.0
        # Water Output
        self.wat_out.s.area=1.0

class MoveWatDynamic(MoveWat):
    default_sp = {'times':(0,50)}
    def static_loading(self, time):
        # Signal Inputs
        if time<5:      self.sig_in.s.power=0.0
        elif time<50:   self.sig_in.s.power=1.0
        else:           self.sig_in.s.power=0.0
        # EE Inputs
        self.ee_in.s.voltage=500.0
        # Water Input
        self.wat_out.s.level=1.0
        # Water Output
        self.wat_out.s.area=1.0


a = MoveWatDynamic()
                    
result, mdlhist = nominal(a, track='all')

plot.hist(mdlhist, 'flows.sig_in.s.power', 'flows.wat_out.s.flowrate')

result, mdlhist = one_fault(a, "short", time=10, track='all')

plot.hist(mdlhist, 'flows.sig_in.s.power', 'flows.wat_out.s.flowrate')
                    

            
        
    


            