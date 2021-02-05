# -*- coding: utf-8 -*-
"""
Dynamical implementation of a tank system with contingency management.

The functions of the system are:
    - ImportLiquid (Inlet Valve)
    - StoreLiquid (Tank)
    - Export Liquid (Outlet Valve)
The Tank stores a set amount of water, the level of which is controlled by 
inlet and outlet valves. 
"""

from fmdtools.modeldef import Model, FxnBlock, Component
import numpy as np

class ImportLiquid(FxnBlock):
    def __init__(self,flows, turnup):
        super().__init__(['Watout', 'Sig'],flows,{'open':1})
        self.assoc_modes({'Leak':[1e-5,[1,0],0], 'Clogged':[1e-5,[1,0],0]}, units='hr')
        self.turnup = turnup
    def behavior(self,time):
        if   self.Sig.action>=1:        self.open= 1 + self.turnup
        elif self.Sig.action==0:        self.open = 1
        elif self.Sig.action==-1:       self.open = 0
        
        if self.has_fault('Clogged'):   self.Watout.effort=0.0;                 self.Sig.indicator =1
        elif self.has_fault('Leak'):    self.Watout.effort = self.open - 1.0;   self.Sig.indicator =-1
        else:                           self.Watout.effort = self.open;         self.Sig.indicator =0
        
class ExportLiquid(FxnBlock):
    def __init__(self,flows, turnup):
        super().__init__(['Watin', 'Sig'],flows,{'open':1})
        self.assoc_modes({'Leak':[1e-5,[1,0],0], 'Clogged':[1e-5,[1,0],0]}, units='hr')
        self.turnup = turnup
    def behavior(self,time):
        if   self.Sig.action>=1:      self.open=  1 + self.turnup
        elif self.Sig.action==0:      self.open = 1
        elif self.Sig.action==-1:     self.open = 0
        
        if self.has_fault('Clogged'):   self.Watin.rate=0.0;                                    self.Sig.indicator =1
        elif self.has_fault('Leak'):    self.Watin.rate = self.open*self.Watin.effort - 1.0;    self.Sig.indicator =-1
        else:                           self.Watin.rate = self.open*self.Watin.effort;          self.Sig.indicator =0    
    
class StoreLiquid(FxnBlock):
    def __init__(self,flows, capacity):
        super().__init__(['Watin','Watout', 'Sig'],flows, {'level':capacity/2, 'net_flow': 0.0, 'coolingbuffer': capacity/2})
        self.assoc_modes({'Leak':[1e-5,[1,0],0]})
        self.capacity=capacity
    def behavior(self, time):
        if self.level >= self.capacity:
            self.Watin.rate = 0.0 * self.Watin.effort
            self.Watout.effort = 2.0 * self.Watin.effort
            self.level = self.capacity
        elif self.level <=0.0:
            self.Watout.effort = 0.0
            self.Watin.rate = self.Watin.effort
        else:
            self.Watin.rate = self.Watin.effort
            self.Watout.effort = 1.0
        if self.level > self.capacity/2+5:        self.Sig.indicator = -1
        elif self.level < self.capacity/2-5:      self.Sig.indicator = 1
        else:                                   self.Sig.indicator = 0
        
        if self.has_fault('Leak'):  self.net_flow = self.Watin.rate - self.Watout.rate - 1.0
        else:                       self.net_flow = self.Watin.rate - self.Watout.rate
        if time>self.time:          
            self.level = self.level + self.net_flow
            self.coolingbuffer = max(self.coolingbuffer - 1.0 + self.Watin.rate, 0)
        
class ContingencyActions(FxnBlock):
    def __init__(self,flows, faultpolicy):

        super().__init__(['Input_Sig','Tank_Sig', 'Output_Sig'], flows, timers={'t1'})
        #self.assoc_modes({'FalseDetection_low':[1e-4,[1,1],0],'FalseDetection_high':[1e-4,[1,1],0]}, probtype='rate')
        self.faultpolicy=faultpolicy
    def behavior(self,time):        
        if time > self.time:
            
            self.Input_Sig.action=self.faultpolicy[self.Input_Sig.indicator,self.Tank_Sig.indicator,self.Output_Sig.indicator][0]
            self.Output_Sig.action=self.faultpolicy[self.Input_Sig.indicator,self.Tank_Sig.indicator,self.Output_Sig.indicator][1]
  
class Tank(Model):
    def __init__(self, params={'capacity':20,'turnup':1.0, 'faultpolicy':{(a-1,b-1,c-1):(1,1) for a,b,c in np.ndindex((3,3,3))}}):
        super().__init__(params = params, modelparams = {'phases':{'na':[0,1],'operation':[1,20]}, 'times':[0,5,10,15,20], 'tstep':1, 'units':'min'})
        
        self.add_flow('Wat_in', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out', {'effort':1.0, 'rate':1.0})
        self.add_flow('Input_Sig', {'indicator':1, 'action':0})
        self.add_flow('Tank_Sig', {'indicator':0, 'action':0})
        self.add_flow('Output_Sig', {'indicator':1, 'action':0})
        
        self.add_fxn('Import_Water', ['Wat_in', 'Input_Sig'], fclass = ImportLiquid, fparams = params['turnup'])
        self.add_fxn('Store_Water', ['Wat_in', 'Wat_out', 'Tank_Sig'], fclass = StoreLiquid, fparams=params['capacity'])
        self.add_fxn('Export_Water', ['Wat_out', 'Output_Sig'], fclass =ExportLiquid, fparams = params['turnup'])
        self.add_fxn('Contingency', ['Input_Sig', 'Tank_Sig', 'Output_Sig'], fclass =ContingencyActions, fparams = params['faultpolicy'])
        
        self.construct_graph()
    def find_classification(self,resgraph, endfaults, endflows, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        overfullcost, emptycost, buffercost = 0, 0, 0
        sum(mdlhists['faulty']['functions']['Store_Water']['level']>=self.params['capacity'])*10000        #time the tank is overfull
        if any(mdlhists['faulty']['functions']['Store_Water']['level']<=0):     emptycost = 1000000     #if the tank lacks any water
        buffercost = sum(mdlhists['faulty']['functions']['Store_Water']['coolingbuffer']<=0)*100000     #if the buffer is 'spent'
        totcost = overfullcost + emptycost + buffercost
        rate=scen['properties']['rate']
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

        
        
        