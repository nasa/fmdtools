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
from fmdtools.modeldef.model import Model
from fmdtools.modeldef.block import FxnBlock, Component
import numpy as np

class ImportLiquid(FxnBlock):
    def __init__(self,name,flows, turnup):
        super().__init__(name,flows,['Watout', 'Sig'],{'open':1.0})
        self.assoc_modes({'Leak':[1e-5,[1,0],0], 'Blockage':[1e-5,[1,0],0]}, units='hr')
        self.turnup = turnup
    def behavior(self,time):
        if   self.Sig.action>=1:        self.open= 1 + self.turnup
        elif self.Sig.action==0:        self.open = 1
        elif self.Sig.action==-1:       self.open = 0
        
        if self.has_fault('Blockage'):   self.Watout.effort=0.0;                 self.Sig.indicator =1
        elif self.has_fault('Leak'):    self.Watout.effort = self.open - 1.0;   self.Sig.indicator =-1
        else:                           self.Watout.effort = self.open;         self.Sig.indicator =0
        
class ExportLiquid(FxnBlock):
    def __init__(self,name, flows, turnup):
        super().__init__(name,flows, ['Watin', 'Sig'],{'open':1.0})
        self.assoc_modes({'Leak':[1e-5,[1,0],0], 'Blockage':[1e-5,[1,0],0]}, units='hr')
        self.turnup = turnup
    def behavior(self,time):
        if   self.Sig.action>=1:      self.open=  1 + self.turnup
        elif self.Sig.action==0:      self.open = 1
        elif self.Sig.action==-1:     self.open = 0
        
        if self.has_fault('Blockage'):   self.Watin.rate=0.0;                                    self.Sig.indicator =1
        elif self.has_fault('Leak'):    self.Watin.rate = self.open*self.Watin.effort + 1.0;    self.Sig.indicator =-1
        else:                           self.Watin.rate = self.open*self.Watin.effort;          self.Sig.indicator =0    
    
class StoreLiquid(FxnBlock):
    def __init__(self,name, flows, capacity):
        super().__init__(name,flows,['Watin','Watout', 'Sig'], {'level':capacity/2, 'net_flow': 0.0, 'coolingbuffer': capacity/2})
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
    def dynamic_behavior(self,time):         
            self.inc(level=self.net_flow)
            self.coolingbuffer = max(self.coolingbuffer - 1.0 + self.Watin.rate, 0)
        
class ContingencyActions(FxnBlock):
    def __init__(self,name, flows, faultpolicy):

        super().__init__(name, flows, ['Input_Sig','Tank_Sig', 'Output_Sig'], timers={'t1'})
        #self.assoc_modes({'FalseDetection_low':[1e-4,[1,1],0],'FalseDetection_high':[1e-4,[1,1],0]}, probtype='rate')
        self.faultpolicy=faultpolicy
    def behavior(self,time):        
        if time > self.time:
            
            self.Input_Sig.action=self.faultpolicy[self.Input_Sig.indicator,self.Tank_Sig.indicator,self.Output_Sig.indicator][0]
            self.Output_Sig.action=self.faultpolicy[self.Input_Sig.indicator,self.Tank_Sig.indicator,self.Output_Sig.indicator][1]
  
class Tank(Model):
    def __init__(self, params={'capacity':20,'turnup':1.0, **{(a-1,b-1,c-1,ul):0 for ul in ["l","u"] for a,b,c in np.ndindex((3,3,3))} },\
                 modelparams = {'phases':{'na':[0],'operation':[1,20]}, 'times':[0,5,10,15,20], 'tstep':1, 'units':'min'}, valparams={}):
        super().__init__(params, modelparams, valparams)
        faultpolicy = {(a-1,b-1,c-1):(params[a-1,b-1,c-1,"l"], params[a-1,b-1,c-1,"u"]) for a,b,c in np.ndindex((3,3,3))}
        
        self.add_flow('Coolant_in', {'effort':1.0, 'rate':1.0})
        self.add_flow('Coolant_out', {'effort':1.0, 'rate':1.0})
        self.add_flow('Input_Sig', {'indicator':1.0, 'action':0.0})
        self.add_flow('Tank_Sig', {'indicator':0.0, 'action':0.0})
        self.add_flow('Output_Sig', {'indicator':1.0, 'action':0.0})
        
        self.add_fxn('Import_Coolant', ['Coolant_in', 'Input_Sig'], fclass = ImportLiquid, fparams = params['turnup'])
        self.add_fxn('Store_Coolant', ['Coolant_in', 'Coolant_out', 'Tank_Sig'], fclass = StoreLiquid, fparams=params['capacity'])
        self.add_fxn('Export_Coolant', ['Coolant_out', 'Output_Sig'], fclass =ExportLiquid, fparams = params['turnup'])
        self.add_fxn('Contingency', ['Input_Sig', 'Tank_Sig', 'Output_Sig'], fclass =ContingencyActions, fparams = faultpolicy)
        
        self.build_model()
    def find_classification(self, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        overfullcost, emptycost, buffercost = 0, 0, 0
        sum(mdlhists['faulty']['functions']['Store_Coolant']['level']>=self.params['capacity'])*10000        #time the tank is overfull
        if any(mdlhists['faulty']['functions']['Store_Coolant']['level']<=0):     emptycost = 1000000     #if the tank lacks any water
        buffercost = sum(mdlhists['faulty']['functions']['Store_Coolant']['coolingbuffer']<=0)*100000     #if the buffer is 'spent'
        mitigationcost = (sum(mdlhists['faulty']['flows']['Input_Sig']['action']!=0)+ sum(mdlhists['faulty']['flows']['Output_Sig']['action']!=0))*1000
        totcost = overfullcost + emptycost + buffercost + mitigationcost
        rate=scen['properties']['rate']
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

if __name__=="__main__":
    import fmdtools.faultsim.propagate as propagate
    import fmdtools.resultdisp as rd
    from fmdtools.modeldef import SampleApproach
    mdl=Tank()
    
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass','bipartite'])
    rd.plot.mdlhists(mdlhist, fxnflowvals='Store_Coolant')

    
    ## faulty run
    resgraph, mdlhist = propagate.one_fault(mdl,'Export_Coolant','Blockage', time=2, desired_result='bipartite')
    
    rd.plot.mdlhists(mdlhist, title='NotVisible', fxnflowvals='Store_Coolant', time_slice=2)
    rd.graph.show(resgraph,faultscen='NotVisible', time=2)
    
        
        
        