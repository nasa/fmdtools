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
from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.mode import Mode
from fmdtools.define.model import Model
from fmdtools.define.block import FxnBlock
import numpy as np

from tank_model import TransportLiquidState, Signal, Water

class TankParam(Parameter, readonly=True):
    capacity:       np.float64 = np.float64(20.0)
    turnup:         np.float64 = np.float64(1.0)
    faultpolicy:    tuple = tuple((a-1,b-1,c-1,ul, 0) for ul in ["l","u"] for a,b,c in np.ndindex((3,3,3)))
    policymap:      dict=dict()
    def __init__(self, *args, **kwargs):
        args = self.get_true_fields(*args, **kwargs)
        super().__init__(*args, strict_immutability=False)
        if not self.policymap: 
            self.policymap.update(self.get_faultpolicy())
    def get_faultpolicy(self):
        fd = {(v[0], v[1], v[2], v[3]): v[4] for v in self.faultpolicy}
        return {(a-1,b-1,c-1):(fd[a-1,b-1,c-1,"l"], fd[a-1,b-1,c-1,"u"]) for a,b,c in np.ndindex((3,3,3))}
def make_tankparam(*args,**kwargs):
    if args: 
        fp = tuple((v[0], v[1], v[2], v[3], args[i]) for i,v in enumerate(TankParam.__defaults__[2]))
        kwargs['faultpolicy']=fp
    return kwargs

class TransportLiquidMode(Mode):
    faultparams={'stuck':(1e-5,[1,0],0),
                 'blockage':(1e-5,[1,0],0)}
    units='hr'
    key_phases_by='global'

class ImportLiquid(FxnBlock):
    __slots__=('sig', 'wat_out')
    _init_p = TankParam
    _init_s = TransportLiquidState
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_wat_out = Water
    flownames = {'coolant_in':'wat_out', 'input_sig':'sig'}
    def behavior(self,time):
        if   self.sig.s.action>=1:        self.s.amt_open= 1 + self.p.turnup
        elif self.sig.s.action==0:        self.s.amt_open = 1
        elif self.sig.s.action==-1:       self.s.amt_open = 0
        
        if self.m.has_fault('blockage'):   
            self.wat_out.s.effort=0.0
            self.sig.s.indicator =1
        elif self.m.has_fault('leak'):    
            self.wat_out.s.effort = self.s.amt_open - 1.0 
            self.sig.s.indicator =-1
        else:                           
            self.wat_out.s.effort = self.s.amt_open 
            self.sig.s.indicator =0

class ExportLiquid(FxnBlock):
    __slots__=('sig', 'wat_in')
    _init_p = TankParam
    _init_s = TransportLiquidState
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_wat_in = Water
    flownames = {'coolant_out':'wat_in', 'output_sig':'sig'}
    def behavior(self,time):
        if   self.sig.s.action>=1:      self.s.amt_open=  1 + self.p.turnup
        elif self.sig.s.action==0:      self.s.amt_open = 1
        elif self.sig.s.action==-1:     self.s.amt_open = 0
        
        if self.m.has_fault('blockage'):   
            self.wat_in.s.rate=0.0 
            self.sig.s.indicator =1
        elif self.m.has_fault('leak'):    
            self.wat_in.s.rate = self.s.amt_open*self.wat_in.s.effort + 1.0 
            self.sig.s.indicator =-1
        else:   
            self.wat_in.s.rate = self.s.amt_open*self.wat_in.s.effort 
            self.sig.s.indicator =0    
 
class StoreLiquidState(State):
    level:          float=10.0
    net_flow:       float=0.0
    coolingbuffer:  float=10.0
from tank_model import StoreLiquidMode 
    
class StoreLiquid(FxnBlock):
    __slots__=('wat_in', 'wat_out', 'sig')
    _init_s = StoreLiquidState
    _init_m = StoreLiquidMode
    _init_p = TankParam
    _init_wat_in = Water
    _init_wat_out = Water
    _init_sig = Signal
    flownames = {'coolant_in':'wat_in', 'coolant_out':'wat_out', 'tank_sig':'sig'}
    def behavior(self, time):
        if self.s.level >= self.p.capacity:
            self.wat_in.s.rate = 0.0 * self.wat_in.s.effort
            self.wat_out.s.effort = 2.0 * self.wat_in.s.effort
            self.s.level = self.p.capacity
        elif self.s.level <=0.0:
            self.wat_out.s.effort = 0.0
            self.wat_in.s.rate = self.wat_in.s.effort
        else:
            self.wat_in.s.rate = self.wat_in.s.effort
            self.wat_out.s.effort = 1.0
        if self.s.level > self.p.capacity/2+5:      self.sig.s.indicator = -1
        elif self.s.level < self.p.capacity/2-5:    self.sig.s.indicator = 1
        else:                                       self.sig.s.indicator = 0
        
        if self.m.has_fault('leak'):    self.s.net_flow = self.wat_in.s.rate - self.wat_out.s.rate - 1.0
        else:                           self.s.net_flow = self.wat_in.s.rate - self.wat_out.s.rate
    def dynamic_behavior(self,time):         
            self.s.inc(level=self.s.net_flow)
            self.s.coolingbuffer = max(self.s.coolingbuffer - 1.0 + self.wat_in.s.rate, 0)

class ContingencyActions(FxnBlock):
    _init_p = TankParam
    _init_input_sig = Signal
    _init_output_sig = Signal 
    _init_tank_sig = Signal
    def dynamic_behavior(self,time):        
        self.input_sig.s.action=self.p.policymap[self.input_sig.s.indicator,self.tank_sig.s.indicator,self.output_sig.s.indicator][0]
        self.output_sig.s.action=self.p.policymap[self.input_sig.s.indicator,self.tank_sig.s.indicator,self.output_sig.s.indicator][1]



class Tank(Model):
    __slots__=()
    _init_p = TankParam 
    default_sp = dict(phases=(('na',0,0),('operation',1,20)),times=(0,5,10,15,20),units='min')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_flow('coolant_in',     Water)
        self.add_flow('coolant_out',    Water)
        self.add_flow('input_sig',      Signal)
        self.add_flow('tank_sig',       Signal)
        self.add_flow('output_sig',     Signal)
        
        
        
        self.add_fxn('import_coolant',  ImportLiquid, 'coolant_in', 'input_sig', p=self.p)
        self.add_fxn('store_coolant',   StoreLiquid, 'coolant_in', 'coolant_out', 'tank_sig', 
                     p=self.p, s={'level':self.p.capacity/2, 'coolingbuffer':self.p.capacity/2})
        
        
        self.add_fxn('export_coolant',  ExportLiquid, 'coolant_out', 'output_sig', p=self.p)
        self.add_fxn('contingency',     ContingencyActions, 'input_sig', 'tank_sig', 'output_sig', p=self.p)
        
        self.build()
    def find_classification(self, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        overfullcost, emptycost, buffercost = 0, 0, 0
        sum(self.h.fxns.store_coolant.s.level>=self.p.capacity)*10000        #time the tank is overfull
        if any(self.h.fxns.store_coolant.s.level<=0):     emptycost = 1000000     #if the tank lacks any water
        buffercost = sum(self.h.fxns.store_coolant.s.coolingbuffer<=0)*100000     #if the buffer is 'spent'
        mitigationcost = (sum(self.h.flows.input_sig.s.action!=0)+ sum(self.h.flows.output_sig.s.action!=0))*1000
        totcost = overfullcost + emptycost + buffercost + mitigationcost
        rate=scen.rate
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

if __name__=="__main__":
    import fmdtools.sim.propagate as propagate
    import fmdtools.analyze as an
    from fmdtools.sim.approach import SampleApproach
    mdl=Tank()
    
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass','fxnflowgraph'])
    an.plot.mdlhists(mdlhist, fxnflowvals={'fxns':'store_coolant'})

    
    ## faulty run
    resgraph, mdlhist = propagate.one_fault(mdl,'export_coolant','blockage', time=2, desired_result='fxnflowgraph')
    
    an.plot.mdlhists(mdlhist, title='NotVisible', fxnflowvals={'fxns':'store_coolant'}, time_slice=2)
    an.graph.show(resgraph,faultscen='NotVisible', time=2)
    
        
        
        