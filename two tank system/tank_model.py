# -*- coding: utf-8 -*-
"""
Dynamical implementation of a human-operated tank system to show how fmdtools
can be used to model human errors.

The functions of the system are:
    - ImportLiquid (Inlet Valve)
    - GuideLiquid (Inlet Pipe)
    - StoreLiquid (Tank)
    - GuideLiquid (Outlet Pipe)
    - Export Liquid (Outlet Valve)
The Tank stores a set amount of water, the level of which is controlled by 
inlet and outlet valves. In this model we (will) use an action sequence graph
to model the human interactions with the system.

For more information on this system, see:
    
Irshad, L., Ahmed, S., Demirel, O., & Tumer, I. Y. (2018). Identification of 
human errors during early design stage functional failure analysis. In ASME 
2018 International Design Engineering Technical Conferences and Computers and 
Information in Engineering Conference. American Society of Mechanical Engineers 
Digital Collection.
"""

from fmdtools.modeldef import Model, FxnBlock


class ImportLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watout'],flows,{'open':1.0})
        self.assoc_modes({'Stuck'}) # need to add human-induced?
    def behavior(self,time):
        #if self.time > 10.0: self.open = 0.0
        self.Watout.effort=self.open
        

class ExportLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watin'],flows,{'open':1})
        self.assoc_modes({'Stuck'}) # need to add human-induced?
    def behavior(self,time):
        if self.time > 10.0: self.open = 0.0
        self.Watin.rate=self.open*self.Watin.effort
    
    
class GuideLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watin','Watout'],flows)
        self.assoc_modes({'Leak', 'Clogged'})
    def behavior(self,time):
        if self.has_fault('Clogged'):
            self.Watin.rate=0.0
            self.Watout.effort=0.0
        elif self.has_fault('Leak'):
            self.Watout.effort = self.Watin.effort - 1.0
            self.Watin.rate = self.Watout.rate - 1.0
        else:
            self.Watout.effort = self.Watin.effort
            self.Watin.rate = self.Watout.rate     

    
class StoreLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watin','Watout'],flows, {'level':5.0, 'net_flow': 0.0})
        self.assoc_modes({'Leak'})
    def behavior(self, time):
        if self.level >= 10.0:
            self.Watin.rate = 0.0 * self.Watin.effort
            self.Watout.effort = 2.0 * self.Watin.effort
            self.level = 10.0
        elif self.level <=0.0:
            self.Watout.effort = 0.0
            self.Watin.rate = self.Watin.effort
        else:
            self.Watin.rate = self.Watin.effort
            self.Watout.effort = 1.0
        
        if self.has_fault('Leak'):  self.net_flow = self.Watin.rate - self.Watout.rate - 1.0
        else:                       self.net_flow = self.Watin.rate - self.Watout.rate
        
        if time>self.time:          self.level = self.level + self.net_flow
            
class Tank(Model):
    def __init__(self, params={}):
        super().__init__(modelparams = {'phases':{'na':[0,20]}, 'times':[0,20], 'tstep':1})
        
        self.add_flow('Wat_in_1', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_in_2', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_1', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_2', 'Water', {'effort':1.0, 'rate':1.0})
        
        self.add_fxn('Import_Water', ImportLiquid, ['Wat_in_1'])
        self.add_fxn('Guide_Water_In', GuideLiquid, ['Wat_in_1', 'Wat_in_2'])
        self.add_fxn('Store_Water', StoreLiquid, ['Wat_in_2', 'Wat_out_1'])
        self.add_fxn('Guide_Water_Out', GuideLiquid, ['Wat_out_1', 'Wat_out_2'])
        self.add_fxn('Export_Water', ExportLiquid, ['Wat_out_2'])
        
        self.construct_graph()
        
        
        