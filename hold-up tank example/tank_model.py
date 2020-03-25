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

from fmdtools.modeldef import Model, FxnBlock, Component


class ImportLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watout', 'Sig'],flows,{'open':1})
        self.assoc_modes({'Stuck':[1e-5,[1,1],0]}, units='hr') # need to add human-induced?
    def behavior(self,time):
        if not self.has_fault('Stuck'):
            if   self.Sig.action>=2:    self.open=2
            elif self.Sig.action==1:      self.open = 1
            elif self.Sig.action==-1:   self.open = 0
        self.Watout.effort=self.open
        self.Sig.indicator = self.open
        

class ExportLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watin', 'Sig'],flows,{'open':1})
        self.assoc_modes({'Stuck':[1e-5,[1,1],0]}) # need to add human-induced?
    def behavior(self,time):
        if not self.has_fault('Stuck'):
            if self.Sig.action==1:      self.open = 1
            elif self.Sig.action==-1:   self.open = 0
        self.Watin.rate=self.open*self.Watin.effort
        self.Sig.indicator = self.open
    
    
class GuideLiquid(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Watin','Watout'],flows)
        self.assoc_modes({'Leak':[1e-5,[1,1],0], 'Clogged':[1e-5,[1,1],0]})
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
        super().__init__(['Watin','Watout', 'Sig'],flows, {'level':10.0, 'net_flow': 0.0})
        self.assoc_modes({'Leak':[1e-5,[1,1],0]})
    def behavior(self, time):
        if self.level >= 20.0:
            self.Watin.rate = 0.0 * self.Watin.effort
            self.Watout.effort = 2.0 * self.Watin.effort
            self.level = 20.0
        elif self.level <=0.0:
            self.Watout.effort = 0.0
            self.Watin.rate = self.Watin.effort
        else:
            self.Watin.rate = self.Watin.effort
            self.Watout.effort = 1.0
        if self.level > 12:      self.Sig.indicator = -1
        elif self.level < 8:    self.Sig.indicator = 1
        else:                   self.Sig.indicator = 0
        
        if self.has_fault('Leak'):  self.net_flow = self.Watin.rate - self.Watout.rate - 1.0
        else:                       self.net_flow = self.Watin.rate - self.Watout.rate
        
        if time>self.time:          self.level = self.level + self.net_flow
        
class HumanActions(FxnBlock):
    def __init__(self,flows):
        actions = {'look':Look('look'),'detect':Detect('detect'),\
                   'reach':Reach('reach'), 'grasp':Grasp('grasp'),\
                   'turn':Turn('turn')}
        super().__init__(['Valve_Sig','Tank_Sig', 'OtherValve_Sig'], flows, components=actions,timers={'t1'})
        self.assoc_modes({'FalseDetection_low':[1e-4,[1,1],0],'FalseDetection_high':[1e-4,[1,1],0]}, probtype='rate')
    def behavior(self,time):        
        if time > self.time:
            if self.t1.time == 0:
                self.look = self.components['look'].behavior()
                self.detect = self.components['detect'].behavior(self.look,self.Tank_Sig.indicator)
                if self.detect or self.has_fault('FalseDetection_low') or self.has_fault('FalseDetection_high'): 
                    self.t1.inc(1)
            elif self.t1.time < 2: self.t1.inc(1)
            elif self.t1.time >=2:
                
                if self.Tank_Sig.indicator == 1 or self.has_fault('FalseDetection_low'):
                    if self.Valve_Sig.indicator >= 1:   intended_turn = 2
                    elif self.Valve_Sig.indicator == 0: intended_turn = 1
                elif self.Tank_Sig.indicator == -1 or self.has_fault('FalseDetection_high'):
                    if self.Valve_Sig.indicator >= 2:   intended_turn = 1
                    else:                               intended_turn = -1
                else: intended_turn = 0
                
                self.reach = self.components['reach'].behavior()
                self.grasp = self.components['grasp'].behavior()
                
                
                self.turn  = self.components['turn'].behavior(self.grasp, intended_turn)
                if self.reach[0]:
                    self.Valve_Sig.action = self.turn
                if self.reach[1]:
                    self.OtherValve_Sig.action = self.turn
                self.t1.reset()
                self.remove_fault('FalseDetection_low')
                self.remove_fault('FalseDetection_high')
            
class Look(Component):
    def __init__(self, name):
        super().__init__(name)
        self.add_he_rate(0.02, EPCs=[[4,0.1],[4,0.6],[1.1,0.9]]) #using lists as inputs leaves the EPCs unlabeled
        self.assoc_modes({'NotVisible':[1,[1,0],0]}, probtype='prob')
    def behavior(self):
        if self.has_fault('NotVisible'):    return 0
        else:                               return 1

class Detect(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.03, EPCs={2:[11,0.1],10:[10,0.2],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0.6]})
        self.assoc_modes({'NotDetected':[1,[1,0],0]}, probtype='prob') # add failed detection, etc modes
    def behavior(self, look, signal):
        if self.has_fault('NotDetected'):   detect = 0
        else:                               detect = look * signal
        return detect
class Reach(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.09, EPCs={2:[11,0.1],10:[10,0.0],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0]})
        self.assoc_modes({'FalseReach':[0.5,[1,0],0], 'CannotReach':[0.5,[1,0],0]}, probtype='prob')
    def behavior(self):
        if self.has_fault('CannotReach'):   return 0,0
        elif self.has_fault('FalseReach'):  return 0,1
        else:                               return 1,0
class Grasp(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.02) #in the case with no EPCs, we can just leave it out
        self.assoc_modes({'CannotGrasp':[1,[1,0],0]}, probtype='prob')
    def behavior(self):
        if self.has_fault('CannotGrasp'):   return 0
        else:                               return 1
class Turn(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.009, EPCs={2:[11,0.4],10:[10,0.2],13:[4,0],14:[4,0],17:[3,0.6],34:[1.1,0]})
        self.assoc_modes({'CannotTurn':[1,[1,0],0]}, probtype='prob')
    def behavior(self,grasp, intended_turn):
        if self.has_fault('CannotTurn') or grasp == 0:  return 0
        else:                                           return intended_turn


            
class Tank(Model):
    def __init__(self, params={}):
        super().__init__(modelparams = {'phases':{'na':[0,1],'operation':[1,20]}, 'times':[0,5,10,15,20], 'tstep':1, 'units':'min'})
        
        self.add_flow('Wat_in_1', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_in_2', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_1', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_2', 'Water', {'effort':1.0, 'rate':1.0})
        self.add_flow('Valve1_Sig', 'Signal', {'indicator':1, 'action':0})
        self.add_flow('Tank_Sig', 'Signal', {'indicator':0, 'action':0})
        self.add_flow('Valve2_Sig', 'Signal', {'indicator':1, 'action':0})
        
        self.add_fxn('Import_Water', ImportLiquid, ['Wat_in_1', 'Valve1_Sig'])
        self.add_fxn('Guide_Water_In', GuideLiquid, ['Wat_in_1', 'Wat_in_2'])
        self.add_fxn('Store_Water', StoreLiquid, ['Wat_in_2', 'Wat_out_1', 'Tank_Sig'])
        self.add_fxn('Guide_Water_Out', GuideLiquid, ['Wat_out_1', 'Wat_out_2'])
        self.add_fxn('Export_Water', ExportLiquid, ['Wat_out_2', 'Valve2_Sig'])
        self.add_fxn('Human', HumanActions, ['Valve1_Sig', 'Tank_Sig', 'Valve2_Sig'])
        
        self.construct_graph()
        
        
        