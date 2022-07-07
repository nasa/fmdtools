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
import sys, os
sys.path.insert(0, os.path.join('..'))
from fmdtools.modeldef import Model, FxnBlock, Component


class ImportLiquid(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['Watout', 'Sig'],{'open':1})
        self.assoc_modes({'Stuck':[1e-5,[1,0],0]}, units='hr', key_phases_by='global') # need to add human-induced?
    def static_behavior(self,time):
        if not self.has_fault('Stuck'):
            if   self.Sig.action>=2:    self.open=2
            elif self.Sig.action==1:      self.open = 1
            elif self.Sig.action==-1:   self.open = 0
        self.Watout.effort=self.open
        self.Sig.indicator = self.open
        

class ExportLiquid(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name,flows,['Watin', 'Sig'],{'open':1})
        self.assoc_modes({'Stuck':[1e-5,[1,0],0]}, key_phases_by='global') # need to add human-induced?
    def static_behavior(self,time):
        if not self.has_fault('Stuck'):
            if self.Sig.action==1:      self.open = 1
            elif self.Sig.action==-1:   self.open = 0
        self.Watin.rate=self.open*self.Watin.effort
        self.Sig.indicator = self.open
    
    
class GuideLiquid(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['Watin','Watout'])
        self.assoc_modes({'Leak':[1e-5,[1,0],0], 'Clogged':[1e-5,[1,0],0]}, key_phases_by='global')
    def static_behavior(self,time):
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
    def __init__(self,name,flows, dt):
        super().__init__(name,flows,['Watin','Watout', 'Sig'], {'level':10.0, 'net_flow': 0.0}, dt=dt)
        self.assoc_modes({'Leak':[1e-5,[1,0],0]}, key_phases_by='global')
    def static_behavior(self, time):
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
    def dynamic_behavior(self,time):
        self.inc(level=self.net_flow*self.dt)
        #self.level = self.level + self.net_flow*self.dt
        
class HumanActions(FxnBlock):
    def __init__(self,name, flows, reacttime):
        actions = {'look':Look('look'),'detect':Detect('detect'),\
                   'reach':Reach('reach'), 'grasp':Grasp('grasp'),\
                   'turn':Turn('turn')}
        super().__init__(name, flows,['Valve_Sig','Tank_Sig', 'OtherValve_Sig'], components=actions,timers={'t1'})
        self.assoc_modes({'FalseDetection_low':[1e-4,[1,1],0],'FalseDetection_high':[1e-4,[1,1],0]}, probtype='rate', key_phases_by='global')
        self.reacttime=reacttime
    def dynamic_behavior(self,time):        
        if self.t1.time == 0:
            self.looked = self.look.behavior()
            self.detected = self.detect.behavior(self.looked,self.Tank_Sig.indicator)
            if self.detected or self.has_fault('FalseDetection_low') or self.has_fault('FalseDetection_high'): 
                self.t1.inc(1)
        elif self.t1.time < self.reacttime: self.t1.inc(1)
        elif self.t1.time >= self.reacttime:
            
            if self.Tank_Sig.indicator == 1 or self.has_fault('FalseDetection_low'):
                if self.Valve_Sig.indicator >= 1:   intended_turn = 2
                elif self.Valve_Sig.indicator == 0: intended_turn = 1
            elif self.Tank_Sig.indicator == -1 or self.has_fault('FalseDetection_high'):
                if self.Valve_Sig.indicator >= 2:   intended_turn = 1
                else:                               intended_turn = -1
            else: intended_turn = 0
            
            self.reached = self.reach.behavior()
            self.grasped = self.grasp.behavior()
            self.turned  = self.turn.behavior(self.grasp, intended_turn)
            if self.reached[0]:
                self.Valve_Sig.action = self.turned
            if self.reached[1]:
                self.OtherValve_Sig.action = self.turned
            self.t1.reset()
            self.remove_fault('FalseDetection_low')
            self.remove_fault('FalseDetection_high')
            
class Look(Component):
    def __init__(self, name):
        super().__init__(name)
        self.add_he_rate(0.02, EPCs=[[4,0.1],[4,0.6],[1.1,0.9]]) #using lists as inputs leaves the EPCs unlabeled
        self.assoc_modes({'NotVisible':[1,[1,0],0]}, probtype='prob', key_phases_by='global')
    def behavior(self):
        if self.has_fault('NotVisible'):    return 0
        else:                               return 1
class Detect(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.03, EPCs={2:[11,0.1],10:[10,0.2],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0.6]})
        self.assoc_modes({'NotDetected':[1,[1,0],0]}, probtype='prob', key_phases_by='global') # add failed detection, etc modes
    def behavior(self, look, signal):
        if self.has_fault('NotDetected'):   detect = 0
        else:                               detect = look * signal
        return detect
class Reach(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.09, EPCs={2:[11,0.1],10:[10,0.0],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0]})
        self.assoc_modes({'FalseReach':[0.5,[1,0],0], 'CannotReach':[0.5,[1,0],0]}, probtype='prob', key_phases_by='global')
    def behavior(self):
        if self.has_fault('CannotReach'):   return 0,0
        elif self.has_fault('FalseReach'):  return 0,1
        else:                               return 1,0
class Grasp(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.02) #in the case with no EPCs, we can just leave it out
        self.assoc_modes({'CannotGrasp':[1,[1,0],0]}, probtype='prob', key_phases_by='global')
    def behavior(self):
        if self.has_fault('CannotGrasp'):   return 0
        else:                               return 1
class Turn(Component):
    def __init__(self,name):
        super().__init__(name)
        self.add_he_rate(0.009, EPCs={2:[11,0.4],10:[10,0.2],13:[4,0],14:[4,0],17:[3,0.6],34:[1.1,0]})
        self.assoc_modes({'CannotTurn':[1,[1,0],0]}, probtype='prob', key_phases_by='global')
    def behavior(self,grasp, intended_turn):
        if self.has_fault('CannotTurn') or grasp == 0:  return 0
        else:                                           return intended_turn
        
class Tank(Model):
    def __init__(self, params={'reacttime':2, 'store_tstep':1.0},\
                 modelparams = {'phases':{'na':[0],'operation':[1,20]}, 'times':[0,5,10,15,20], 'tstep':1, 'units':'min'},\
                 valparams = {'functions':{'Store_Water':'level'}}):
        super().__init__(params = params,modelparams=modelparams, valparams=valparams )
        
        self.add_flow('Wat_in_1', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_in_2', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_1', {'effort':1.0, 'rate':1.0})
        self.add_flow('Wat_out_2',{'effort':1.0, 'rate':1.0})
        self.add_flow('Valve1_Sig', {'indicator':1, 'action':0})
        self.add_flow('Tank_Sig', {'indicator':0, 'action':0})
        self.add_flow('Valve2_Sig', {'indicator':1, 'action':0})
        
        self.add_fxn('Import_Water', ['Wat_in_1', 'Valve1_Sig'], fclass = ImportLiquid)
        self.add_fxn('Guide_Water_In', ['Wat_in_1', 'Wat_in_2'], fclass = GuideLiquid)
        self.add_fxn('Store_Water', ['Wat_in_2', 'Wat_out_1', 'Tank_Sig'], fclass = StoreLiquid, fparams=params['store_tstep'])
        self.add_fxn('Guide_Water_Out', ['Wat_out_1', 'Wat_out_2'], fclass =GuideLiquid)
        self.add_fxn('Export_Water', ['Wat_out_2', 'Valve2_Sig'], fclass =ExportLiquid)
        self.add_fxn('Human', ['Valve1_Sig', 'Tank_Sig', 'Valve2_Sig'], fclass =HumanActions, fparams = params['reacttime'])
        
        self.build_model()
    def find_classification(self, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        if any(mdlhists['faulty']['functions']['Store_Water']['level']>=20):    totcost = 1000000
        elif any(mdlhists['faulty']['functions']['Store_Water']['level']<=0):   totcost = 1000000
        else:                                                                   totcost = 0
        rate=scen['properties'].get('rate',0.0)
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

if __name__ == '__main__':
    import fmdtools.faultsim.propagate as propagate
    import fmdtools.resultdisp as rd
    from fmdtools.modeldef import SampleApproach
    
    mdl = Tank()
    
    ## nominal run
    endresults, resgraph, mdlhist = propagate.nominal(mdl)
    rd.plot.mdlhists(mdlhist, fxnflowvals='Store_Water')
    rd.graph.show(resgraph)
    
    
    ## faulty run
    endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Human','NotVisible', time=2)
    
    rd.plot.mdlhists(mdlhist, title='NotVisible', fxnflowvals='Store_Water', time_slice=2)
    rd.graph.show(resgraph,faultscen='NotVisible', time=2)
    
    endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Human','FalseReach', time=2, gtype='component')
    
    rd.plot.mdlhists(mdlhist,title='FalseReach', fxnflowvals='Store_Water', time_slice=2)
    rd.graph.show(resgraph,gtype='component',faultscen='FalseReach', time=2)
    
    
    mdl = Tank(params={'reacttime':2, 'store_tstep':3.0})
    endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Store_Water','Leak', time=2)
    rd.plot.mdlhists(mdlhist, title='Leak Response', fxnflowvals='Store_Water', time_slice=2)
    
    ## run all faults - note: all faults get caught!
    endclasses, mdlhists = propagate.single_faults(mdl)
    
    app_full = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app_full)
    
    mdl.fxns['Human'].dt=2.0
    rd.graph.exec_order(mdl, renderer='graphviz')
    rd.graph.exec_order(mdl, show_dyn_tstep=False)
    rd.graph.exec_order(mdl, show_dyn_order=False)
    rd.graph.exec_order(mdl, show_dyn_order=False, show_dyn_tstep=False)
             