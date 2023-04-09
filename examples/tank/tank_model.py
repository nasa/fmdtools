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
from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.mode import Mode
from fmdtools.define.flow import Flow
from fmdtools.define.model import Model
from fmdtools.define.block import FxnBlock, Action, ASG

class WatState(State):
    effort: float=1.0
    rate:   float=1.0
class Water(Flow):
    _init_s=WatState

class SigState(State):
    indicator:  int=1
    action:     int=0
class Signal(Flow):
    _init_s = SigState

class TransportLiquidState(State):
    amt_open: int=1
class TransportLiquidMode(Mode):
    faultparams={'stuck':(1e-5,[1,0],0)}
    units='hr'
    key_phases_by='global'
class ImportLiquid(FxnBlock):
    __slots__=('sig', 'watout')
    _init_s = TransportLiquidState
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_watout = Water
    flownames = {'wat_in_1':'watout', 'valve1_sig':'sig'}
    def static_behavior(self,time):
        if not self.m.has_fault('Stuck'):
            if   self.sig.s.action>=2:    self.s.amt_open=2
            elif self.sig.s.action==1:    self.s.amt_open = 1
            elif self.sig.s.action==-1:   self.s.amt_open = 0
        self.watout.s.effort=self.s.amt_open
        self.sig.s.indicator = self.s.amt_open

class ExportLiquid(FxnBlock):
    __slots__=('sig', 'watin')
    _init_s = TransportLiquidState 
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_watin = Water
    flownames = {'wat_out_2':'watin', 'valve2_sig':'sig'}
    def static_behavior(self,time):
        if not self.m.has_fault('Stuck'):
            if self.sig.s.action>=1:      self.s.open = 1
            elif self.sig.s.action==-1:   self.s.open = 0
        self.watin.s.rate=self.s.amt_open*self.watin.s.effort
        self.sig.s.indicator = self.s.amt_open

class GuideLiquidMode(Mode):
    faultparams = {'Leak':(1e-5,[1,0],0), 
                   'Clogged':(1e-5,[1,0],0)}
    key_phases_by='global'
class GuideLiquid(FxnBlock):
    __slots__=('watin', 'watout')
    _init_watin=Water
    _init_watout=Water 
    def static_behavior(self,time):
        if self.m.has_fault('Clogged'):
            self.watin.s.put(rate=0.0,effort=0.0)
        elif self.m.has_fault('Leak'):
            self.watout.s.effort = self.watin.s.effort - 1.0
            self.watin.s.rate = self.watout.s.rate - 1.0
        else:
            self.watout.s.effort = self.watin.s.effort
            self.watin.s.rate = self.watout.s.rate
class GuideLiquidIn(GuideLiquid):
    flownames = {'wat_in_1':'watin', 'wat_in_2':'watout'}
class GuideLiquidOut(GuideLiquid):
    flownames = {'wat_out_1':'watin', 'wat_out_2':'watout'}

class StoreLiquidState(State):
    level:      float=10.0
    net_flow:   float=0.0
class StoreLiquidMode(Mode):
    faultparams={'leak':(1e-5,[1,0],0)}
    key_phases_by='global'
class StoreLiquid(FxnBlock):
    __slots__=('watin', 'watout', 'sig')
    _init_s = StoreLiquidState
    _init_m = StoreLiquidMode
    _init_watin = Water
    _init_watout = Water
    _init_sig = Signal
    flownames = {'wat_in_2':'watin', 'wat_out_1':'watout', 'tank_sig':'sig'}
    def static_behavior(self, time):
        if self.s.level >= 20.0:
            self.watin.s.rate = 0.0 * self.watin.s.effort
            self.watout.s.effort = 2.0 * self.watin.s.effort
            self.s.level = 20.0
        elif self.s.level <=0.0:
            self.watout.s.effort = 0.0
            self.watin.s.rate = self.watin.s.effort
        else: 
            self.watin.s.put(rate=self.watin.s.effort, effort = 1.0)
        if self.s.level > 12:   self.sig.s.indicator = -1
        elif self.s.level < 8:  self.sig.s.indicator = 1
        else:                   self.sig.s.indicator = 0
        
        if self.m.has_fault('leak'):    
            self.s.net_flow = self.watin.s.rate - self.watout.s.rate - 1.0
        else:                           
            self.s.net_flow = self.watin.s.rate - self.watout.s.rate
    def dynamic_behavior(self,time):
        self.s.inc(level=self.s.net_flow*self.t.dt)
        self.s.limit(level=(0.0, 25))
        #self.s.level = self.s.level + self.s.net_flow*self.dt

class HumanParam(Parameter):
    reacttime: int=1
class HumanASG(ASG):
    reacttime:  float=0.0
    _init_tank_sig = Signal
    _init_valve1_sig = Signal 
    _init_valve2_sig = Signal
    _init_detect_sig = Signal
    initial_action = "look"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_act('look',  Look)
        self.add_act('detect',Detect, 'detect_sig','tank_sig','valve1_sig', duration=self.reacttime)
        self.add_act('reach', Reach)
        self.add_act('grasp', Grasp)
        self.add_act('turn',  Turn, 'detect_sig','valve1_sig', 'valve2_sig', duration=1.0)
        
        self.add_cond('look', 'detect', 'looked',    condition=self.actions['look'].looked)
        self.add_cond('detect', 'reach', 'detected', condition=self.actions['detect'].detected)
        self.add_cond('reach', 'grasp', 'reached',   condition=self.actions['reach'].reached)
        self.add_cond('grasp', 'turn', 'grasped',    condition=self.actions['grasp'].grasped)
        self.add_cond('turn', 'look', 'done',        condition=self.actions['turn'].turned)
class HumanActions(FxnBlock): 
    _init_p = HumanParam
    _init_a = HumanASG
    _init_valve1_sig = Signal
    _init_tank_sig = Signal
    _init_valve2_sig = Signal

class LookMode(Mode):
    faultparams={'not_visible':(1,[1,0],0)}
    proptype='prob'
    he_args=(0.02, [[4,0.1],[4,0.6],[1.1,0.9]]) #using lists as inputs leaves the EPCs unlabeled
class Look(Action):
    _init_m=LookMode
    def looked(self):
        return not self.m.has_fault('not_visible')

class DetectMode(Mode):
    faultparams = {'not_detected':(1,[1,0],0),
                   'false_high':(1,[1,1],0),
                   'false_low': (1,[1,1],0)}
    probtype='prob'
    he_args = (0.03, {2:[11,0.1],10:[10,0.2],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0.6]})
class Detect(Action):
    _init_m = DetectMode
    _init_detect_sig = Signal
    _init_tank_sig = Signal
    _init_valve1_sig = Signal
    def behavior(self, time):
        if self.m.has_fault('not_detected'):    self.detect_sig.s.put(indicator = 0, action=0)
        elif self.m.has_fault('false_high'):    self.detect_sig.s.put(indicator = 1, action=-1)
        elif self.m.has_fault('false_low'):     self.detect_sig.s.put(indicator = -1, action=2)
        else:                                   
            self.detect_sig.s.indicator = self.tank_sig.s.indicator
            if self.detect_sig.s.indicator >= 1: 
                self.detect_sig.s.action=2
            elif self.detect_sig.s.indicator <= -1:
                self.detect_sig.s.action=-1
            else:
                self.detect_sig.s.action=1
    def detected(self):
        return self.detect_sig.s.indicator

class ReachMode(Mode):
    faultparams = {'unable':(0.5,[1,0],0)}
    probtype='prob'
    he_args = (0.09, {2:[11,0.1],10:[10,0.0],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0]})
class Reach(Action):
    _init_m = ReachMode
    def reached(self):
        return not self.m.has_fault('unable')

class GraspMode(Mode):
    faultparams = {'cannot':(1,[1,0],0)}
    probtype = 'prob'
    failrate=0.02
class Grasp(Action):
    _init_m = GraspMode
    def grasped(self):
        return not self.m.has_fault('cannot')

class TurnMode(Mode):
    faultparams = {'cannot':(1,[1,0],0), 'wrong_valve':(0.5,[1,0],0)}
    probtype = 'prob'
    he_args = (0.009, {2:[11,0.4],10:[10,0.2],13:[4,0],14:[4,0],17:[3,0.6],34:[1.1,0]})
class Turn(Action):
    _init_m = TurnMode
    _init_detect_sig = Signal
    _init_valve1_sig = Signal
    _init_valve2_sig = Signal
    def behavior(self, time):
        if self.m.has_fault('cannot'):  turned =  0
        else:                           turned = 1
        if turned and self.m.has_fault('wrong_valve'):
            self.valve2_sig.s.assign(self.detect_sig.s, 'action')
        elif turned:
            self.valve1_sig.s.assign(self.detect_sig.s, 'action')
    def turned(self):
        return not self.m.has_fault('cannot')
    
class TankParam(Parameter, readonly=True):
    reacttime:      int = 2
    store_tstep:    float = 1.0

class Tank(Model):
    __slots__=()
    _init_p = TankParam
    default_sp = dict(phases=(('na',0,0),('operation',1,20)),times=(0,5,10,15,20),units='min')
    default_track = {'fxns':{'store_water':{'s':'level'}}}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_flow('wat_in_1',  Water)
        self.add_flow('wat_in_2',  Water)
        self.add_flow('wat_out_1', Water)
        self.add_flow('wat_out_2', Water)
        self.add_flow('valve1_sig',Signal, s={'indicator':1, 'action':0})
        self.add_flow('tank_sig',  Signal, s={'indicator':0, 'action':0})
        self.add_flow('valve2_sig',Signal, s={'indicator':1, 'action':0})
        
        self.add_fxn('import_water',    ImportLiquid,  'wat_in_1', 'valve1_sig')
        self.add_fxn('guide_water_in',  GuideLiquidIn, 'wat_in_1', 'wat_in_2')
        self.add_fxn('store_water',     StoreLiquid,   'wat_in_2', 'wat_out_1', 'tank_sig')
        self.add_fxn('guide_water_out', GuideLiquidOut,'wat_out_1','wat_out_2')
        self.add_fxn('export_water',    ExportLiquid,  'wat_out_2','valve2_sig')
        self.add_fxn('human',           HumanActions,  'valve1_sig','tank_sig', 'valve2_sig', a={'reacttime':self.p.reacttime})
        
        self.build()
    def find_classification(self, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        if any(self.h.fxns.store_water.s.level>=20):    totcost = 1000000
        elif any(self.h.fxns.store_water.s.level<=0):   totcost = 1000000
        else:                                           totcost = 0
        rate=scen['properties'].get('rate',0.0)
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

if __name__ == '__main__':
    import fmdtools.sim.propagate as propagate
    import fmdtools.analyze as an
    from fmdtools.sim.approach import SampleApproach
    
    mdl = Tank()
    
    endclass, mdlhist = propagate.one_fault(mdl,'human','look_not_visible', time=2)
    
    ## nominal run
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass','fxnflowgraph'])
    an.plot.mdlhists(mdlhist, fxnflowvals={'fxns':'store_water'})
    an.graph.show(endresults['fxnflowgraph'])
    
    
    ## faulty run
    resgraph, mdlhist = propagate.one_fault(mdl,'store_water','leak', time=2, desired_result='fxnflowgraph')
    an.plot.mdlhists(mdlhist, title='Leak Response', fxnflowvals={'fxns':'store_water'}, time_slice=2)
    an.graph.show(resgraph,faultscen='leak response', time='end')
    
    
    resgraph, mdlhist = propagate.one_fault(mdl,'human','detect_false_high', time=2, desired_result='fxnflowgraph')
    
    an.plot.mdlhists(mdlhist, title='detect_false_high', fxnflowvals={'fxns':'store_water'}, time_slice=2)
    an.graph.show(resgraph,faultscen='detect_false_high', time=2)
    
    resgraph, mdlhist = propagate.one_fault(mdl,'human','turn_wrong_valve', time=2, desired_result='fxnflowgraph')
    
    an.plot.mdlhists(mdlhist,title='turn_wrong_valve', fxnflowvals={'fxns':'store_water'}, time_slice=2)
    an.graph.show(resgraph,faultscen='turn_wrong_valve', time=2)
    
    
    mdl = Tank(p=TankParam(reacttime=2), sp = dict(dt=3.0))
    resgraph, mdlhist = propagate.one_fault(mdl,'store_water','leak', time=2, desired_result='fxnflowgraph')
    an.plot.mdlhists(mdlhist, title='Leak Response', fxnflowvals={'fxns':'store_water'}, time_slice=2)
    an.graph.show(resgraph,faultscen='turn_wrong_valve', time='end')
    
    ## run all faults - note: all faults get caught!
    endclasses, mdlhists = propagate.single_faults(mdl)
    
    app_full = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app_full)
    
    mdl.fxns['human'].dt=2.0
    an.graph.exec_order(mdl, renderer='graphviz')
    an.graph.exec_order(mdl, show_dyn_tstep=False)
    an.graph.exec_order(mdl, show_dyn_order=False)
    an.graph.exec_order(mdl, show_dyn_order=False, show_dyn_tstep=False)
    
    a = HumanASG("hi")
    a.show()
             