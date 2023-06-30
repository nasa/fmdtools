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
    effort: float = 1.0
    rate: float = 1.0
class Liquid(Flow):
    _init_s=WatState

class SigState(State):
    indicator: int = 1
    action: int = 0
class Signal(Flow):
    _init_s = SigState

class TransportLiquidState(State):
    amt_open: int = 1
class TransportLiquidMode(Mode):
    faultparams={'stuck':(1e-5,[1,0],0)}
    units='hr'
    key_phases_by='global'
class ImportLiquid(FxnBlock):
    __slots__=('sig', 'watout')
    _init_s = TransportLiquidState
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_watout = Liquid
    flownames = {'wat_in_1':'watout', 'valve1_sig':'sig'}
    def static_behavior(self,time):
        if not self.m.has_fault('stuck'):
            if   self.sig.s.action>=2:    
                self.s.amt_open = 2
            elif self.sig.s.action==1:    
                self.s.amt_open = 1
            elif self.sig.s.action==-1:   
                self.s.amt_open = 0
        self.watout.s.effort=self.s.amt_open
        self.sig.s.indicator = self.s.amt_open

class ExportLiquid(FxnBlock):
    __slots__=('sig', 'watin')
    _init_s = TransportLiquidState 
    _init_m = TransportLiquidMode
    _init_sig = Signal
    _init_watin = Liquid
    flownames = {'wat_out_2':'watin', 'valve2_sig':'sig'}
    def static_behavior(self,time):
        if not self.m.has_fault('stuck'):
            if self.sig.s.action>=1:      
                self.s.amt_open = 1
            elif self.sig.s.action==-1:   
                self.s.amt_open = 0
        self.watin.s.rate=self.s.amt_open*self.watin.s.effort
        self.sig.s.indicator = self.s.amt_open

class GuideLiquidMode(Mode):
    faultparams = {'leak':(1e-5,[1,0],0), 
                   'clogged':(1e-5,[1,0],0)}
    key_phases_by = 'global'
class GuideLiquid(FxnBlock):
    __slots__=('watin', 'watout')
    _init_watin=Liquid
    _init_watout=Liquid 
    _init_m = GuideLiquidMode
    def static_behavior(self,time):      
        if self.m.has_fault('clogged'):
            self.watin.s.put(rate=0.0)
            self.watout.s.put(effort=0.0)
        elif self.m.has_fault('leak'):
            self.watout.s.effort = self.watin.s.effort - 1.0
            self.watin.s.rate = self.watout.s.rate - 1.0
        else:
            self.watout.s.effort = self.watin.s.effort
            self.watin.s.rate = self.watout.s.rate
class GuideLiquidIn(GuideLiquid):
    __slots__=()
    flownames = {'wat_in_1':'watin', 'wat_in_2':'watout'}
class GuideLiquidOut(GuideLiquid):
    __slots__=()
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
    _init_watin = Liquid
    _init_watout = Liquid
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
        if self.s.level > 12:  
            self.sig.s.indicator = -1
        elif self.s.level < 8:
            self.sig.s.indicator = 1
        else:
            self.sig.s.indicator = 0
        
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
    initial_action = "look"
    def __init__(self, *args, reacttime=0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.add_flow("tank_sig", Signal)
        self.add_flow("valve1_sig", Signal)
        self.add_flow("valve2_sig", Signal)
        self.add_flow("detect_sig", Signal)
        
        self.add_act('look', Look)
        self.add_act('detect', Detect, 'detect_sig', 'tank_sig', duration=reacttime)
        self.add_act('reach', Reach)
        self.add_act('grasp', Grasp)
        self.add_act('turn', Turn, 'detect_sig', 'valve1_sig', 'valve2_sig', duration=1.0)
        
        self.add_cond('look', 'detect', 'looked', condition=self.actions['look'].looked)
        self.add_cond('detect', 'reach', 'detected', condition=self.actions['detect'].detected)
        self.add_cond('reach', 'grasp', 'reached', condition=self.actions['reach'].reached)
        self.add_cond('grasp', 'turn', 'grasped', condition=self.actions['grasp'].grasped)
        self.add_cond('turn', 'look', 'done', condition=self.actions['turn'].turned)
        
        self.build()
class HumanActions(FxnBlock): 
    _init_p = HumanParam
    _init_a = HumanASG
    _init_valve1_sig = Signal
    _init_tank_sig = Signal
    _init_valve2_sig = Signal
    def dynamic_behavior(self, time):
        #if self.valve1_sig.s.indicator:
            #print(self.a.flows['valve1_sig'].s.indicator==self.valve1_sig.s.indicator, flush=True)
            #print(self.a.flows['valve2_sig'].s.indicator==self.valve2_sig.s.indicator, flush=True)
        
        #assert self.a.actions['turn'].valve1_sig.s.indicator==self.valve1_sig.s.indicator
        #assert self.a.actions['turn'].valve2_sig.s.indicator==self.valve2_sig.s.indicator
        
        if self.a.actions['look'].looked.__self__.__hash__()!=self.a.conditions['looked'].__self__.__hash__():
            raise Exception("Condition not passed")
        
        if self.m.has_fault("detect_false_high") and time==5.0 and not self.h.m.faults.detect_false_high[4]:
            if 'turn' not in self.a.active_actions:
                raise Exception("Invalid behavior, detect.t.t_loc="+str(self.a.actions['detect'].t.t_loc))
            print(self.a.actions['detect'].t.t_loc, flush=True)
            print(self.a.active_actions)
            
            if not self.a.actions['detect'].m.has_fault("false_high"):
                raise Exception("detect_false_high")
        
        if self.a.flows['valve1_sig'].__hash__()!=self.valve1_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.a.actions['detect'].tank_sig.__hash__()!=self.tank_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.a.flows['detect_sig'].__hash__()!=self.a.actions['detect'].detect_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.a.flows['valve2_sig'].__hash__()!=self.valve2_sig.__hash__():
            raise Exception("Invalid connection hash in asg.flows")
        if self.a.actions['turn'].valve2_sig.__hash__()!=self.valve2_sig.__hash__():
            raise Exception("Invalid connection hash")
        
        if not  self.a.actions['turn'].valve2_sig.s.action==self.valve2_sig.s.action:
            raise Exception("invalid connection: valve2_sig")
        
        if not  self.a.actions['turn'].valve1_sig.s.action==self.valve1_sig.s.action:
            raise Exception("invalid connection: valve1_sig")

class LookMode(Mode):
    faultparams={'not_visible':(1,[1,0],0)}
    proptype='prob'
    he_args=(0.02, [[4,0.1],[4,0.6],[1.1,0.9]]) #using lists as inputs leaves the EPCs unlabeled
class Look(Action):
    _init_m=LookMode
    def looked(self):
        return not self.m.has_fault('not_visible')

class DetectMode(Mode):
    faultparams = {'not_detected': (1,[1,0],0),
                   'false_high': (1,[1,1],0),
                   'false_low': (1,[1,1],0)}
    probtype='prob'
    he_args = (0.03, {2:[11,0.1],10:[10,0.2],13:[4,0],14:[4,0.1],17:[3,0],34:[1.1,0.6]})
class Detect(Action):
    _init_m = DetectMode
    _init_detect_sig = Signal
    _init_tank_sig = Signal
    def behavior(self, time):
        if self.m.has_fault('not_detected'):    
            self.detect_sig.s.put(indicator = 0, action=0)
        elif self.m.has_fault('false_high'):    
            self.detect_sig.s.put(indicator = 1, action=-1)
        elif self.m.has_fault('false_low'):     
            self.detect_sig.s.put(indicator = -1, action=2)
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
        if self.m.has_fault('cannot'):
            turned =  0
        else:
            turned = 1
        if turned and self.m.has_fault('wrong_valve'):
            self.valve2_sig.s.assign(self.detect_sig.s, 'action')
        elif turned:
            self.valve1_sig.s.assign(self.detect_sig.s, 'action')
    def turned(self):
        return not self.m.has_fault('cannot')
    
class TankParam(Parameter, readonly=True):
    reacttime: int = 2
    store_tstep: float = 1.0

class Tank(Model):
    __slots__=()
    _init_p = TankParam
    default_sp = dict(phases=(('na',0,0),('operation',1,20)),times=(0,5,10,15,20),units='min')
    default_track = {'fxns':{'store_water':{'s':'level'}}}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_flow('wat_in_1', Liquid)
        self.add_flow('wat_in_2', Liquid)
        self.add_flow('wat_out_1', Liquid)
        self.add_flow('wat_out_2', Liquid)
        self.add_flow('valve1_sig',Signal, s={'indicator':1, 'action':0}) # Need to make sure this info is passed during copy!!
        self.add_flow('tank_sig',  Signal, s={'indicator':0, 'action':0})
        self.add_flow('valve2_sig',Signal, s={'indicator':1, 'action':0})
        
        self.add_fxn('import_water', ImportLiquid, 'wat_in_1', 'valve1_sig')
        self.add_fxn('guide_water_in', GuideLiquidIn, 'wat_in_1', 'wat_in_2')
        self.add_fxn('store_water', StoreLiquid, 'wat_in_2', 'wat_out_1', 'tank_sig')
        self.add_fxn('guide_water_out', GuideLiquidOut,'wat_out_1','wat_out_2')
        self.add_fxn('export_water', ExportLiquid, 'wat_out_2','valve2_sig')
        self.add_fxn('human', HumanActions, 'valve1_sig','tank_sig', 'valve2_sig', a={'reacttime':self.p.reacttime})
        
        self.build()
    def find_classification(self, scen, hist):
        # here we define failure in terms of the water level getting too low or too high
        if any(self.h.fxns.store_water.s.level>=20):
            totcost = 1000000
        elif any(self.h.fxns.store_water.s.level<=0):
            totcost = 1000000
        else: 
            totcost = 0
        rate=scen.rate
        life=1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': rate*life*totcost}

if __name__ == '__main__':
    import fmdtools.sim.propagate as propagate
    import fmdtools.analyze as an
    from fmdtools.sim.approach import SampleApproach
    
    mdl = Tank()
    
    app = SampleApproach(mdl)
    
    app = SampleApproach(mdl, defaultsamp={'samp':'evenspacing','numpts':4})
    import multiprocessing as mp
    print("normal")
    endclasses, mdlhists = propagate.approach(mdl, app, showprogress=False, track='all', staged=True)
    print("staged")
    endclasses_staged, mdlhists_staged = propagate.approach(mdl, app, showprogress=False, track='all', staged=True)
    
    assert endclasses==endclasses_staged
    print("parallel")
    endclasses_par, mdlhists_par = propagate.approach(mdl, app, showprogress=False,pool=mp.Pool(4), staged=False, track='all')
    
    assert endclasses==endclasses_par
    print("staged-parallel")
    endclasses_par_staged, mdlhists_par_staged = propagate.approach(mdl, app, showprogress=False,pool=mp.Pool(4), staged=True, track='all')
    
    mc_diff = mdlhists.get_different(mdlhists_par_staged)
    ec_diff = endclasses.get_different(endclasses_par_staged)
    
    mc_diff.guide_water_out_leak_t0p0.flows.wat_in_2.s.effort
    
    #mc_diff.guide_water_in_leak_t0p0.flows.wat_in_2.s.effort
    
    mc_diff.human_detect_false_low_t16p0.fxns.human.a.active_actions[16]
    
    assert endclasses==endclasses_par_staged
    
    
    """
    endclass, mdlhist = propagate.one_fault(mdl,'human','look_not_visible', time=2)
    
    ## nominal run
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass','graph'])
    an.plot.hist(mdlhist, "fxns.store_water.s.level")
    endresults.graph.draw()
    
    
    ## faulty run
    endres, mdlhist = propagate.one_fault(mdl,'store_water','leak', time=2, desired_result='graph')
    an.plot.hist(mdlhist,  "fxns.store_water.s.level",title='Leak Response', time_slice=2)
    endres.graph.draw(title="leak response at time=end")
    
    
    resgraph, mdlhist = propagate.one_fault(mdl,'human','detect_false_high', time=2, desired_result='graph')
    
    an.plot.hist(mdlhist, "fxns.store_water.s.level", title='detect_false_high', time_slice=2)
    resgraph.graph.draw(title='detect_false_high, t=2')
    
    resgraph, mdlhist = propagate.one_fault(mdl,'human','turn_wrong_valve', time=2, desired_result='graph')
    
    an.plot.hist(mdlhist, "fxns.store_water.s.level", title='turn_wrong_valve', time_slice=2)
    resgraph.graph.draw(title='turn_wrong_valve, t=2')
    
    
    mdl = Tank(p=TankParam(reacttime=2), sp = dict(dt=3.0))
    resgraph, mdlhist = propagate.one_fault(mdl,'store_water','leak', time=2, desired_result='graph')
    an.plot.hist(mdlhist, "fxns.store_water.s.level", title='Leak Response', time_slice=2)
    resgraph.graph.draw(title='turn_wrong_valve, t=end')
    
    ## run all faults - note: all faults get caught!
    endclasses, hist = propagate.single_faults(mdl)
    
    app_full = SampleApproach(mdl)
    endclasses, hist = propagate.approach(mdl, app_full)
    
    from fmdtools.analyze.graph import ModelGraph
    mdl.fxns['human'].t.dt=2.0
    mg = ModelGraph(mdl)
    mg.set_exec_order(mdl)
    mg.draw()
    
    from fmdtools.analyze.graph import ASGGraph
    ag = ASGGraph(mdl.fxns['human'].a)
    ag.draw()
    """
             