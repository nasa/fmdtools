# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019, revised Nov 2022
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.mode import Mode
from fmdtools.define.block import FxnBlock, Component, CompArch
from fmdtools.define.flow import Flow
from fmdtools.define.model import Model
from fmdtools.sim.approach import SampleApproach
from fmdtools.sim import propagate
from fmdtools.sim.search import ProblemInterface
from fmdtools.analyze.result import History

import fmdtools.analyze as an
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from examples.multirotor.drone_mdl_dynamic import finddist,vectdist, inrange
from recordclass import asdict

# DEFINE PARAMETERS
class ResPolicy(Parameter, readonly=True):
    bat :   str= 'to_home'
    bat_set = ('to_nearest', 'to_home', 'emland', 'land', 'move', 'continue')
    line:   str= 'emland'
    line_set = ('to_nearest', 'to_home', 'emland', 'land', 'move', 'continue')

class DroneParam(Parameter, readonly=True):
    """Parameters for the Drone optimization model"""
    bat:        str='monolithic'
    bat_set=('monolithic', 'series-split', 'parallel-split', 'split-both')
    linearch:   str='quad'
    linearch_set = ('quad','hex','oct')
    respolicy:  ResPolicy=ResPolicy()
    flightplan: tuple = ((0,0,0),           #flies through a few points and back to the start location
                         (0,0,100),
                         (100,0,100),
                         (100,100,100),
                         (150, 150, 100),
                         (0,0,100),
                         (0,0,0))
    start:      tuple = (0.0, 0.0, 10.0, 10.0)
    target:     tuple = (0.0, 150.0, 160.0, 160.0)
    safe:       tuple = (0.0, 50.0, 10.0, 10.0)
    batweight:  float=0.4
    archweight: float=1.2
    archdrag:   float=0.95
    loc :       str='rural'
    def __init__(self, *args, **kwargs):
        args = self.get_true_fields(*args, **kwargs)
        args[7] = {'monolithic':0.4, 'series-split':0.5, 'parallel-split':0.5, 'split-both':0.6}[args[0]]
        args[8] = {'quad':1.2, 'hex':1.6, 'oct':2.0}[args[1]]
        args[9] = {'quad':0.95, 'hex':0.85, 'oct':0.75}[args[1]]
        super().__init__(*args)
        

# DEFINE FLOWS
from examples.multirotor.drone_mdl_static import EE, Force, Control
class DOFstate(State):
    vertvel:    float=1.0
    planvel:    float=1.0
    planpwr:    float=1.0
    uppwr:      float=1.0
    x:          float=0.0
    y:          float=0.0
    z:          float=0.0
class DOFs(Flow):
    _init_s = DOFstate

class HSigState(State):
    hstate: str='nominal'
class HSig(Flow):
    _init_s = HSigState

class DesTrajState(State):
    x:      float=0.0
    y:      float=0.0
    z:      float=0.0
    power:  float=1.0
class DesTraj(Flow):
    _init_s = DesTrajState
    
class RSigState(State):
    mode:   str='continue'
class RSig(Flow):
    _init_s = RSigState

#DEFINE FUNCTIONS
from examples.multirotor.drone_mdl_static import DistEE

class BatState(State):
    soc:  float=100.0
    ee_e: float=1.0
    e_t:  float=1.0
class BatMode(Mode):
    failrate=1e-4
    faultparams = {'short':(0.2,[0.3,0.3,0.3],100),
                   'degr':(0.2,[0.3,0.3,0.3],100),
                   'break':(0.2,[0.2,0.2,0.2],100), 
                   'nocharge':(0.6,[0.6,0.2,0.2],100),
                   'lowcharge':(0,[0.6,0.2,0.2],100)}
    key_phases_by = 'plan_path'
class BatParam(Parameter):
    avail_eff:  float=0.0
    maxa:       float=0.0
    amt:        float=0.0
    weight:     float=0.1
    drag:       float=0.0
    series:     int=1
    parallel:   int=1
    voltage:    float=12.0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avail_eff=1/self.parallel
        self.maxa= 2/self.series
        self.amt=60*4.200/(self.weight*170/(self.drag*self.voltage))
class Battery(Component):
    _init_s = BatState
    _init_m = BatMode
    _init_p = BatParam
    def behavior(self, fs, ee_outr, time):
        if fs <1.0:                         self.m.add_fault('break')
        if ee_outr>self.p.maxa:             self.m.add_fault('break')
        if self.m.has_fault('short'):       self.s.e_t=0.0
        elif self.m.has_fault('break'):     self.s.e_t=0.0
        elif self.m.has_fault('degr'):      self.s.e_t=0.5*self.p.avail_eff
        else:                               self.s.e_t=self.p.avail_eff
        
        if time > self.t.time:
            self.s.inc(soc=-100*ee_outr*self.p.parallel*self.p.series*(time-self.t.time)/self.p.amt)
            self.t.time=time
        if self.s.soc<20:         self.m.add_fault('lowcharge')
        if self.s.soc<1:          
            self.m.replace_fault('lowcharge','nocharge')
            self.s.put(soc=0.0, e_t=0.0)
            er_res = ee_outr
        else: er_res=0.0
        return self.s.e_t, self.s.soc, er_res

class BatArch(CompArch):
    archtype:   str='monolithic'
    batparams:  dict={} #weight, cap, voltage, drag_factor
    weight:     float = 0.0
    drag:       float = 0.0
    series:     int =1
    parallel:   int=1
    voltage:    float=12.0
    drag:       float=0.0
    def __init__(self, *args, **kwargs):
        archtype=self.get_true_field('archtype', *args, **kwargs)
        weight = self.get_true_field('weight', *args, **kwargs)
        drag = self.get_true_field('drag', *args, **kwargs)
        if archtype=='monolithic':
            batparams = {"series":1, "parallel":1, "voltage":12.0, "weight":weight, "drag":drag}
            compnames = ['s1p1']
        elif archtype=='series-split':
            batparams ={'series':2,'parallel':1,'voltage':12.0, "weight":weight, "drag":drag}
            compnames = ['s1p1', 's2p1']
        elif archtype=='parallel-split':
            batparams ={'series':1,'parallel':2,'voltage':12.0, "weight":weight, "drag":drag}
            compnames = ['s1p1', 's1p2']
        elif archtype=='split-both':
            batparams ={'series':2,'parallel':2,'voltage':12.0, "weight":weight, "drag":drag}
            compnames = ['s1p1', 's1p2', 's2p1', 's2p2']
        else: raise Exception("Invalid battery architecture")
        kwargs.update(batparams)
        super().__init__(*args, **kwargs)
        self.make_components(Battery, *compnames, p=batparams)
from examples.multirotor.drone_mdl_static import StoreEEState
class StoreEEMode(Mode):
    failrate=1e-4
    faultparams = {'nocharge':  (0.2,[0.6,0.2,0.2],0),
                   'lowcharge': (0.7,[0.6,0.2,0.2],0)}
    key_phases_by="plan_path"

class StoreEE(FxnBlock):
    __slots__=('hsig_bat', 'ee_1', 'force_st')
    _init_s = StoreEEState
    _init_m = StoreEEMode
    _init_c = BatArch
    _init_hsig_bat = HSig
    _init_ee_1 = EE
    _init_force_st = Force
    def condfaults(self, time):
        if self.s.soc<20:                   self.m.add_fault('lowcharge')
        if self.s.soc<1:                    self.m.replace_fault('lowcharge','nocharge')
        if self.m.has_fault('lowcharge'):   
            for batname, bat in self.c.components.items(): bat.s.soc=19
        elif self.m.has_fault('nocharge'):
            for batname, bat in self.c.components.items(): bat.s.soc=0
    def behavior(self, time):
        ee, soc = {}, {}
        rate_res=0
        for batname, bat in self.c.components.items():
            ee[bat.name], soc[bat.name], rate_res = \
                bat.behavior(self.force_st.s.support, self.ee_1.s.rate/(self.c.series*self.c.parallel)+rate_res, time)
        #need to incorporate max current draw somehow + draw when reconfigured
        if self.c.archtype == 'monolithic':           self.ee_1.s.effort = ee['s1p1']
        elif self.c.archtype == 'series-split':       self.ee_1.s.effort = np.max(list(ee.values()))
        elif self.c.archtype == 'parallel-split':     self.ee_1.s.effort = np.sum(list(ee.values()))
        elif self.c.archtype == 'split-both':          
            e=list(ee.values())
            e.sort()
            self.ee_1.effort = e[-1]+e[-2]  
        self.s.soc=np.mean(list(soc.values()))
        if self.m.any_faults() and not self.m.has_fault("dummy"):   self.hsig_bat.s.hstate = 'faulty'
        else:                                                       self.hsig_bat.s.hstate = 'nominal'

class HoldPayloadMode(Mode):
    failrate=1e-6
    faultparams = {'break':(0.2, [0.33, 0.33, 0.33], 1000), 
                   'deform':(0.8, [0.33, 0.33, 0.33], 1000)} 
    key_phases_by = 'plan_path'      
class HoldPayloadState(State):  
    force_gr:   float=1.0
class HoldPayload(FxnBlock):
    __slots__=('dofs', 'force_st', 'force_lin')
    _init_m = HoldPayloadMode
    _init_s = HoldPayloadState
    _init_dofs = DOFs
    _init_force_st = Force
    _init_force_lin = Force
    def dynamic_behavior(self, time):
        if self.dofs.s.z<=0.0:  self.s.force_gr=min(-0.5, (self.dofs.s.vertvel-self.dofs.s.planvel)/(60*7.5))
        else:                   self.s.force_gr=0.0
        if abs(self.s.force_gr/2)>0.8:      self.m.add_fault('break')
        elif abs(self.s.force_gr/2)>1.0:    self.m.add_fault('deform')

        #need to transfer FG to FA & FS???
        if self.m.has_fault('break'):       self.force_st.s.support = 0.0
        elif self.m.has_fault('deform'):    self.force_st.s.support = 0.5
        else:                               self.force_st.s.support = 1.0
        self.force_lin.s.assign(self.force_st.s, 'support')

class ManageHealthMode(Mode):
    failrate=1e-6
    faultparams = {'falsemasking':(0.1,[0.5,0.5,0.5],1000),
                   'falseemland':(0.05,[0.0, 1.0, 0.0],1000),
                   'lostfunction':(0.05,[0.5,0.5,0.5],1000)}
    key_phases_by="plan_path"
class ManageHealth(FxnBlock):
    __slots__=('force_st', 'ee_ctl', 'hsig_dofs', 'hsig_bat', 'rsig_traj')
    _init_m = ManageHealthMode
    _init_p = ResPolicy
    _init_force_st = Force 
    _init_ee_ctl = EE
    _init_hsig_dofs = HSig 
    _init_hsig_bat = HSig 
    _init_rsig_traj = RSig
    def condfaults(self, time):
        if self.force_st.s.support<0.5 or self.ee_ctl.s.effort>2.0: 
            self.m.add_fault('lostfunction')
    def behavior(self, time):
        if self.m.has_fault('lostfunction'):      self.rsig_traj.s.mode = 'continue'
        elif  self.hsig_dofs.s.hstate=='faulty':  self.rsig_traj.s.mode = self.p.line
        elif  self.hsig_bat.s.hstate=='faulty':   self.rsig_traj.s.mode = self.p.bat
        else:                                     self.rsig_traj.s.mode = 'continue'

from examples.multirotor.drone_mdl_hierarchical import AffectDOFArch, OverallAffectDOFState
class AffectMode(Mode):
    key_phases_by='plan_path'
class AffectDOF(FxnBlock): #ee_mot,ctl,dofs,force_lin hsig_dofs, RSig_dofs
    __slots__=('ee_mot', 'ctl', 'hsig_dofs', 'dofs', 'des_traj', 'force_lin')
    _init_c = AffectDOFArch
    _init_s = OverallAffectDOFState
    _init_m = AffectMode
    _init_ee_mot = EE 
    _init_ctl = Control 
    _init_hsig_dofs = HSig 
    _init_dofs = DOFs 
    _init_des_traj = DesTraj
    _init_force_lin = Force
    def behavior(self, time):
        air,ee_in={},{}
        for linname,lin in self.c.components.items():
            air[lin.name], ee_in[lin.name] = lin.behavior(self.ee_mot.s.effort, self.ctl, self.c.forward[linname], self.force_lin.s.support) 
        
        if any(value>=10 for value in ee_in.values()):      self.ee_mot.s.rate=10
        elif any(value!=0.0 for value in ee_in.values()):   self.ee_mot.s.rate=sum(ee_in.values())/len(ee_in) #should it really be max?
        else:                                               self.ee_mot.s.rate=0.0
        
        self.s.lrstab = (sum([air[comp] for comp in self.c.lr_dict['l']])-sum([air[comp] for comp in self.c.lr_dict['r']]))/len(air)
        self.s.frstab = (sum([air[comp] for comp in self.c.fr_dict['r']])-sum([air[comp] for comp in self.c.fr_dict['f']]))/len(air)
        
        if abs(self.s.lrstab) >=0.25 or abs(self.s.frstab)>=0.75:   
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:                                                   
            self.dofs.s.put(uppwr=np.mean(list(air.values())), planpwr=self.ctl.s.forward)
        
        if self.m.any_faults(): self.hsig_dofs.s.hstate='faulty'
        else:                   self.hsig_dofs.s.hstate='nominal'
    def dynamic_behavior(self, time):
        #calculate velocities from power
        self.dofs.s.put(vertvel = 300*(self.dofs.s.uppwr-1.0), planvel = 600*self.dofs.s.planpwr) # 600 m/m = 23 mph
        #can only take off at ground
        if self.dofs.s.z<=0.0:        self.dofs.s.put(planvel=0.0, vertvel=max(0,self.dofs.s.vertvel))
        #if falling, it can't reach the destination if it hits the ground first
        plan_dist = np.sqrt(self.des_traj.s.x**2 + self.des_traj.s.y**2+0.0001)
        if self.dofs.s.vertvel<-self.dofs.s.z and -self.dofs.s.vertvel>self.dofs.s.planvel: 
            plan_dist = plan_dist*self.dofs.s.z/(-self.dofs.s.vertvel+0.001)
        self.dofs.s.limit(vertvel=(-self.dofs.s.z, 300.0), planvel=(0.0, plan_dist))
        #increment x,y,z
        norm_vel = self.dofs.s.planvel/np.sqrt(self.des_traj.s.x**2 + self.des_traj.s.y**2+0.0001)
        self.dofs.s.inc(x=norm_vel*self.des_traj.s.x, y=norm_vel*self.des_traj.s.y, z=self.dofs.s.vertvel)
    
class CtlDOFState(State):
    cs:              float = 1.0
    vel:             float = 0.0    
    upthrottle:      float=0.0
    throttle: float=0.0
class CtlDOFMode(Mode):
    failrate=1e-5
    faultparams={'noctl':   (0.2, [0.6, 0.3, 0.1], 1000), 
                 'degctl':  (0.8, [0.6, 0.3, 0.1], 1000)}
    exclusive=True
    key_phases_by = 'plan_path'
    mode:   str='nominal'
class CtlDOF(FxnBlock):
    __slots__ = ('ctl', 'ee_ctl', 'des_traj', 'force_st', 'dofs')
    _init_s = CtlDOFState
    _init_m = CtlDOFMode
    _init_ctl = Control 
    _init_ee_ctl = EE 
    _init_des_traj = DesTraj
    _init_force_st = Force
    _init_dofs = DOFs
    def condfaults(self, time):
        if self.force_st.s.support<0.5: self.m.add_fault('noctl')
    def behavior(self, time):
        if self.m.has_fault('noctl'):     self.s.cs=0.0
        elif self.m.has_fault('degctl'):  self.s.cs=0.5
        else:                             self.s.cs=1.0
        
        # throttle settings: 0 is off (-50 m/s), 1 is hover, 2 is max climb (5 m/s)
        self.s.upthrottle=1+self.des_traj.s.z/(50*5)
        self.s.throttle=np.sqrt(self.des_traj.s.x**2+self.des_traj.s.y**2)/(60*10)
        self.s.limit(throttle=(0,1), upthrottle=(0,2))
        
        self.ctl.s.forward=self.ee_ctl.s.effort*self.s.cs*self.s.throttle*self.des_traj.s.power
        self.ctl.s.upward=self.ee_ctl.s.effort*self.s.cs*self.s.upthrottle*self.des_traj.s.power
    def dynamic_behavior(self, time):
        self.s.vel=self.dofs.s.vertvel

class PlanPathMode(Mode):
    failrate=1e-5
    faultparams = {'noloc':(0.2, [0.6, 0.3, 0.1], 1000), 
                  'degloc':(0.8, [0.6, 0.3, 0.1], 1000)}
    opermodes = ('taxi', 'to_nearest', 'to_home', 'emland', 'land', 'move')
    mode: str = 'taxi'
    exclusive = True
    key_phases_by = 'self'
class PlanPathState(State):
    dx: float=0.0
    dy: float=0.0
    dz: float=0.0
    dist: float=0.0
    pt: int=1
    x: float=0.0
    y: float=0.0
    z: float=0.0

class PlanPath(FxnBlock):
    __slots__=('force_st', 'rsig_traj', 'dofs', 'ee_ctl', 'des_traj', 'goals')
    _init_s = PlanPathState
    _init_m = PlanPathMode
    _init_p = DroneParam
    _init_force_st = Force
    _init_rsig_traj = RSig
    _init_dofs = DOFs
    _init_ee_ctl = EE
    _init_des_traj = DesTraj
    def __init__(self, name, flows, **kwargs):
        super().__init__(name, flows, **kwargs)
        self.goals = {i: list(vals) for i, vals in enumerate(self.p.flightplan)}
    def condfaults(self, time):
        if self.force_st.s.support<0.5:   self.m.add_fault('noloc')
    def behavior(self, t):
        if not self.m.any_faults():
            # if in reconfigure mode, copy that mode, otherwise complete mission
            if self.rsig_traj.s.mode !='continue':        self.m.set_mode(self.rsig_traj.s.mode)
            elif self.m.in_mode('taxi') and t<5 and t>1:  self.m.set_mode("move")
            # if mission is over, enter landing mode when you get close
            if self.mission_over():
                if self.dofs.s.z<1:                       self.m.set_mode('taxi')
                elif self.s.dist<10:                      self.m.set_mode('land')
        # if close to the given point, go to the next point
        if self.m.in_mode('move') and self.s.dist<10: 
            self.s.pt+=1
        # set the new goal based on the mode
        if self.m.in_mode('emland','land'):       self.calc_dist_to_goal([self.dofs.s.x,self.dofs.s.y,-self.dofs.s.z/2])
        elif self.m.in_mode('to_home','taxi'):    self.calc_dist_to_goal(self.goals[0])
        elif self.m.in_mode('to_nearest'):        self.calc_dist_to_goal([*self.p.safe[:2],0.0])
        elif self.m.in_mode('move'):              self.calc_dist_to_goal(self.goals[self.s.pt])
        elif self.m.in_mode('noloc'):             self.calc_dist_to_goal(self.dofs.s.get('x','y','z'))
        elif self.m.in_mode('degloc'):            self.calc_dist_to_goal([self.dofs.s.x,self.dofs.s.y,self.dofs.s.z-1])
        # send commands (des_traj) if power
        if self.ee_ctl.s.effort<0.5 or self.m.in_mode('taxi'): 
            self.des_traj.s.assign([0.0,0.0,0.0,0.0],'x','y','z','power')  
        else:                                             
            self.des_traj.s.power=1.0
            self.des_traj.s.assign(self.s,x='dx',y='dy',z='dz')  
    def calc_dist_to_goal(self, goal):
        self.s.assign(goal, 'x','y','z')
        self.s.dist = finddist(self.dofs.s.get('x','y','z'), self.s.get('x','y','z'))
        dx, dy, dz = vectdist(self.s.get('x','y','z'), self.dofs.s.get('x','y','z'))
        self.s.put(dx=dx,dy=dy,dz=dz)
    def mission_over(self):
        return self.s.pt>=max(self.goals) or self.m.in_mode('to_nearest', 'to_home', 'land', 'emland')


class Drone(Model):
    __slots__=('start_area', 'safe_area', 'target_area')
    _init_p = DroneParam
    default_sp = dict(phases=(('ascend',0,0),('forward',1,11),('taxi',12, 20)),times=(0,30),units='min')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.start_area = square(self.p.start[0:2],self.p.start[2],self.p.start[3] )
        self.safe_area = square(self.p.safe[0:2],self.p.safe[2],self.p.safe[3] )
        self.target_area = square(self.p.target[0:2],self.p.target[2],self.p.target[3] )
        
        #add flows to the model
        self.add_flow('force_st',   Force)
        self.add_flow('force_lin',  Force)
        self.add_flow('hsig_dofs',  HSig)
        self.add_flow('hsig_bat',   HSig)
        self.add_flow('rsig_traj',  RSig)
        self.add_flow('ee_1',       EE)
        self.add_flow('ee_mot',     EE)
        self.add_flow('ee_ctl',     EE)
        self.add_flow('ctl',        Control)
        self.add_flow('dofs',       DOFs)
        self.add_flow('des_traj',   DesTraj)
        #add functions to the model
        
        flows=['ee_ctl', 'force_st', 'hsig_dofs', 'hsig_bat', 'rsig_traj']
        self.add_fxn('manage_health', ManageHealth, *flows, p=asdict(self.p.respolicy))
        
        store_ee_p = {'archtype':self.p.bat, 'weight':(self.p.batweight+self.p.archweight)/2.2 , 'drag': self.p.archdrag }
        self.add_fxn('store_ee',    StoreEE, 'ee_1', 'force_st', 'hsig_bat', c=store_ee_p)
        self.add_fxn('dist_ee',     DistEE,   'ee_1','ee_mot','ee_ctl', 'force_st')
        self.add_fxn('affect_dof',  AffectDOF,'ee_mot','ctl','dofs','des_traj','force_lin', 'hsig_dofs',  c={'archtype':self.p.linearch})
        self.add_fxn('ctl_dof',     CtlDOF,   'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path',   PlanPath, 'ee_ctl', 'dofs','des_traj', 'force_st', 'rsig_traj', p=asdict(self.p))
        self.add_fxn('hold_payload',HoldPayload, 'dofs', 'force_lin', 'force_st')
        
        self.build()
        
    def find_classification(self, scen, mdlhist):
        #landing costs
        viewed = env_viewed(mdlhist.faulty.flows.dofs.s.x, mdlhist.faulty.flows.dofs.s.y,mdlhist.faulty.flows.dofs.s.z, self.target_area)
        viewed_value = sum([0.5+2*view for k,view in viewed.items() if view!='unviewed'])
        
        # to fix: need to find fault time more efficiently (maybe in the toolkit?)
        faulttime = self.h.get_fault_time(metric='total')
        
        dofs=self.flows['dofs']
        if  inrange(self.start_area, dofs.s.x, dofs.s.y):     landloc = 'nominal' # nominal landing
        elif inrange(self.safe_area, dofs.s.x, dofs.s.y):     landloc = 'designated' # emergency safe
        elif inrange(self.target_area, dofs.s.x, dofs.s.y):   landloc = 'over target' # emergency dangerous
        else:                                           landloc = 'outside target' # emergency unsanctioned
        # need a way to differentiate horizontal and vertical crashes/landings
        if landloc in ['over target', 'outside target']: 
            if landloc=="outside target" and self.p.loc=='congested':    loc='urban'
            else:                                                        loc=self.p.loc
            body_strikes = density_categories[loc]['body strike']['horiz']
            head_strikes = density_categories[loc]['head strike']['horiz']
            property_restrictions = 1
        else: body_strikes =0.0; head_strikes=0.0; property_restrictions=0
        
        safecost = safety_categories['hazardous']['cost'] * (head_strikes + body_strikes) + unsafecost[self.p.loc] * faulttime
        landcost = property_restrictions*propertycost[self.p.loc]
        #repair costs
        repcost=self.calc_repaircost(max_cost=1500)
        rate=scen.rate
        p_safety = 1-np.exp(-(body_strikes+head_strikes) * 60/(faulttime+0.001)) #convert to pfh
        classifications = {'hazardous':rate*p_safety, 'minor':rate*(1-p_safety)}

        totcost=repcost+landcost+safecost-viewed_value
        
        expcost=totcost*rate*1e5
        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost, 'repcost':repcost, 
                'landcost':landcost,'safecost':safecost,'viewed value': viewed_value, 'viewed':viewed, 
                'landloc':landloc,'body strikes':body_strikes, 'head strikes':head_strikes, 
                'property restrictions': property_restrictions, 'severities':classifications, 
                'unsafe flight time':faulttime}

pos = {'manage_health': [0.23793980988102348, 1.0551602632416588],
       'store_ee': [-0.9665780995752296, -0.4931538151692423],
       'dist_ee': [-0.1858834234148632, -0.20479989209711924],
       'affect_dof': [1.0334916329507422, 0.6317263653616103],
       'ctl_dof': [0.1835014208949617, 0.32084893189175423],
       'plan_path': [-0.7427736219526058, 0.8569475547950892],
       'hold_payload': [0.74072970715511, -0.7305391093272489]}

bippos = {'manage_health': [-0.23403572483176666, 0.8119063670455383],
          'store_ee': [-0.7099736148158298, 0.2981652748232978],
          'dist_ee': [-0.28748133634190726, 0.32563569654296287],
          'affect_dof': [0.9073412427515959, 0.0466423266443633],
          'ctl_dof': [0.498663257339388, 0.44284186573420836],
          'plan_path': [0.5353654708147643, 0.7413936186204868],
          'hold_payload': [0.329334798653681, -0.17443414674339652],
          'force_st': [-0.2364754675127569, -0.18801548176633154],
          'force_lin': [0.7206415618571647, -0.17552020772024013],
          'hsig_dofs': [0.3209028709788254, 0.04984245810974697],
          'hsig_bat': [-0.6358884586093769, 0.7311076416371343],
          'rsig_traj': [0.18430501738656657, 0.856472541655958],
          'ee_1': [-0.48288657418004555, 0.3017533207866233],
          'ee_mot': [-0.0330582435936827, 0.2878069006385988],
          'ee_ctl': [0.13195069534343862, 0.4818116953414546],
          'ctl': [0.5682663453757308, 0.23385244312813386],
          'dofs': [0.8194232270836169, 0.3883256382522293],
          'des_traj': [0.9276094920710914, 0.6064107724557304]}

## BASE FUNCTIONS
def find_landtime(mdlhist):
    return min([i for i,a in enumerate(mdlhist['functions']['plan_path']['mode']) if a=='taxi']+[15])
# creates list of corner coordinates for a square, given a center, xwidth, and ywidth
def square(center,xw,yw):
    square=[[center[0]-xw/2,center[1]-yw/2],\
            [center[0]+xw/2,center[1]-yw/2], \
            [center[0]+xw/2,center[1]+yw/2],\
            [center[0]-xw/2,center[1]+yw/2]]
    return square

def rect(x1, y1, x2, y2, width, height):
    vec = [x1-x2, y1-y2]
    vec = vec/(np.sum([v**2 for v in vec])+0.00001)
    normvec = np.array([vec[1], -vec[0]])
    rec = [[x1, y1]+normvec*width/2+vec*height/2,[x1, y1]-normvec*width/2+vec*height/2,[x2, y2]-normvec*width/2-vec*height/2,[x2, y2]+normvec*width/2-vec*height/2]
    return rec

import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
def env_viewed(xhist, yhist,zhist, square):
    viewed = {(x,y):'unviewed' for x in range(int(square[0][0]),int(square[1][0])+10,10) for y in range(int(square[0][1]),int(square[2][1])+10,10)}
    for i,x in enumerate(xhist[1:len(xhist)]):
        w,h,d = viewable_area(zhist[i+1])
        viewed_area = rect(xhist[i],yhist[i],xhist[i+1],yhist[i+1], w+5,h+5)
        if abs(xhist[i]-xhist[i+1]) + abs(yhist[i]-yhist[i+1]) > 0.1 and w >0.01:
            polygon=Polygon(viewed_area)
            #plt.plot(*polygon.exterior.xy) (displays area to debug code)
            #plt.plot([xhist[i],xhist[i+1]],[yhist[i],yhist[i+1]])
            if not polygon.is_valid:    print('invalid points')
            for spot in viewed:
                if polygon.contains(Point(spot)): 
                    viewed[spot]=d
    return viewed

def viewable_area(z):
    width = z
    height = z #* 0.75 # 4/3 camera with ~45 mm lens st dist = width
    detail = 1/(width*height+0.00001)
    return width, height, detail

## PLOTTING
def plot_nomtraj(mdlhist, params, title='Trajectory'):
    xnom=mdlhist.flows.dofs.s.x
    ynom=mdlhist.flows.dofs.s.y
    znom=mdlhist.flows.dofs.s.z
    
    time = mdlhist.time
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in enumerate(params.flightplan):
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    plt.show()

def plot_faulttraj(mdlhist, params, title='Fault response to RFpropbreak fault at t=20'):
    xnom=mdlhist.nominal.flows.dofs.s.x
    ynom=mdlhist.nominal.flows.dofs.s.y
    znom=mdlhist.nominal.flows.dofs.s.z
    #
    x=mdlhist.faulty.flows.dofs.s.x
    y=mdlhist.faulty.flows.dofs.s.y
    z=mdlhist.faulty.flows.dofs.s.z
    
    time = mdlhist.nominal.time
    
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)
    ax2.plot(x,y,z)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal, loc in enumerate(params.flightplan):
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    ax2.legend(['Nominal Flightpath','Faulty Flightpath'], loc=4)
    return fig2, ax2
    
def plot_xy(mdlhist, endresults, mdl, title='', legend=False):
    plt.figure()
    plot_one_xy(mdlhist, endresults)
    
    plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue', label='Starting Area')
    plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red', label='Target Area')
    plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow', label='Emergency Landing Area')
    
    plt.title(title)
    if legend: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return plt.gcf(), plt.gca()
def plot_one_xy(mdlhist,endresults):
    xnom=mdlhist.flows.dofs.x
    ynom=mdlhist.flows.dofs.y
    
    plt.plot(xnom,ynom)
    
    
    xviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    yviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    xunviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    yunviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    
    plt.scatter(xviewed,yviewed, color='red', label='Viewed')
    plt.scatter(xunviewed,yunviewed, color='grey', label='Unviewed')
    return plt.gca(), plt.gcf()
    
def plot_xys(mdlhists, endresultss, mdl, cols=2, title='', legend=False):
    num_plots = len(mdlhists)
    fig, axs = plt.subplots(nrows=int(np.ceil((num_plots)/cols)), ncols=cols, figsize=(cols*6, 5*num_plots/cols))
    n=1
    
    for paramlab, mdlhist in mdlhists.items():
        plt.subplot(int(np.ceil((num_plots)/cols)),cols,n, label=paramlab)
        a, _= plot_one_xy(mdlhist, endresultss[paramlab])
        b= plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue', label='Starting Area')
        c=plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red', label='Target Area')
        d=plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow', label='Emergency Landing Area')
        plt.title(paramlab)
        n+=1
    plt.suptitle(title)
    if legend: 
        plt.subplot(np.ceil((num_plots+1)/cols),cols,n, label='legend')
        plt.axis('off')
        legend_elements = [Line2D([0], [0], color='b', lw=1, label='Flightpath'),
                   Line2D([0], [0], marker='o', color='r', label='Viewed',
                          markerfacecolor='r', markersize=8),
                   Line2D([0], [0], marker='o', color='grey', label='Unviewed',
                          markerfacecolor='grey', markersize=8),
                   Patch(facecolor='red', edgecolor='red', alpha=0.2,
                         label='Target Area'),
                   Patch(facecolor='blue', edgecolor='blue',label='Landing Area'),
                   Patch(facecolor='yellow', edgecolor='yellow',label='Emergency Landing Area')]
        plt.legend( handles=legend_elements, loc='center')
    plt.subplots_adjust(top=1-0.05-0.05/(num_plots/cols))
    
    return fig

# likelihood class schedule (pfh)
p_allowable = {'small airplane':{'no requirement':'na', 'probable':1e-3, 'remote':1e-4, 'extremely remote':1e-5, 'extremely improbable':1e-6},
               'small helicopter':{'no requirement':'na', 'probable':1e-3, 'remote':1e-5, 'extremely remote':1e-7, 'extremely improbable':1e-9}}

# population schedule
density_categories = {'congested':{'density':0.006194, 'body strike':{'vert':0.1, 'horiz':0.73},'head strike':{'vert':0.0375,'horiz':0.0375}},
                      'urban':{'density':0.002973, 'body strike':{'vert':0.0004, 'horiz':0.0003},'head strike':{'vert':0.0002,'horiz':0.0002}},
                      'suburban':{'density':0.001042, 'body strike':{'vert':0.0001, 'horiz':0.0011},'head strike':{'vert':0.0001,'horiz':0.0001}},
                      'rural':{'density':0.0001042, 'body strike':{'vert':0.0000, 'horiz':0.0001},'head strike':{'vert':0.000,'horiz':0.000}},
                      'remote':{'density':1.931e-6, 'body strike':{'vert':0.0000, 'horiz':0.0000},'head strike':{'vert':0.000,'horiz':0.000}}}

unsafecost = {'congested': 1000,'urban': 100, 'suburban':25, 'rural':5, 'remote':1}
propertycost = {'congested': 100000,'urban': 10000, 'suburban':1000, 'rural':1000, 'remote':1000}
# safety class schedule
safety_categories = {'catastrophic':{'injuries':'multiple fatalities', 'safety margins':'na', 'crew workload': 'na', 'cost':2000000},
                     'hazardous':{'injuries':'single fatality and/or multiple serious injuries', 'safety margins':'large decrease', 'crew workload': 'compromises safety', 'cost':9600000},
                     'major': {'injuries':'non-serious injuries', 'safety margins':'significant decrease', 'crew workload': 'significant increase', 'cost':2428800},
                     'minor': {'injuries':'na', 'safety margins':'slight decrease', 'crew workload': 'slight increase', 'cost':28800},
                     'no effect': {'injuries':'na', 'safety margins':'na', 'crew workload': 'na','cost': 0}}

hazards = {'VH-1':'loss of control', 'VH-2':'fly-away / non-conformance', 'VH-3':'loss of communication', 'VH-4':'loss of navigation', 'VH-5':'unsuccessful landing',
           'VH-6':'unintentional flight termination', 'VH-7':'collision'}

respols = ['continue', 'to_home', 'to_nearest', 'emland']

target = [0, 150, 160, 160]
safe = [0, 50, 10, 10]
start = [0.0,0.0, 10, 10]
def_mdl = Drone()

def plan_flight(z):
    sq = square(def_mdl.p.target[0:2],def_mdl.p.target[2],def_mdl.p.target[3])
    landing = [*def_mdl.p.start[0:2], 0]
    
    flightplan = {0:landing, 1: landing[0:2]+[z]}
    
    width, height, detail = viewable_area(z)
    # x,y, z
    startpt = [sq[0][0]+width/2, sq[0][1]+height/2, z]
    endpt = [sq[1][0]-width/2, sq[1][1]+height/2, z]
    
    num_rows = int(np.ceil((sq[2][1]-sq[0][1])/width))
    
    leftpts = [[startpt[0] , startpt[1]+ r*width] for r in range(num_rows)]
    rightpts = [[endpt[0], endpt[1]+ r*width] for r in range(num_rows)]
    leftpts.sort(reverse=True)
    rightpts.sort(reverse=True)
    
    vec1 = leftpts
    vec2 = rightpts
    vec=[]
    n=2
    newplan = {}
    while len(vec1+vec2+vec)>0:
        newplan[n]=vec1.pop()+[z]
        n+=1
        if len(vec1)< len(vec2) or n==0:
            vec = vec2
            vec2 = vec1
            vec1 = vec
    
    flightplan.update(newplan)
    flightplan.update({max(flightplan)+1:flightplan[1], max(flightplan)+2:flightplan[0]})
    return {'flightplan': tuple(tuple(v) for v in flightplan.values())}

## Optimization Functions
bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
linarchs = ['quad', 'hex', 'oct']
batcostdict = {'monolithic':0, 'series-split':300, 'parallel-split':300, 'split-both':600}
linecostdict = {'quad':0, 'hex':1000, 'oct':2000}
def x_to_dcost(xdes):
    descost = batcostdict[bats[int(xdes[0])]] + linecostdict[linarchs[int(xdes[1])]]
    return descost
def xd_paramfunc(xdes):
    return {'bat':bats[int(xdes[0])],'linearch':linarchs[int(xdes[1])]}

opt_prob = ProblemInterface("drone_problem", def_mdl)
opt_prob.add_simulation("dcost", "external", x_to_dcost)
opt_prob.add_objectives("dcost", cd="cd")
opt_prob.add_variables("dcost",('batteryarch',(0,3)),('linearch',(0,3)))

opt_prob.add_simulation("ocost", "single", {}, staged=False,
                        upstream_sims = {"dcost":{'paramfunc':xd_paramfunc}})
opt_prob.add_objectives("ocost", co="expected cost")
opt_prob.add_constraints("ocost", g_soc=("store_ee.s.soc", "vars", "end",("greater", 20)),
                                  g_max_height=("dofs.s.z", "vars", "all", ("less", 122)),
                                  g_faults=("repcost", "endclass", "end", ("less", 0.1)))
opt_prob.add_variables("ocost", "height", vartype=plan_flight)
#opt_prob.cd([2,2])
#opt_prob.co([10])

respols = ['continue', 'to_home', 'to_nearest', 'emland']
def spec_respol(bat, line):
    return {'respolicy': ResPolicy(bat=respols[int(bat)], line=respols[int(line)])}

app = SampleApproach(def_mdl,  phases={'forward'}, faults=('single-component', 'store_ee'))
opt_prob.add_simulation("rcost", "multi", app.scenlist, include_nominal=False,\
                        upstream_sims={'ocost':{'phases':{'plan_path':'move'},'pass_mdl':[]}},\
                        app_args={'faults':('single-component', 'store_ee')},\
                        staged=True)
opt_prob.add_objectives("rcost", cr="expected cost")
opt_prob.add_variables("rcost", "bat","line", vartype=spec_respol)

#opt_prob.cr([1,0])

#an.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'dofs'}, time_slice=6)
#an.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'store_ee'}, time_slice=6)
#(variablename, objtype (optional), t (optional))


def calc_oper(mdl):
    endresults_nom, mdlhist =propagate.nominal(mdl)
    opercost = endresults_nom.endclass['expected cost']
    g_soc = 20 - mdlhist.fxns.store_ee.s.soc[-1] 
    #g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist.flows.dofs.s.z-122 if i>0])
    
    phases, modephases=mdlhist.get_modephases()
    return opercost, g_soc, g_max_height, phases
def x_to_ocost(xdes, xoper, loc='rural'):
    fp = plan_flight(xoper[0], def_mdl)
    params = DroneParam(bat=bats[xdes[0]], linearch=linarchs[xdes[1]], respolicy=ResPolicy(bat='continue', line='continue'))
    mdl = Drone(params=params)
    return calc_oper(mdl)

def calc_res(mdl, fullcosts=False, faultmodes = 'all', include_nominal=True, pool=False, phases={}, staged=True):
    #app = SampleApproach(mdl, faults=('single-component', faultmodes), phases={'forward'})
    app = SampleApproach(mdl, faults=('single-component', 'store_ee'), phases={'move':phases['plan_path']['move']})
    result, mdlhists = propagate.approach(mdl, app, staged=staged, pool=pool, showprogress=False) #, staged=False)
    rescost = result.total('expected cost')-(not include_nominal)*result.nominal.endclass['expected cost']
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'dofs'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=7.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'store_ee'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'plan_path'}, time_slice=6)
    #an.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'rsig_traj', 'hsig_bat','hsig_dofs'})
    #[ec['expected cost'] for ec in endclasses.values()]
    #[ec['endclass']['expected cost'] for ec in opt_prob._sims['rcost']['results'].values()]
    #plot_faulttraj({'nominal':mdlhists['nominal'], 'faulty':mdlhists['store_ee lowcharge, t=7.0']}, mdl.params, title='Fault response to store_ee lowcharge, t=6.0')
    #phases, modephases = an.process.modephases(mdlhists['nominal'])
    #an.plot.phases({p:ph for p,ph in phases.items() if p=='plan_path'}, modephases)
    return rescost
def x_to_rcost(xdes, xoper, xres, loc='rural', fullcosts=False, faultmodes = 'all', include_nominal=False, pool=False, phases={},staged=True):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    fp = plan_flight(xoper[0])
    
    
    params = DroneParam(bat=bats[xdes[0]], linearch=linarchs[xdes[1]], respolicy=ResPolicy(bat=respols[xres[0]], line=respols[xres[1]]))
    mdl = Drone(p=params)
    if not phases: _,_,_, phases = calc_oper(mdl)
    return calc_res(mdl, fullcosts=fullcosts, faultmodes = faultmodes, include_nominal=include_nominal, pool=pool, phases=phases, staged=staged)

if __name__=="__main__":
    import fmdtools.sim.propagate as prop
    import matplotlib.pyplot as plt
    
    mdl = Drone()
    ec, mdlhist = prop.nominal(mdl)
    phases, modephases = mdlhist.get_modephases()
    an.plot.phases(phases, modephases)
    
    mdl = Drone()
    app = SampleApproach(mdl,  phases={'forward'})
    endclasses, mdlhists = prop.approach(mdl, app, staged=True)
    plot_faulttraj(History(nominal=mdlhists.nominal, 
                           faulty=mdlhists.store_ee_lowcharge_t6p0), 
                   mdl.p, title='Fault response to RFpropbreak fault at t=20')

    #opt_prob.add_combined_objective("total_cost", 'cd', 'co', 'cr')
    #opt_prob.total_cost([1,1],[100],[1,1])
    #opt_prob.total_cost([1,1,100,1,1])
    #opt_prob.time_sims([1,1,100,1,1])
    
    opt_prob.cr([2,2, 100, 0,0])
    
    x_to_rcost([2,2],[100],[0,0], faultmodes='store_ee')
    x_to_rcost([0,0],[100],[0,0], faultmodes='store_ee')
    opt_prob.show_architecture()
    
    #opt_prob.update_sim_options("ocost", track={"functions":{"plan_path":"all"}, "flows":{"dofs":"all"}})
    #opt_prob.update_sim_options("rcost", log_iter_hist=True, pool=mp.Pool(4), track={"functions":{"store_ee":"faults"}, "flows":{"dofs":"all"}})
    
    #opt_prob.total_cost([1,1,120,1,1])
    #opt_prob.total_cost([1,1,60,1,1])
    
    opt_prob.time_sims([1,1,100,1,1])
    
    opt_prob.iter_hist
    
    plt.show()
    
    