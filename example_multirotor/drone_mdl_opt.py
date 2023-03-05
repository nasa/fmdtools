# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019, revised Nov 2022
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
from fmdtools.modeldef.common import Parameter, State
from fmdtools.modeldef.block import FxnBlock, Component, CompArch, Mode
from fmdtools.modeldef.flow import Flow
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.approach import SampleApproach
from fmdtools.faultsim import propagate
from fmdtools.faultsim.search import ProblemInterface
from shapely import Point, Polygon

import fmdtools.resultdisp as rd
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from drone_mdl_static import m2to1
from drone_mdl_static import finddist,vectdist, inrange

from drone_mdl_static import DistEE

#Define functions
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
        archtype=self.get_true_field(*args, **kwargs)
        weight = self.get_true_field(*args, **kwargs)
        drag = self.get_true_field(*args, **kwargs)
        if archtype=='monolithic':
            batparams = {"series":1, "parallel":2, "voltage":12.0, "weight":weight, "drag":drag}
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
        self.make_components(Battery, *compnames, batparams)

from drone_mdl_static import StoreEEState
class StoreEEMode(Mode):
    failrate=1e-4
    faultparams = {'nocharge':  (0.2,[0.6,0.2,0.2],0),
                   'lowcharge': (0.7,[0.6,0.2,0.2],0)}
    key_phases_by="plan_path"

class StoreEE(FxnBlock):
    _init_s = StoreEEState
    _init_m = StoreEEMode
    _init_c = BatArch
    def condfaults(self, time):
        if self.s.soc<20:                   self.m.add_fault('lowcharge')
        if self.s.soc<1:                    self.m.replace_fault('lowcharge','nocharge')
        if self.m.has_fault('lowcharge'):   
            for batname, bat in self.c.components.items(): bat.s.soc=19
        elif self.m.fault('nocharge'):
            for batname, bat in self.c.components.items(): bat.s.soc=0
    def behavior(self, time):
        EE, soc = {}, {}
        rate_res=0
        for batname, bat in self.ccomponents.items():
            EE[bat.name], soc[bat.name], rate_res = \
                bat.behavior(self.s.force_st.support, self.s.ee_1.rate/(self.c.series*self.c.parallel)+rate_res, time)
        #need to incorporate max current draw somehow + draw when reconfigured
        if self.c.archtype == 'monolithic':           self.ee_1.s.effort = EE['S1P1']
        elif self.c.archtype == 'series-split':       self.ee_1.s.effort = np.max(list(EE.values()))
        elif self.c.archtype == 'parallel-split':     self.ee_1.s.effort = np.sum(list(EE.values()))
        elif self.c.archtype == 'split-both':          
            e=list(EE.values())
            e.sort()
            self.ee_1.effort = e[-1]+e[-2]  
        self.s.soc=np.mean(list(soc.values()))
        if self.m.any_faults() and not self.m.has_fault("dummy"):   self.hsig_bat.s.hstate = 'faulty'
        else:                                                       self.hsig_bat.s.hstate = 'nominal'
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
    weight:     float=0.0
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
    def behavior(self, FS, EEoutr, time):
        if FS <1.0:             self.m.add_fault('break')
        if EEoutr>self.maxa:    self.m.add_fault('break')
        if self.m.has_fault('short'):       self.s.Et=0.0
        elif self.m.has_fault('break'):     self.s.Et=0.0
        elif self.m.has_fault('degr'):      self.s.Et=0.5*self.s.avail_eff
        else:                               self.Et=self.s.avail_eff
        
        if time > self.time:
            self.s.inc(soc=-100*EEoutr*self.parallel*self.series*(time-self.time)/self.p.amt)
            self.time=time
        if self.s.soc<20:         self.m.add_fault('lowcharge')
        if self.s.soc<1:          
            self.m.replace_fault('lowcharge','nocharge')
            self.s.put(soc=0.0, Et=0.0)
            Er_res=EEoutr
        else: Er_res=0.0
        return self.s.Et, self.s.soc, Er_res

            
class HoldPayload(FxnBlock):
    def __init__(self,  name, flows):
        super().__init__(name, flows, states={'Force_GR':1.0})
        self.failrate=1e-6
        self.assoc_modes({'break':[0.2, [0.33, 0.33, 0.33], 1000], 'deform':[0.8, [0.33, 0.33, 0.33], 1000]}, key_phases_by="plan_path")
    def dynamic_behavior(self, time):
        if self.dofs.z<=0.0:    self.Force_GR=min(-0.5, (self.dofs.vertvel-self.dofs.planvel)/(60*7.5))
        else:                   self.Force_GR=0.0
        if abs(self.Force_GR/2)>0.8:      self.m.add_fault('break')
        elif abs(self.Force_GR/2)>1.0:    self.m.add_fault('deform')

        #need to transfer FG to FA & FS???
        if self.m.has_fault('break'):     self.force_lin.support, self.force_st.support = 0.0,0.0
        elif self.m.has_fault('deform'):  self.force_lin.support, self.force_st.support = 0.5,0.5
        else:                           self.force_lin.support, self.force_st.support = 1.0,1.0
    
class ManageHealth(FxnBlock):
    def __init__(self,  name, flows,respolicy):
        self.respolicy = respolicy
        super().__init__(name, flows)
        self.failrate=1e-6 #{'falsemaintenance':[0.8,[1.0, 0.0,0.0,0.0,0.0],1000],\
        self.assoc_modes({'falsemasking':[0.1,[0.5,0.5,0.5],1000],'falseemland':[0.05,[0.0, 1.0, 0.0],1000],\
                         'lostfunction':[0.05,[0.5,0.5,0.5],1000]}, key_phases_by="plan_path")
    def condfaults(self, time):
        if self.force_st.support<0.5 or self.ee_ctl.effort>2.0: self.m.add_fault('lostfunction')
    def behavior(self, time):
        if self.m.has_fault('lostfunction'):      self.rsig_traj.mode = 'continue'
        elif  self.hsig_dofs.hstate=='faulty':  self.rsig_traj.mode = self.respolicy.line
        elif  self.hsig_bat.hstate=='faulty':   self.rsig_traj.mode = self.respolicy.bat
        else:                                   self.rsig_traj.mode = 'continue'
    
class AffectDOF(FxnBlock): #ee_mot,ctl,dofs,force_lin hsig_dofs, RSig_dofs
    def __init__(self, name, flows, archtype):     
        self.archtype=archtype
        if archtype=='quad':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR')}
            self.f_fact={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
            self.LR_dict = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR_dict = {'F':{'LF', 'RF'}, 'R':{'LR', 'RR'}}
        elif archtype=='hex':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR'),'R':Line('R'), 'F':Line('F')}
            self.f_fact={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5, 'R':-0.75, 'F':0.75}
            self.LR_dict = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR_dict = {'F':{'LF', 'RF', 'F'}, 'R':{'LR', 'RR', 'R'}}
        elif archtype=='oct':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR'),'RF2':Line('RF2'), 'LF2':Line('LF2'), 'LR2':Line('LR2'), 'RR2':Line('RR2')}
            self.f_fact={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5,'RF2':0.5,'LF2':0.5,'LR2':-0.5,'RR2':-0.5}
            self.LR_dict = {'L':{'LF', 'LR','LF2', 'LR2'}, 'R':{'RF','RR','RF2','RR2'}}
            self.FR_dict = {'F':{'LF', 'RF','LF2', 'RF2'}, 'R':{'LR', 'RR','LR2', 'RR2'}}
        super().__init__(name, flows, states={'LRstab':0.0, 'FRstab':0.0}, components=components) 
        self.assoc_modes(key_phases_by="plan_path")
    def behavior(self, time):
        Air,EEin={},{}
        for linname,lin in self.components.items():
            lin.behavior(self.ee_mot.effort, self.ctl, self.f_fact[linname], self.force_lin.support) 
            Air[lin.name]=lin.Airout
            EEin[lin.name]=lin.EE_in
        
        if any(value>=10 for value in EEin.values()):    self.ee_mot.rate=10
        elif any(value!=0.0 for value in EEin.values()): self.ee_mot.rate=sum(EEin.values())/len(EEin) 
        else:                                            self.ee_mot.rate=0.0
        
        self.LRstab = (sum([Air[comp] for comp in self.LR_dict['L']])-sum([Air[comp] for comp in self.LR_dict['R']]))/len(Air)
        self.FRstab = (sum([Air[comp] for comp in self.FR_dict['R']])-sum([Air[comp] for comp in self.FR_dict['F']]))/len(Air)
        
        if abs(self.LRstab) >=0.25 or abs(self.FRstab)>=0.75:   self.dofs.put(uppwr=0.0, planpwr=0.0)
        else:                                                   self.dofs.put(uppwr=np.mean(list(Air.values())), planpwr=self.ctl.forward)
        
        if self.any_faults():   self.hsig_dofs.hstate='faulty'
        else:                   self.hsig_dofs.hstate='nominal'
    def dynamic_behavior(self, time):
        #calculate velocities from power
        self.dofs.put(vertvel = 300*(self.dofs.uppwr-1), planvel = 600*self.dofs.planpwr) # 600 m/m = 23 mph
        #can only take off at ground
        if self.dofs.z<=0.0:        self.dofs.put(planvel=0.0, vertvel=max(0,self.dofs.vertvel))
        #if falling, it can't reach the destination if it hits the ground first
        plan_dist = np.sqrt(self.des_traj.x**2 + self.des_traj.y**2+0.0001)
        if self.dofs.vertvel<-self.dofs.z and -self.dofs.vertvel>self.dofs.planvel: 
            plan_dist = plan_dist*self.dofs.z/(-self.dofs.vertvel+0.001)
        self.dofs.limit(vertvel=(-self.dofs.z, 300.0), planvel=(0.0, plan_dist))
        #increment x,y,z
        norm_vel = self.dofs.planvel/np.sqrt(self.des_traj.x**2 + self.des_traj.y**2+0.0001)
        self.dofs.inc(x=norm_vel*self.des_traj.x, y=norm_vel*self.des_traj.y, z=self.dofs.vertvel)
class Line(Component):
    def __init__(self, name):
        super().__init__(name,{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0})
        self.failrate=1e-5
        self.assoc_modes({'short':[0.1, [0.33, 0.33, 0.33], 200],'openc':[0.1, [0.33, 0.33, 0.33], 200],\
                          'ctlbreak':[0.2, [0.33, 0.33, 0.33], 100], 'mechbreak':[0.1, [0.33, 0.33, 0.33], 500],\
                          'mechfriction':[0.05, [0.0, 0.5,0.5], 500], 'stuck':[0.02, [0.0, 0.5,0.5], 200]},name=name)
    def behavior(self, EEin, Ctlin, f_fact, Force):
        if Force<=0.0:   self.m.add_fault('mechbreak')
        elif Force<=0.5: self.m.add_fault('mechfriction')
            
        if self.m.has_fault('short'):                   self.put(Eti=0.0, Eto= np.inf)
        elif self.m.has_fault('openc'):                 self.put(Eti=0.0, Eto= 0.0)
        elif Ctlin.upward==0 and Ctlin.forward == 0:  self.put(Eto= 0.0)
        else:                                         self.put(Eto= 1.0)
        if self.m.has_fault('ctlbreak'):                self.put(Ct=0.0)
        if self.m.has_fault('mechbreak'):               self.put(Mt=0.0)
        elif self.m.has_fault('mechfriction'):          self.put(Mt=0.5, Eti= 2.0) 
        if self.m.has_fault('stuck'):                   self.put(Pt=0.0, Mt=0.0, Eti= 4.0) 
        
        self.Airout=m2to1([EEin,self.Eti,Ctlin.upward+Ctlin.forward*f_fact,self.Ct,self.Mt,self.Pt])
        self.EE_in=m2to1([EEin,self.Eto])  
    
class CtlDOF(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, states={'vel':0.0, 'Cs':1.0, 'upthrottle':0.0, 'forwardthrottle':0.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, [0.6, 0.3, 0.1], 1000], 'degctl':[0.8, [0.6, 0.3, 0.1], 1000]}, exclusive=True, key_phases_by="plan_path")
    def condfaults(self, time):
        if self.force_st.support<0.5: self.m.add_fault('noctl')
    def behavior(self, time):
        if self.m.has_fault('noctl'):     self.Cs=0.0
        elif self.m.has_fault('degctl'):  self.Cs=0.5
        else:                           self.Cs=1.0
        
        # throttle settings: 0 is off (-50 m/s), 1 is hover, 2 is max climb (5 m/s)
        self.upthrottle=1+self.des_traj.z/(50*5)
        self.forwardthrottle=np.sqrt(self.des_traj.x**2+self.des_traj.y**2)/(60*10)
        self.limit(forwardthrottle=(0,1), upthrottle=(0,2))
        
        self.ctl.forward=self.ee_ctl.effort*self.Cs*self.forwardthrottle*self.des_traj.power
        self.ctl.upward=self.ee_ctl.effort*self.Cs*self.upthrottle*self.des_traj.power
    def dynamic_behavior(self, time):
        self.vel=self.dofs.vertvel

class PlanPath(FxnBlock):
    def __init__(self, name, flows, params):
        self.nearest = [*params.safe[0:2],0]
        self.goals = {i: list(vals) for i, vals in enumerate(params.flightplan)}
        super().__init__(name, flows, states={'dx':0.0, 'dy':0.0, 'dz':0.0, 'dist':0.0, 'pt':1})
        self.add_flow('goal', {'x':self.goals[1][0],'y':self.goals[1][1],'z':self.goals[1][2]})
        self.failrate=1e-5
        self.assoc_modes({'noloc':[0.2, [0.6, 0.3, 0.1], 1000], 'degloc':[0.8, [0.6, 0.3, 0.1], 1000]},
                         ['taxi', 'to_nearest', 'to_home', 'emland', 'land', 'move'], initmode='taxi', exclusive=True,
                         key_phases_by='self')
    def condfaults(self, time):
        if self.force_st.support<0.5:   self.m.add_fault('noloc')
    def behavior(self, t):
        if not self.any_faults():
            # if in reconfigure mode, copy that mode, otherwise complete mission
            if self.rsig_traj.mode !='continue':        self.set_mode(self.rsig_traj.mode)
            elif self.in_mode('taxi') and t<5 and t>1:  self.set_mode("move")
            # if mission is over, enter landing mode when you get close
            if self.mission_over():
                if self.dofs.z<1:                   self.set_mode('taxi')
                elif self.dist<10:                  self.set_mode('land')
        # if close to the given point, go to the next point
        if self.in_mode('move') and self.dist<10: self.pt+=1
        # set the new goal based on the mode
        if self.in_mode('emland','land'):       self.calc_dist_to_goal([self.dofs.x,self.dofs.y,-self.dofs.z/2])
        elif self.in_mode('to_home','taxi'):    self.calc_dist_to_goal(self.goals[0])
        elif self.in_mode('to_nearest'):        self.calc_dist_to_goal(self.nearest)
        elif self.in_mode('move'):              self.calc_dist_to_goal(self.goals[self.pt])
        elif self.in_mode('noloc'):             self.calc_dist_to_goal(self.dofs.get('x','y','z'))
        elif self.in_mode('degloc'):            self.calc_dist_to_goal([self.dofs.x,self.dofs.y,self.dofs.z-1])
        # send commands (des_traj) if power
        if self.ee_ctl.effort<0.5 or self.in_mode('taxi'): 
            self.des_traj.assign([0.0,0.0,0.0,0.0],'x','y','z','power')  
        else:                                             
            self.des_traj.power=1.0
            self.des_traj.assign(self,x='dx',y='dy',z='dz')  
    def calc_dist_to_goal(self, goal):
        self.goal.assign(goal, 'x','y','z')
        self.dist = finddist(self.dofs, self.goal)        
        [self.dx,self.dy, self.dz] = vectdist(self.goal, self.dofs)
    def mission_over(self):
        return self.pt>=max(self.goals) or self.in_mode('to_nearest', 'to_home', 'land', 'emland')

#def finddist(f1, f2):
#    return np.sqrt((f1.x-f2.x)**2+(f1.y-f2.y)**2+(f1.z-f2.z)**2)
#def vectdist(f1, f2):
#    return [f1.x-f2.x,f1.y-f2.y,f1.z-f2.z]
#def vectdir(f1, f2):
#    return vectdist(f1,f2)/(finddist(f1,f2)+0.0001)

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

class Drone(Model):
    def __init__(self, params=DroneParam(), 
                 modelparams=ModelParam(phases=(('ascend',0,0),('forward',1,11),('taxi',12, 20)),times=(0,30),units='min'),
                 valparams={'loc':'rural'}):
        super().__init__(params, modelparams, valparams)
        
        self.start_area = square(params.start[0:2],params.start[2],params.start[3] )
        self.safe_area = square(params.safe[0:2],params.safe[2],params.safe[3] )
        self.target_area = square(params.target[0:2],params.target[2],params.target[3] )
        
        #add flows to the model
        self.add_flow('force_st', {'support':1.0})
        self.add_flow('force_lin', {'support':1.0} )
        self.add_flow('hsig_dofs', {'hstate':'nominal', 'config':1.0})
        self.add_flow('hsig_bat', {'hstate':'nominal', 'config':1.0} )
        self.add_flow('rsig_traj', {'mode':'continue'})
        self.add_flow('ee_1', {'rate':0.0, 'effort':1.0})
        self.add_flow('ee_mot', {'rate':1.0, 'effort':1.0})
        self.add_flow('ee_ctl', {'rate':1.0, 'effort':1.0})
        self.add_flow('ctl', {'forward':0.0, 'upward':1.0})
        self.add_flow('dofs', {'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0, 'x':0.0,'y':0.0,'z':0.0})
        self.add_flow('des_traj', {'x':0.0, 'y':0.0, 'z':0.0, 'power': 1.0})
        #add functions to the model
        flows=['ee_ctl', 'force_st', 'hsig_dofs', 'hsig_bat', 'rsig_traj']
        # trajconfig: continue, to_home, to_nearest, emland
        self.add_fxn('manage_health',flows,fclass = ManageHealth, fparams=params.respolicy)
        batweight = {'monolithic':0.4, 'series-split':0.5, 'parallel-split':0.5, 'split-both':0.6}[params.bat]
        archweight = {'quad':1.2, 'hex':1.6, 'oct':2.0}[params.linearch]
        archdrag = {'quad':0.95, 'hex':0.85, 'oct':0.75}[params.linearch]
        self.add_fxn('store_ee',['ee_1', 'force_st', 'hsig_bat'],fclass = StoreEE, fparams= {'bat':params.bat, 'weight':(batweight+archweight)/2.2 , 'drag': archdrag })
        self.add_fxn('dist_ee', ['ee_1','ee_mot','ee_ctl', 'force_st'], fclass = DistEE)
        self.add_fxn('affect_dof',['ee_mot','ctl','dofs','des_traj','force_lin', 'hsig_dofs'], fclass=AffectDOF, fparams = params.linearch)
        self.add_fxn('ctl_dof',['ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st'], fclass = CtlDOF)
        self.add_fxn('plan_path', ['ee_ctl', 'dofs','des_traj', 'force_st', 'rsig_traj'], fclass=PlanPath, fparams=params)
        self.add_fxn('HoldPayload',['dofs', 'force_lin', 'force_st'], fclass = HoldPayload)
        
        pos = {'manage_health': [0.23793980988102348, 1.0551602632416588],
               'store_ee': [-0.9665780995752296, -0.4931538151692423],
               'dist_ee': [-0.1858834234148632, -0.20479989209711924],
               'affect_dof': [1.0334916329507422, 0.6317263653616103],
               'ctl_dof': [0.1835014208949617, 0.32084893189175423],
               'plan_path': [-0.7427736219526058, 0.8569475547950892],
               'HoldPayload': [0.74072970715511, -0.7305391093272489]}
        
        bippos = {'manage_health': [-0.23403572483176666, 0.8119063670455383],
                  'store_ee': [-0.7099736148158298, 0.2981652748232978],
                  'dist_ee': [-0.28748133634190726, 0.32563569654296287],
                  'affect_dof': [0.9073412427515959, 0.0466423266443633],
                  'ctl_dof': [0.498663257339388, 0.44284186573420836],
                  'plan_path': [0.5353654708147643, 0.7413936186204868],
                  'HoldPayload': [0.329334798653681, -0.17443414674339652],
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
        
        self.build_model(graph_pos=pos, bipartite_pos=bippos)
        
    def find_classification(self, scen, mdlhist):
        #landing costs
        viewed = env_viewed(mdlhist['faulty']['flows']['dofs']['x'], mdlhist['faulty']['flows']['dofs']['y'],mdlhist['faulty']['flows']['dofs']['z'], self.target_area)
        viewed_value = sum([0.5+2*view for k,view in viewed.items() if view!='unviewed'])
        
        # to fix: need to find fault time more efficiently (maybe in the toolkit?)
        reshist,_,_ = rd.process.hist(mdlhist)
        faulttime = np.sum(reshist['stats']['total faults']>0)
        
        dofs=self.flows['dofs']
        if  inrange(self.start_area, dofs.x, dofs.y):     landloc = 'nominal' # nominal landing
        elif inrange(self.safe_area, dofs.x, dofs.y):     landloc = 'designated' # emergency safe
        elif inrange(self.target_area, dofs.x, dofs.y):   landloc = 'over target' # emergency dangerous
        else:                                           landloc = 'outside target' # emergency unsanctioned
        # need a way to differentiate horizontal and vertical crashes/landings
        if landloc in ['over target', 'outside target']: 
            if landloc=="outside target" and self.valparams['loc']=='congested':    loc='urban'
            else:                                                                   loc=self.valparams['loc']
            body_strikes = density_categories[loc]['body strike']['horiz']
            head_strikes = density_categories[loc]['head strike']['horiz']
            property_restrictions = 1
        else: body_strikes =0.0; head_strikes=0.0; property_restrictions=0
        
        safecost = safety_categories['hazardous']['cost'] * (head_strikes + body_strikes) + unsafecost[self.valparams['loc']] * faulttime
        landcost = property_restrictions*propertycost[self.valparams['loc']]
        #repair costs
        repcost=self.calc_repaircost(max_cost=1500)
        rate=scen['properties']['rate']
        p_safety = 1-np.exp(-(body_strikes+head_strikes) * 60/(faulttime+0.001)) #convert to pfh
        classifications = {'hazardous':rate*p_safety, 'minor':rate*(1-p_safety)}

        totcost=repcost+landcost+safecost-viewed_value
        
        expcost=totcost*rate*1e5
        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost, 'repcost':repcost, 
                'landcost':landcost,'safecost':safecost,'viewed value': viewed_value, 'viewed':viewed, 
                'landloc':landloc,'body strikes':body_strikes, 'head strikes':head_strikes, 
                'property restrictions': property_restrictions, 'severities':classifications, 
                'unsafe flight time':faulttime}

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
    xnom=mdlhist['flows']['dofs']['x']
    ynom=mdlhist['flows']['dofs']['y']
    znom=mdlhist['flows']['dofs']['z']
    
    time = mdlhist['time']
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    plt.show()

def plot_faulttraj(mdlhist, params, title='Fault response to RFpropbreak fault at t=20'):
    xnom=mdlhist['nominal']['flows']['dofs']['x']
    ynom=mdlhist['nominal']['flows']['dofs']['y']
    znom=mdlhist['nominal']['flows']['dofs']['z']
    #
    x=mdlhist['faulty']['flows']['dofs']['x']
    y=mdlhist['faulty']['flows']['dofs']['y']
    z=mdlhist['faulty']['flows']['dofs']['z']
    
    time = mdlhist['nominal']['time']
    
    
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
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    ax2.legend(['Nominal Flightpath','Faulty Flighpath'], loc=4)
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
    xnom=mdlhist['flows']['dofs']['x']
    ynom=mdlhist['flows']['dofs']['y']
    
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
    sq = square(def_mdl.params.target[0:2],def_mdl.params.target[2],def_mdl.params.target[3])
    landing = [*def_mdl.params.start[0:2], 0]
    
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
opt_prob.add_constraints("ocost", g_soc=("store_ee.soc", "vars", "end",("greater", 20)),
                                  g_max_height=("dofs.z", "vars", "all", ("less", 122)),
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

#rd.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'dofs'}, time_slice=6)
#rd.plot.mdlhists(opt_prob._sims['rcost']['mdlhists']['store_ee lowcharge, t=7.0'], fxnflowvals={'store_ee'}, time_slice=6)
#(variablename, objtype (optional), t (optional))


def calc_oper(mdl):
    endresults_nom, mdlhist =propagate.nominal(mdl)
    opercost = endresults_nom['expected cost']
    g_soc = 20 - mdlhist['functions']['store_ee']['soc'][-1] 
    #g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist['flows']['dofs']['z']-122 if i>0])
    
    phases, modephases=rd.process.modephases(mdlhist)
    return opercost, g_soc, g_max_height, phases
def x_to_ocost(xdes, xoper, loc='rural'):
    fp = plan_flight(xoper[0], def_mdl)
    params = DroneParam(bat=bats[xdes[0]], linearch=linarchs[xdes[1]], respolicy=ResPolicy(bat='continue', line='continue'))
    mdl = Drone(params=params)
    return calc_oper(mdl)

def calc_res(mdl, fullcosts=False, faultmodes = 'all', include_nominal=True, pool=False, phases={}, staged=True):
    #app = SampleApproach(mdl, faults=('single-component', faultmodes), phases={'forward'})
    app = SampleApproach(mdl, faults=('single-component', 'store_ee'), phases={'move':phases['plan_path']['move']})
    endclasses, mdlhists = propagate.approach(mdl, app, staged=staged, pool=pool, showprogress=False) #, staged=False)
    rescost = rd.process.totalcost(endclasses)-(not include_nominal)*endclasses['nominal']['expected cost']
    #rd.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'dofs'}, time_slice=6)
    #rd.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=7.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'store_ee'}, time_slice=6)
    #rd.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'plan_path'}, time_slice=6)
    #rd.plot.mdlhists({'faulty':mdlhists['store_ee lowcharge, t=6.0'], 'nominal':mdlhists['nominal']}, fxnflowvals={'rsig_traj', 'hsig_bat','hsig_dofs'})
    #[ec['expected cost'] for ec in endclasses.values()]
    #[ec['endclass']['expected cost'] for ec in opt_prob._sims['rcost']['results'].values()]
    #plot_faulttraj({'nominal':mdlhists['nominal'], 'faulty':mdlhists['store_ee lowcharge, t=7.0']}, mdl.params, title='Fault response to store_ee lowcharge, t=6.0')
    #phases, modephases = rd.process.modephases(mdlhists['nominal'])
    #rd.plot.phases({p:ph for p,ph in phases.items() if p=='plan_path'}, modephases)
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
    mdl = Drone(params=params)
    if not phases: _,_,_, phases = calc_oper(mdl)
    return calc_res(mdl, fullcosts=fullcosts, faultmodes = faultmodes, include_nominal=include_nominal, pool=pool, phases=phases, staged=staged)

if __name__=="__main__":
    import fmdtools.faultsim.propagate as prop
    import matplotlib.pyplot as plt
    

    #opt_prob.add_combined_objective("total_cost", 'cd', 'co', 'cr')
    #opt_prob.total_cost([1,1],[100],[1,1])
    #opt_prob.total_cost([1,1,100,1,1])
    #opt_prob.time_sims([1,1,100,1,1])
    
    opt_prob.cr([2,2, 50, 0,0])
    
    x_to_rcost([2,2],[50],[0,0], faultmodes='store_ee')
    x_to_rcost([0,0],[50],[0,0], faultmodes='store_ee')
    opt_prob.show_architecture()
    
    #opt_prob.update_sim_options("ocost", track={"functions":{"plan_path":"all"}, "flows":{"dofs":"all"}})
    #opt_prob.update_sim_options("rcost", log_iter_hist=True, pool=mp.Pool(4), track={"functions":{"store_ee":"faults"}, "flows":{"dofs":"all"}})
    
    #opt_prob.total_cost([1,1,120,1,1])
    #opt_prob.total_cost([1,1,60,1,1])
    
    opt_prob.time_sims([1,1,100,1,1])
    
    opt_prob.iter_hist
    
    mdl = Drone()
    ec, mdlhist = prop.nominal(mdl)
    phases, modephases = rd.process.modephases(mdlhist)
    rd.plot.phases(phases, modephases)

    app = SampleApproach(mdl,  phases={'forward'})
    plt.show()
    
    #endclasses, mdlhists = prop.approach(mdl, app, staged=True)
    #plot_faulttraj({'nominal':mdlhists['nominal'], 'faulty':mdlhists['store_ee lowcharge, t=6.0']}, mdl.params, title='Fault response to RFpropbreak fault at t=20')

