# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
import fmdtools.sim as fs
import fmdtools.analyze as an

from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.block import FxnBlock, Component, CompArch
from fmdtools.sim.approach import SampleApproach

from examples.multirotor.drone_mdl_static import m2to1, EngageLand, HoldPayload, DistEE
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, Env, Dir
from examples.multirotor.drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, Trajectory, ViewEnvironment
            
class OverallAffectDOFState(State):
    lrstab:     float=0.0
    frstab:     float=0.0

class AffectDOFArch(CompArch):
    archtype:   str='quad'
    forward:    dict=dict()
    lr_dict:    dict=dict()
    fr_dict:    dict=dict()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.archtype=="quad":
            self.make_components(Line,'lf', 'lr','rf','rr')
            self.forward.update({'rf':0.5,'lf':0.5,'lr':-0.5,'rr':-0.5})
            self.lr_dict.update({'l':{'lf', 'lr'}, 'r':{'rf','rr'}})
            self.fr_dict.update({'f':{'lf', 'rf'}, 'r':{'lr', 'rr'}})
        elif self.archtype=="hex":
            self.make_components(Line,'rf', 'lf','lr','rr', 'r', 'f')
            self.forward.update({'rf':0.5,'lf':0.5,'lr':-0.5,'rr':-0.5, 'r':-0.75, 'f':0.75})
            self.lr_dict.update({'l':{'lf', 'lr'}, 'r':{'rf','rr'}})
            self.fr_dict.update({'f':{'lf', 'rf', 'f'}, 'r':{'lr', 'rr', 'r'}})
        elif self.archtype=="oct":
            self.make_components(Line,'lf', 'rf','lf2', 'rf2', 'lr', 'rr','lr2', 'rr2')
            self.forward.update({'rf':0.5,'lf':0.5,'lr':-0.5,'rr':-0.5,'rf2':0.5,'lf2':0.5,'lr2':-0.5,'rr2':-0.5})
            self.lr_dict.update({'l':{'lf', 'lr','lf2', 'lr2'}, 'r':{'rf','rr','rf2','rr2'}})
            self.fr_dict.update({'f':{'lf', 'rf','lf2', 'rf2'}, 'r':{'lr', 'rr','lr2', 'rr2'}})

class AffectDOF(FxnBlock): #EEmot,ctl,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    __slots__=('ee_in', 'ctl_in', 'dofs', 'force')
    _init_s = OverallAffectDOFState
    _init_c = AffectDOFArch
    _init_ee_in = EE
    _init_ctl_in = Control
    _init_dofs = DOFs
    _init_force = Force
    flownames = {'ee_mot':'ee_in', 'ctl':'ctl_in','force_lin':'force'}
    def behavior(self, time):
        air, ee_in={},{}
        #injects faults into lines
        for linname,lin in self.c.components.items():
            air[lin.name], ee_in[lin.name] = lin.behavior(self.ee_in.s.effort, self.ctl_in, self.c.forward[linname], self.force.s.support) 
        
        if any(value>=10 for value in ee_in.values()):      self.ee_in.s.rate=10
        elif any(value!=0.0 for value in ee_in.values()):   self.ee_in.s.rate=sum(ee_in.values())/len(ee_in) #should it really be max?
        else:                                               self.ee_in.s.rate=0.0
        
        self.s.lrstab = (sum([air[comp] for comp in self.c.lr_dict['l']])-sum([air[comp] for comp in self.c.lr_dict['r']]))/len(air)
        self.s.frstab = (sum([air[comp] for comp in self.c.fr_dict['r']])-sum([air[comp] for comp in self.c.fr_dict['f']]))/len(air)
        
        if abs(self.s.lrstab) >=0.4 or abs(self.s.frstab)>=0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            airs=list(air.values())
            self.dofs.s.uppwr=np.mean(airs)
            self.dofs.s.planpwr=-2*self.s.frstab

from examples.multirotor.drone_mdl_static import AffectDOFMode, AffectDOFState
class Line(Component):
    _init_s = AffectDOFState
    _init_m = AffectDOFMode
    def behavior(self,ee_in, ctlin, f_fact, force):
        if force<=0.0:   self.m.add_fault('mechbreak','propbreak')
        elif force<=0.5: self.m.add_fault('mechfriction')
            
        if self.m.has_fault('short'):                   self.s.put(e_ti=0.0, e_to= np.inf)
        elif self.m.has_fault('openc'):                 self.s.put(e_ti=0.0, e_to= 0.0)
        elif ctlin.s.upward==0 and ctlin.s.forward == 0:self.s.put(e_to= 0.0)
        else:                                           self.s.put(e_to= 1.0)
        
        if self.m.has_fault('ctlbreak'):                self.s.put(ct=0.0)
        elif self.m.has_fault('ctldn'):                 self.s.put(ct=0.5)
        elif self.m.has_fault('ctlup'):                 self.s.put(ct=2.0)
        
        if self.m.has_fault('mechbreak'):               self.s.put(mt=0.0)
        elif self.m.has_fault('mechfriction'):          self.s.put(mt=0.5, e_ti= 2.0) 
        
        if self.m.has_fault('propstuck'):               self.s.put(pt=0.0, mt=0.0, e_ti= 4.0) 
        elif self.m.has_fault('propbreak'):             self.s.put(pt=0.0)
        elif self.m.has_fault('propwarp'):              self.s.put(pt=0.5)
        
        airout=m2to1([ee_in,self.s.e_ti,ctlin.s.upward+ctlin.s.forward*f_fact,self.s.ct,self.s.mt,self.s.pt])
        ee_in=m2to1([ee_in,self.s.e_to])   
        return airout, ee_in

from examples.multirotor.drone_mdl_dynamic import Drone as DynDrone

class DroneParam(Parameter, readonly=True):
    arch:   str='quad'
    arch_set = ('quad', 'oct', 'hex')

class Drone(DynDrone):
    __slots__=()
    _init_p = DroneParam
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #add flows to the model
        self.add_flow('force_st',   Force)
        self.add_flow('force_lin',  Force)
        self.add_flow('force_gr' ,  Force)
        self.add_flow('force_lg',   Force)
        self.add_flow('ee_1',       EE)
        self.add_flow('ee_mot',     EE)
        self.add_flow('ee_ctl',     EE)
        self.add_flow('ctl',        Control)
        self.add_flow('dofs',       DOFs)
        self.add_flow('env',        Env, s={'z':0.0})
        self.add_flow('dir',        Dir)
        #add functions to the model
        self.add_fxn('store_ee',    StoreEE,    'ee_1', 'force_st')
        self.add_fxn('dist_ee',     DistEE,     'ee_1','ee_mot','ee_ctl', 'force_st')
        self.add_fxn('affect_dof',  AffectDOF,  'ee_mot','ctl','dofs','force_lin', c={'archtype':self.p.arch})
        self.add_fxn('ctl_dof',     CtlDOF,     'ee_ctl', 'dir', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path',   PlanPath,   'ee_ctl', 'env','dir', 'force_st')
        self.add_fxn('trajectory',  Trajectory, 'env','dofs','dir', 'force_gr')
        self.add_fxn('engage_land', EngageLand, 'force_gr', 'force_lg')
        self.add_fxn('hold_payload',HoldPayload,'force_lg', 'force_lin', 'force_st')
        self.add_fxn('view_env',    ViewEnvironment, 'env')
        
        self.build()


fxnflowgraph_pos = {'store_ee': [-1.067135163123663, 0.32466987344741055],
 'dist_ee': [-0.617149602161968, 0.3165981670924663],
 'affect_dof': [0.11827439153655106, 0.10792528450121897],
 'ctl_dof': [-0.2636856982162134, 0.42422600969836144],
 'plan_path': [-0.9347151173753852, 0.6943421719257798],
 'trajectory': [0.6180477286739998, 0.32930706399226856],
 'engage_land': [0.0015917696269229786, -0.2399760932810826],
 'hold_payload': [-0.8833099612826893, -0.247201580673997],
 'view_env': [0.5725955705698363, 0.6901513410348765],
 'force_st': [-0.8925771348524384, -0.025638904424547027],
 'force_lin': [-0.5530952425102891, -0.10380834289626095],
 'force_gr': [0.568921162299461, -0.22991830334765573],
 'force_lg': [-0.37244114591548894, -0.2355298479531287],
 'ee_1': [-0.809433489993954, 0.319191761486317],
 'ee_mot': [-0.33469985340998853, 0.1307636433702345],
 'ee_ctl': [-0.48751243650229525, 0.4852032717825657],
 'ctl': [-0.06913038312848868, 0.2445174568603189],
 'dofs': [0.2606664304933561, 0.3243482171363975],
 'env': [0.06157634305459603, 0.7099922980251693],
 'dir': [-0.13617863906968142, 0.6037252153639261]}

graph_pos = {'store_ee': [-1.0787279392101061, -0.06903523859088145],
 'dist_ee': [-0.361531174332526, -0.0935883732235363],
 'affect_dof': [0.36541282312106205, -0.09674444529230719],
 'ctl_dof': [0.4664934329906758, 0.5822138245848214],
 'plan_path': [-0.7095750728126631, 0.8482786785038505],
 'trajectory': [1.1006824683444765, -0.10423208715241583],
 'engage_land': [0.8423521094741182, -0.8813666134484857],
 'hold_payload': [-0.5857395187723944, -0.86974898769837],
 'view_env': [1.1035500215472247, 0.9373523025760659]}

if __name__=="__main__":
    
    hierarchical_model = Drone(p=DroneParam(arch='quad'))
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,'affect_dof', 'rf_mechbreak', time=50)
    
    mdl = Drone(p=DroneParam(arch='oct'))
    app = SampleApproach(mdl, faults=[('affect_dof', 'rr2_propstuck')])
    endclasses, mdlhists = fs.propagate.approach(mdl, app, staged=True)
    an.plot.hist(mdlhists.get('nominal', 'affect_dof_rr2_propstuck_t49p0').flatten(),
                 'flows.env.s.x', 'env.s.y', 'env.s.z', 'store_ee.s.soc')









