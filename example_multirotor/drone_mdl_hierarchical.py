# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fmdtools.faultsim as fs
import fmdtools.resultdisp as rd
from IPython.display import HTML

from fmdtools.modeldef.common import Parameter, State
from fmdtools.modeldef.block import FxnBlock, Component, CompArch
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.approach import SampleApproach

from drone_mdl_static import m2to1, EngageLand, HoldPayload, DistEE
from drone_mdl_static import Force, EE, Control, DOFs, Env, Dir
from drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, Trajectory, ViewEnvironment
            
class OverallAffectDOFState(State):
    lrstab:     float=0.0
    FRstab:     float=0.0

class AffectDOFArch(CompArch):
    archtype:   str='quad'
    upward:     dict= dict()
    forward:    dict=dict()
    lr_dict:    dict=dict()
    FR_dict:    dict=dict()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.archtype=="quad":
            kwargs['components'], kwargs['faultmodes'] = self.make_components(Line,'lf', 'lr','rf','rr')
            self.upward.update({'rf':1,'lf':1,'lr':1,'rr':1})
            self.forward.update({'rf':0.5,'lf':0.5,'lr':-0.5,'rr':-0.5})
            self.lr_dict.update({'l':{'lf', 'lr'}, 'r':{'rf','rr'}})
            self.fr_dict.update({'f':{'lf', 'rf'}, 'r':{'lr', 'rr'}})
        elif self.archtype=="oct":
            kwargs['components'], kwargs['faultmodes'] = self.make_components(Line,'lf', 'rf','lf2', 'rf2', 'lr', 'rr','lr2', 'rr2')
            self.upward.update({'rf':1,'lf':1,'lr':1,'rr':1,'rf2':1,'lf2':1,'lr2':1,'rr2':1})
            self.forward.update({'rf':0.5,'lf':0.5,'lr':-0.5,'rr':-0.5,'rf2':0.5,'lf2':0.5,'lr2':-0.5,'rr2':-0.5})
            self.lr_dict.update({'L':{'lf', 'lr','lf2', 'lr2'}, 'R':{'rf','rr','rf2','rr2'}})
            self.fr_dict.update({'F':{'lf', 'rf','lf2', 'rf2'}, 'R':{'lr', 'rr','lr2', 'rr2'}})

class AffectDOF(FxnBlock): #EEmot,ctl,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    _init_s = OverallAffectDOFState
    _init_c = AffectDOFArch
    _init_ee_in = EE
    _init_ctl_in = Control
    _init_dofs = DOFs
    _init_force = Force
    flownames = {'ee_lin':'ee_in', 'ctl':'ctl_in','force_st':'force'}
    def behavior(self, time):
        air, ee_in={},{}
        #injects faults into lines
        for linname,lin in self.c.components.items():
            cmds={'up':self.c.upward[linname], 'for':self.c.forward[linname]}
            air[lin.name], ee_in[lin.name] = lin.behavior(self.ee_in.s.effort, self.ctl_in, cmds, self.force.s.support) 
        
        if any(value>=10 for value in ee_in.values()):      self.ee_in.s.rate=10
        elif any(value!=0.0 for value in ee_in.values()):   self.ee_in.s.rate=sum(ee_in.values())/len(ee_in) #should it really be max?
        else:                                               self.ee_in.s.rate=0.0
        
        self.s.lrstab = (sum([air[comp] for comp in self.c.lr_dict['L']])-sum([air[comp] for comp in self.c.lr_dict['R']]))/len(air)
        self.s.FRstab = (sum([air[comp] for comp in self.c.FR_dict['R']])-sum([air[comp] for comp in self.c.fr_dict['F']]))/len(air)
        
        if abs(self.s.lrstab) >=0.4 or abs(self.s.FRstab)>=0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            airs=list(air.values())
            self.dofs.s.uppwr=np.mean(airs)
            self.dofs.s.planpwr=-2*self.s.FRstab

from drone_mdl_static import AffectDOFMode, AffectDOFState
class Line(Component):
    _init_s = AffectDOFState
    _init_m = AffectDOFMode
    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.m.add_fault('mechbreak','propbreak')
        elif Force<=0.5: self.m.add_fault('mechfriction')
            
        if self.m.has_fault('short'):                   self.s.put(Eti=0.0, Eto= np.inf)
        elif self.m.has_fault('openc'):                 self.s.put(Eti=0.0, Eto= 0.0)
        elif Ctlin.s.upward==0 and Ctlin.s.forward == 0:self.s.put(Eto= 0.0)
        else:                                           self.s.put(Eto= 1.0)
        
        if self.m.has_fault('ctlbreak'):                self.s.put(Ct=0.0)
        elif self.m.has_fault('ctldn'):                 self.s.put(Ct=0.5)
        elif self.m.has_fault('ctlup'):                 self.s.put(Ct=2.0)
        
        if self.m.has_fault('mechbreak'):               self.s.put(Mt=0.0)
        elif self.m.has_fault('mechfriction'):          self.s.put(Mt=0.5, Eti= 2.0) 
        
        if self.m.has_fault('propstuck'):               self.s.put(Pt=0.0, Mt=0.0, Eti= 4.0) 
        elif self.m.has_fault('propbreak'):             self.s.put(Pt=0.0)
        elif self.m.has_fault('propwarp'):              self.s.put(Pt=0.5)
        
        Airout=m2to1([EEin,self.s.Eti,Ctlin.s.upward*cmds['up']+Ctlin.s.forward*cmds['for'],self.s.Ct,self.s.Mt,self.s.Pt])
        EE_in=m2to1([EEin,self.s.Eto])   
        return Airout, EE_in

class DroneParam(Parameter, readonly=True):
    arch:   str='quad'
    arch_set = ('quad', 'oct', 'hex')


class Drone(Model):
    def __init__(self, params=DroneParam(),\
            modelparams=ModelParam(phases=(('ascend',0,4),('forward',5,94),('descend',95, 100)),times=(0,135),units='sec'), 
            valparams={}):
        super().__init__(params, modelparams, valparams)
        #add flows to the model
        self.add_flow('force_st',   Force)
        self.add_flow('force_lin',  Force)
        self.add_flow('force_gr' ,  Force)
        self.add_flow('force_lg',   Force)
        self.add_flow('ee_1',       EE)
        self.add_flow('ee_mot',     EE)
        self.add_flow('ee_ctl',     EE)
        self.add_flow('ctl',       Control)
        self.add_flow('dofs',       DOFs)
        self.add_flow('env',        Env)
        self.add_flow('dir',        Dir)
        #add functions to the model
        self.add_fxn('store_ee',    ['ee_1', 'force_st'],                           fclass=StoreEE)
        self.add_fxn('dist_ee',     ['ee_1','ee_mot','ee_ctl', 'force_st'],         fclass=DistEE)
        self.add_fxn('affect_dof',  ['ee_mot','ctl','dofs','force_lin'],           fclass=AffectDOF, fkwargs={'c':{'archtype':params.arch}})
        self.add_fxn('ctl_dof',     ['ee_ctl', 'dir', 'ctl', 'dofs', 'force_st'],  fclass=CtlDOF)
        self.add_fxn('plan_path',   ['ee_ctl', 'env','dir', 'force_st'],            fclass=PlanPath)
        self.add_fxn('trajectory',  ['env','dofs','dir', 'force_gr'],               fclass=Trajectory)
        self.add_fxn('engage_land', ['force_gr', 'force_lg'],                       fclass=EngageLand)
        self.add_fxn('hold_payload',['force_lg', 'force_lin', 'force_st'],          fclass=HoldPayload)
        self.add_fxn('view_env',    ['env'],                                        fclass=ViewEnvironment)
        
        bipartite_pos = {'store_ee': [-1.067135163123663, 0.32466987344741055],
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
        
        self.build_model(graph_pos=graph_pos, bipartite_pos=bipartite_pos)
    def find_classification(self,scen, mdlhists):
        if -5 >mdlhists['faulty']['flows']['env']['x'][-1] or 5<mdlhists['faulty']['flows']['env']['x'][-1]:
            lostcost=50000
        elif -5 >mdlhists['faulty']['flows']['env']['y'][-1] or 5<mdlhists['faulty']['flows']['env']['y'][-1]:
            lostcost=50000
        elif mdlhists['faulty']['flows']['env']['z'][-1] >5:
            lostcost=50000
        else:
            lostcost=0
        
        if any(abs(mdlhists['faulty']['flows']['Force_GR']['support'])>2.0):
            crashcost = 100000
        else:
            crashcost = 0
        repcost = self.calc_repaircost()
        
        totcost=repcost + crashcost + lostcost
        rate=scen['properties']['rate']
        expcost=totcost*rate*1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

if __name__=="__main__":
    
    hierarchical_model = Drone(params=DroneParam(arch='quad'))
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,'affect_dof', 'rfmechbreak', time=50)
    
    mdl = Drone(params=DroneParam(arch='oct'))
    app = SampleApproach(mdl, faults=[('affect_dof', 'rr2propstuck')])
    endclasses, mdlhists = fs.propagate.approach(mdl, app, staged=False)
    rd.plot.mdlhists({'nominal': mdlhists['nominal'],'faulty': mdlhists['affect_dof rr2propstuck, t=49.0']},fxnflowvals='env')









