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
from drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, Trajectory, ViewEnvironment
            
class OverallAffectDOFState(State):
    LRstab:     float=0.0
    FRstab:     float=0.0

class AffectDOFArch(CompArch):
    archtype: str='quad'
    upward = dict()
    forward = dict()
    LR_dict = dict()
    FR_dict = dict()
    def __init__(self, *args, **kwargs):
        if self.archtype=="quad":
            kwargs['components'], kwargs['faultmodes'] = self.make_components(Line,'LF', 'LR','RF','RR')
            self.upward.update({'RF':1,'LF':1,'LR':1,'RR':1})
            self.forward.update({'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5})
            self.LR_dict.update({'L':{'LF', 'LR'}, 'R':{'RF','RR'}})
            self.FR_dict.update({'F':{'LF', 'RF'}, 'R':{'LR', 'RR'}})
        elif self.archtype=="oct":
            kwargs['components'], kwargs['faultmodes'] = self.make_components(Line,'LF', 'RF','LF2', 'RF2', 'LR', 'RR','LR2', 'RR2')
            self.upward.update({'RF':1,'LF':1,'LR':1,'RR':1,'RF2':1,'LF2':1,'LR2':1,'RR2':1})
            self.forward.update({'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5,'RF2':0.5,'LF2':0.5,'LR2':-0.5,'RR2':-0.5})
            self.LR_dict.update({'L':{'LF', 'LR','LF2', 'LR2'}, 'R':{'RF','RR','RF2','RR2'}})
            self.FR_dict.update({'F':{'LF', 'RF','LF2', 'RF2'}, 'R':{'LR', 'RR','LR2', 'RR2'}})
        super().__init__(*args, **kwargs)

class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    _init_s = OverallAffectDOFState
    _init_c = AffectDOFArch
    def __init__(self, name, flows, params={}, **kwargs):     
        super().__init__(name, flows, ['EEin', 'Ctlin','DOF','Force'], **kwargs)
    def behavior(self, time):
        Air,EEin={},{}
        #injects faults into lines
        for linname,lin in self.c.components.items():
            cmds={'up':self.c.upward[linname], 'for':self.c.forward[linname]}
            Air[lin.name] ,EEin[lin.name] = lin.behavior(self.EEin.s.effort, self.Ctlin, cmds, self.Force.s.support) 
        
        if any(value>=10 for value in EEin.values()):       self.EEin.s.rate=10
        elif any(value!=0.0 for value in EEin.values()):    self.EEin.s.rate=sum(EEin.values())/len(EEin) #should it really be max?
        else:                                               self.EEin.s.rate=0.0
        
        self.s.LRstab = (sum([Air[comp] for comp in self.c.LR_dict['L']])-sum([Air[comp] for comp in self.c.LR_dict['R']]))/len(Air)
        self.s.FRstab = (sum([Air[comp] for comp in self.c.FR_dict['R']])-sum([Air[comp] for comp in self.c.FR_dict['F']]))/len(Air)
        
        if abs(self.s.LRstab) >=0.4 or abs(self.s.FRstab)>=0.75:
            self.DOF.s.uppwr=0.0
            self.DOF.s.planpwr=0.0
        else:
            Airs=list(Air.values())
            self.DOF.s.uppwr=np.mean(Airs)
            self.DOF.s.planpwr=-2*self.s.FRstab

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

from drone_mdl_static import Force, EE, Control, DOFs, Env, Dir
class Drone(Model):
    def __init__(self, params=DroneParam(),\
            modelparams=ModelParam(phases=(('ascend',0,4),('forward',5,94),('descend',95, 100)),times=(0,135),units='sec'), 
            valparams={}):
        
        super().__init__(params, modelparams, valparams)
                                     
        #add flows to the model
        self.add_flow('Force_ST',   Force)
        self.add_flow('Force_Lin',  Force)
        self.add_flow('Force_GR' ,  Force)
        self.add_flow('Force_LG',   Force)
        self.add_flow('EE_1',       EE)
        self.add_flow('EEmot',      EE)
        self.add_flow('EEctl',      EE)
        self.add_flow('Ctl1',       Control)
        self.add_flow('DOFs',       DOFs)
        self.add_flow('Env1',       Env, s={'z':0.0} )
        self.add_flow('Dir1',       Dir)
        #add functions to the model
        flows=['EEctl', 'Force_ST']
        self.add_fxn('StoreEE',['EE_1', 'Force_ST'], fclass=StoreEE)
        self.add_fxn('DistEE', ['EE_1','EEmot','EEctl', 'Force_ST'], fclass=DistEE)
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Force_Lin'], fclass=AffectDOF, fkwargs={'c':{'archtype':params.arch}})
        self.add_fxn('CtlDOF', ['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST'], fclass=CtlDOF)
        self.add_fxn('Planpath', ['EEctl', 'Env1','Dir1', 'Force_ST'], fclass=PlanPath)
        self.add_fxn('Trajectory', ['Env1','DOFs','Dir1', 'Force_GR'], fclass=Trajectory)
        self.add_fxn('EngageLand',['Force_GR', 'Force_LG'], fclass=EngageLand)
        self.add_fxn('HoldPayload',['Force_LG', 'Force_Lin', 'Force_ST'], fclass=HoldPayload)
        self.add_fxn('ViewEnv', ['Env1'], fclass=ViewEnvironment)
        
        bipartite_pos = {'StoreEE': [-1.067135163123663, 0.32466987344741055],
         'DistEE': [-0.617149602161968, 0.3165981670924663],
         'AffectDOF': [0.11827439153655106, 0.10792528450121897],
         'CtlDOF': [-0.2636856982162134, 0.42422600969836144],
         'Planpath': [-0.9347151173753852, 0.6943421719257798],
         'Trajectory': [0.6180477286739998, 0.32930706399226856],
         'EngageLand': [0.0015917696269229786, -0.2399760932810826],
         'HoldPayload': [-0.8833099612826893, -0.247201580673997],
         'ViewEnv': [0.5725955705698363, 0.6901513410348765],
         'Force_ST': [-0.8925771348524384, -0.025638904424547027],
         'Force_Lin': [-0.5530952425102891, -0.10380834289626095],
         'Force_GR': [0.568921162299461, -0.22991830334765573],
         'Force_LG': [-0.37244114591548894, -0.2355298479531287],
         'EE_1': [-0.809433489993954, 0.319191761486317],
         'EEmot': [-0.33469985340998853, 0.1307636433702345],
         'EEctl': [-0.48751243650229525, 0.4852032717825657],
         'Ctl1': [-0.06913038312848868, 0.2445174568603189],
         'DOFs': [0.2606664304933561, 0.3243482171363975],
         'Env1': [0.06157634305459603, 0.7099922980251693],
         'Dir1': [-0.13617863906968142, 0.6037252153639261]}

        graph_pos = {'StoreEE': [-1.0787279392101061, -0.06903523859088145],
         'DistEE': [-0.361531174332526, -0.0935883732235363],
         'AffectDOF': [0.36541282312106205, -0.09674444529230719],
         'CtlDOF': [0.4664934329906758, 0.5822138245848214],
         'Planpath': [-0.7095750728126631, 0.8482786785038505],
         'Trajectory': [1.1006824683444765, -0.10423208715241583],
         'EngageLand': [0.8423521094741182, -0.8813666134484857],
         'HoldPayload': [-0.5857395187723944, -0.86974898769837],
         'ViewEnv': [1.1035500215472247, 0.9373523025760659]}
        
        self.build_model(graph_pos=graph_pos, bipartite_pos=bipartite_pos)
    def find_classification(self,scen, mdlhists):
        if -5 >mdlhists['faulty']['flows']['Env1']['x'][-1] or 5<mdlhists['faulty']['flows']['Env1']['x'][-1]:
            lostcost=50000
        elif -5 >mdlhists['faulty']['flows']['Env1']['y'][-1] or 5<mdlhists['faulty']['flows']['Env1']['y'][-1]:
            lostcost=50000
        elif mdlhists['faulty']['flows']['Env1']['z'][-1] >5:
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
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,'AffectDOF', 'RFmechbreak', time=50)
    
    mdl = Drone(params=DroneParam(arch='oct'))
    app = SampleApproach(mdl, faults=[('AffectDOF', 'RR2propstuck')])
    endclasses, mdlhists = fs.propagate.approach(mdl, app, staged=False)
    rd.plot.mdlhists({'nominal': mdlhists['nominal'],'faulty': mdlhists['AffectDOF RR2propstuck, t=49.0']},fxnflowvals='Env1')









