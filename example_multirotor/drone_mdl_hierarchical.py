# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""

import sys, os
sys.path.insert(1, os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fmdtools.faultsim as fs
import fmdtools.resultdisp as rd
import quadpy
from IPython.display import HTML

from fmdtools.modeldef import FxnBlock
from fmdtools.modeldef import Flow
from fmdtools.modeldef import Model
from fmdtools.modeldef import m2to1
from fmdtools.modeldef import Component
from fmdtools.modeldef import SampleApproach

class StoreEE(FxnBlock):
    def __init__(self, name, flows):
        self.failrate=1e-5
        super().__init__(name, flows, ['EEout', 'FS'], {'soc': 100.0})
        self.assoc_modes({'nocharge':[1,[0.6,0.1,0.1],300]})
    def condfaults(self,time):
        if self.soc<1:
            self.soc=0
            self.add_fault('nocharge')
    def behavior(self, time):
        if      self.has_fault('nocharge'):   self.EEout.effort=0.0
        else: self.EEout.effort=1.0
        if time > self.time:
            self.soc=self.soc-self.EEout.effort*self.EEout.rate*(time-self.time)/2
class DistEE(FxnBlock):
    def __init__(self, name,flows):
        super().__init__(name, flows, ['EEin','EEmot','EEctl','ST'], {'EEtr':1.0, 'EEte':1.0})
        self.failrate=1e-5
        self.assoc_modes({'short':[0.3,[0.33, 0.33, 0.33],3000], 'degr':[0.5,[0.33, 0.33, 0.33],1000],\
                          'break':[0.2,[0.33, 0.33, 0.33],2000]})
    def condfaults(self, time):
        if self.ST.support<0.5 or max(self.EEmot.rate,self.EEctl.rate)>2: 
            self.add_fault('break')
        if self.EEin.rate>2:
            self.add_fault('short')
    def behavior(self, time):
        if self.has_fault('short'): 
            self.EEte=0.0
            self.EEre=10
        elif self.has_fault('break'): 
            self.EEte=0.0
            self.EEre=0.0
        elif self.has_fault('degr'): self.EEte=0.5
        self.EEmot.effort=self.EEte*self.EEin.effort
        self.EEctl.effort=self.EEte*self.EEin.effort
        self.EEin.rate=m2to1([ self.EEin.effort, self.EEtr, 0.9*self.EEmot.rate+0.1*self.EEctl.rate])
class EngageLand(FxnBlock):
    def __init__(self, name,flows):
        super().__init__(name, flows, ['forcein', 'forceout'])
        self.failrate=1e-5
        self.assoc_modes({'break':[0.2,[0.5,0.0,0.5], 1000], 'deform':[0.8,[0.5,0.0,0.5], 1000]})
    def condfaults(self, time):
        if abs(self.forcein.value)>=2.0:      self.add_fault('break')
        elif abs(self.forcein.value)>1.5:    self.add_fault('deform')
    def behavior(self, time):
        self.forceout.value=self.forcein.value/2
            
class HoldPayload(FxnBlock):
    def __init__(self, name,flows):
        super().__init__(name, flows, ['FG', 'Lin', 'ST'])
        self.failrate=1e-6
        self.assoc_modes({'break':[0.2, [0.33, 0.33, 0.33], 10000], 'deform':[0.8, [0.33, 0.33, 0.33], 10000]})
    def condfaults(self, time):
        if abs(self.FG.value)>0.8:      self.add_fault('break')
        elif abs(self.FG.value)>1.0:    self.add_fault('deform')
    def behavior(self, time):
        #need to transfer FG to FA & FS???
        if self.has_fault('break'):     self.Lin.support, self.ST.support = 0,0
        elif self.has_fault('deform'):  self.Lin.support, self.ST.support = 0.5,0.5
        else:                           self.Lin.support, self.ST.support = 1.0,1.0
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    def __init__(self, name, flows, archtype):     
        self.archtype=archtype
        if archtype=='quad':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
            self.LR_dict = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR_dict = {'F':{'LF', 'RF'}, 'R':{'LR', 'RR'}}
        elif archtype=='oct':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR'),'RF2':Line('RF2'), 'LF2':Line('LF2'), 'LR2':Line('LR2'), 'RR2':Line('RR2')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1,'RF2':1,'LF2':1,'LR2':1,'RR2':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5,'RF2':0.5,'LF2':0.5,'LR2':-0.5,'RR2':-0.5}
            self.LR_dict = {'L':{'LF', 'LR','LF2', 'LR2'}, 'R':{'RF','RR','RF2','RR2'}}
            self.FR_dict = {'F':{'LF', 'RF','LF2', 'RF2'}, 'R':{'LR', 'RR','LR2', 'RR2'}}
        super().__init__(name, flows, ['EEin', 'Ctlin','DOF','Force'], {'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0}, components)
        self.assoc_modes()
    def behavior(self, time):
        Air,EEin={},{}
        #injects faults into lines
        for linname,lin in self.components.items():
            cmds={'up':self.upward[linname], 'for':self.forward[linname]}
            lin.behavior(self.EEin.effort, self.Ctlin, cmds, self.Force.support) 
            Air[lin.name]=lin.Airout
            EEin[lin.name]=lin.EE_in
        
        if any(value>=10 for value in EEin.values()): self.EEin.rate=10
        elif any(value!=0.0 for value in EEin.values()): self.EEin.rate=sum(EEin.values())/len(EEin) #should it really be max?
        else: self.EEin.rate=0.0
        
        self.LRstab = (sum([Air[comp] for comp in self.LR_dict['L']])-sum([Air[comp] for comp in self.LR_dict['R']]))/len(Air)
        self.FRstab = (sum([Air[comp] for comp in self.FR_dict['R']])-sum([Air[comp] for comp in self.FR_dict['F']]))/len(Air)
        
        if abs(self.LRstab) >=0.4 or abs(self.FRstab)>=0.75:
            self.DOF.uppwr=0
            self.DOF.planpwr=0
        else:
            Airs=list(Air.values())
            self.DOF.uppwr=np.mean(Airs)
            self.DOF.planpwr=-2*self.FRstab

class Line(Component):
    def __init__(self, name):
        super().__init__(name,{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0})
        self.failrate=1e-5
        self.assoc_modes({'short':[0.1, [0.33, 0.33, 0.33], 200],'openc':[0.1, [0.33, 0.33, 0.33], 200],\
                          'ctlup':[0.2, [0.33, 0.33, 0.33], 500],'ctldn':[0.2, [0.33, 0.33, 0.33], 500],\
                          'ctlbreak':[0.2, [0.33, 0.33, 0.33], 1000], 'mechbreak':[0.1, [0.33, 0.33, 0.33], 500],\
                          'mechfriction':[0.05, [0.0, 0.5,0.5], 500],'propwarp':[0.01, [0.0, 0.5,0.5], 200],\
                          'propstuck':[0.02, [0.0, 0.5,0.5], 200], 'propbreak':[0.03, [0.0, 0.5,0.5], 200]},name=name)

    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.add_fault('mechbreak','propbreak')
        elif Force<=0.5: self.add_fault('mechfriction')
            
        if self.has_fault('short'):
            self.Eti=0.0
            self.Eto=np.inf
        elif self.has_fault('openc'):
            self.Eti=0.0
            self.Eto=0.0
        elif Ctlin.upward==0 and Ctlin.forward == 0:
            self.Eto = 0.0
        if self.has_fault('ctlbreak'): self.Ct=0.0
        elif self.has_fault('ctldn'):  self.Ct=0.5
        elif self.has_fault('ctlup'):  self.Ct=2.0
        if self.has_fault('mechbreak'): self.Mt=0.0
        elif self.has_fault('mechfriction'):
            self.Mt=0.5
            self.Eti=2.0
        if self.has_fault('propstuck'):
            self.Pt=0.0
            self.Mt=0.0
            self.Eti=4.0
        elif self.has_fault('propbreak'): self.Pt=0.0
        elif self.has_fault('propwarp'):  self.Pt=0.5
        
        self.Airout=m2to1([EEin,self.Eti,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.Ct,self.Mt,self.Pt])
        self.EE_in=m2to1([EEin,self.Eto])   
        
class CtlDOF(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEin','Dir','Ctl','DOFs','FS'], {'vel':0.0, 'Cs':1.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, [0.6, 0.3, 0.1], 10000], 'degctl':[0.8, [0.6, 0.3, 0.1], 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noctl')
    def behavior(self, time):
        if self.has_fault('noctl'):    self.Cs=0.0
        elif self.has_fault('degctl'): self.Cs=0.5
        
        upthrottle=1.0
        if self.Dir.traj[2]>1:     upthrottle=2.0
        elif 0<self.Dir.traj[2]<=1:  upthrottle= self.Dir.traj[2] + 1.0
        elif self.Dir.traj[2]==0:
            damp=np.sign(self.vel)
            damp2=damp*min(1.0, np.power(self.vel, 2))
            upthrottle=1.0-0.2*damp2
        elif -1<self.Dir.traj[2]<=0.0:
            damp=min(1.0, np.power(self.vel+0.5, 2))
            upthrottle=0.75+0.25*damp
        elif self.Dir.traj[2]<=-1.0:
            damp=min(0.75, np.power(self.vel+5.0, 2))
            upthrottle=0.75+0.15*damp
            
        if self.Dir.traj[0]==0 and self.Dir.traj[1]==0: forwardthrottle=0.0
        else: forwardthrottle=1.0
        
        self.Ctl.forward=self.EEin.effort*self.Cs*forwardthrottle*self.Dir.power
        self.Ctl.upward=self.EEin.effort*self.Cs*self.Dir.power*upthrottle

class PlanPath(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEin','Env','Dir','FS'], states={'dx':0.0, 'dy':0.0, 'dz':0.0, 'pt':1, 'mode':'taxi'},timers={'pause'})
        self.goals = {1:[0,0,50], 2:[100, 0, 50], 3:[100, 100, 50], 4:[150, 150, 50], 5:[0,0,50], 6:[0,0,0]}
        self.goal = self.goals[1]
        self.failrate=1e-5
        self.assoc_modes({'noloc':[0.2, [0.6, 0.3, 0.1], 10000], 'degloc':[0.8, [0.6, 0.3, 0.1], 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noloc')
    def behavior(self, t):
        self.goal = self.goals[self.pt]
        loc = [self.Env.x, self.Env.y, self.Env.elev]
        dist = finddist(loc, self.goal)        
        [self.dx,self.dy, self.dz] = vectdist(self.goal,loc)
        
        if self.mode=='taxi' and t>5: self.mode=='taxi'
        elif dist<5 and {'move', 'hover'}.issuperset({self.mode}):
            self.mode='hover'
            if t>self.time:
                self.pause.inc(1)
                if self.pause.t() > 2:
                    self.pt=self.pt+1
                    self.goal = self.goals[self.pt]
                    self.pause.reset()
        elif self.Env.elev<1 and self.pt==6: self.mode = 'taxi'
        elif dist<5 and self.pt==6:         self.mode = 'land'
        elif self.pt==6 and {'move', 'hover'}.issuperset({self.mode}): self.mode = 'descend'
        elif dist>5 and not(self.mode=='descend'):                       self.mode='move'
        # nominal behaviors
        self.Dir.power=1.0
        if self.mode=='taxi':       self.Dir.power=0.0
        elif self.mode=='hover':    self.Dir.assign([0,0,0])           
        elif self.mode=='move':     self.Dir.assign(vectdir(self.goal, loc))     
        elif self.mode=='descend':  self.Dir.assign([0,0,-0.5])
        elif self.mode=='land':     self.Dir.assign([0,0,-0.1])
        # faulty behaviors    
        if self.has_fault('noloc'):     self.Dir.assign([0,0,0])
        elif self.has_fault('degloc'):  self.Dir.assign([0,0,-1])
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.assign([0,0,0])

class Trajectory(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env','DOF', 'Dir', 'Force_GR'])
        #self.assoc_modes({'crash':[0, 100000], 'lost':[0.0, 50000]})
    def behavior(self, time):
        if time>self.time:            
            if self.Env.elev<=0.0:  
                self.Force_GR.value=min(-0.5, (self.DOF.vertvel-self.DOF.planvel)/7.5)
                acc=10*self.DOF.uppwr
            else:                   
                self.Force_GR.value=0.0
                acc=10*(self.DOF.uppwr-1.0) 
            
            sign=np.sign(self.DOF.vertvel)
            damp=(-0.02*sign*np.power(self.DOF.vertvel, 2)-0.1*self.DOF.vertvel)
            self.DOF.vertvel=self.DOF.vertvel+(acc+damp)
            self.DOF.planvel=10*self.DOF.planpwr            
            if self.Env.elev<=0.0:  
                self.DOF.vertvel=max(0,self.DOF.vertvel)
                self.DOF.planvel=0.0
            
            self.Env.elev=max(0.0, self.Env.elev+self.DOF.vertvel)
            self.Env.x=self.Env.x+self.DOF.planvel*self.Dir.traj[0]
            self.Env.y=self.Env.y+self.DOF.planvel*self.Dir.traj[1]

class ViewEnvironment(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env'])
        sq=square([0,150], 160, 160)
        self.viewingarea = {(x,y):'unviewed' for x in range(int(sq[0][0]),int(sq[1][0])+10,10) for y in range(int(sq[0][1]),int(sq[2][1])+10,10)}
    def behavior(self, time):
        area = square((self.Env.x, self.Env.y), 10, 10)
        for spot in self.viewingarea:
            if inrange(area, spot[0],spot[1]): self.viewingarea[spot]='viewed'

class Direc(Flow):
    def __init__(self):
        self.traj=[0.0,0.0,0.0]
        super().__init__({'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': 1.0}, 'Trajectory')
    def assign(self, traj):
        self.x=traj[0]
        self.y=traj[1]
        self.z=traj[2]
        self.traj=traj
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()
        
class Drone(Model):
    def __init__(self, params={'graph_pos':{}, 'bipartite_pos':{},'arch':'quad'},\
            modelparams={'phases': {'ascend':[0,4],'forward':[5,94],'descend':[95, 100]}, 'times':[0,135],'units':'sec'}, valparams={}):
        
        super().__init__(params, modelparams, valparams)
                                     
        #add flows to the model
        self.add_flow('Force_ST', {'support':1.0})
        self.add_flow('Force_Lin', {'support':1.0})
        self.add_flow('Force_GR' , {'value':0.0})
        self.add_flow('Force_LG', {'value':0.0})
        self.add_flow('EE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEmot', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEctl', {'rate':1.0, 'effort':1.0})
        self.add_flow('Ctl1', {'forward':0.0, 'upward':1.0})
        self.add_flow('DOFs', {'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0})
        self.add_flow('Env1', {'x':0.0,'y':0.0,'elev':0.0} )
        # custom flows
        self.add_flow('Dir1', Direc())
        #add functions to the model
        flows=['EEctl', 'Force_ST']
        self.add_fxn('StoreEE',['EE_1', 'Force_ST'], fclass=StoreEE)
        self.add_fxn('DistEE', ['EE_1','EEmot','EEctl', 'Force_ST'], fclass=DistEE)
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Force_Lin'], fclass=AffectDOF, fparams=params['arch'])
        self.add_fxn('CtlDOF', ['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST'], fclass=CtlDOF)
        self.add_fxn('Planpath', ['EEctl', 'Env1','Dir1', 'Force_ST'], fclass=PlanPath)
        self.add_fxn('Trajectory', ['Env1','DOFs','Dir1', 'Force_GR'], fclass=Trajectory)
        self.add_fxn('EngageLand',['Force_GR', 'Force_LG'], fclass=EngageLand)
        self.add_fxn('HoldPayload',['Force_LG', 'Force_Lin', 'Force_ST'], fclass=HoldPayload)
        self.add_fxn('ViewEnv', ['Env1'], fclass=ViewEnvironment)
        
        self.build_model(graph_pos=params['graph_pos'], bipartite_pos=params['bipartite_pos'])
    def find_classification(self,scen, mdlhists):
        if -5 >mdlhists['faulty']['flows']['Env1']['x'][-1] or 5<mdlhists['faulty']['flows']['Env1']['x'][-1]:
            lostcost=50000
        elif -5 >mdlhists['faulty']['flows']['Env1']['y'][-1] or 5<mdlhists['faulty']['flows']['Env1']['y'][-1]:
            lostcost=50000
        elif mdlhists['faulty']['flows']['Env1']['elev'][-1] >5:
            lostcost=50000
        else:
            lostcost=0
        
        if any(abs(mdlhists['faulty']['flows']['Force_GR']['value'])>2.0):
            crashcost = 100000
        else:
            crashcost = 0
        repcost = self.calc_repaircost()
        
        totcost=repcost + crashcost + lostcost
        rate=scen['properties']['rate']
        expcost=totcost*rate*1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}
def square(center,xw,yw):
    square=[[center[0]-xw/2,center[1]-yw/2],\
            [center[0]+xw/2,center[1]-yw/2], \
            [center[0]+xw/2,center[1]+yw/2],\
            [center[0]-xw/2,center[1]+yw/2]]
    return square
#checks to see if a point with x-y coordinates is in the area a
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
def inrange(area, x, y):
    point=Point(x,y)
    polygon=Polygon(area)
    return polygon.contains(point)

def finddist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def calcdist(p1, p2):
    return np.sqrt((p1[0]-p2.x)**2+(p1[1]-p2.y)**2+(p1[2]-p2.elev)**2)

def vectdist(p1, p2):
    return [p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]]

def vectdir(p1, p2):
    return vectdist(p1,p2)/finddist(p1,p2)

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


if __name__=="__main__":
    
    hierarchical_model = Drone(params={'graph_pos':graph_pos, 'bipartite_pos':bipartite_pos,'arch':'quad'})
    endresults, resgraph, mdlhist = fs.propagate.one_fault(hierarchical_model,'AffectDOF', 'RFmechbreak', time=50)
    
    mdl = Drone(params={'graph_pos':graph_pos, 'bipartite_pos':bipartite_pos,'arch':'oct'})
    app = SampleApproach(mdl, faults=[('AffectDOF', 'RR2propstuck')])
    endclasses, mdlhists = fs.propagate.approach(mdl, app, staged=False)
    rd.plot.mdlhists({'nominal': mdlhists['nominal'],'faulty': mdlhists['AffectDOF RR2propstuck, t=49.0']})









