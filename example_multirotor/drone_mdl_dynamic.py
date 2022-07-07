# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import sys, os
sys.path.insert(1,os.path.join('..'))
import numpy as np
from fmdtools.modeldef import *
import fmdtools.faultsim as fs

class StoreEE(FxnBlock):
    def __init__(self, name, flows):
        self.failrate=1e-5
        super().__init__(name, flows, ['EEout', 'FS'], {'soc': 100.0})
        self.assoc_modes({'nocharge':[1,300]})
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
        self.assoc_modes({'short':[0.3,3000], 'degr':[0.5,1000], 'break':[0.2,2000]})
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
        self.assoc_modes({'break':[0.2, 1000], 'deform':[0.8, 1000]})
    def condfaults(self, time):
        if abs(self.forcein.value)>=2.0:      self.add_fault('break')
        elif abs(self.forcein.value)>1.5:    self.add_fault('deform')
    def behavior(self, time):
        self.forceout.value=self.forcein.value/2
            
class HoldPayload(FxnBlock):
    def __init__(self, name,flows):
        super().__init__(name, flows, ['FG', 'Lin', 'ST'])
        self.failrate=1e-6
        self.assoc_modes({'break':[0.2, 10000], 'deform':[0.8, 10000]})
    def condfaults(self, time):
        if abs(self.FG.value)>0.8:      self.add_fault('break')
        elif abs(self.FG.value)>1.0:    self.add_fault('deform')
    def behavior(self, time):
        #need to transfer FG to FA & FS???
        if self.has_fault('break'):     self.Lin.support, self.ST.support = 0,0
        elif self.has_fault('deform'):  self.Lin.support, self.ST.support = 0.5,0.5
        else:                           self.Lin.support, self.ST.support = 1.0,1.0
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    def __init__(self, name, flows):     
        super().__init__(name, flows, ['EEin', 'Ctlin','DOF','Force'], {'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0})
        self.failrate=1e-5
        self.assoc_modes({'short':[0.1, [0.33, 0.33, 0.33], 200],'openc':[0.1, [0.33, 0.33, 0.33], 200],\
                          'ctlup':[0.2, [0.33, 0.33, 0.33], 500],'ctldn':[0.2, [0.33, 0.33, 0.33], 500],\
                          'ctlbreak':[0.2, [0.33, 0.33, 0.33], 1000], 'mechbreak':[0.1, [0.33, 0.33, 0.33], 500],\
                          'mechfriction':[0.05, [0.0, 0.5,0.5], 500],'propwarp':[0.01, [0.0, 0.5,0.5], 200],\
                          'propstuck':[0.02, [0.0, 0.5,0.5], 200], 'propbreak':[0.03, [0.0, 0.5,0.5], 200]})
    def behavior(self, time):
        self.Eti=1.0
        self.Eto=1.0
        if self.has_fault('short'):
            self.Eti=10
            self.Eto=0.0
        elif self.has_fault('openc'):
            self.Eti=0.0
            self.Eto=0.0
        elif self.Ctlin.upward==0 and self.Ctlin.forward == 0:
            self.Eti = 0.0
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
        
        self.EEin.rate=self.Eti

        self.DOF.uppwr=self.Eto*self.Eti*self.Ctlin.upward*self.Ct*self.Mt*self.Pt
        self.DOF.planpwr=self.Eto*self.Eti*self.Ctlin.forward*self.Ct*self.Mt*self.Pt    
        
class CtlDOF(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEin','Dir','Ctl','DOFs','FS'], {'vel':0.0, 'Cs':1.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, 10000], 'degctl':[0.8, 10000]})
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
            self.DOF.planvel=10.0*self.DOF.planpwr            
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
    def __init__(self, params={'graph_pos':{}, 'bipartite_pos':{}},\
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
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Force_Lin'], fclass=AffectDOF)
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
        
        modes, modeprops = self.return_faultmodes()
        repcost = sum([ c['rcost'] for f,m in modeprops.items() for a, c in m.items()])
        
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

if __name__=="__main__":
    mdl = Drone()
    app = SampleApproach(mdl)
    
    mdl_quad_comp = Drone()
    quad_comp_app = SampleApproach(mdl_quad_comp, faults=[('AffectDOF', 'mechbreak')],defaultsamp={'samp':'evenspacing','numpts':5})
    quad_comp_endclasses, quad_comp_mdlhists = fs.propagate.approach(mdl_quad_comp, quad_comp_app, staged=True)
    quad_comp_endclasses_1, quad_comp_mdlhists_1 = fs.propagate.approach(mdl_quad_comp, quad_comp_app)
    
    cost_tests = [quad_comp_endclasses[ec]['expected cost']==quad_comp_endclasses_1[ec]['expected cost'] for ec in quad_comp_endclasses]
    dist_tests = [all(quad_comp_mdlhists[ec]['flows']['Env1']['x']==quad_comp_mdlhists_1[ec]['flows']['Env1']['x']) for ec in quad_comp_mdlhists]
    dist_tests2 = [all(quad_comp_mdlhists[ec]['flows']['Env1']['y']==quad_comp_mdlhists_1[ec]['flows']['Env1']['y']) for ec in quad_comp_mdlhists]
    