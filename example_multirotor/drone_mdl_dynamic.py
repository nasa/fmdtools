# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
from fmdtools.modeldef.common import Parameter, State
from fmdtools.modeldef.block import FxnBlock, Mode
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.approach import SampleApproach

from drone_mdl_static import m2to1
import fmdtools.faultsim as fs

from drone_mdl_static import DistEE, EngageLand, HoldPayload, AffectDOF

from drone_mdl_static import StoreEE as StaticStoreEE
class StoreEE(StaticStoreEE):
    def condfaults(self,time):
        if self.s.soc<1:
            self.s.soc=0
            self.m.add_fault('nocharge')
    def behavior(self, time):
        if self.m.has_fault('nocharge'):    self.EEout.s.effort=0.0
        else:                               self.EEout.s.effort=1.0
        if time > self.time:
            self.s.inc(soc=self.EEout.s.mul('rate', 'effort')*(time-self.time)/2)
            
from drone_mdl_static import CtlDOFMode
class CtlDOFState(State):
    Cs:     float = 1.0
    vel:    float = 0.0     
class CtlDOF(FxnBlock):
    _init_s = CtlDOFState
    _init_m = CtlDOFMode
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEin','Dir','Ctl','DOFs','FS'])
    def condfaults(self, time):
        if self.FS.s.support<0.5: self.m.add_fault('noctl')
    def behavior(self, time):
        if self.m.has_fault('noctl'):    self.s.Cs=0.0
        elif self.m.has_fault('degctl'): self.s.Cs=0.5
        
        upthrottle=1.0
        if self.Dir.s.z>1:      upthrottle=2.0
        elif 0<self.Dir.s.z<=1: upthrottle= self.Dir.s.z + 1.0
        elif self.Dir.s.z==0:
            damp=np.sign(self.s.vel)
            damp2=damp*min(1.0, np.power(self.s.vel, 2))
            upthrottle=1.0-0.2*damp2
        elif -1<self.Dir.s.z<=0.0:
            damp=min(1.0, np.power(self.s.vel+0.5, 2))
            upthrottle=0.75+0.25*damp
        elif self.Dir.s.z<=-1.0:
            damp=min(0.75, np.power(self.s.vel+5.0, 2))
            upthrottle=0.75+0.15*damp
            
        if self.Dir.s.x==0 and self.Dir.s.y==0: forwardthrottle=0.0
        else: forwardthrottle=1.0
        
        self.Ctl.s.forward=self.EEin.s.effort*self.s.Cs*forwardthrottle*self.Dir.s.power
        self.Ctl.s.upward=self.EEin.s.effort*self.s.Cs*self.Dir.s.power*upthrottle

class PlanPathMode(Mode):
    failrate=1e-5
    faultparams = {'noloc': (0.2, 10000),
                   'degloc':(0.8, 10000)}
    opermodes = ('taxi', 'hover', 'move', 'descend', 'land')
    mode: int = 'taxi'
class PlanPathStates(State):
    """ """
    dx:     float=0.0
    dy:     float=0.0
    dz:     float=0.0
    pt:     int=0
    goal:   tuple=(0.0,0.0,50.0)
class PlanPathParams(Parameter):
    goals = ((0.0,      0.0,    50.0),
             (100.0,    0.0,    50.0),
             (100.0,    100.0,  50.0),
             (150.0,    150.0,  50.0),
             (0.0,      0.0,    50.0),
             (0.0,      0.0,    0.0)) 

class PlanPath(FxnBlock):
    _init_m = PlanPathMode
    _init_s = PlanPathStates
    _init_p = PlanPathParams
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEin','Env','Dir','FS'], timers={'pause'})
    def condfaults(self, time):
        if self.FS.s.support<0.5: self.m.add_fault('noloc')
    def behavior(self, t):
        self.s.goal = self.p.goals[self.s.pt]
        loc =self.Env.s.get('x','y','z')
        dist = finddist(loc, self.s.goal) 
        self.s.assign(vectdist(self.s.goal,loc), 'dz', 'dy', 'dz')
        
        if self.m.mode=='taxi' and t>5: self.m.mode='taxi'
        elif dist<5 and self.m.in_mode({'move', 'hover'}):
            self.m.mode='hover'
            if t>self.time:
                self.pause.inc(1)
                if self.pause.t() > 2:
                    self.s.inc(pt=1)
                    self.s.goal = self.s.goals[self.pt]
                    self.pause.reset()
        elif self.Env.s.z<1 and self.s.pt==6:                       self.m.mode = 'taxi'
        elif dist<5 and self.s.pt==6:                               self.m.mode = 'land'
        elif self.s.pt==6 and self.m.in_mode({'move', 'hover'}):    self.m.mode = 'descend'
        elif dist>5 and not(self.m.mode=='descend'):                self.m.mode = 'move'
        # nominal behaviors
        self.Dir.s.power=1.0
        if self.m.mode=='taxi':           self.Dir.s.power=0.0
        elif self.m.mode=='hover':        self.Dir.s.assign([0,0,0], "x","y","z")           
        elif self.m.mode=='move':         self.Dir.s.assign(vectdir(self.s.goal, loc), "x","y","z")     
        elif self.m.mode=='descend':      self.Dir.s.assign([0,0,-0.5], "x","y","z")
        elif self.m.mode=='land':         self.Dir.s.assign([0,0,-0.1], "x","y","z")
        # faulty behaviors    
        if self.m.has_fault('noloc'):     self.Dir.s.assign([0,0,0], "x","y","z")
        elif self.m.has_fault('degloc'):  self.Dir.s.assign([0,0,-1], "x","y","z")
        if self.EEin.s.effort<0.5:        self.Dir.s.assign([0,0,0,0], "x","y","z", "power")

class Trajectory(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env','DOF', 'Dir', 'Force_GR'])
    def dynamic_behavior(self, time):            
        if self.Env.s.z<=0.0:  
            self.Force_GR.s.support=min(-0.5, (self.DOF.s.vertvel-self.DOF.s.planvel)/7.5)
            acc=10*self.DOF.s.uppwr
        else:                   
            self.Force_GR.s.support=0.0
            acc=10*(self.DOF.s.uppwr-1.0) 
        
        sign=np.sign(self.DOF.s.vertvel)
        damp=(-0.02*sign*np.power(self.DOF.s.vertvel, 2)-0.1*self.DOF.s.vertvel)
        self.DOF.s.vertvel=self.DOF.s.vertvel+(acc+damp)
        self.DOF.s.planvel=10.0*self.DOF.s.planpwr            
        if self.Env.s.z<=0.0:  
            self.DOF.s.vertvel=max(0,self.DOF.s.vertvel)
            self.DOF.s.planvel=0.0
        
        self.Env.s.inc(x=self.DOF.s.planvel*self.Dir.s.x,
                       y=self.DOF.s.planvel*self.Dir.s.y,
                       z=self.DOF.s.vertvel)
        self.Env.s.limit(z=(0.0,np.inf))

class ViewEnvironment(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env'])
        sq=square([0,150], 160, 160)
        self.viewingarea = {(x,y):'unviewed' for x in range(int(sq[0][0]),int(sq[1][0])+10,10) for y in range(int(sq[0][1]),int(sq[2][1])+10,10)}
    def behavior(self, time):
        area = square((self.Env.s.x, self.Env.s.y), 10, 10)
        for spot in self.viewingarea:
            if inrange(area, spot[0],spot[1]): self.viewingarea[spot]='viewed'

from drone_mdl_static import Force, EE, Control, DOFs, Env, Dir
class Drone(Model):
    def __init__(self, params=Parameter(),\
            modelparams=ModelParam(phases=(('ascend',0,4),('forward',5,94),('descend',95, 100)),times=(0,135),units='sec'), valparams={}):
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
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Force_Lin'], fclass=AffectDOF)
        self.add_fxn('CtlDOF', ['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST'], fclass=CtlDOF)
        self.add_fxn('Planpath', ['EEctl', 'Env1','Dir1', 'Force_ST'], fclass=PlanPath)
        self.add_fxn('Trajectory', ['Env1','DOFs','Dir1', 'Force_GR'], fclass=Trajectory)
        self.add_fxn('EngageLand',['Force_GR', 'Force_LG'], fclass=EngageLand)
        self.add_fxn('HoldPayload',['Force_LG', 'Force_Lin', 'Force_ST'], fclass=HoldPayload)
        self.add_fxn('ViewEnv', ['Env1'], fclass=ViewEnvironment)
        
        self.build_model()
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
    return np.sqrt((p1[0]-p2.x)**2+(p1[1]-p2.y)**2+(p1[2]-p2.z)**2)

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
    