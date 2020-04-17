# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
from fmdtools.modeldef import *

#Define specialized flows
class Direc(Flow):
    def __init__(self):
        self.traj=[0,0,0]
        super().__init__({'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': 1}, 'Trajectory')
    def assign(self, traj):
        self.x=traj[0]
        self.y=traj[1]
        self.z=traj[2]
        self.traj=traj
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()

#Define functions
class StoreEE(FxnBlock):
    def __init__(self, flows, archtype):
        if archtype=='normal':
            #architecture: 1 for controllers? + cells in Series & Parallel
            #Batctl=battery('ctl')
            components={'00':Battery('00'), '01':Battery('01'), '10':Battery('10'), '11':Battery('11')}
        #failrate for function w- component only applies to function modes
        self.failrate=1e-3
        self.assoc_modes({'nocharge':[0.2,[0.6,0.2,0.2],300],'lowcharge':[0.7,[0.6,0.2,0.2],200]})
        super().__init__(['EEout', 'FS', 'Hsig'], flows, {'soc': 2000}, components)
    def condfaults(self, time):
        if self.soc<20: self.add_fault('lowcharge')
        if self.soc<1: self.replacefault('lowcharge','nocharge')
        return 0
    def behavior(self, time):
        EE={}
        soc={}
        for batname, bat in self.components.items():
            bat.behavior(self.FS.support, self.EEout.rate, time)
            EE[bat.name]=bat.Et
            soc[bat.name]=bat.soc
            
        self.EEout.effort=(np.mean([EE['00'],EE['01']])+np.mean([EE['10'],EE['11']]))/2.0
        self.soc=np.mean(list(soc.values()))

class Battery(Component):
    def __init__(self, name):
        super().__init__(name, {'soc':2000, 'EEe':1.0, 'Et':1.0})
        self.failrate=1e-3
        self.assoc_modes({'short':[0.02,[0.3,0.3,0.3],2000], 'degr':[0.06,[0.3,0.3,0.3],2000],
                          'break':[0.02,[0.2,0.2,0.2],2000], 'nocharge':[0.2,[0.6,0.2,0.2],300],
                          'lowcharge':[0.7,[0.6,0.2,0.2],200]}, name=name)
    def behavior(self, FS, EEoutr, time):
        if FS <1.0:     self.add_fault(self.name+'break')
        if EEoutr>2:    self.add_fault(self.name+'break')
        if self.soc<20: self.add_fault(self.name+'lowcharge')
        if self.soc<1:  self.replace_fault(self.name+'lowcharge',self.name+'nocharge')
        self.Et=1.0 #default
        if self.has_fault(self.name+'short'):       self.Et=0.0
        elif self.has_fault(self.name+'break'):     self.Et=0.0
        elif self.has_fault(self.name+'degr'):      self.Et=0.5
        
        if self.has_fault(self.name+'nocharge'):    self.soc, self.Et = 0.0,0.0
            
        if time > self.time:
            self.soc=self.soc-EEoutr*(time-self.time)
            self.time=time
        return self.Et

class DistEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEmot','EEctl','ST'],flows, {'EEtr':1.0, 'EEte':1.0}, timely=False)
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
            
class HoldPayload(FxnBlock):
    def __init__(self,flows):
        super().__init__(['DOF', 'Lin', 'ST'],flows, timely=False, states={'Force_GR':1.0})
        self.failrate=1e-6
        self.assoc_modes({'break':[0.2, [0.33, 0.33, 0.33], 10000], 'deform':[0.8, [0.33, 0.33, 0.33], 10000]})
    def condfaults(self, time):
        if self.DOF.elev<=0.0:  self.Force_GR=min(-0.5, (self.DOF.vertvel/60-self.DOF.planvel/60)/7.5)
        else:                   self.Force_GR=0.0
        if abs(self.Force_GR/2)>0.8:      self.add_fault('break')
        elif abs(self.Force_GR/2)>1.0:    self.add_fault('deform')
    def behavior(self, time):
        #need to transfer FG to FA & FS???
        if self.has_fault('break'):     self.Lin.support, self.ST.support = 0,0
        elif self.has_fault('deform'):  self.Lin.support, self.ST.support = 0.5,0.5
        else:                           self.Lin.support, self.ST.support = 1.0,1.0
    
class ManageHealth(FxnBlock):
    def __init__(self,flows):
        flownames=['EECtl','FS','DOFshealth', 'Bathealth','Ctlconfig', 'Trajconfig' ]
        super().__init__(flownames, flows)
        
        self.failrate=1e-5
        self.assoc_modes({'falsemaintenance':[0.8,[1.0, 0.0,0.0,0.0,0.0],10000],\
                         'falsemasking':[0.1,[1.0, 0.2,0.4,0.4,0.0],10000],\
                         'falseemland':[0.05,[0.0, 0.2,0.4,0.4,0.0],10000],\
                         'lostfunction':[0.05,[0.2, 0.2,0.2,0.2,0.2],10000]})
    def condfaults(self, time):
        if self.FS.support<0.5 or self.EECtl.effort>2.0: self.add_fault('lostfunction')
    def behavior(self, time):
        if self.EECtl.effort>0.5 or self.has_fault('lostfunction'):
            self.Ctlconfig.mode=1
            self.Trajconfig.mode=1
        else:
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2    
    
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    def __init__(self, flows, archtype):     
        self.archtype=archtype
        if archtype=='quad':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
            self.LR = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR = {'F':{'LF', 'RF'}, 'R':{'LR', 'RR'}}
        elif archtype=='oct':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR'),'RF2':Line('RF2'), 'LF2':Line('LF2'), 'LR2':Line('LR2'), 'RR2':Line('RR2')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1,'RF2':1,'LF2':1,'LR2':1,'RR2':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5,'RF2':0.5,'LF2':0.5,'LR2':-0.5,'RR2':-0.5}
            self.LR = {'L':{'LF', 'LR','LF2', 'LR2'}, 'R':{'RF','RR','RF2','RR2'}}
            self.FR = {'F':{'LF', 'RF','LF2', 'RF2'}, 'R':{'LR', 'RR','LR2', 'RR2'}}
        super().__init__(['EEin', 'Ctlin','DOF','Dir','Force','Hsig'], flows,{'LRstab':0.0, 'FRstab':0.0}, components) 
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
        
        self.LRstab = (sum([Air[comp] for comp in self.LR['L']])-sum([Air[comp] for comp in self.LR['R']]))/len(Air)
        self.FRstab = (sum([Air[comp] for comp in self.FR['R']])-sum([Air[comp] for comp in self.FR['F']]))/len(Air)
        
        if abs(self.LRstab) >=0.4 or abs(self.FRstab)>=0.75:
            self.DOF.uppwr=0
            self.DOF.planpwr=0
        else:
            Airs=list(Air.values())
            self.DOF.uppwr=np.mean(Airs)
            self.DOF.planpwr=self.Ctlin.forward
            
        if time> self.time:
            if self.DOF.uppwr > 1:      self.DOF.vertvel = 60*min([(self.DOF.uppwr-1)*5, 5])
            elif self.DOF.uppwr < 1:    self.DOF.vertvel = 60*max([(self.DOF.uppwr-1)*50, -50])
            else:                       self.DOF.vertvel = 0.0
                
            self.DOF.planvel=60*min([10*self.DOF.planpwr, 10])            
            if self.DOF.elev<=0.0:  
                self.DOF.vertvel=max(0,self.DOF.vertvel)
                self.DOF.planvel=0.0
            
            self.DOF.elev=max(0.0, self.DOF.elev+self.DOF.vertvel)
            vect = np.sqrt(np.power(self.Dir.traj[0], 2)+ np.power(self.Dir.traj[1], 2))+0.001
            self.DOF.x=self.DOF.x+self.DOF.planvel*self.Dir.traj[0]/vect
            self.DOF.y=self.DOF.y+self.DOF.planvel*self.Dir.traj[1]/vect

class Line(Component):
    def __init__(self, name):
        super().__init__(name,{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0}, timely=False)
        self.failrate=1e-4
        self.assoc_modes({'short':[0.1, [0.33, 0.33, 0.33], 200],'openc':[0.1, [0.33, 0.33, 0.33], 200],\
                          'ctlup':[0.2, [0.33, 0.33, 0.33], 500],'ctldn':[0.2, [0.33, 0.33, 0.33], 500],\
                          'ctlbreak':[0.2, [0.33, 0.33, 0.33], 1000], 'mechbreak':[0.1, [0.33, 0.33, 0.33], 500],\
                          'mechfriction':[0.05, [0.0, 0.5,0.5], 500],'propwarp':[0.01, [0.0, 0.5,0.5], 200],\
                          'propstuck':[0.02, [0.0, 0.5,0.5], 200], 'propbreak':[0.03, [0.0, 0.5,0.5], 200]},name=name)

    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.add_faults([self.name+'mechbreak', self.name+'propbreak'])
        elif Force<=0.5: self.add_fault(self.name+'mechfriction')
            
        if self.has_fault(self.name+'short'):
            self.Eti=0.0
            self.Eto=np.inf
        elif self.has_fault(self.name+'openc'):
            self.Eti=0.0
            self.Eto=0.0
        elif Ctlin.upward==0 and Ctlin.forward == 0:
            self.Eto = 0.0
        if self.has_fault(self.name+'ctlbreak'): self.Ct=0.0
        elif self.has_fault(self.name+'ctldn'):  self.Ct=0.5
        elif self.has_fault(self.name+'ctlup'):  self.Ct=2.0
        if self.has_fault(self.name+'mechbreak'): self.Mt=0.0
        elif self.has_fault(self.name+'mechfriction'):
            self.Mt=0.5
            self.Eti=2.0
        if self.has_fault(self.name+'propstuck'):
            self.Pt=0.0
            self.Mt=0.0
            self.Eti=4.0
        elif self.has_fault(self.name+'propbreak'): self.Pt=0.0
        elif self.has_fault(self.name+'propwarp'):  self.Pt=0.5
        
        self.Airout=m2to1([EEin,self.Eti,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.Ct,self.Mt,self.Pt])
        self.EE_in=m2to1([EEin,self.Eto])  
    
class CtlDOF(FxnBlock):
    def __init__(self, flows):
        super().__init__(['EEin','Dir','Ctl','DOFs','FS','Rsig'],flows, {'vel':0.0, 'Cs':1.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, [0.6, 0.3, 0.1], 10000], 'degctl':[0.8, [0.6, 0.3, 0.1], 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noctl')
    def behavior(self, time):
        if self.has_fault('noctl'):    self.Cs=0.0
        elif self.has_fault('degctl'): self.Cs=0.5
        if time>self.time: self.vel=self.DOFs.vertvel
        # throttle settings: 0 is off (-50 m/s), 1 is hover, 2 is max climb (5 m/s)
        if self.Dir.traj[2]>0:      upthrottle = 1+np.min([self.Dir.traj[2]/(50*5), 1])
        elif self.Dir.traj[2] <0:   upthrottle = 1+np.max([self.Dir.traj[2]/(50*50), -1])
        else:                        upthrottle = 1.0
        
        vect = np.sqrt(np.power(self.Dir.traj[0], 2)+ np.power(self.Dir.traj[1], 2))+0.001
        forwardthrottle = np.min([vect/(60*9), 1])
        
        self.Ctl.forward=self.EEin.effort*self.Cs*forwardthrottle*self.Dir.power
        self.Ctl.upward=self.EEin.effort*self.Cs*upthrottle*self.Dir.power

class PlanPath(FxnBlock):
    def __init__(self, flows, params):
        super().__init__(['EEin','Env','Dir','FS','Rsig'], flows, states={'dx':0.0, 'dy':0.0, 'dz':0.0, 'pt':1, 'mode':'taxi'})
        self.goals = params['flightplan']
        self.goal=self.goals[1]
        self.failrate=1e-5
        self.assoc_modes({'noloc':[0.2, [0.6, 0.3, 0.1], 10000], 'degloc':[0.8, [0.6, 0.3, 0.1], 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noloc')
    def behavior(self, t):
        loc = [self.Env.x, self.Env.y, self.Env.elev]
        if self.pt <= max(self.goals):   self.goal = self.goals[self.pt]
        dist = finddist(loc, self.goal)        
        [self.dx,self.dy, self.dz] = vectdist(self.goal,loc)
        
        if self.mode=='taxi' and t>5: self.mode=='taxi'
        elif dist<5 and {'move'}.issuperset({self.mode}):
            if self.pt < max(self.goals):   
                self.pt+=1
                self.goal = self.goals[self.pt]
        elif self.Env.elev<1 and self.pt>=max(self.goals):  self.mode = 'taxi'
        elif dist<5 and self.pt>=max(self.goals):           self.mode = 'land'
        elif dist>5 and not(self.mode=='descend'):          self.mode='move'
        # nominal behaviors
        self.Dir.power=1.0
        if self.mode=='taxi':       self.Dir.power=0.0          
        elif self.mode=='move':     self.Dir.assign([self.dx,self.dy, self.dz])     
        elif self.mode=='land':     self.Dir.assign([0,0,-0.1])
        # faulty behaviors    
        if self.has_fault('noloc'):     self.Dir.assign([0,0,0])
        elif self.has_fault('degloc'):  self.Dir.assign([0,0,-1])
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.assign([0,0,0])            
            
class Drone(Model):
    def __init__(self, params={'flightplan':{1:[0,0,100], 2:[100, 0,100], 3:[100, 100,100], 4:[150, 150,100], 5:[0,0,100], 6:[0,0,0]} }):
        super().__init__()
        super().__init__(modelparams={'phases': {'ascend':[0,1],'forward':[1,29],'descend':[29, 30]},
                                     'times':[0,30],'units':'min'}, params=params)
        
        self.start_area=square([0.0,0.0],10, 10) # coordinates, xwidth, ywidth
        self.dang_area=square([0,150], 160, 160)
        self.safe1_area=square([-25,100], 10, 10)
        self.safe2_area=square([25,50], 10, 10)
        
        #add flows to the model
        self.add_flow('Force_ST', {'support':1.0})
        self.add_flow('Force_Lin', {'support':1.0} )
        self.add_flow('HSig_DOFs', {'hstate':'nominal', 'config':1.0})
        self.add_flow('HSig_Bat', {'hstate':'nominal', 'config':1.0} )
        self.add_flow('RSig_Ctl', {'mode':1})
        self.add_flow('RSig_Traj', {'mode':1})
        self.add_flow('EE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEmot', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEctl', {'rate':1.0, 'effort':1.0})
        self.add_flow('Ctl1', {'forward':0.0, 'upward':1.0})
        self.add_flow('DOFs', {'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0, 'x':0.0,'y':0.0,'elev':0.0})
        # custom flows
        self.add_flow('Dir1', Direc())
        #add functions to the model
        flows=['EEctl', 'Force_ST', 'HSig_DOFs', 'HSig_Bat', 'RSig_Ctl', 'RSig_Traj']
        self.add_fxn('ManageHealth',flows,fclass = ManageHealth)
        self.add_fxn('StoreEE',['EE_1', 'Force_ST', 'HSig_Bat'],fclass = StoreEE, fparams= 'normal')
        self.add_fxn('DistEE', ['EE_1','EEmot','EEctl', 'Force_ST'], fclass = DistEE)
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Dir1','Force_Lin', 'HSig_DOFs'], fclass=AffectDOF, fparams = 'quad')
        self.add_fxn('CtlDOF',['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST', 'RSig_Ctl'], fclass = CtlDOF)
        self.add_fxn('Planpath', ['EEctl', 'DOFs','Dir1', 'Force_ST', 'RSig_Traj'], fclass=PlanPath, fparams=params)
        self.add_fxn('HoldPayload',['DOFs', 'Force_Lin', 'Force_ST'], fclass = HoldPayload)
        
        self.construct_graph()
        
    def find_classification(self, g, endfaults, endflows, scen, mdlhist):
        #landing costs
        viewed = env_viewed(mdlhist['faulty']['flows']['DOFs']['x'], mdlhist['faulty']['flows']['DOFs']['y'], self.dang_area)
        num_viewed = sum([1 for k,view in viewed.items() if view=='viewed'])
        viewed_value = num_viewed*100
        
        Env=self.flows['DOFs']
        if  inrange(self.start_area, Env.x, Env.y): landcost = 1 # nominal landing
        elif inrange(self.safe1_area, Env.x, Env.y) or inrange(self.safe2_area, Env.x, Env.y): landcost=1000 # emergency safe
        elif inrange(self.dang_area, Env.x, Env.y):  landcost=100000 # emergency dangerous
        else:                                    landcost=10000 # emergency unsanctioned
        #repair costs
        repcost=sum([ c['rcost'] for f,m in endfaults.items() for a, c in m.items()])

        totcost=repcost+landcost-viewed_value
        rate=scen['properties']['rate']
        expcost=totcost*rate*1e5
        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost, 'viewed':viewed}

## BASE FUNCTIONS

# creates list of corner coordinates for a square, given a center, xwidth, and ywidth
def square(center,xw,yw):
    square=[[center[0]-xw/2,center[1]-yw/2],\
            [center[0]+xw/2,center[1]-yw/2], \
            [center[0]+xw/2,center[1]+yw/2],\
            [center[0]-xw/2,center[1]+yw/2]]
    return square

def rect(x1, y1, x2, y2, width, height):
    vec = vectdir([x1, y1,0], [x2,y2+0.00001,0])[0:2]
    normvec = np.array([vec[1], -vec[0]])
    rec = [[x1, y1]+normvec*width/2+vec*height/2,[x1, y1]-normvec*width/2+vec*height/2,[x2, y2]-normvec*width/2-vec*height/2,[x2, y2]+normvec*width/2-vec*height/2]
    return rec

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
    return vectdist(p1,p2)/(finddist(p1,p2)+0.0001)

import matplotlib.pyplot as plt
def env_viewed(xhist, yhist, square):
    viewed = {(x,y):'unviewed' for x in range(int(square[0][0]),int(square[1][0])+10,10) for y in range(int(square[0][1]),int(square[2][1])+10,10)}
    for i,x in enumerate(xhist[1:len(xhist)]):
        viewed_area = rect(xhist[i],yhist[i],xhist[i+1],yhist[i+1], 10,10)
        polygon=Polygon(viewed_area)
        #plt.plot(*polygon.exterior.xy) (displays area to debug code)
        #plt.plot([xhist[i],xhist[i+1]],[yhist[i],yhist[i+1]])
        if not polygon.is_valid:    print('invalid points')
        for spot in viewed:
            if polygon.contains(Point(spot)): 
                viewed[spot]='viewed'
    return viewed


