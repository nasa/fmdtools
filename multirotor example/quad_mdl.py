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
        if archtype[0]=='normal':
            #architecture: 1 for controllers? + cells in Series & Parallel
            #Batctl=battery('ctl')
            components={'00':Battery('00'), '01':Battery('01'), '10':Battery('10'), '11':Battery('11')}
        #failrate for function w- component only applies to function modes
        self.failrate=1e-3
        self.assoc_modes({'nocharge':[0.2,[0.6,0.1,0.1,0.1,0.1],300],'lowcharge':[0.7,[0.6,0.1,0.1,0.1,0.1],200]})
        super().__init__(['EEout', 'FS', 'Hsig', 'Rsig'], flows, {'soc': 2000}, components)
    def condfaults(self, time):
        if self.soc<20: self.add_fault('lowcharge')
        if self.soc<1: self.replacefault('lowcharge','nocharge')
        return 0
    def behavior(self, time):
        EE={}
        soc={}
        for batname, bat in self.components.items():
            for fault in self.faults:
                if fault in bat.faultmodes:
                    bat.faults.update([fault])
            
            bat.behavior(self.FS.value, self.EEout.rate, time)
            self.faults.update(bat.faults)
            EE[bat.name]=bat.Et
            soc[bat.name]=bat.soc
            
        self.EEout.effort=(np.mean([EE['00'],EE['01']])+np.mean([EE['10'],EE['11']]))/2.0
        self.soc=np.mean(list(soc.values()))

class Battery(Component):
    def __init__(self, name):
        super().__init__(name, {'soc':2000, 'EEe':1.0, 'Et':1.0})
        self.failrate=1e-3
        self.assoc_modes({'short':[0.02,[0.2,0.2,0.2,0.2,0.2],2000], 'degr':[0.06,[0.2,0.2,0.2,0.2,0.2],2000],
                          'break':[0.02,[0.2,0.2,0.2,0.2,0.2],2000], 'nocharge':[0.2,[0.6,0.1,0.1,0.1,0.1],300],
                          'lowcharge':[0.7,[0.6,0.1,0.1,0.1,0.1],200]}, name=name)
    def behavior(self, FS, EEoutr, time):
        if FS <1.0: self.add_fault(self.name+'break')
        if EEoutr>2: self.add_fault(self.name+'break')
        if self.soc<20: self.add_fault(self.name+'lowcharge')
        if self.soc<1: self.replace_fault(self.name+'lowcharge',self.name+'nocharge')
        self.Et=1.0 #default
        if self.has_fault(self.name+'short'): self.Et=0.0
        elif self.has_fault(self.name+'break'): self.Et=0.0
        elif self.has_fault(self.name+'degr'): self.Et=0.5
        
        if self.has_fault(self.name+'nocharge'):
            self.soc=0.0
            self.Et=0.0
            
        if time > self.time:
            self.soc=self.soc-EEoutr*(time-self.time)
            self.time=time
        return self.Et

class DistEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEmot','EEctl','FS'],flows, {'EEtr':1.0, 'EEte':1.0}, timely=False)
        self.failrate=1e-5
        self.assoc_modes({'short':[0.3,[0.2, 0.2,0.2,0.2,0.2],3000], 'degr':[0.5,[0.2, 0.2,0.2,0.2,0.2],1000],\
                          'break':[0.2,[0.2, 0.2,0.2,0.2,0.2],2000]})
    def condfaults(self, time):
        if self.FS.value<0.5 or max(self.EEmot.rate,self.EEctl.rate)>2:
            self.add_fault('break')
    def behavior(self, time):
        if self.has_fault('short'): 
            self.EEte=0.0
            self.EEre=np.inf
        elif self.has_fault('break'): 
            self.EEte=0.0
            self.EEre=0.0
        elif self.has_fault('degr'): self.EEte=0.5
        self.EEmot.effort=self.EEte*self.EEin.effort
        self.EEctl.effort=self.EEte*self.EEin.effort
        self.EEin.rate=m2to1([ self.EEin.effort, self.EEtr, max(self.EEmot.rate,self.EEctl.rate)])

class EngageLand(FxnBlock):
    def __init__(self,flows):
        super().__init__(['forcein', 'forceout'],flows, {'Ft':1.0}, timely=False)
        self.failrate=1e-5
        self.assoc_modes({'break':[0.2,[0.5,0.0,0.0,0.0,0.5], 1000], 'deform':[0.8,[0.5,0.0,0.0,0.0,0.5], 1000]})
    def condfaults(self, time):
        if self.forceout.value<-1.4: self.add_fault('break')
        elif self.forceout.value<-1.2: self.add_fault('deform')
    def behavior(self, time):
        if self.has_fault('break'): self.Ft=4.0
        elif self.has_fault('deform'): self.Ft=2.0
        else: self.Ft=1.0
        self.forceout.value=self.Ft*min([-2.0,self.forcein.value])*0.2
    
class ManageHealth(FxnBlock):
    def __init__(self,flows):
        flownames=['EECtl','FS','DOFshealth', 'Bathealth','DOFconfig','Batconfig','Ctlconfig', 'Trajconfig' ]
        super().__init__(flownames, flows)
        
        self.failrate=1e-5
        self.assoc_modes({'falsemaintenance':[0.8,[1.0, 0.0,0.0,0.0,0.0],10000],\
                         'falsemasking':[0.1,[1.0, 0.2,0.4,0.4,0.0],10000],\
                         'falseemland':[0.05,[0.0, 0.2,0.4,0.4,0.0],10000],\
                         'lostfunction':[0.05,[0.2, 0.2,0.2,0.2,0.2],10000]})
        
        #need to add joint-fault modes of not catching faults?
    def condfaults(self, time):
        if self.FS.value<0.5 or self.EECtl.effort>2.0: self.add_fault('lostfunction')
    def behavior(self, time):
        if self.EECtl.effort>0.5 or self.has_fault('lostfunction'):
            self.DOFconfig=1
            self.Batconfig=1
            self.Ctlconfig=1
            self.Trajconfig=1
        else:
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2    
            
class HoldPayload(FxnBlock):
    def __init__(self,flows):
        super().__init__(['FG', 'FA', 'FS'],flows, {'Ft': 1.0}, timely=False)
        self.failrate=1e-6
        self.assoc_modes({'break':[0.2, [0.2, 0.2, 0.2, 0.2,0.2], 10000], 'deform':[0.8, [0.2, 0.2, 0.2, 0.2,0.2], 10000]})
    def condfaults(self, time):
        if abs(self.FG.value)>1.6: self.add_fault('break')
        elif abs(self.FG.value)>1.4: self.add_fault('deform')
    def behavior(self, time):
        if self.has_fault('break'): self.Ft=0.0
        elif self.has_fault('deform'): self.Ft=0.5
        else: self.Ft=1.0
        self.FA.value=self.Ft
        self.FS.value=self.Ft
    
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Air, HSig_DOFs, RSig_DOFs
    def __init__(self, flows, archtype):     
        self.archtype=archtype
        if archtype[0]=='quad':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
        super().__init__(['EEin', 'Ctlin','DOF','Force','Hsig', 'Rsig'], flows,{}, components, timely=False) 
    def behavior(self, time):
        Air={}
        EEin={}
        #injects faults into lines
        for linname,lin in self.components.items():
            for fault in self.faults:
                if fault in lin.faultmodes:
                    lin.add_fault(fault)
            
            cmds={'up':self.upward[linname], 'for':self.forward[linname]}
            lin.behavior(self.EEin.effort, self.Ctlin, cmds, self.Force.value)
            self.faults.update(lin.faults)  
            Air[lin.name]=lin.Airout
            EEin[lin.name]=lin.EE_in
        
        if any(value==np.inf for value in EEin.values()): self.EEin.rate=np.inf
        elif any(value!=0.0 for value in EEin.values()): self.EEin.rate=np.max(list(EEin.values()))
        else: self.EEin.rate=0.0
        
        if all(value==1.0 for value in Air.values()):   self.DOF.stab=1.0
        elif all(value==0.5 for value in Air.values()): self.DOF.stab=1.0
        elif all(value==2.0 for value in Air.values()): self.DOF.stab=1.0
        elif all(value==0.0 for value in Air.values()): self.DOF.stab=1.0
        elif any(value==0.0 for value in Air.values()): self.DOF.stab=0.0
        elif any(value>2.5 for value in Air.values()):  self.DOF.stab=0.0
        Airs=list(Air.values())
        #if not(self.Force.value==1.0):
        #    self.DOF.stab=self.Force.value
        
        self.DOF.uppwr=np.mean(Airs)
        
        list1=Airs[:len(Airs)//2]
        list2=Airs[len(Airs)//2:]
        vect=np.array([list1,list2])
        self.DOF.planpwr=np.sum(vect[0]-vect[1])/3
        
        #need to expand on this, add directional velocity, etc
        return

class Line(Component):
    def __init__(self, name):
        super().__init__(name,{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0}, timely=False)
        self.failrate=1e-4
        self.assoc_modes({'short':[0.1, [0.2, 0.2, 0.2, 0.2,0.2], 200],'openc':[0.1, [0.2, 0.2, 0.2, 0.2,0.2], 200],\
                          'ctlup':[0.2, [0.2, 0.2, 0.2, 0.2,0.2], 500],'ctldn':[0.2, [0.2, 0.2, 0.2, 0.2,0.2], 500],\
                          'ctlbreak':[0.2, [0.2, 0.2, 0.2, 0.2,0.2], 1000], 'mechbreak':[0.1, [0.2, 0.2, 0.2, 0.2,0.2], 500],\
                          'mechfriction':[0.05, [0.0, 0.2, 0.2, 0.2,0.2], 500],'propwarp':[0.01, [0.0, 0.2, 0.2, 0.2,0.2], 200],\
                          'propstuck':[0.02, [0.0, 0.2, 0.2, 0.2,0.2], 200], 'propbreak':[0.03, [0.0, 0.2, 0.2, 0.2,0.2], 200]},name=name)

    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.add_faults([self.name+'mechbreak', self.name+'propbreak'])
        elif Force<=0.5: self.add_fault(self.name+'mechfriction')
            
        if self.has_fault(self.name+'short'):
            self.Eti=0.0
            self.Eto=np.inf
        elif self.has_fault(self.name+'openc'):
            self.Eti=0.0
            self.Et0=0.0
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
        self.assoc_modes({'noctl':[0.2, [0.4, 0.2, 0.3, 0.1,0.0], 10000], 'degctl':[0.8, [0.4, 0.2, 0.3, 0.1,0.0], 10000]})
    def condfaults(self, time):
        if self.FS.value<0.5: self.add_fault('noctl')
    def behavior(self, time):
        if self.has_fault('noctl'):    self.Cs=0.0
        elif self.has_fault('degctl'): self.Cs=0.5
        
        if time>self.time: self.vel=self.DOFs.vertvel
        
        upthrottle=1.0
        if self.Dir.traj[2]>=1:     upthrottle=1.5
        elif 0<self.Dir.traj[2]<1:  upthrottle= 0.5 * self.Dir.traj[2] + 1.0
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
        self.Ctl.upward=self.EEin.effort*self.Cs*upthrottle*self.Dir.power

class PlanPath(FxnBlock):
    def __init__(self, flows):
        super().__init__(['EEin','Env','Dir','FS','Rsig'], flows, states={'dx':0.0, 'dy':0.0, 'dz':0.0, 'pt':1, 'mode':'taxi'},timers={'pause'})
        
        self.goals = { 1:[0,0,50], 2:[100, 0, 50], 3:[100, 100, 50], 4:[150, 150, 50], 5:[0,0,50], 6:[0,0,0] }
        self.queue = list(self.goals.keys())
        self.queue.reverse()
        self.goal = self.goals[1]
        self.failrate=1e-5
        self.assoc_modes({'noloc':[0.2, [0.4, 0.2, 0.3, 0.1,0.0], 10000], 'degloc':[0.8, [0.4, 0.2, 0.3, 0.1,0.0], 10000]})
    def condfaults(self, time):
        if self.FS.value<0.5: self.add_fault('noloc')
    def behavior(self, t):
        loc = [self.Env.x, self.Env.y, self.Env.elev]
        dist = finddist(loc, self.goal)        
        [self.dx,self.dy, self.dz] = vectdist(self.goal,loc)
        
        
        if self.mode=='taxi' and t>5: self.mode=='taxi'
        elif dist<5 and {'move', 'hover'}.issuperset({self.mode}):
            self.mode='hover'
            if t>self.time:
                self.pause.inc(1)
                if self.pause.t() > 5:
                    self.goal = self.goals[self.queue.pop()]
                    self.pause.reset()
        elif self.Env.elev<1 and len(self.queue)==0: self.mode = 'taxi'
        elif dist<10 and len(self.queue)==0:         self.mode = 'land'
        elif len(self.queue)==0 and {'move', 'hover'}.issuperset({self.mode}): self.mode = 'descend'
        elif dist>10 and not(self.mode=='descend'):                       self.mode='move'
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
    def __init__(self, flows):
        super().__init__(['Env','DOF','Land', 'Dir', 'Force_LG'], flows)
    def behavior(self, time):
        
        if time>self.time:
            maxvel=20.0
            maxpvel=5.0
            
            if self.Env.elev<=0.0:  
                self.Force_LG.value=min(-2.0, (self.DOF.vertvel-self.DOF.planvel)/3)
                acc=10*self.DOF.uppwr
            else:                   
                self.Force_LG.value=0.0
                acc=10*(self.DOF.uppwr-1.0) 
            
            sign=np.sign(self.DOF.vertvel)
            damp=(-0.02*sign*np.power(self.DOF.vertvel, 2)-0.1*self.DOF.vertvel)
            self.DOF.vertvel=self.DOF.vertvel+(acc+damp)
            self.DOF.planvel=maxpvel*self.DOF.planpwr            
            if self.Env.elev<=0.0:  
                self.DOF.vertvel=max(0,self.DOF.vertvel)
                self.DOF.planvel=0.0
            
            self.Env.elev=max(0.0, self.Env.elev+self.DOF.vertvel)
            self.Env.x=self.Env.x+self.DOF.planvel*self.Dir.traj[0]
            self.Env.y=self.Env.y+self.DOF.planvel*self.Dir.traj[1]
            
class Quadrotor(Model):
    def __init__(self, params={}):
        super().__init__()
        self.phases={'taxi':[0,10], 'climb':[10,15],'forward':[15, 45], 'descend':[45,50], 'land':[50,55]}
        #Declare time range to run model over
        self.times=[0,300]
        self.tstep = 1 #Stepsize: (change at your own risk--any accumulated value will need to change)
        
        #add flows to the model
        self.add_flow('Force_ST', 'Force', {'value':1.0})
        self.add_flow('Force_Air','Force', {'value':1.0} )
        self.add_flow('Force_GR','Force', {'value':1.0} )
        self.add_flow('Force_LG','Force', {'value':1.0})
        self.add_flow('HSig_DOFs','Health Signal', {'hstate':'nominal'})
        self.add_flow('HSig_Bat','Health Signal', {'hstate':'nominal'} )
        self.add_flow('RSig_DOFs','Reconfiguration Signal', {'mode':1} )
        self.add_flow('RSig_Bat','Reconfiguration Signal', {'mode':1} )
        self.add_flow('RSig_Ctl','Reconfiguration Signal', {'mode':1})
        self.add_flow('RSig_Traj','Reconfiguration Signal', {'mode':1})
        self.add_flow('EE_1', 'EE', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEmot', 'EE', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEctl', 'EE', {'rate':1.0, 'effort':1.0})
        self.add_flow('Ctl1','Direction Signal', {'forward':0.0, 'upward':1.0})
        self.add_flow('DOFs', 'DOFs',{'stab':1.0, 'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0})
        self.add_flow('Land1','Land', {'state':'landed', 'area':'start'} )
        self.add_flow('Env1','Environment', {'x':0.0,'y':0.0,'elev':0.0} )
        # custom flows
        self.add_flow('Dir1', 'Direction', Direc())
        #add functions to the model
        flows=['EEctl', 'Force_ST', 'HSig_DOFs', 'HSig_Bat', 'RSig_DOFs', 'RSig_Bat', 'RSig_Ctl', 'RSig_Traj']
        self.add_fxn('ManageHealth',ManageHealth,flows)
        self.add_fxn('StoreEE',StoreEE,['EE_1', 'Force_ST', 'HSig_Bat', 'RSig_Bat'], 'normal')
        self.add_fxn('DistEE',DistEE, ['EE_1','EEmot','EEctl', 'Force_ST'])
        self.add_fxn('AffectDOF',AffectDOF,['EEmot','Ctl1','DOFs','Force_Air', 'HSig_DOFs', 'RSig_DOFs'], 'quad')
        self.add_fxn('CtlDOF', CtlDOF,['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST', 'RSig_Ctl'])
        self.add_fxn('Planpath', PlanPath, ['EEctl', 'Env1','Dir1', 'Force_ST', 'RSig_Traj'])
        self.add_fxn('Trajectory', Trajectory,['Env1','DOFs','Land1','Dir1', 'Force_GR'] )
        self.add_fxn('EngageLand', EngageLand,['Force_GR', 'Force_LG'])
        self.add_fxn('HoldPayload', HoldPayload,['Force_LG', 'Force_Air', 'Force_ST'])
        
        self.construct_graph()
    def find_classification(self, g, endfaults, endflows, scen, mdlhist):
        
        start=[0.0,0.0]
        start_xw=10
        start_yw=10
        start_area=square(start,start_xw, start_yw)
        flyelev=30
        poi_center=[0,150]
        poi_xw=50
        poi_yw=50
        dang_center=[0,150]
        dang_xw=150
        dang_yw=150
        dang_area=square(dang_center, dang_xw, dang_yw)
        safe1_center=[-25,100]
        safe1_xw=10
        safe1_yw=10
        safe1_area=square(safe1_center, safe1_xw, safe1_yw)
        safe2_center=[25,50]
        safe2_xw=10
        safe2_yw=10
        safe2_area=square(safe2_center, safe2_xw, safe2_yw)
        
        Env=self.flows['Env1']
        
        #may need to redo this
        if  inrange(start_area, Env.x, Env.y):
            landloc='nominal'
            area=1
        elif inrange(safe1_area, Env.x, Env.y) or inrange(safe2_area, Env.x, Env.y):
            landloc='emsafe'
            area=1000
        elif inrange(dang_area, Env.x, Env.y):
            landloc='emdang'
            area=100000
        else:
            landloc='emunsanc'
            area=10000
        modes, modeprops = self.return_faultmodes()
        repaircosts=[ c['rcost'] for f,m in modeprops.items() for a, c in m.items()]
        maxcost=textmax(repaircosts)
        
        if maxcost=='major':
            repcost=10000
        elif maxcost=='moderate':
            repcost=3000
        elif maxcost=='minor':
            repcost=500
        elif maxcost=='replacement':
            repcost=250
        else:
            repcost=0
    
        totcost=repcost+area
        
        rate=scen['properties']['rate']
        
        expcost=totcost*rate*1e5
        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

## BASE FUNCTIONS

# creates list of corner coordinates for a square, given a center, xwidth, and ywidth
def square(center,xw,yw):
    square=[[center[0]-xw/2,center[1]-yw/2],\
            [center[0]+xw/2,center[1]-yw/2], \
            [center[0]+xw/2,center[1]+yw/2],\
            [center[0]-xw/2,center[1]+yw/2]]
    return square

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#checks to see if a point with x-y coordinates is in the area a
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

#takes the maximum of a variety of classifications given a list of strings
def textmax(texts):
    if 'major' in texts:
        maxt='major'
    elif 'moderate' in texts:
        maxt='moderate'
    elif 'minor' in texts:
        maxt='minor'
    elif 'replacement' in texts:
        maxt='replacement'
    else:
        maxt='none'
    return maxt