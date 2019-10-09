# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""

import networkx as nx
import numpy as np

import faultprop as fp
from modeldef import *

#Declare time range to run model over
times=[0,3, 55]

#Define specialized flows
class Env:
    def __init__(self):
        self.type = 'flow'
        self.flow='Environment'
        self.elev=0.0
        self.x=0.0
        self.y=0.0
        self.start=[0.0,0.0]
        self.start_xw=5
        self.start_yw=5
        self.start_area=square(self.start,self.start_xw, self.start_yw)
        self.flyelev=30
        self.poi_center=[0,150]
        self.poi_xw=50
        self.poi_yw=50
        self.poi_area=square(self.poi_center, self.poi_xw, self.poi_yw)
        self.dang_center=[0,150]
        self.dang_xw=150
        self.dang_yw=150
        self.dang_area=square(self.dang_center, self.dang_xw, self.dang_yw)
        self.safe1_center=[-25,100]
        self.safe1_xw=10
        self.safe1_yw=10
        self.safe1_area=square(self.safe1_center, self.safe1_xw, self.safe1_yw)
        self.safe2_center=[25,50]
        self.safe2_xw=10
        self.safe2_yw=10
        self.safe2_area=square(self.safe2_center, self.safe2_xw, self.safe2_yw)
        self.nominal={'elev':1.0, 'x':1.0, 'y':1.0}
    def status(self):
        status={'elev':self.elev, 'x':self.x, 'y':self.y}
        return status.copy()
class Direc(flow):
    def __init__(self):
        self.traj=[0,0,0]
        super().__init__({'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': 1}, 'Trajectory')
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()

#Define functions
class storeEE(fxnblock):
    def __init__(self, name,EEout, FS, Hsig, Rsig, archtype):
        super().__init__({'EEout':EEout, 'FS':FS, 'Hsig':Hsig, 'Rsig': Rsig})
        self.faultmodes={'nocharge':{'rate':'moderate','rcost':'minor'}, \
                         'lowcharge':{'rate':'moderate','rcost':'minor'}}
        self.soc=2000
        if archtype=='normal':
            #architecture: 1 for controllers? + cells in Series & Parallel
            #Batctl=battery('ctl')
            Bat00=battery('00')
            Bat01=battery('01')
            Bat10=battery('10')
            Bat11=battery('11')
            self.batteries=[Bat00, Bat01, Bat10, Bat11]
            
        for bat in self.batteries:
            self.faultmodes.update(bat.faultmodes)  
    def condfaults(self, time):
        if self.soc<20: self.addfault('lowcharge')
        if self.soc<1: self.replacefault('lowcharge','nocharge')
        return 0
    def behavior(self, time):
        EE={}
        soc={}
        for bat in self.batteries:
            for fault in self.faults:
                if fault in bat.faultmodes:
                    bat.faults.update([fault])
            
            bat.behavior(self.FS.value, self.EEout.rate, time)
            self.faults.update(bat.faults)
            EE[bat.name]=bat.EEoute
            soc[bat.name]=bat.soc
            
        self.EEout.effort=(np.mean([EE['00'],EE['01']])+np.mean([EE['10'],EE['11']]))/2.0
        self.soc=np.mean(list(soc.values()))

class battery(component):
    def __init__(self, name):
        super().__init__(name)
        self.soc=2000
        self.t1=1.0
        self.EEoute=1.0
        self.faultmodes={name+'short':{'rate':'moderate', 'rcost':'major'}, \
                         name+'degr':{'rate':'moderate', 'rcost':'minor'}, \
                         name+'break':{'rate':'common', 'rcost':'moderate'}, \
                         name+'nocharge':{'rate':'moderate','rcost':'minor'}, \
                         name+'lowcharge':{'rate':'moderate','rcost':'minor'}}
        self.effstate=1.0
    def behavior(self, FS, EEoutr, time):
        if FS <1.0: self.faults.update([self.name+'break'])
        if EEoutr>2: self.faults.add(self.name+'break')
        if self.soc<20: self.faults.add(self.name+'lowcharge')
        if self.soc<1:
            self.faults.remove(self.name+'lowcharge')
            self.faults.add(self.name+'nocharge')

        if self.faults.intersection(set([self.name+'short'])): self.effstate=0.0
        elif self.faults.intersection(set([self.name+'break'])): self.effstate=0.0
        elif self.faults.intersection(set([self.name+'degr'])): self.effstate=0.5
        
        if self.faults.intersection(set([self.name+'nocharge'])):
            self.soc=0.0
            self.effstate=0.0
            
        self.EEoute=self.effstate
        if time > self.t1:
            self.soc=self.soc-EEoutr*(time-self.t1)
            self.t1=time
        return self.EEoute

class distEE(fxnblock):
    def __init__(self,EEin,EEmot,EEctl,FS):
        super().__init__({'EEin':EEin, 'EEmot':EEmot,'EEctl':EEctl,'FS':FS}, {'EEtr':1.0, 'EEte':1.0})
        self.faultmodes={'short':{'rate':'moderate', 'rcost':'major'}, \
                         'degr':{'rate':'moderate', 'rcost':'minor'}, \
                         'break':{'rate':'common', 'rcost':'moderate'}}
    def condfaults(self, time):
        if self.FS.value<0.5 or max(self.EEmot.rate,self.EEctl.rate)>2:
            self.addfault('break')
    def behavior(self, time):
        if self.hasfault('short'): 
            self.EEte=0.0
            self.EEre=np.inf
        elif self.hasfault('break'): 
            self.EEte=0.0
            self.EEre=0.0
        elif self.hasfault('degr'): self.EEte=0.5
        self.EEmot.effort=self.EEte*self.EEin.effort
        self.EEctl.effort=self.EEte*self.EEin.effort
        self.EEin.rate=m2to1([ self.EEin.effort, self.EEtr, max(self.EEmot.rate,self.EEctl.rate)])

class engageLand(fxnblock):
    def __init__(self,name, Forcein, Forceout):
        super().__init__({'forcein':Forcein, 'forceout':Forceout})
        self.name=name
        self.fstate=1.0
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}}
    def condfaults(self, time):
        if self.forceout.value<-1.4: self.addfault('break')
        elif self.forceout.value<-1.2: self.addfault('deform')
    def behavior(self, time):
        if self.hasfault('break'): self.fstate=4.0
        elif self.hasfault('deform'): self.fstate=2.0
        else: self.fstate=1.0
        self.forceout.value=self.fstate*min([-2.0,self.forcein.value])*0.2
    
class manageHealth(fxnblock):
    def __init__(self, EECtl, Force_ST, HSig_DOFs, HSig_Bat, RSig_DOFs, RSig_Bat, RSig_Ctl, RSig_Traj):
        
        flows={'DOFshealth':HSig_DOFs, 'Bathealth':HSig_Bat, 'DOFconfig':RSig_DOFs, 'Batconfig':RSig_Bat, \
               'Ctlconfig':RSig_Ctl, 'Trajconfig':RSig_Traj, 'FS':Force_ST, 'EECtl':EECtl}
        super().__init__(flows)
        
        self.faultmodes={'falsemaintenance':{'rate':'moderate', 'rcost':'minor'},\
                         'falsemasking':{'rate':'rare', 'rcost': 'major'},\
                         'falseemland':{'rate':'rare', 'rcost': 'major'},\
                         'lostfunction':{'rate':'rare', 'rcost':'minor'}} 
        #need to add joint-fault modes of not catching faults?
    def condfaults(self, time):
        if self.FS.value<0.5 or self.EECtl.effort>2.0: self.addfault('lostfunction')
    def behavior(self, time):
        if self.EECtl.effort>0.5 or self.faults.intersection(set(['lostfunction'])):
            self.DOFconfig=1
            self.Batconfig=1
            self.Ctlconfig=1
            self.Trajconfig=1
        else:
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2    
            
class holdPayload(fxnblock):
    def __init__(self,name, Force_gr,Force_air, Force_struct):
        self.name=name
        super().__init__({'FG':Force_gr, 'FA':Force_air, 'FS':Force_struct})
        self.fstate=1.0
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}, }
    def condfaults(self, time):
        if abs(self.FG.value)>1.6: self.addfault('break')
        elif abs(self.FG.value)>1.4: self.addfault('deform')
    def behavior(self, time):
        if self.hasfault('break'): self.fstate=0.0
        elif self.hasfault('deform'): self.fstate=0.5
        else: self.fstate=1.0
        self.FA.value=self.fstate
        self.FS.value=self.fstate
    
class affectDOF(fxnblock):
    def __init__(self, name, EEin, Ctlin, DOFout,Force, Hsig, Rsig, archtype):
        flows={'Hsig':Hsig, 'Rsig':Rsig, 'EEin':EEin, 'Ctlin':Ctlin,'DOF':DOFout,'Force':Force}
        super().__init__(flows)
        self.archtype=archtype
        self.faultmodes={}
        if archtype=='quad':
            LineRF=line('RF')
            LineLF=line('LF')
            LineLR=line('LR')
            LineRR=line('RR')
            self.lines=[LineRF,LineLF,LineLR, LineRR]
            self.upward=[1,1,1,1]
            self.forward=[0.5,0.5,-0.5,-0.5]
        for lin in self.lines:
            self.faultmodes.update(lin.faultmodes)  
    def behavior(self, time):
        Air={}
        EEin={}
        #injects faults into lines
        for lin in self.lines:
            for fault in self.faults:
                if fault in lin.faultmodes:
                    lin.faults.update([fault])
            
            ind=self.lines.index(lin)
            cmds={'up':self.upward[ind], 'for':self.forward[ind]}
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

class line(component):
    def __init__(self, name):
        super().__init__(name)
        self.elecstate=1.0
        self.elecstate_in=1.0
        self.ctlstate=1.0
        self.mechstate=1.0
        self.propstate=1.0
        self.Airout=1.0
        self.faultmodes={name+'short':{'rate':'moderate', 'rcost':'major'}, \
                         name+'openc':{'rate':'moderate', 'rcost':'major'}, \
                         name+'ctlup':{'rate':'moderate', 'rcost':'minor'}, \
                         name+'ctldn':{'rate':'moderate', 'rcost':'minor'}, \
                         name+'ctlbreak':{'rate':'common', 'rcost':'moderate'}, \
                         name+'mechbreak':{'rate':'common', 'rcost':'moderate'}, \
                         name+'mechfriction':{'rate':'common', 'rcost':'moderate'}, \
                         name+'propwarp':{'rate':'veryrare', 'rcost':'replacement'}, \
                         name+'propstuck':{'rate':'veryrare', 'rcost':'replacement'}, \
                         name+'propbreak':{'rate':'veryrare', 'rcost':'replacement'}
                         }
    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.faults.update([self.name+'mechbreak', self.name+'propbreak'])
        elif Force<=0.5: self.faults.update([self.name+'mechfriction'])
            
        if self.faults.intersection(set([self.name+'short'])):
            self.elecstate=0.0
            self.elecstate_in=np.inf
        elif self.faults.intersection(set([self.name+'openc'])):
            self.elecstate=0.0
            self.elecstate_in=0.0
        if self.faults.intersection(set([self.name+'ctlbreak'])):
            self.ctlstate=0.0
        elif self.faults.intersection(set([self.name+'ctldn'])):
            self.ctlstate=0.5
        elif self.faults.intersection(set([self.name+'ctlup'])):
            self.ctlstate=2.0
        if self.faults.intersection(set([self.name+'mechbreak'])):
            self.mechstate=0.0
        elif self.faults.intersection(set([self.name+'mechfriction'])):
            self.mechstate=0.5
            self.elecstate_in=2.0
        if self.faults.intersection(set([self.name+'propstuck'])):
            self.propstate=0.0
            self.mechstate=0.0
            self.elecstate_in=4.0
        elif self.faults.intersection(set([self.name+'propbreak'])):
            self.propstate=0.0
        elif self.faults.intersection(set([self.name+'propwarp'])):
            self.propstate=0.5
        
        self.Airout=m2to1([EEin,self.elecstate,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.ctlstate,self.mechstate,self.propstate])
        self.EE_in=m2to1([EEin,self.elecstate_in])     
    
class ctlDOF(fxnblock):
    def __init__(self, name,EEin, Dir, Ctl, DOFs, FS, Rsig):
        super().__init__({'EEin':EEin,'Rsig':Rsig,'Ctl':Ctl,'Dir':Dir,'DOFs':DOFs,'FS':FS})
        self.vel=0.0
        self.t1=0
        self.ctlstate=1.0
        self.faultmodes={'noctl':{'rate':'rare', 'rcost':'high'}, \
                         'degctl':{'rate':'rare', 'rcost':'high'}}
    def condfaults(self, time):
        if self.FS.value<0.5: self.addfault('noctl')
    def behavior(self, time):
        if self.hasfault('noctl'):    self.ctlstate=0.0
        elif self.hasfault('degctl'): self.ctlstate=0.5
        
        if time>self.t1:
            self.vel=self.DOFs.vertvel
            self.t1=time
        
        upthrottle=1.0
        
        if self.Dir.traj[2]>=1: upthrottle=1.5
        elif self.Dir.traj[2]>0 and self.Dir.traj[2]>1:
            upthrottle= 0.5 * self.Dir.traj[2] + 1.0
        elif self.Dir.traj[2]==0:
            damp=np.sign(self.vel)
            damp2=damp*min(1.0, np.power(self.vel, 2))
            upthrottle=1.0-0.2*damp2
        elif self.Dir.traj[2]<=0.0 and self.Dir.traj[2]>-1.0:
            maxdesc=-0.5
            damp=min(1.0, np.power(self.vel-maxdesc, 2))
            upthrottle=0.75+0.4*damp
        elif self.Dir.traj[2]<=-1.0:
            maxdesc=-5.0
            damp=min(0.75, np.power(self.vel-maxdesc, 2))
            upthrottle=0.75+0.15*damp
            
        if self.Dir.traj[0]==0 and self.Dir.traj[1]==0: forwardthrottle=0.0
        else: forwardthrottle=1.0
        
        pwr=self.Dir.power
        self.Ctl.forward=self.EEin.effort*self.ctlstate*forwardthrottle*pwr
        self.Ctl.upward=self.EEin.effort*self.ctlstate*upthrottle*pwr

class planpath(fxnblock):
    def __init__(self, name,EEin, Env, Dir, FS, Rsig):
        super().__init__({'EEin':EEin,'Rsig':Rsig,'Env':Env,'Dir':Dir,'FS':FS})
        self.mode='taxi'
        self.faultmodes={'noloc':{'rate':'rare', 'rcost':'high'}, \
                         'degloc':{'rate':'rare', 'rcost':'high'}}
    def condfaults(self, time):
        if self.FS.value<0.5: self.addfault('noloc')
    def behavior(self, t):
            
        if t<1: self.mode='taxi'
        elif self.mode=='taxi' and t<2: self.mode='climb'
        elif self.mode=='climb' and self.Env.elev>=50: self.mode='hover'
        elif self.mode=='hover' and self.Env.y==0 and t<20: self.mode='forward'
        elif self.mode=='forward' and self.Env.y>50: self.mode='hover'
        elif self.mode=='hover' and self.Env.y>50: self.mode='backward'
        elif self.mode=='backward' and self.Env.y<0: self.mode='hover'
        elif self.mode=='hover' and self.Env.y<0: self.mode='descend'
        elif self.mode=='descend' and self.Env.elev<10: self.mode='land'
        elif self.mode=='land' and self.Env.elev<1: self.mode='taxi'
        
        if self.mode=='taxi':  self.Dir.power=0.0
        elif self.mode=='takeoff':
            self.Dir.power=1.0
            self.Dir.traj=[0,0,1]
        elif self.mode=='climb':
            self.Dir.power=1.0
            self.Dir.traj=[0,0,1]
        elif self.mode=='hover':
            self.Dir.power=1.0
            self.Dir.traj=[0,0,0]
        elif self.mode=='forward':
            self.Dir.power=1.0
            self.Dir.traj=[0,1,0]
        elif self.mode=='backward':
            self.Dir.power=1.0
            self.Dir.traj=[0,-1,0]
        elif self.mode=='descend':
            self.Dir.power=1.0
            self.Dir.traj=[0,0,-1]
        elif self.mode=='land':
            self.Dir.power=1.0
            self.Dir.traj=[0,0,-0.1]
            
        if self.hasfault('noloc'): self.Dir.traj=[0,0,0]
        elif self.hasfault('degloc'): self.Dir.traj=[0,0,-1]
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.traj=[0,0,0]

class trajectory(fxnblock):
    def __init__(self, name, Env, DOF, Land, Dir, Force_LG):
        self.type='environment'
        super().__init__({'Env':Env,'DOF':DOF,'Land':Land, 'Dir': Dir, 'Force_LG': Force_LG})
        self.lasttime=0
        self.t1=0.0
        self.faultmodes={'nom':{'rate':'common', 'rcost':'NA'}, }
    def behavior(self, time):
        
        if time>self.lasttime:
            maxvel=20.0
            maxpvel=5.0
            
            if self.Env.elev<=0.0:
                self.Force_LG.value=min(-2.0, (self.DOF.vertvel-self.DOF.planvel)/3)
                flight=0.0
            else:
                self.Force_LG.value=0.0
                flight=1.0
            
            if time>self.t1:
                sign=np.sign(self.DOF.vertvel)
                damp=-0.02*sign*np.power(self.DOF.vertvel, 2)-0.1*self.DOF.vertvel
                acc=10*(self.DOF.uppwr-flight)
                self.DOF.vertvel=self.DOF.vertvel+acc+damp
                if self.Env.elev<=0.0:
                    self.DOF.vertvel=max(0,self.DOF.vertvel)
                self.t1=time
            
            self.DOF.planvel=flight*maxpvel*self.DOF.planpwr
                    
            self.Env.elev=max(0.0, self.Env.elev+self.DOF.vertvel)
            self.Env.x=self.Env.x+self.DOF.planvel*self.Dir.traj[0]
            self.Env.y=self.Env.y+self.DOF.planvel*self.Dir.traj[1]
            
            self.lasttime=time

##future: try to automate this part so you don't have to do it in a wierd order
def initialize():
    
    #initialize graph
    g=nx.DiGraph()
    ##Define flows for model
    #initialize one-line flows
    Force_ST=flow({'value':1.0},'Force')
    Force_Air=flow({'value':1.0},'Force')
    Force_GR=flow({'value':1.0},'Force')
    Force_LG=flow({'value':1.0},'Force')
    HSig_DOFs=flow({'hstate':'nominal'}, 'Health Signal')
    HSig_Bat=flow({'hstate':'nominal'}, 'Health Signal')
    RSig_DOFs=flow({'mode':1}, 'Reconfiguration Signal')
    RSig_Bat=flow({'mode':1}, 'Reconfig Signal')
    RSig_Ctl=flow({'mode':1}, 'Reconfig Signal')
    RSig_Traj=flow({'mode':1}, 'Reconfig Signal')
    EE_1=flow({'rate':1.0, 'effort':1.0}, 'EE')
    EEmot=flow({'rate':1.0, 'effort':1.0}, 'EE')
    EEctl=flow({'rate':1.0, 'effort':1.0}, 'EE')
    Ctl1=flow({'forward':0.0, 'upward':1.0}, 'Direction Signal')
    DOFs=flow({'stab':1.0, 'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0}, 'DOFs')
    Land1=flow({'status':'landed', 'area':'start'}, 'Land')
    #specialized flows
    Dir1=Direc()
    Env1=Env()

    ManageHealth=manageHealth(EEctl, Force_ST, HSig_DOFs, HSig_Bat, RSig_DOFs, RSig_Bat, RSig_Ctl, RSig_Traj)
    g.add_node('ManageHealth', obj=ManageHealth)
    
    StoreEE=storeEE('StoreEE',EE_1, Force_ST, HSig_Bat, RSig_Bat, 'normal')
    g.add_node('StoreEE', obj=StoreEE)
    
    DistEE=distEE(EE_1,EEmot,EEctl, Force_ST)
    g.add_node('DistEE', obj=DistEE)
    g.add_edge('StoreEE','DistEE', EE_1=EE_1)
    
    AffectDOF=affectDOF('AffectDOF',EEmot,Ctl1,DOFs,Force_Air, HSig_DOFs, RSig_DOFs, 'quad')
    g.add_node('AffectDOF', obj=AffectDOF)
    
    CtlDOF=ctlDOF('CtlDOF',EEctl, Dir1, Ctl1, DOFs, Force_ST, RSig_Ctl)
    g.add_node('CtlDOF', obj=CtlDOF)
    g.add_edge('DistEE','AffectDOF', EEmot=EEmot)
    g.add_edge('DistEE','CtlDOF', EEctl=EEctl)
    g.add_edge('CtlDOF','AffectDOF', Ctl1=Ctl1,DOFs=DOFs)

    Planpath=planpath('Planpath',EEctl, Env1,Dir1, Force_ST, RSig_Traj)
    g.add_node('Planpath', obj=Planpath)
    g.add_edge('DistEE','Planpath', EEctl=EEctl)
    g.add_edge('Planpath','CtlDOF', Dir1=Dir1)
    
    Trajectory=trajectory('Trajectory',Env1,DOFs,Land1,Dir1, Force_GR)
    g.add_node('Trajectory', obj=Trajectory)
    g.add_edge('Trajectory','AffectDOF',DOFs=DOFs)
    g.add_edge('Planpath', 'Trajectory', Dir1=Dir1, Env1=Env1)
    
    EngageLand=engageLand('EngageLand',Force_GR, Force_LG)
    g.add_node('EngageLand', obj=EngageLand)
    g.add_edge('Trajectory', 'EngageLand', Force_GR=Force_GR)
    
    HoldPayload=holdPayload('HoldPayload',Force_LG, Force_Air, Force_ST)
    g.add_node('HoldPayload', obj=HoldPayload)
    g.add_edge('EngageLand','HoldPayload', Force_LG=Force_LG)
    g.add_edge('HoldPayload', 'AffectDOF', Force_Air=Force_Air)
    g.add_edge('HoldPayload', 'StoreEE', Force_ST=Force_ST)
    g.add_edge('HoldPayload', 'DistEE', Force_ST=Force_ST)
    g.add_edge('HoldPayload', 'Planpath', Force_ST=Force_ST)
    g.add_edge('HoldPayload', 'CtlDOF', Force_ST=Force_ST)
    g.add_edge('HoldPayload', 'ManageHealth', Force_ST=Force_ST)
    
    g.add_edge('DistEE', 'ManageHealth', EEctl=EEctl)
    g.add_edge('AffectDOF','ManageHealth', HSig_DOFs=HSig_DOFs, RSig_DOFs=RSig_DOFs)
    g.add_edge('StoreEE','ManageHealth', HSig_Bat=HSig_Bat, RSig_Bat=RSig_Bat)
    g.add_edge('ManageHealth', 'CtlDOF', RSig_Ctl=RSig_Ctl)
    g.add_edge('ManageHealth', 'Planpath', RSig_Traj=RSig_Traj)
    
    return g

    
def findclassification(g, endfaults, endflows, scen):
    
    Env=fp.getflow('Env1', g)
    
    #may need to redo this
    if  inrange(Env.start_area, Env.x, Env.y):
        landloc='nominal'
        area=1
    elif inrange(Env.safe1_area, Env.x, Env.y) or inrange(Env.safe2_area, Env.x, Env.y):
        landloc='emsafe'
        area=1000
    elif inrange(Env.dang_area, Env.x, Env.y):
        landloc='emdang'
        area=100000
    else:
        landloc='emunsanc'
        area=10000
        
    repaircosts=fp.listfaultsprops(endfaults, g, 'rcost')
    maxcost=textmax(repaircosts.values())
    
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
    
    rate=1e-6
    
    expcost=totcost*rate*1e5
    
    return {'rate':rate, 'cost': totcost, 'expected cost': expcost}

## BASE FUNCTIONS

#translates L, R, and C into Left, Right, and Center
def rlc(x):
    y='NA'
    if x=='R': y='Right'
    if x=='L': y='Left'
    if x=='C': y='Center'
    return y

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