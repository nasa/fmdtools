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
class Direc(flow):
    def __init__(self):
        self.traj=[0,0,0]
        super().__init__({'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': 1}, 'Trajectory')
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()

#Define functions
class storeEE(fxnblock):
    def __init__(self, flows, archtype):
        if archtype[0]=='normal':
            #architecture: 1 for controllers? + cells in Series & Parallel
            #Batctl=battery('ctl')
            components={'00':battery('00'), '01':battery('01'), '10':battery('10'), '11':battery('11')}
        self.faultmodes={'nocharge':{'rate':'moderate','rcost':'minor'}, \
                         'lowcharge':{'rate':'moderate','rcost':'minor'}} 
        super().__init__(['EEout', 'FS', 'Hsig', 'Rsig'], flows, {'soc': 2000}, components)
    def condfaults(self, time):
        if self.soc<20: self.addfault('lowcharge')
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

class battery(component):
    def __init__(self, name):
        super().__init__(name, {'soc':2000, 'EEe':1.0, 'Et':1.0})
        self.faultmodes={name+'short':{'rate':'moderate', 'rcost':'major'}, \
                         name+'degr':{'rate':'moderate', 'rcost':'minor'}, \
                         name+'break':{'rate':'common', 'rcost':'moderate'}, \
                         name+'nocharge':{'rate':'moderate','rcost':'minor'}, \
                         name+'lowcharge':{'rate':'moderate','rcost':'minor'}}
    def behavior(self, FS, EEoutr, time):
        if FS <1.0: self.addfault(self.name+'break')
        if EEoutr>2: self.addfault(self.name+'break')
        if self.soc<20: self.addfault(self.name+'lowcharge')
        if self.soc<1: self.replacefault(self.name+'lowcharge',self.name+'nocharge')

        if self.hasfault(self.name+'short'): self.Et=0.0
        elif self.hasfault(self.name+'break'): self.Et=0.0
        elif self.hasfault(self.name+'degr'): self.Et=0.5
        
        if self.hasfault(self.name+'nocharge'):
            self.soc=0.0
            self.Et=0.0
            
        if time > self.time:
            self.soc=self.soc-EEoutr*(time-self.time)
            self.time=time
        return self.Et

class distEE(fxnblock):
    def __init__(self,flows):
        super().__init__(['EEin','EEmot','EEctl','FS'],flows, {'EEtr':1.0, 'EEte':1.0})
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
    def __init__(self,flows):
        super().__init__(['forcein', 'forceout'],flows, {'Ft':1.0})
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}}
    def condfaults(self, time):
        if self.forceout.value<-1.4: self.addfault('break')
        elif self.forceout.value<-1.2: self.addfault('deform')
    def behavior(self, time):
        if self.hasfault('break'): self.Ft=4.0
        elif self.hasfault('deform'): self.Ft=2.0
        else: self.Ft=1.0
        self.forceout.value=self.Ft*min([-2.0,self.forcein.value])*0.2
    
class manageHealth(fxnblock):
    def __init__(self,flows):
        flownames=['EECtl','FS','DOFshealth', 'Bathealth','DOFconfig','Batconfig','Ctlconfig', 'Trajconfig' ]
        super().__init__(flownames, flows)
        
        self.faultmodes={'falsemaintenance':{'rate':'moderate', 'rcost':'minor'},\
                         'falsemasking':{'rate':'rare', 'rcost': 'major'},\
                         'falseemland':{'rate':'rare', 'rcost': 'major'},\
                         'lostfunction':{'rate':'rare', 'rcost':'minor'}} 
        #need to add joint-fault modes of not catching faults?
    def condfaults(self, time):
        if self.FS.value<0.5 or self.EECtl.effort>2.0: self.addfault('lostfunction')
    def behavior(self, time):
        if self.EECtl.effort>0.5 or self.hasfault('lostfunction'):
            self.DOFconfig=1
            self.Batconfig=1
            self.Ctlconfig=1
            self.Trajconfig=1
        else:
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2
            if self.DOFshealth=='degraded': self.DOFconfig=2    
            
class holdPayload(fxnblock):
    def __init__(self,flows):
        super().__init__(['FG', 'FA', 'FS'],flows, {'Ft': 1.0})
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}, }
    def condfaults(self, time):
        if abs(self.FG.value)>1.6: self.addfault('break')
        elif abs(self.FG.value)>1.4: self.addfault('deform')
    def behavior(self, time):
        if self.hasfault('break'): self.Ft=0.0
        elif self.hasfault('deform'): self.Ft=0.5
        else: self.Ft=1.0
        self.FA.value=self.Ft
        self.FS.value=self.Ft
    
class affectDOF(fxnblock): #EEmot,Ctl1,DOFs,Force_Air, HSig_DOFs, RSig_DOFs
    def __init__(self, flows, archtype):     
        self.archtype=archtype
        self.faultmodes={}
        if archtype[0]=='quad':
            components={'RF':line('RF'), 'LF':line('LF'), 'LR':line('LR'), 'RR':line('RR')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
        super().__init__(['EEin', 'Ctlin','DOF','Force','Hsig', 'Rsig'], flows,{}, components) 
    def behavior(self, time):
        Air={}
        EEin={}
        #injects faults into lines
        for linname,lin in self.components.items():
            for fault in self.faults:
                if fault in lin.faultmodes:
                    lin.addfault(fault)
            
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

class line(component):
    def __init__(self, name):
        super().__init__(name,{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0})
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
        if Force<=0.0:   self.addfaults([self.name+'mechbreak', self.name+'propbreak'])
        elif Force<=0.5: self.addfault(self.name+'mechfriction')
            
        if self.hasfault(self.name+'short'):
            self.Eti=0.0
            self.Eto=np.inf
        elif self.hasfault(self.name+'openc'):
            self.Eti=0.0
            self.Et0=0.0
        if self.hasfault(self.name+'ctlbreak'): self.Ct=0.0
        elif self.hasfault(self.name+'ctldn'):  self.Ct=0.5
        elif self.hasfault(self.name+'ctlup'):  self.Ct=2.0
        if self.hasfault(self.name+'mechbreak'): self.Mt=0.0
        elif self.hasfault(self.name+'mechfriction'):
            self.Mt=0.5
            self.Eti=2.0
        if self.hasfault(self.name+'propstuck'):
            self.Pt=0.0
            self.Mt=0.0
            self.Eti=4.0
        elif self.hasfault(self.name+'propbreak'): self.Pt=0.0
        elif self.hasfault(self.name+'propwarp'):  self.Pt=0.5
        
        self.Airout=m2to1([EEin,self.Eti,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.Ct,self.Mt,self.Pt])
        self.EE_in=m2to1([EEin,self.Eto])     
    
class ctlDOF(fxnblock):
    def __init__(self, flows):
        super().__init__(['EEin','Dir','Ctl','DOFs','FS','Rsig'],flows, {'vel':0.0, 'Cs':1.0})
        self.faultmodes={'noctl':{'rate':'rare', 'rcost':'high'}, \
                         'degctl':{'rate':'rare', 'rcost':'high'}}
    def condfaults(self, time):
        if self.FS.value<0.5: self.addfault('noctl')
    def behavior(self, time):
        if self.hasfault('noctl'):    self.Cs=0.0
        elif self.hasfault('degctl'): self.Cs=0.5
        
        if time>self.time:
            self.vel=self.DOFs.vertvel
        
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
        self.Ctl.forward=self.EEin.effort*self.Cs*forwardthrottle*pwr
        self.Ctl.upward=self.EEin.effort*self.Cs*upthrottle*pwr

class planpath(fxnblock):
    def __init__(self, flows):
        super().__init__(['EEin','Env','Dir','FS','Rsig'], flows)
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
        
        self.Dir.power=1.0
        if self.mode=='taxi':  self.Dir.power=0.0
        elif self.mode=='takeoff': self.Dir.traj=[0,0,1]
        elif self.mode=='climb': self.Dir.traj=[0,0,1]
        elif self.mode=='hover': self.Dir.traj=[0,0,0]           
        elif self.mode=='forward': self.Dir.traj=[0,1,0]           
        elif self.mode=='backward': self.Dir.traj=[0,-1,0]
        elif self.mode=='descend': self.Dir.traj=[0,0,-1]
        elif self.mode=='land': self.Dir.traj=[0,0,-0.1]
            
        if self.hasfault('noloc'): self.Dir.traj=[0,0,0]
        elif self.hasfault('degloc'): self.Dir.traj=[0,0,-1]
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.traj=[0,0,0]

class trajectory(fxnblock):
    def __init__(self, flows):
        super().__init__(['Env','DOF','Land', 'Dir', 'Force_LG'], flows, {'flight':0.0})
        self.faultmodes={'nom':{'rate':'common', 'rcost':'NA'}, }
    def behavior(self, time):
        
        if time>self.time:
            maxvel=20.0
            maxpvel=5.0
            
            if self.Env.elev<=0.0:
                self.Force_LG.value=min(-2.0, (self.DOF.vertvel-self.DOF.planvel)/3)
                self.flight=0.0
            else:
                self.Force_LG.value=0.0
                self.flight=1.0
            
            sign=np.sign(self.DOF.vertvel)
            damp=-0.02*sign*np.power(self.DOF.vertvel, 2)-0.1*self.DOF.vertvel
            acc=10*(self.DOF.uppwr-self.flight)
            self.DOF.vertvel=self.DOF.vertvel+acc+damp
            if self.Env.elev<=0.0:
                self.DOF.vertvel=max(0,self.DOF.vertvel)
            
            self.DOF.planvel=self.flight*maxpvel*self.DOF.planpwr
                    
            self.Env.elev=max(0.0, self.Env.elev+self.DOF.vertvel)
            self.Env.x=self.Env.x+self.DOF.planvel*self.Dir.traj[0]
            self.Env.y=self.Env.y+self.DOF.planvel*self.Dir.traj[1]
            
class quadrotor(model):
    def __init__(self):
        super().__init__()
        #add flows to the model
        self.addflow('Force_ST', 'Force', {'value':1.0})
        self.addflow('Force_Air','Force', {'value':1.0} )
        self.addflow('Force_GR','Force', {'value':1.0} )
        self.addflow('Force_LG','Force', {'value':1.0})
        self.addflow('HSig_DOFs','Health Signal', {'hstate':'nominal'})
        self.addflow('HSig_Bat','Health Signal', {'hstate':'nominal'} )
        self.addflow('RSig_DOFs','Reconfiguration Signal', {'mode':1} )
        self.addflow('RSig_Bat','Reconfiguration Signal', {'mode':1} )
        self.addflow('RSig_Ctl','Reconfiguration Signal', {'mode':1})
        self.addflow('RSig_Traj','Reconfiguration Signal', {'mode':1})
        self.addflow('EE_1', 'EE', {'rate':1.0, 'effort':1.0})
        self.addflow('EEmot', 'EE', {'rate':1.0, 'effort':1.0})
        self.addflow('EEctl', 'EE', {'rate':1.0, 'effort':1.0})
        self.addflow('Ctl1','Direction Signal', {'forward':0.0, 'upward':1.0})
        self.addflow('DOFs', 'DOFs',{'stab':1.0, 'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0})
        self.addflow('Land1','Land', {'status':'landed', 'area':'start'} )
        self.addflow('Env1','Environment', {'x':0.0,'y':0.0,'elev':0.0} )
        # custom flows
        self.addflow('Dir1', 'Direction', Direc())
        #add functions to the model
        flows=['EEctl', 'Force_ST', 'HSig_DOFs', 'HSig_Bat', 'RSig_DOFs', 'RSig_Bat', 'RSig_Ctl', 'RSig_Traj']
        self.addfxn('ManageHealth',manageHealth,flows)
        self.addfxn('StoreEE',storeEE,['EE_1', 'Force_ST', 'HSig_Bat', 'RSig_Bat'], 'normal')
        self.addfxn('DistEE',distEE, ['EE_1','EEmot','EEctl', 'Force_ST'])
        self.addfxn('AffectDOF',affectDOF,['EEmot','Ctl1','DOFs','Force_Air', 'HSig_DOFs', 'RSig_DOFs'], 'quad')
        self.addfxn('CtlDOF', ctlDOF,['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST', 'RSig_Ctl'])
        self.addfxn('Planpath', planpath, ['EEctl', 'Env1','Dir1', 'Force_ST', 'RSig_Traj'])
        self.addfxn('Trajectory', trajectory,['Env1','DOFs','Land1','Dir1', 'Force_GR'] )
        self.addfxn('EngageLand', engageLand,['Force_GR', 'Force_LG'])
        self.addfxn('HoldPayload', holdPayload,['Force_LG', 'Force_Air', 'Force_ST'])

def initialize():
    q=quadrotor()
    return q.constructgraph()

##future: try to automate this part so you don't have to do it in a wierd order
#def initialize():
#    
#    #initialize graph
#    g=nx.DiGraph()
#    ##Define flows for model
#    #initialize one-line flows
#    Force_ST=flow({'value':1.0},'Force')
#    Force_Air=flow({'value':1.0},'Force')
#    Force_GR=flow({'value':1.0},'Force')
#    Force_LG=flow({'value':1.0},'Force')
#    HSig_DOFs=flow({'hstate':'nominal'}, 'Health Signal')
#    HSig_Bat=flow({'hstate':'nominal'}, 'Health Signal')
#    RSig_DOFs=flow({'mode':1}, 'Reconfiguration Signal')
#    RSig_Bat=flow({'mode':1}, 'Reconfig Signal')
#    RSig_Ctl=flow({'mode':1}, 'Reconfig Signal')
#    RSig_Traj=flow({'mode':1}, 'Reconfig Signal')
#    EE_1=flow({'rate':1.0, 'effort':1.0}, 'EE')
#    EEmot=flow({'rate':1.0, 'effort':1.0}, 'EE')
#    EEctl=flow({'rate':1.0, 'effort':1.0}, 'EE')
#    Ctl1=flow({'forward':0.0, 'upward':1.0}, 'Direction Signal')
#    DOFs=flow({'stab':1.0, 'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0}, 'DOFs')
#    Land1=flow({'status':'landed', 'area':'start'}, 'Land')
#    Env1=flow({'x':0.0,'y':0.0,'elev':0.0}, 'Environment')
#    
#    #specialized flows
#    Dir1=Direc()
#
#    ManageHealth=manageHealth([EEctl, Force_ST, HSig_DOFs, HSig_Bat, RSig_DOFs, RSig_Bat, RSig_Ctl, RSig_Traj])
#    g.add_node('ManageHealth', obj=ManageHealth)
#    
#    StoreEE=storeEE([EE_1, Force_ST, HSig_Bat, RSig_Bat], ['normal'])
#    g.add_node('StoreEE', obj=StoreEE)
#    
#    DistEE=distEE([EE_1,EEmot,EEctl, Force_ST])
#    g.add_node('DistEE', obj=DistEE)
#    g.add_edge('StoreEE','DistEE', EE_1=EE_1)
#    
#    AffectDOF=affectDOF([EEmot,Ctl1,DOFs,Force_Air, HSig_DOFs, RSig_DOFs], ['quad'])
#    g.add_node('AffectDOF', obj=AffectDOF)
#    
#    CtlDOF=ctlDOF([EEctl, Dir1, Ctl1, DOFs, Force_ST, RSig_Ctl])
#    g.add_node('CtlDOF', obj=CtlDOF)
#    g.add_edge('DistEE','AffectDOF', EEmot=EEmot)
#    g.add_edge('DistEE','CtlDOF', EEctl=EEctl)
#    g.add_edge('CtlDOF','AffectDOF', Ctl1=Ctl1,DOFs=DOFs)
#
#    Planpath=planpath([EEctl, Env1,Dir1, Force_ST, RSig_Traj])
#    g.add_node('Planpath', obj=Planpath)
#    g.add_edge('DistEE','Planpath', EEctl=EEctl)
#    g.add_edge('Planpath','CtlDOF', Dir1=Dir1)
#    
#    Trajectory=trajectory([Env1,DOFs,Land1,Dir1, Force_GR])
#    g.add_node('Trajectory', obj=Trajectory)
#    g.add_edge('Trajectory','AffectDOF',DOFs=DOFs)
#    g.add_edge('Planpath', 'Trajectory', Dir1=Dir1, Env1=Env1)
#    
#    EngageLand=engageLand([Force_GR, Force_LG])
#    g.add_node('EngageLand', obj=EngageLand)
#    g.add_edge('Trajectory', 'EngageLand', Force_GR=Force_GR)
#    
#    HoldPayload=holdPayload([Force_LG, Force_Air, Force_ST])
#    g.add_node('HoldPayload', obj=HoldPayload)
#    g.add_edge('EngageLand','HoldPayload', Force_LG=Force_LG)
#    g.add_edge('HoldPayload', 'AffectDOF', Force_Air=Force_Air)
#    g.add_edge('HoldPayload', 'StoreEE', Force_ST=Force_ST)
#    g.add_edge('HoldPayload', 'DistEE', Force_ST=Force_ST)
#    g.add_edge('HoldPayload', 'Planpath', Force_ST=Force_ST)
#    g.add_edge('HoldPayload', 'CtlDOF', Force_ST=Force_ST)
#    g.add_edge('HoldPayload', 'ManageHealth', Force_ST=Force_ST)
#    
#    g.add_edge('DistEE', 'ManageHealth', EEctl=EEctl)
#    g.add_edge('AffectDOF','ManageHealth', HSig_DOFs=HSig_DOFs, RSig_DOFs=RSig_DOFs)
#    g.add_edge('StoreEE','ManageHealth', HSig_Bat=HSig_Bat, RSig_Bat=RSig_Bat)
#    g.add_edge('ManageHealth', 'CtlDOF', RSig_Ctl=RSig_Ctl)
#    g.add_edge('ManageHealth', 'Planpath', RSig_Traj=RSig_Traj)
#    
#    return g


    
def findclassification(g, endfaults, endflows, scen):
    
    start=[0.0,0.0]
    start_xw=5
    start_yw=5
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
    
    Env=fp.getflow('Env1', g)
    
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