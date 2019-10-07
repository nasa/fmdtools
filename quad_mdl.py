# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""

import networkx as nx
import numpy as np

import auxfunctions as aux
import faultprop as fp

#Declare time range to run model over
times=[0,3, 55]

##Define flows for model
class EE:
    def __init__(self,name):
        self.rate=1.0
        self.effort=1.0
    def status(self):
        status={'rate':self.rate, 'effort':self.effort}
        return status.copy()
    
class Force:
    def __init__(self,name):
        self.flowtype='Force'
        self.name=name
        self.value=1.0
    def status(self):
        status={'value':self.value}
        return status.copy()

class ME:
    def __init__(self,name):
        self.flowtype='ME'
        self.name=name
        self.rate=1.0
        self.effort=1.0
        self.nominal={'rate':1.0, 'effort':1.0}
    def status(self):
        status={'rate':self.rate, 'effort':self.effort}
        return status.copy() 

class Sig:
    def __init__(self,name):
        self.flowtype='Sig'
        self.name=name
        self.forward=0.0
        self.upward=0.0
    def status(self):
        status={'forward':self.forward, 'upward':self.upward}
        return status.copy() 

class HSig:
    def __init__(self,name):
        self.flowtype='Health Signal'
        self.name=name
        self.health={"nominal":"nominal"}
        self.healths={1:{"nominal":"nominal"}, 2:{"degraded":"mode"},3:{"failed":"mode"}}
    def status(self):
        status={'Health State':self.health}
        return status.copy() 

class RSig:
    def __init__(self,name):
        self.flowtype='Reconfiguration Signal'
        self.name=name
        self.mode=1 # 1 nominal, 2+ reconfigured
    def status(self):
        status={'Reconfiguration Mode': self.mode}
        return status.copy() 

class DOF:
    def __init__(self,name):
        self.flowtype='DOF'
        self.name=name
        self.stab=1.0
        self.vertvel=0.0
        self.planvel=0.0
        self.uppwr=0.0
        self.planpwr=0.0
    def status(self):
        status={'stab':self.stab, 'vertvel':self.vertvel, 'planvel':self.planvel, 'planpwr':self.planpwr, 'uppwr':self.uppwr}
        return status.copy() 
class Land:
    def __init__(self,name):
        self.flowtype='Land'
        self.name=name
        self.stat='landed'
        self.area='start'
        self.nominal={'status':'landed', 'area':'start'}
    def status(self):
        status={'status':self.stat, 'area':self.area}
        return status.copy() 

class Env:
    def __init__(self,name):
        self.flowtype='Env'
        self.name=name
        self.elev=0.0
        self.x=0.0
        self.y=0.0
        self.start=[0.0,0.0]
        self.start_xw=5
        self.start_yw=5
        self.start_area=aux.square(self.start,self.start_xw, self.start_yw)
        self.flyelev=30
        self.poi_center=[0,150]
        self.poi_xw=50
        self.poi_yw=50
        self.poi_area=aux.square(self.poi_center, self.poi_xw, self.poi_yw)
        self.dang_center=[0,150]
        self.dang_xw=150
        self.dang_yw=150
        self.dang_area=aux.square(self.dang_center, self.dang_xw, self.dang_yw)
        self.safe1_center=[-25,100]
        self.safe1_xw=10
        self.safe1_yw=10
        self.safe1_area=aux.square(self.safe1_center, self.safe1_xw, self.safe1_yw)
        self.safe2_center=[25,50]
        self.safe2_xw=10
        self.safe2_yw=10
        self.safe2_area=aux.square(self.safe2_center, self.safe2_xw, self.safe2_yw)
        self.nominal={'elev':1.0, 'x':1.0, 'y':1.0}
    def status(self):
        status={'elev':self.elev, 'x':self.x, 'y':self.y}
        return status.copy()

class Direc:
    def __init__(self,name):
        self.flowtype='Dir'
        self.name=name
        self.traj=[0,0,0]
        self.power=1
        self.nominal={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': 1}
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()

class storeEE:
    def __init__(self, name,EEout, FS, Hsig, Rsig):
        self.type='function'
        self.EEout=EEout
        self.Hsig=Hsig
        self.Rsig=Rsig
        self.FS=FS
        self.effstate=1.0
        self.ratestate=1.0
        self.soc=2000
        self.faultmodes={'short':{'rate':'moderate', 'rcost':'major'}, \
                         'degr':{'rate':'moderate', 'rcost':'minor'}, \
                         'break':{'rate':'common', 'rcost':'moderate'}, \
                         'nocharge':{'rate':'moderate','rcost':'minor'}, \
                         'lowcharge':{'rate':'moderate','rcost':'minor'}}
        self.faults=set(['nom'])
    def condfaults(self):
        if self.FS.value<1.0:
            self.faults.update(['break'])
        if self.EEout.rate>2:
            self.faults.add('break')
        if self.soc<20:
            self.faults.add('lowcharge')
        if self.soc<1:
            self.faults.remove('lowcharge')
            self.faults.add('nocharge')
        return 0
    def behavior(self, time):
        if self.faults.intersection(set(['short'])):
            self.effstate=0.0
        elif self.faults.intersection(set(['break'])):
            self.effstate=0.0
        elif self.faults.intersection(set(['degr'])):
            self.effstate=0.5
        
        if self.faults.intersection(set(['nocharge'])):
            self.soc=0.0
            self.effstate=0.0
            
        self.EEout.effort=self.effstate
        self.soc=self.soc-self.EEout.rate*time
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.condfaults()
        self.behavior(time)
        return 

class distEE:
    def __init__(self,EEin,EEmot,EEctl,FS):
        self.useprop=1.0
        self.type='function'
        self.EEin=EEin
        self.EEmot=EEmot
        self.EEctl=EEctl
        self.FS=FS
        self.effstate=1.0
        self.ratestate=1.0
        self.faultmodes={'short':{'rate':'moderate', 'rcost':'major'}, \
                         'degr':{'rate':'moderate', 'rcost':'minor'}, \
                         'break':{'rate':'common', 'rcost':'moderate'}}
        self.faults=set(['nom'])
    def condfaults(self):
        if self.FS.value<0.5:
            self.faults.update(['break'])
        if max(self.EEmot.rate,self.EEctl.rate)>2:
            self.faults.add('break') 
    def behavior(self, time):
        if self.faults.intersection(set(['short'])):
            self.ratestate=np.inf
            self.effstate=0.0
        elif self.faults.intersection(set(['break'])):
            self.effstate=0.0
        elif self.faults.intersection(set(['degr'])):
            self.effstate=0.5
        self.EEin.rate=self.ratestate*self.EEin.effort
        self.EEmot.effort=self.effstate*self.EEin.effort
        self.EEctl.effort=self.effstate*self.EEin.effort
        
        self.EEin.rate=aux.m2to1([ self.EEin.effort, self.ratestate, max(self.EEmot.rate,self.EEctl.rate)])
        
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.condfaults()
        self.behavior(time)
        return 

class engageLand:
    def __init__(self,name, Forcein, Forceout):
        self.useprop=1.0
        self.name=name
        self.type='function'
        self.forcein=Forcein
        self.forceout=Forceout
        self.fstate=1.0
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}}
        self.faults=set(['nom'])
    def condfaults(self):
        if self.forceout.value<-1.4:
            self.faults.update(['break'])
        elif self.forceout.value<-1.2:
            self.faults.update(['deform'])
    def behavior(self, time):
        if self.faults.intersection(set(['break'])):
            self.fstate=4.0
        elif self.faults.intersection(set(['deform'])):
            self.fstate=2.0
        else:
            self.fstate=1.0
            
        self.forceout.value=self.fstate*min([-2.0,self.forcein.value])*0.2
            
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.condfaults()
        self.behavior(time)
        return 
    
class manageHealth:
    def __init__(self, EECtl, Force_ST, HSig_DOFs, HSig_Bat, RSig_DOFs, RSig_Bat, RSig_Ctl, RSig_Traj):
        self.DOFshealth=HSig_DOFs
        self.Bathealth=HSig_Bat
        self.DOFconfig=RSig_DOFs
        self.Batconfig=RSig_Bat
        self.Ctlconfig=RSig_Ctl
        self.Trajconfig=RSig_Traj
        self.FS=Force_ST
        self.EECtl=EECtl
        
        self.faultmodes={'falsemaintenance':{'rate':'moderate', 'rcost':'minor'},\
                         'falsemasking':{'rate':'rare', 'rcost': 'major'},\
                         'falseemland':{'rate':'rare', 'rcost': 'major'},\
                         'lostfunction':{'rate':'rare', 'rcost':'minor'}} 
        #need to add joint-fault modes of not catching faults?
        self.faults=set(['nom'])
    def condfaults(self):
        if self.FS.value<0.5 or self.EECtl.effort>2.0:
            self.faults.update(['lostfunction'])
    def behavior(self):
        
        if self.EECtl.effort>0.5 or self.faults.intersection(set(['lostfunction'])):
            self.DOFconfig=1
            self.Batconfig=1
            self.Ctlconfig=1
            self.Trajconfig=1
        else:
            if self.DOFshealth=='degraded':
                self.DOFconfig=2
            if self.DOFshealth=='degraded':
                self.DOFconfig=2
            if self.DOFshealth=='degraded':
                self.DOFconfig=2
        
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.condfaults()
        self.behavior()
        return     
            
class holdPayload:
    def __init__(self,name, Force_gr,Force_air, Force_struct):
        self.name=name
        self.useprop=1.0
        self.type='function'
        self.FG=Force_gr
        self.FA=Force_air
        self.FS=Force_struct
        self.fstate=1.0
        self.faultmodes={'break':{'rate':'moderate', 'rcost':'major'}, \
                         'deform':{'rate':'moderate', 'rcost':'minor'}, }
        self.faults=set(['nom'])
    def condfaults(self):
        if abs(self.FG.value)>1.6:
            self.faults.update(['break'])
        elif abs(self.FG.value)>1.4:
            self.faults.update(['deform'])
    def behavior(self, time):
        if self.faults.intersection(set(['break'])):
            self.fstate=0.0
        elif self.faults.intersection(set(['deform'])):
            self.fstate=0.5
        else:
            self.fstate=1.0
        self.FA.value=self.fstate
        self.FS.value=self.fstate
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.condfaults()
        self.behavior(time)
        return 
    
class affectDOF:
    def __init__(self, name, EEin, Ctlin, DOFout,Force, Hsig, Rsig, archtype):
        self.type='function'
        self.Hsig=Hsig
        self.Rsig=Rsig
        self.EEin=EEin
        self.Ctlin=Ctlin
        self.DOF=DOFout
        self.Force=Force
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
        self.faults={'nom'}
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
        
        if any(value==np.inf for value in EEin.values()):
            self.EEin.rate=np.inf
        elif any(value!=0.0 for value in EEin.values()):
            self.EEin.rate=np.max(list(EEin.values()))
        else:
            self.EEin.rate=0.0
        
        if all(value==1.0 for value in Air.values()):
            self.DOF.stab=1.0
        elif all(value==0.5 for value in Air.values()):
            self.DOF.stab=1.0
        elif all(value==2.0 for value in Air.values()):
            self.DOF.stab=1.0
        elif all(value==0.0 for value in Air.values()):
            self.DOF.stab=1.0
        elif any(value==0.0 for value in Air.values()):
            self.DOF.stab=0.0
        elif any(value>2.5 for value in Air.values()):
            self.DOF.stab=0.0
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
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.faults.update(faults)
        self.behavior(time)
        return 

class line:
    def __init__(self, name):
        self.type='component'
        self.name=name 
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
        self.faults=set(['nom'])
    def behavior(self, EEin, Ctlin, cmds, Force):
        
        if Force<=0.0:
            self.faults.update([self.name+'mechbreak', self.name+'propbreak'])
        elif Force<=0.5:
            self.faults.update([self.name+'mechfriction'])
            
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
        
        self.Airout=aux.m2to1([EEin,self.elecstate,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.ctlstate,self.mechstate,self.propstate])
        self.EE_in=aux.m2to1([EEin,self.elecstate_in])     
    
class ctlDOF:
    def __init__(self, name,EEin, Dir, Ctl, DOFs, FS, Rsig):
        self.type='function'
        self.EEin=EEin
        self.Rsig=Rsig
        self.Ctl=Ctl
        self.Dir=Dir
        self.DOFs=DOFs
        self.FS=FS
        self.vel=0.0
        self.t1=0
        self.ctlstate=1.0
        self.faultmodes={'noctl':{'rate':'rare', 'rcost':'high'}, \
                         'degctl':{'rate':'rare', 'rcost':'high'}}
        self.faults=set(['nom'])
    def condfaults(self):
        if self.FS.value<0.5:
            self.faults.update(['noctl'])
    def behavior(self, time):
        if self.faults.intersection(set(['noctl'])):
            self.ctlstate=0.0
        elif self.faults.intersection(set(['degctl'])):
            self.ctlstate=0.5
        
        if time>self.t1:
            self.vel=self.DOFs.vertvel
            self.t1=time
        
        upthrottle=1.0
        
        if self.Dir.traj[2]>=1:
            upthrottle=1.5
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
            
        if self.Dir.traj[0]==0 and self.Dir.traj[1]==0:
            forwardthrottle=0.0
        else:
            forwardthrottle=1.0
        
        pwr=self.Dir.power
        self.Ctl.forward=self.EEin.effort*self.ctlstate*forwardthrottle*pwr
        self.Ctl.upward=self.EEin.effort*self.ctlstate*upthrottle*pwr

    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.condfaults()
        self.faults.update(faults)
        self.behavior(time)

class planpath:
    def __init__(self, name,EEin, Env, Dir, FS, Rsig):
        self.type='function'
        self.EEin=EEin
        self.Rsig=Rsig
        self.Env=Env
        self.Dir=Dir
        self.FS=FS
        self.mode='taxi'
        self.faultmodes={'noloc':{'rate':'rare', 'rcost':'high'}, \
                         'degloc':{'rate':'rare', 'rcost':'high'}}
        self.faults=set(['nom'])
    def condfaults(self):
        if self.FS.value<0.5:
            self.faults.update(['noloc'])
    def behavior(self, t):
            
        if t<1:
            self.mode='taxi'
        elif self.mode=='taxi' and t<2:
            self.mode='climb'
        elif self.mode=='climb' and self.Env.elev>=50:
            self.mode='hover'
        elif self.mode=='hover' and self.Env.y==0 and t<20:
            self.mode='forward'
        elif self.mode=='forward' and self.Env.y>50:
            self.mode='hover'
        elif self.mode=='hover' and self.Env.y>50:
            self.mode='backward'
        elif self.mode=='backward' and self.Env.y<0:
            self.mode='hover'
        elif self.mode=='hover' and self.Env.y<0:
            self.mode='descend'
        elif self.mode=='descend' and self.Env.elev<10:
            self.mode='land'
        elif self.mode=='land' and self.Env.elev<1:
            self.mode='taxi'
        
        if self.mode=='taxi':
            self.Dir.power=0.0
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
            
        if self.faults.intersection(set(['noloc'])):
            self.Dir.traj=[0,0,0]
        elif self.faults.intersection(set(['degloc'])):
            self.Dir.traj=[0,0,-1]
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.traj=[0,0,0]
        
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        self.condfaults()
        self.faults.update(faults)
        self.behavior(time)

class trajectory:
    def __init__(self, name, Env, DOF, Land, Dir, Force_LG):
        self.type='environment'
        self.Env=Env
        self.DOF=DOF
        self.Land=Land
        self.Dir=Dir
        self.Force_LG=Force_LG
        self.lasttime=0
        self.t1=0.0
        self.faultmodes={'nom':{'rate':'common', 'rcost':'NA'}, }
        self.faults=set(['nom'])
    def condfaults(self):
        return 0
    def behavior(self, time):
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
        
    def updatefxn(self,faults=['nom'],opermode=[], time=0):
        if time>self.lasttime:
            self.behavior(time)
            self.lasttime=time
        self.condfaults()

##future: try to automate this part so you don't have to do it in a wierd order
def initialize():
    
    #initialize graph
    g=nx.DiGraph()
    
    Force_ST=Force('Force_ST')
    Force_Air=Force('Force_Air')
    
    HSig_DOFs=HSig("HSig_DOFs")
    HSig_Bat=HSig("HSig_Bat")
    
    RSig_DOFs=RSig("RSig_DOFs")
    RSig_Bat=RSig("RSig_Bat")
    RSig_Ctl=RSig("RSig_Ctl")
    RSig_Traj=RSig("RSig_Traj")
    
    EE_1=EE('EE_1')
    EEmot=EE('EEmot')
    EEctl=EE('EEctl')
    
    Ctl1=Sig('Ctl1')
    DOFs=DOF('DOFs')
    Dir1=Direc('Dir1')
    Env1=Env('Env1')
    
    ManageHealth=manageHealth(EEctl, Force_ST, HSig_DOFs, HSig_Bat, RSig_DOFs, RSig_Bat, RSig_Ctl, RSig_Traj)
    g.add_node('ManageHealth', obj=ManageHealth)
    
    StoreEE=storeEE('StoreEE',EE_1, Force_ST, HSig_Bat, RSig_Bat)
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
    
    Land1=Land('Land')
    Force_GR=Force('Force_GR')
    Force_LG=Force('Force_LG')
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

#def environment(DOF,t):
#    if DOF.stab
    
def findclassification(g, endfaults, endflows, scen):
    
    Env=fp.getflow('Env1', g)
    
    #may need to redo this
    if  aux.inrange(Env.start_area, Env.x, Env.y):
        landloc='nominal'
        area=1
    elif aux.inrange(Env.safe1_area, Env.x, Env.y) or aux.inrange(Env.safe2_area, Env.x, Env.y):
        landloc='emsafe'
        area=1000
    elif aux.inrange(Env.dang_area, Env.x, Env.y):
        landloc='emdang'
        area=100000
    else:
        landloc='emunsanc'
        area=10000
        
    repaircosts=fp.listfaultsprops(endfaults, g, 'rcost')
    maxcost=aux.textmax(repaircosts.values())
    
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