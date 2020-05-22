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
        self.x, self.y, self.z = traj[0], traj[1], traj[2]
        self.traj=traj
    def status(self):
        status={'x': self.traj[0], 'y': self.traj[1], 'z': self.traj[2], 'power': self.power}
        return status.copy()

#Define functions
class StoreEE(FxnBlock):
    def __init__(self, flows, params):
        self.archtype=params['bat']
        
        #weight, cap, voltage, drag_factor
        
        if self.archtype == 'monolithic':
            self.batparams ={'s':1,'p':1,'w':params['weight'],'v':12,'d':params['drag']}
            components = {'S1P1': Battery('S1P1', self.batparams)}
        elif self.archtype =='series-split':
            self.batparams ={'s':2,'p':1,'w':params['weight'],'v':12,'d':params['drag']}
            components = {'S1P1': Battery('S1P1', self.batparams), 'S2P1': Battery('S2P1', self.batparams)}
        elif self.archtype == 'parallel-split':
            self.batparams ={'s':1,'p':2,'w':params['weight'],'v':12,'d':params['drag']}
            components = {'S1P1': Battery('S1P1', self.batparams),'S1P2': Battery('S1P2', self.batparams)}
        elif self.archtype == 'split-both':
            self.batparams ={'s':2,'p':2,'w':params['weight'],'v':12,'d':params['drag']}
            components = {'S1P1': Battery('S1', self.batparams), 'S2P1': Battery('S2', self.batparams),'S1P2': Battery('S1P2', self.batparams), 'S2P2': Battery('S2P2', self.batparams)}
        else: raise Exception("Invalid battery architecture")
        #failrate for function w- component only applies to function modes
        self.failrate=1e-3
        self.assoc_modes({'nocharge':[0.2,[0.6,0.2,0.2],300],'lowcharge':[0.7,[0.6,0.2,0.2],200]})
        super().__init__(['EEout', 'FS', 'HSig'], flows, {'soc': 100}, components)
    def condfaults(self, time):
        if self.soc<20:                     self.add_fault('lowcharge')
        if self.soc<1:                      self.replace_fault('lowcharge','nocharge')
    def behavior(self, time):
        EE, soc = {}, {}
        rate_res=0
        for batname, bat in self.components.items():
            EE[bat.name], soc[bat.name], rate_res = bat.behavior(self.FS.support, self.EEout.rate/(self.batparams['s']*self.batparams['p'])+rate_res, time)
        #need to incorporate max current draw somehow + draw when reconfigured
        if self.archtype == 'monolithic':           self.EEout.effort = EE['S1P1']
        elif self.archtype == 'series-split':       self.EEout.effort = np.max(list(EE.values()))
        elif self.archtype == 'parallel-split':     self.EEout.effort = np.sum(list(EE.values()))
        elif self.archtype == 'split-both':          
            e=list(EE.values())
            e.sort()
            self.EEout.effort = e[-1]+e[-2]  
        self.soc=np.mean(list(soc.values()))
        if self.any_faults():     self.HSig.hstate = 'faulty'
        else:                   self.HSig.hstate = 'nominal'

class Battery(Component):
    def __init__(self, name, batparams):
        super().__init__(name, {'soc':100, 'EEe':1.0, 'Et':1.0})
        self.failrate=1e-3
        self.avail_eff = 1/batparams['p']
        self.maxa = 2/batparams['s']
        self.p=batparams['p']
        self.s=batparams['s']
        self.amt = 60*4.200/(batparams['w']*170/(batparams['d']*batparams['v']))
        self.assoc_modes({'short':[0.02,[0.3,0.3,0.3],2000], 'degr':[0.06,[0.3,0.3,0.3],2000],
                          'break':[0.02,[0.2,0.2,0.2],2000], 'nocharge':[0.2,[0.6,0.2,0.2],300],
                          'lowcharge':[0.7,[0.6,0.2,0.2],200]}, name=name)
    def behavior(self, FS, EEoutr, time):
        if FS <1.0:     self.add_fault(self.name+'break')
        if EEoutr>self.maxa:    self.add_fault(self.name+'break')
        if self.soc<20: self.add_fault(self.name+'lowcharge')
        if self.soc<1:  self.replace_fault(self.name+'lowcharge',self.name+'nocharge')
        Et=1.0 #default
        if self.has_fault(self.name+'short'):       Et=0.0
        elif self.has_fault(self.name+'break'):     Et=0.0
        elif self.has_fault(self.name+'degr'):      Et=0.5
        self.Et = Et*self.avail_eff
        Er_res=0.0
        if time > self.time:
            self.soc=self.soc-100*EEoutr*self.p*self.s*(time-self.time)/self.amt
            self.time=time
        if self.has_fault(self.name+'nocharge'):    self.soc, self.Et, Er_res = 0.0,0.0, EEoutr
        return self.Et, self.soc, Er_res

class DistEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEmot','EEctl','ST'],flows, {'EEtr':1.0, 'EEte':1.0}, timely=False)
        self.failrate=1e-5
        self.assoc_modes({'short':[0.3,[0.33, 0.33, 0.33],3000], 'degr':[0.5,[0.33, 0.33, 0.33],1000],\
                          'break':[0.2,[0.33, 0.33, 0.33],2000]})
    def condfaults(self, time):
        if self.ST.support<0.5 or max(self.EEmot.rate,self.EEctl.rate)>2:   self.add_fault('break')
        if self.EEin.rate>2:                                                self.add_fault('short')
    def behavior(self, time):
        if self.has_fault('short'):                                         self.EEte, self.EEre = 0.0,10.0
        elif self.has_fault('break'):                                       self.EEte, self.EEre = 0.0,0.0
        elif self.has_fault('degr'):                                        self.EEte=0.5
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
    def __init__(self,flows,respolicy):
        self.respolicy = respolicy
        flownames=['EECtl','FS','DOFshealth', 'Bathealth', 'Trajconfig' ]
        super().__init__(flownames, flows)
        
        self.failrate=1e-5
        self.assoc_modes({'falsemaintenance':[0.8,[1.0, 0.0,0.0,0.0,0.0],10000],\
                         'falsemasking':[0.1,[1.0, 0.2,0.4,0.4,0.0],10000],\
                         'falseemland':[0.05,[0.0, 0.2,0.4,0.4,0.0],10000],\
                         'lostfunction':[0.05,[0.2, 0.2,0.2,0.2,0.2],10000]})
    def condfaults(self, time):
        if self.FS.support<0.5 or self.EECtl.effort>2.0: self.add_fault('lostfunction')
    def behavior(self, time):
        if self.has_fault('lostfunction'):      self.Trajconfig.mode = 'continue'
        elif  self.DOFshealth.hstate=='faulty': self.Trajconfig.mode = self.respolicy['line']
        elif  self.Bathealth.hstate=='faulty':   self.Trajconfig.mode = self.respolicy['bat']
        else:                                   self.Trajconfig.mode = 'continue'
            # trajconfig: continue, to_home, to_nearest, emland
    
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    def __init__(self, flows, archtype):     
        self.archtype=archtype
        if archtype=='quad':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5}
            self.LR = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR = {'F':{'LF', 'RF'}, 'R':{'LR', 'RR'}}
        elif archtype=='hex':
            components={'RF':Line('RF'), 'LF':Line('LF'), 'LR':Line('LR'), 'RR':Line('RR'),'R':Line('R'), 'F':Line('F')}
            self.upward={'RF':1,'LF':1,'LR':1,'RR':1,'R':1,'F':1}
            self.forward={'RF':0.5,'LF':0.5,'LR':-0.5,'RR':-0.5, 'R':-0.75, 'F':0.75}
            self.LR = {'L':{'LF', 'LR'}, 'R':{'RF','RR'}}
            self.FR = {'F':{'LF', 'RF', 'F'}, 'R':{'LR', 'RR', 'R'}}
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
        
        if abs(self.LRstab) >=0.4 or abs(self.FRstab)>=0.75:    self.DOF.uppwr, self.DOF.planpwr = 0.0, 0.0
        else:
            Airs=list(Air.values())
            self.DOF.uppwr=np.mean(Airs)
            self.DOF.planpwr=self.Ctlin.forward
        if time> self.time:
            if self.DOF.uppwr > 1:      self.DOF.vertvel = 60*min([(self.DOF.uppwr-1)*5, 5])
            elif self.DOF.uppwr < 1:    self.DOF.vertvel = 60*max([(self.DOF.uppwr-1)*50, -50])
            else:                       self.DOF.vertvel = 0.0
                
            self.DOF.planvel=60*min([10*self.DOF.planpwr, 10]) # 600 m/m = 23 mph 
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
        self.failrate=1e-5
        self.assoc_modes({'short':[0.1, [0.33, 0.33, 0.33], 200],'openc':[0.1, [0.33, 0.33, 0.33], 200],\
                          'ctlbreak':[0.2, [0.33, 0.33, 0.33], 1000], 'mechbreak':[0.1, [0.33, 0.33, 0.33], 500],\
                          'mechfriction':[0.05, [0.0, 0.5,0.5], 500], 'stuck':[0.02, [0.0, 0.5,0.5], 200]},name=name)
    def behavior(self, EEin, Ctlin, cmds, Force):
        if Force<=0.0:   self.add_fault(self.name+'mechbreak')
        elif Force<=0.5: self.add_fault(self.name+'mechfriction')
            
        if self.has_fault(self.name+'short'):                   self.Eti, self.Eto = 0.0, np.inf
        elif self.has_fault(self.name+'openc'):                 self.Eti, self.Eto =0.0, 0.0
        elif Ctlin.upward==0 and Ctlin.forward == 0:            self.Eto = 0.0
        if self.has_fault(self.name+'ctlbreak'):                self.Ct=0.0
        if self.has_fault(self.name+'mechbreak'):               self.Mt=0.0
        elif self.has_fault(self.name+'mechfriction'):          self.Mt, self.Eti = 0.5, 2.0
        if self.has_fault(self.name+'stuck'):                   self.Pt, self.Mt, self.Eti = 0.0, 0.0, 4.0
        
        self.Airout=m2to1([EEin,self.Eti,Ctlin.upward*cmds['up']+Ctlin.forward*cmds['for'],self.Ct,self.Mt,self.Pt])
        self.EE_in=m2to1([EEin,self.Eto])  
    
class CtlDOF(FxnBlock):
    def __init__(self, flows):
        super().__init__(['EEin','Dir','Ctl','DOFs','FS'],flows, {'vel':0.0, 'Cs':1.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, [0.6, 0.3, 0.1], 10000], 'degctl':[0.8, [0.6, 0.3, 0.1], 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noctl')
    def behavior(self, time):
        if self.has_fault('noctl'):    self.Cs=0.0
        elif self.has_fault('degctl'): self.Cs=0.5
        if time>self.time: self.vel=self.DOFs.vertvel
        # throttle settings: 0 is off (-50 m/s), 1 is hover, 2 is max climb (5 m/s)
        if self.Dir.traj[2]>0:          upthrottle = 1+np.min([self.Dir.traj[2]/(50*5), 1])
        elif self.Dir.traj[2] <0:       upthrottle = 1+np.max([self.Dir.traj[2]/(50*50), -1])
        else:                           upthrottle = 1.0
        
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
        
        if self.mode=='taxi' and t>5:   self.mode=='taxi'
        elif self.Rsig.mode == 'to_home': # add a to_nearest option
            self.pt = 0
            self.goal =self.goals[self.pt]
        elif self.Rsig.mode== 'emland': self.mode = 'land'
        elif self.Env.elev<1 and (self.pt>=max(self.goals) or self.mode=='land'):  self.mode = 'taxi'
        elif dist<5 and self.pt>=max(self.goals):           self.mode = 'land'
        elif dist<5 and {'move'}.issuperset({self.mode}):
            if self.pt < max(self.goals):   
                self.pt+=1
                self.goal = self.goals[self.pt]
        elif dist>5 and not(self.mode=='descend'):          self.mode='move'
        # nominal behaviors
        self.Dir.power=1.0
        if self.mode=='taxi':       self.Dir.power=0.0          
        elif self.mode=='move':     self.Dir.assign([self.dx,self.dy, self.dz])     
        elif self.mode=='land':     self.Dir.assign([0,0,-self.Env.elev/2])
        # faulty behaviors    
        if self.has_fault('noloc'):     self.Dir.assign([0,0,0])
        elif self.has_fault('degloc'):  self.Dir.assign([0,0,-1])
        if self.EEin.effort<0.5:
            self.Dir.power=0.0
            self.Dir.assign([0,0,0])            
            
class Drone(Model):
    def __init__(self, params={'flightplan':{1:[0,0,100], 2:[100, 0,100], 3:[100, 100,100], 4:[150, 150,100], 5:[0,0,100], 6:[0,0,0]},'bat':'monolithic', 'linearch':'quad','respolicy':{'bat':'to_home','line':'emland'}, 'start': [0.0,0.0, 10, 10], 'target': [0, 150, 160, 160], 'safe': [0, 50, 10, 10]}):
        super().__init__()
        super().__init__(modelparams={'phases': {'ascend':[0,1],'forward':[1,19],'descend':[19, 20]},
                                     'times':[0,30],'units':'min'}, params=params)
        
        self.start_area = square(self.params['start'][0:2],self.params['start'][2],self.params['start'][3] )
        self.safe_area = square(self.params['safe'][0:2],self.params['safe'][2],self.params['safe'][3] )
        self.target_area = square(self.params['target'][0:2],self.params['target'][2],self.params['target'][3] )
        
        #add flows to the model
        self.add_flow('Force_ST', {'support':1.0})
        self.add_flow('Force_Lin', {'support':1.0} )
        self.add_flow('HSig_DOFs', {'hstate':'nominal', 'config':1.0})
        self.add_flow('HSig_Bat', {'hstate':'nominal', 'config':1.0} )
        self.add_flow('RSig_Traj', {'mode':'continue'})
        self.add_flow('EE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEmot', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEctl', {'rate':1.0, 'effort':1.0})
        self.add_flow('Ctl1', {'forward':0.0, 'upward':1.0})
        self.add_flow('DOFs', {'vertvel':0.0, 'planvel':0.0, 'planpwr':0.0, 'uppwr':0.0, 'x':0.0,'y':0.0,'elev':0.0})
        # custom flows
        self.add_flow('Dir1', Direc())
        #add functions to the model
        flows=['EEctl', 'Force_ST', 'HSig_DOFs', 'HSig_Bat', 'RSig_Traj']
        # trajconfig: continue, to_home, to_nearest, emland
        self.add_fxn('ManageHealth',flows,fclass = ManageHealth, fparams=params['respolicy'])
        batweight = {'monolithic':0.4, 'series-split':0.5, 'parallel-split':0.5, 'split-both':0.6}[params['bat']]
        archweight = {'quad':1.2, 'hex':1.6, 'oct':2.0}[params['linearch']]
        archdrag = {'quad':0.95, 'hex':0.85, 'oct':0.75}[params['linearch']]
        self.add_fxn('StoreEE',['EE_1', 'Force_ST', 'HSig_Bat'],fclass = StoreEE, fparams= {'bat': params['bat'], 'weight':(batweight+archweight)/2.2 , 'drag': archdrag })
        self.add_fxn('DistEE', ['EE_1','EEmot','EEctl', 'Force_ST'], fclass = DistEE)
        self.add_fxn('AffectDOF',['EEmot','Ctl1','DOFs','Dir1','Force_Lin', 'HSig_DOFs'], fclass=AffectDOF, fparams = params['linearch'])
        self.add_fxn('CtlDOF',['EEctl', 'Dir1', 'Ctl1', 'DOFs', 'Force_ST'], fclass = CtlDOF)
        self.add_fxn('Planpath', ['EEctl', 'DOFs','Dir1', 'Force_ST', 'RSig_Traj'], fclass=PlanPath, fparams=params)
        self.add_fxn('HoldPayload',['DOFs', 'Force_Lin', 'Force_ST'], fclass = HoldPayload)
        
        pos = {'ManageHealth': [0.23793980988102348, 1.0551602632416588],
               'StoreEE': [-0.9665780995752296, -0.4931538151692423],
               'DistEE': [-0.1858834234148632, -0.20479989209711924],
               'AffectDOF': [1.0334916329507422, 0.6317263653616103],
               'CtlDOF': [0.1835014208949617, 0.32084893189175423],
               'Planpath': [-0.7427736219526058, 0.8569475547950892],
               'HoldPayload': [0.74072970715511, -0.7305391093272489]}
        
        bippos = {'ManageHealth': [-0.23403572483176666, 0.8119063670455383],
                  'StoreEE': [-0.7099736148158298, 0.2981652748232978],
                  'DistEE': [-0.28748133634190726, 0.32563569654296287],
                  'AffectDOF': [0.9473412427515959, 0.0466423266443633],
                  'CtlDOF': [0.498663257339388, 0.44284186573420836],
                  'Planpath': [0.5353654708147643, 0.7413936186204868],
                  'HoldPayload': [0.329334798653681, -0.17443414674339652],
                  'Force_ST': [-0.2364754675127569, -0.18801548176633154],
                  'Force_Lin': [0.7206415618571647, -0.17552020772024013],
                  'HSig_DOFs': [0.3209028709788254, 0.04984245810974697],
                  'HSig_Bat': [-0.6358884586093769, 0.7311076416371343],
                  'RSig_Traj': [0.18430501738656657, 0.856472541655958],
                  'EE_1': [-0.48288657418004555, 0.3017533207866233],
                  'EEmot': [-0.0330582435936827, 0.2878069006385988],
                  'EEctl': [0.13195069534343862, 0.4818116953414546],
                  'Ctl1': [0.5682663453757308, 0.23385244312813386],
                  'DOFs': [0.8194232270836169, 0.3883256382522293],
                  'Dir1': [0.9276094920710914, 0.6064107724557304]}
        
        self.construct_graph(graph_pos=pos, bipartite_pos=bippos)
        
    def find_classification(self, g, endfaults, endflows, scen, mdlhist):
        #landing costs
        viewed = env_viewed(mdlhist['faulty']['flows']['DOFs']['x'], mdlhist['faulty']['flows']['DOFs']['y'],mdlhist['faulty']['flows']['DOFs']['elev'], self.target_area)
        viewed_value = 10*sum([view for k,view in viewed.items() if view!='unviewed'])
        
        Env=self.flows['DOFs']
        if  inrange(self.start_area, Env.x, Env.y): landcost = 0 # nominal landing
        elif inrange(self.safe_area, Env.x, Env.y): landcost=1000 # emergency safe
        elif inrange(self.target_area, Env.x, Env.y):  landcost=100000 # emergency dangerous
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
def env_viewed(xhist, yhist,zhist, square):
    viewed = {(x,y):'unviewed' for x in range(int(square[0][0]),int(square[1][0])+10,10) for y in range(int(square[0][1]),int(square[2][1])+10,10)}
    for i,x in enumerate(xhist[1:len(xhist)]):
        w,h,d = viewable_area(zhist[i+1])
        viewed_area = rect(xhist[i],yhist[i],xhist[i+1],yhist[i+1], w,h)
        polygon=Polygon(viewed_area)
        #plt.plot(*polygon.exterior.xy) (displays area to debug code)
        #plt.plot([xhist[i],xhist[i+1]],[yhist[i],yhist[i+1]])
        if not polygon.is_valid:    print('invalid points')
        for spot in viewed:
            if polygon.contains(Point(spot)): 
                viewed[spot]=d
    return viewed

def viewable_area(elev):
    width = elev
    height = elev * 0.75 # 4/3 camera with ~45 mm lens st dist = width
    detail = 1/(width*height+0.00001)
    return width, height, detail

def plan_flight(elev, square, landing):
    
    flightplan = {0:landing, 1: landing[0:2]+[elev]}
    
    width, height, detail = viewable_area(elev)
    # x,y, elev
    startpt = [square[0][0]+width/2, square[0][1]+height/2, elev]
    endpt = [square[1][0]-width/2, square[1][1]+height/2, elev]
    
    num_rows = int(np.ceil((square[2][1]-square[0][1])/width))
    
    leftpts = [[startpt[0] , startpt[1]+ r*width] for r in range(num_rows)]
    rightpts = [[endpt[0], endpt[1]+ r*width] for r in range(num_rows)]
    leftpts.sort(reverse=True)
    rightpts.sort(reverse=True)
    
    addpt = max(flightplan) + 1
    numpts = 2*len(leftpts)
    
    vec1 = leftpts
    vec2 = rightpts
    vec=[]
    n=2
    newplan = {}
    while len(vec1+vec2+vec)>0:
        newplan[n]=vec1.pop()+[elev]
        n+=1
        if len(vec1)< len(vec2) or n==0:
            vec = vec2
            vec2 = vec1
            vec1 = vec
    
    flightplan.update(newplan)
    flightplan.update({max(flightplan)+1:flightplan[1], max(flightplan)+2:flightplan[0]})
    return flightplan


