import numpy as np

from fmdtools.modeldef.common import Parameter, State
from fmdtools.modeldef.block import FxnBlock, Mode
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.flow import Flow

class StoreEEMode(Mode):
    failrate=1e-5
    faultparams = {'nocharge':(1,300)}
class StoreEEState(State):
    soc:    float = 100.0
class StoreEE(FxnBlock):
    _init_s = StoreEEState
    _init_m = StoreEEMode
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEout', 'FS'])
    def behavior(self, time):
        if self.m.has_fault('nocharge'):    self.EEout.s.effort=0.0
        else:                               self.EEout.s.effort=1.0
class DistEEMode(Mode):
    failrate=1e-5
    faultparams = {'short':(0.3,3000), 
                   'degr':(0.5,1000), 
                   'break':(0.2,2000)}
class DistEEState(State):
    EEtr: float=1.0
    EEte: float=1.0
class DistEE(FxnBlock):
    _init_s = DistEEState
    _init_m = DistEEMode
    def __init__(self,name, flows):
        super().__init__(name,flows, ['EEin','EEmot','EEctl','ST'])
    def condfaults(self, time):
        if self.ST.s.support<0.5 or max(self.EEmot.s.rate,self.EEctl.s.rate)>2: 
            self.m.add_fault('break')
        if self.EEin.s.rate>2:
            self.m.add_fault('short')
    def behavior(self, time):
        if self.m.has_fault('short'): 
            self.s.EEtr=0.0
            self.s.EEte=10.0
        elif self.m.has_fault('break'): 
            self.s.EEtr=0.0
            self.s.EEte=0.0
        elif self.m.has_fault('degr'): 
            self.s.EEte=0.5
        self.EEmot.s.effort=self.s.EEte*self.EEin.s.effort
        self.EEctl.s.effort=self.s.EEte*self.EEin.s.effort
        self.EEin.s.rate=m2to1([self.EEin.s.effort, self.s.EEtr, 0.9*self.EEmot.s.rate+0.1*self.EEctl.s.rate])
class EngageLandMode(Mode):
    failrate=1e-5
    faultparams = {'break':(0.2, 1000), 
                   'deform':(0.8, 1000)}
class EngageLand(FxnBlock):
    _init_m=EngageLandMode
    def __init__(self,name, flows):
        super().__init__(name,flows, ['forcein', 'forceout'])
    def condfaults(self, time):
        if abs(self.forcein.s.support)>=2.0:     self.m.add_fault('break')
        elif abs(self.forcein.s.support)>1.5:    self.m.add_fault('deform')
    def behavior(self, time):
        self.forceout.s.support=self.forcein.s.support/2

class HoldPayloadMode(Mode):
    failrate=1e-6
    faultparams = {'break':(0.2, 10000), 
                   'deform':(0.8, 10000)}
class HoldPayload(FxnBlock):
    _init_m = HoldPayloadMode
    def __init__(self,name, flows):
        super().__init__(name,flows, ['FG', 'Lin', 'ST'])
    def condfaults(self, time):
        if abs(self.FG.s.support)>0.8:      self.m.add_fault('break')
        elif abs(self.FG.s.support)>1.0:    self.m.add_fault('deform')
    def behavior(self, time):
        #need to transfer FG to FA & FS???
        if self.m.has_fault('break'):       self.Lin.s.support, self.ST.s.support = 0,0
        elif self.m.has_fault('deform'):    self.Lin.s.support, self.ST.s.support = 0.5,0.5
        else:                               self.Lin.s.support, self.ST.s.support = 1.0,1.0

class AffectDOFState(State):
    Eto:    float = 1.0
    Eti:    float = 1.0
    Ct:     float = 1.0
    Mt:     float = 1.0
    Pt:     float = 1.0
class AffectDOFMode(Mode):
    failrate=1e-5
    faultparams = {'short':         (0.1, 200),
                   'openc':         (0.1, 200),
                   'ctlup':         (0.2, 500),
                   'ctldn':         (0.2, 500),
                   'ctlbreak':      (0.2, 1000), 
                   'mechbreak':     (0.1, 500), 
                   'mechfriction':  (0.05, 500),
                   'propwarp':      (0.01, 200),
                   'propstuck':     (0.02, 200), 
                   'propbreak':     (0.03, 200)}
class AffectDOF(FxnBlock): #EEmot,Ctl1,DOFs,Force_Lin HSig_DOFs, RSig_DOFs
    _init_s = AffectDOFState
    _init_m = AffectDOFMode
    def __init__(self,name, flows):     
        super().__init__(name, flows, ['EEin', 'Ctlin','DOF','Force'])
    def behavior(self, time):
        self.s.put(Eti=1.0, Eto=1.0)
        if self.m.has_fault('short'):           self.s.put(Eti=10, Eto=0.0)
        elif self.m.has_fault('openc'):         self.s.put(Eti=0.0, Eto=0.0)
        if self.m.has_fault('ctlbreak'):        self.s.Ct=0.0
        elif self.m.has_fault('ctldn'):         self.s.Ct=0.5
        elif self.m.has_fault('ctlup'):         self.s.Ct=2.0
        if self.m.has_fault('mechbreak'):       self.s.Mt=0.0
        elif self.m.has_fault('mechfriction'):  self.s.put(Mt=0.5, Eti=2.0)
        if self.m.has_fault('propstuck'):       self.s.put(Pt=0.0, Mt=0.0, Eti=4.0)
        elif self.m.has_fault('propbreak'):     self.s.Pt=0.0
        elif self.m.has_fault('propwarp'):      self.s.Pt=0.5
        
        self.EEin.s.rate=self.s.Eti

        self.DOF.uppwr=self.Ctlin.s.upward*self.s.mul('Eto','Eti','Ct','Mt','Pt')
        self.DOF.planpwr=self.Ctlin.s.forward*self.s.mul('Eto','Eti','Ct','Mt','Pt')  

class CtlDOFState(State):
    Cs: float = 1.0
class CtlDOFMode(Mode):
    failrate=1e-5
    faultparams={'noctl':   (0.2, 10000), 
                 'degctl':  (0.8, 10000)}
class CtlDOF(FxnBlock):
    _init_s = CtlDOFState
    _init_m = CtlDOFMode
    def __init__(self,name,flows):
        super().__init__(name,flows, ['EEin','Dir','Ctl','DOFs','FS'])
    def condfaults(self, time):
        if self.FS.s.support<0.5: self.m.add_fault('noctl')
    def behavior(self, time):
        if self.m.has_fault('noctl'):    self.s.Cs=0.0
        elif self.m.has_fault('degctl'): self.s.Cs=0.5
        
        upthrottle=1.0
        if self.Dir.s.z>1:        upthrottle=2
        elif -1<self.Dir.s.z<1:   upthrottle= 1 + self.Dir.s.z
        elif self.Dir.s.z<=-1.0:  upthrottle = 0
            
        if self.Dir.s.x==0 and self.Dir.s.y==0: forwardthrottle=0.0
        else: forwardthrottle=1.0
        
        self.Ctl.s.forward=self.EEin.s.effort*self.s.Cs*forwardthrottle*self.Dir.s.power
        self.Ctl.s.upward=self.EEin.s.effort*self.s.Cs*self.Dir.s.power*upthrottle

class PlanPathMode(Mode):
    failrate=1e-5
    faultparams = {'noloc': (0.2, 10000),
                   'degloc':(0.8, 10000)}
class PlanPath(FxnBlock):
    _init_m = PlanPathMode
    def __init__(self,name, flows):
        super().__init__(name, flows, ['EEin','Env','Dir','FS'])
    def condfaults(self, time):
        if self.FS.s.support<0.5: self.m.add_fault('noloc')
    def behavior(self, t):
        self.Dir.s.assign([1.0,0.0,0.0], "x","y","z")
        # faulty behaviors    
        if self.m.has_fault('noloc'):     self.Dir.s.assign([0,0,0], "x", "y", "z")
        elif self.m.has_fault('degloc'):  self.Dir.s.assign([0,0,-1], "x", "y", "z")
        if self.EEin.s.effort<0.5:
            self.Dir.s.assign([0.0,0.0,0.0,0.0],'x','y','z','power')  

class TrajectoryMode(Mode):
    faultparams = {'crash':(0, 100000), 
                  'lost':(0.0, 50000)}
class Trajectory(FxnBlock):
    _init_m = TrajectoryMode
    def __init__(self,name, flows):
        super().__init__(name, flows, ['Env','DOF', 'Dir', 'Force_GR'])
    def behavior(self, time):
        self.DOF.vertvel = max(min(-2+2*self.DOF.uppwr, 2), -2)
        self.Force_GR.s.support =self.DOF.vertvel
        self.DOF.planvel=self.DOF.planpwr
        if self.DOF.vertvel>1.5 or self.DOF.vertvel<-1:
            self.m.add_fault('crash')
            self.Env.s.z=0.0
        if self.DOF.planvel>1.5 or self.DOF.planvel<0.5:
            self.m.add_fault('lost')
            self.Env.s.x=0.0
        else:
            self.Env.s.x=1.0

def m2to1(x):
    """
    Multiplies a list of numbers which may take on the values infinity or zero. In deciding if num is inf or zero, the earlier values take precedence

    Parameters
    ----------
    x : list 
        numbers to multiply

    Returns
    -------
    y : float
        result of multiplication
    """
    if np.size(x)>2:    x=[x[0], m2to1(x[1:])]
    if x[0]==np.inf:    y=np.inf
    elif x[1]==np.inf:
        if x[0]==0.0:   y=0.0
        else:           y=np.inf
    else:               y=x[0]*x[1]
    return y

class ViewModes(Mode):
    failrate=1e-5
    faultparams = {'poorview':(0.2, 10000)}
class ViewEnvironment(FxnBlock):
    _init_m = ViewModes
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env'])

class EEState(State):
    rate:   float=1.0
    effort: float=1.0
class EE(Flow):
    _init_s = EEState

class ForceState(State):
    support: float=1.0
class Force(Flow):
    _init_s = ForceState

class ControlState(State):
    forward:    float=1.0
    upward:     float=1.0
class Control(Flow):
    _init_s = ControlState

class DOFState(State):
    vertvel:    float=1.0
    planvel:    float=1.0
    planpwr:    float=1.0
    uppwr:      float=1.0
class DOFs(Flow):
    _init_s = DOFState
    
class EnvState(State):
    x:          float=0.0
    y:          float=0.0
    z:          float=50.0
class Env(Flow):
    _init_s = EnvState

class DirState(State):
    x:          float=1.0
    y:          float=0.0
    z:          float=0.0
    power:      float=1.0
class Dir(Flow):
    _init_s = DirState
        
class Drone(Model):
    def __init__(self, params=Parameter(), modelparams=ModelParam(), valparams={}):
        super().__init__(params, modelparams, valparams)
        self.params=params
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
        self.add_flow('Env1',       Env)
        self.add_flow('Dir1',       Dir)
        #add functions to the model
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
    def find_classification(self,scen, mdlhist):
        modes, modeprops = self.return_faultmodes()
        repcost=sum([ c['rcost'] for f,m in modeprops.items() for a, c in m.items()])
        
        totcost=repcost
        rate=scen['properties']['rate']
        expcost=totcost*rate*1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}
    
if __name__=="__main__":
    from fmdtools.faultsim import propagate
    static_mdl = Drone()
    endclasses, mdlhists = propagate.single_faults(static_mdl)