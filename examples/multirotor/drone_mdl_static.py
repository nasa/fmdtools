import numpy as np

from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.block import FxnBlock
from fmdtools.define.mode import Mode
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow


## MODEL FLOWS
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

class DOFstate(State):
    vertvel:    float=1.0
    planvel:    float=1.0
    planpwr:    float=1.0
    uppwr:      float=1.0
class DOFs(Flow):
    _init_s = DOFstate
    
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
    
## MODEL FUNCTIONS

class StoreEEMode(Mode):
    failrate=1e-5
    faultparams = {'nocharge':(1,300)}
class StoreEEState(State):
    soc:    float = 100.0
class StoreEE(FxnBlock):
    __slots__=('ee_out', 'fs')
    _init_s = StoreEEState
    _init_m = StoreEEMode
    _init_ee_out = EE
    _init_fs = Force
    flownames = {"ee_1":'ee_out', 'force_st':'fs'}
    def behavior(self, time):
        if self.m.has_fault('nocharge'):    self.ee_out.s.effort=0.0
        else:                               self.ee_out.s.effort=1.0
class DistEEMode(Mode):
    failrate=1e-5
    faultparams = {'short':(0.3,3000), 
                   'degr':(0.5,1000), 
                   'break':(0.2,2000)}
class DistEEState(State):
    ee_tr: float=1.0
    ee_te: float=1.0
class DistEE(FxnBlock):
    __slots__=('ee_in', 'ee_mot', 'ee_ctl', 'st')
    _init_s = DistEEState
    _init_m = DistEEMode
    _init_ee_in = EE
    _init_ee_mot = EE
    _init_ee_ctl = EE
    _init_st = Force
    flownames = {"ee_1":"ee_in", "force_st":'st'}
    def condfaults(self, time):
        if self.st.s.support<0.5 or max(self.ee_mot.s.rate,self.ee_ctl.s.rate)>2: 
            self.m.add_fault('break')
        if self.ee_in.s.rate>2:
            self.m.add_fault('short')
    def behavior(self, time):
        if self.m.has_fault('short'):       self.s.put(ee_tr=0.0, ee_te=10.0)
        elif self.m.has_fault('break'):     self.s.put(ee_tr=0.0, ee_te=0.0)
        elif self.m.has_fault('degr'):      self.s.put(ee_te=0.5)
        self.ee_mot.s.effort=self.s.ee_te*self.ee_in.s.effort
        self.ee_ctl.s.effort=self.s.ee_te*self.ee_in.s.effort
        self.ee_in.s.rate=m2to1([self.ee_in.s.effort, self.s.ee_tr, 0.9*self.ee_mot.s.rate+0.1*self.ee_ctl.s.rate])
class EngageLandMode(Mode):
    failrate=1e-5
    faultparams = {'break':(0.2, 1000), 
                   'deform':(0.8, 1000)}
class EngageLand(FxnBlock):
    __slots__=('force_in', 'force_out')
    _init_m=EngageLandMode
    _init_force_in = Force
    _init_force_out = Force 
    flownames = {'force_gr':'force_in', 'force_lg':'force_out'}
    def condfaults(self, time):
        if abs(self.force_in.s.support)>=2.0:     self.m.add_fault('break')
        elif abs(self.force_in.s.support)>1.5:    self.m.add_fault('deform')
    def behavior(self, time):
        self.force_out.s.support=self.force_in.s.support/2

class HoldPayloadMode(Mode):
    failrate=1e-6
    faultparams = {'break':(0.2, 10000), 
                   'deform':(0.8, 10000)}
class HoldPayload(FxnBlock):
    __slots__=('force_lg', 'force_lin', 'force_st')
    _init_m = HoldPayloadMode
    _init_force_lg = Force 
    _init_force_lin = Force 
    _init_force_st = Force 
    def condfaults(self, time):
        if abs(self.force_lg.s.support)>0.8:      self.m.add_fault('break')
        elif abs(self.force_lg.s.support)>1.0:    self.m.add_fault('deform')
    def behavior(self, time):
        #need to transfer FG to FA & FS???
        if self.m.has_fault('break'):       self.force_st.s.support = 0.0
        elif self.m.has_fault('deform'):    self.force_st.s.support = 0.5
        else:                               self.force_st.s.support = 1.0
        self.force_lin.s.assign(self.force_st.s, 'support')

class AffectDOFState(State):
    e_to:    float = 1.0
    e_ti:    float = 1.0
    ct:     float = 1.0
    mt:     float = 1.0
    pt:     float = 1.0
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
class AffectDOF(FxnBlock): #ee_mot,ctl,dofs,force_lin HSig_dofs, RSig_dofs
    __slots__=('ee_in', 'ctl_in', 'dofs', 'force')
    _init_s = AffectDOFState
    _init_m = AffectDOFMode
    _init_ee_in = EE
    _init_ctl_in = Control
    _init_dofs = DOFs
    _init_force = Force
    flownames = {'ee_mot':'ee_in', 'ctl':'ctl_in','force_lin':'force'}
    def behavior(self, time):
        self.s.put(e_ti=1.0, e_to=1.0)
        if self.m.has_fault('short'):           self.s.put(e_ti=10, e_to=0.0)
        elif self.m.has_fault('openc'):         self.s.put(e_ti=0.0,e_to=0.0)
        if self.m.has_fault('ctlbreak'):        self.s.ct=0.0
        elif self.m.has_fault('ctldn'):         self.s.ct=0.5
        elif self.m.has_fault('ctlup'):         self.s.ct=2.0
        if self.m.has_fault('mechbreak'):       self.s.mt=0.0
        elif self.m.has_fault('mechfriction'):  self.s.put(mt=0.5, e_ti=2.0)
        if self.m.has_fault('propstuck'):       self.s.put(pt=0.0, mt=0.0, e_ti=4.0)
        elif self.m.has_fault('propbreak'):     self.s.pt=0.0
        elif self.m.has_fault('propwarp'):      self.s.pt=0.5
        
        self.ee_in.s.rate=self.s.e_ti
        pwr = self.s.mul('e_to','e_ti','ct','mt','pt')
        self.dofs.s.uppwr=self.ctl_in.s.upward*pwr
        self.dofs.s.planpwr=self.ctl_in.s.forward*pwr 

class CtlDOFstate(State):
    cs: float = 1.0
class CtlDOFMode(Mode):
    failrate=1e-5
    faultparams={'noctl':   (0.2, 10000), 
                 'degctl':  (0.8, 10000)}
class CtlDOF(FxnBlock):
    __slots__=('ee_in', 'dir', 'ctl', 'dofs', 'fs')
    _init_s = CtlDOFstate
    _init_m = CtlDOFMode
    _init_ee_in = EE
    _init_dir = Dir
    _init_ctl = Control 
    _init_dofs = DOFs 
    _init_fs = Force 
    flownames = {'ee_ctl':'ee_in', 'force_st':'fs'}
    def condfaults(self, time):
        if self.fs.s.support<0.5: self.m.add_fault('noctl')
    def behavior(self, time):
        if self.m.has_fault('noctl'):       self.s.cs=0.0
        elif self.m.has_fault('degctl'):    self.s.cs=0.5
        else:                               self.s.cs=1.0
        
        upthrottle=1.0
        if self.dir.s.z>1:        upthrottle=2
        elif -1<self.dir.s.z<1:   upthrottle= 1 + self.dir.s.z
        elif self.dir.s.z<=-1.0:  upthrottle = 0
            
        if self.dir.s.same([0.0, 0.0], 'x', 'y'):   forwardthrottle=0.0
        else:                                       forwardthrottle=1.0
        
        power = self.ee_in.s.effort*self.s.cs*self.dir.s.power
        self.ctl.s.put(forward=power*forwardthrottle, upward=power*upthrottle)

class PlanPathMode(Mode):
    failrate=1e-5
    faultparams = {'noloc': (0.2, 10000),
                   'degloc':(0.8, 10000)}
class PlanPath(FxnBlock):
    __slots__=('ee_in', 'env', 'dir', 'fs')
    _init_m = PlanPathMode
    _init_ee_in = EE
    _init_env = Env
    _init_dir = Dir 
    _init_fs = Force 
    flownames = {'ee_ctl':'ee_in', 'force_st': 'fs'}
    def condfaults(self, time):
        if self.fs.s.support<0.5: self.m.add_fault('noloc')
    def behavior(self, t):
        self.dir.s.assign([1.0,0.0,0.0], "x","y","z")
        # faulty behaviors    
        if self.m.has_fault('noloc'):     self.dir.s.assign([0,0,0], "x", "y", "z")
        elif self.m.has_fault('degloc'):  self.dir.s.assign([0,0,-1], "x", "y", "z")
        if self.ee_in.s.effort<0.5:
            self.dir.s.assign([0.0,0.0,0.0,0.0],'x','y','z','power')  

class TrajectoryMode(Mode):
    faultparams = {'crash':(0, 100000), 
                  'lost':(0.0, 50000)}
class Trajectory(FxnBlock):
    __slots__=('env', 'dofs', 'dir', 'force_gr')
    _init_m = TrajectoryMode
    _init_env = Env
    _init_dofs = DOFs 
    _init_dir = Dir 
    _init_force_gr = Force 
    def behavior(self, time):
        self.dofs.s.vertvel = max(min(-2+2*self.dofs.s.uppwr, 2), -2)
        self.force_gr.s.support =self.dofs.s.vertvel
        self.dofs.s.planvel=self.dofs.s.planpwr
        if self.dofs.s.vertvel>1.5 or self.dofs.s.vertvel<-1:
            self.m.add_fault('crash')
            self.env.s.z=0.0
        if self.dofs.s.planvel>1.5 or self.dofs.s.planvel<0.5:
            self.m.add_fault('lost')
            self.env.s.x=0.0
        else:
            self.env.s.x=1.0

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
    _init_env = Env
        
class Drone(Model):
    __slots__=()
    def __init__(self, sp=SimParam(times=(0,1)), **kwargs):
        super().__init__(sp=sp, **kwargs)
        #add flows to the model
        self.add_flow('force_st',   Force)
        self.add_flow('force_lin',  Force)
        self.add_flow('force_gr' ,  Force)
        self.add_flow('force_lg',   Force)
        self.add_flow('ee_1',       EE)
        self.add_flow('ee_mot',     EE)
        self.add_flow('ee_ctl',     EE)
        self.add_flow('ctl',       Control)
        self.add_flow('dofs',       DOFs)
        self.add_flow('env',        Env)
        self.add_flow('dir',        Dir)
        #add functions to the model
        self.add_fxn('store_ee',    StoreEE,        'ee_1', 'force_st')
        self.add_fxn('dist_ee',     DistEE,         'ee_1','ee_mot','ee_ctl', 'force_st')
        self.add_fxn('affect_dof',  AffectDOF,      'ee_mot','ctl','dofs','force_lin')
        self.add_fxn('ctl_dof',     CtlDOF,         'ee_ctl', 'dir', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path',   PlanPath,       'ee_ctl', 'env','dir', 'force_st')
        self.add_fxn('trajectory',  Trajectory,     'env','dofs','dir', 'force_gr')
        self.add_fxn('engage_land', EngageLand,     'force_gr', 'force_lg')
        self.add_fxn('hold_payload',HoldPayload,    'force_lg', 'force_lin', 'force_st')
        self.add_fxn('view_env',    ViewEnvironment,'env')
        
        self.build()
    def find_classification(self,scen, mdlhist):
        modes, modeprops = self.return_faultmodes()
        repcost=sum([ c['rcost'] for f,m in modeprops.items() for a, c in m.items()])
        
        totcost=repcost
        rate=scen.rate
        expcost=totcost*rate*1e5
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}
    
if __name__=="__main__":
    from fmdtools.sim import propagate
    static_mdl = Drone()
    endclasses, mdlhists = propagate.single_faults(static_mdl)