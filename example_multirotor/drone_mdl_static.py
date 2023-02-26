import numpy as np

from fmdtools.modeldef.common import Parameter
from fmdtools.modeldef.block import FxnBlock
from fmdtools.modeldef.model import Model, ModelParam

class StoreEE(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['EEout', 'FS'], {'soc': 2000})
        self.failrate=1e-5
        self.assoc_modes({'nocharge':[1,300]})
    def behavior(self, time):
        if self.has_fault('nocharge'):  self.EEout.effort=0.0
        else:                           self.EEout.effort=1.0
class DistEE(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name,flows, ['EEin','EEmot','EEctl','ST'], {'EEtr':1.0, 'EEte':1.0})
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
    def __init__(self,name, flows):
        super().__init__(name,flows, ['forcein', 'forceout'])
        self.failrate=1e-5
        self.assoc_modes({'break':[0.2, 1000], 'deform':[0.8, 1000]})
    def condfaults(self, time):
        if abs(self.forcein.value)>=2.0:      self.add_fault('break')
        elif abs(self.forcein.value)>1.5:    self.add_fault('deform')
    def behavior(self, time):
        self.forceout.value=self.forcein.value/2
            
class HoldPayload(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name,flows, ['FG', 'Lin', 'ST'])
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
    def __init__(self,name, flows):     
        super().__init__(name, flows, ['EEin', 'Ctlin','DOF','Force'],{'Eto': 1.0, 'Eti':1.0, 'Ct':1.0, 'Mt':1.0, 'Pt':1.0})
        self.failrate=1e-5
        self.assoc_modes({'short':[0.1, 200],'openc':[0.1, 200],'ctlup':[0.2, 500],'ctldn':[0.2, 500],
                          'ctlbreak':[0.2, 1000], 'mechbreak':[0.1, 500], 'mechfriction':[0.05, 500],
                          'propwarp':[0.01, 200],'propstuck':[0.02, 200], 'propbreak':[0.03, 200]})
    def behavior(self, time):
        self.Eti=1.0
        self.Eto=1.0
        if self.has_fault('short'):
            self.Eti=10
            self.Eto=0.0
        elif self.has_fault('openc'):
            self.Eti=0.0
            self.Eto=0.0
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
    def __init__(self,name,flows):
        super().__init__(name,flows, ['EEin','Dir','Ctl','DOFs','FS'], {'Cs':1.0})
        self.failrate=1e-5
        self.assoc_modes({'noctl':[0.2, 10000], 'degctl':[0.8, 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noctl')
    def behavior(self, time):
        if self.has_fault('noctl'):    self.Cs=0.0
        elif self.has_fault('degctl'): self.Cs=0.5
        
        upthrottle=1.0
        if self.Dir.z>1:        upthrottle=2
        elif -1<self.Dir.z<1:   upthrottle= 1 + self.Dir.z
        elif self.Dir.z<=-1.0:  upthrottle = 0
            
        if self.Dir.x==0 and self.Dir.y==0: forwardthrottle=0.0
        else: forwardthrottle=1.0
        
        self.Ctl.forward=self.EEin.effort*self.Cs*forwardthrottle*self.Dir.power
        self.Ctl.upward=self.EEin.effort*self.Cs*self.Dir.power*upthrottle

class PlanPath(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name, flows, ['EEin','Env','Dir','FS'])
        self.failrate=1e-5
        self.assoc_modes({'noloc':[0.2, 10000], 'degloc':[0.8, 10000]})
    def condfaults(self, time):
        if self.FS.support<0.5: self.add_fault('noloc')
    def behavior(self, t):
        self.Dir.assign([1.0,0.0,0.0], "x","y","z")
        # faulty behaviors    
        if self.has_fault('noloc'):     self.Dir.assign([0,0,0], "x", "y", "z")
        elif self.has_fault('degloc'):  self.Dir.assign([0,0,-1], "x", "y", "z")
        if self.EEin.effort<0.5:
            self.Dir.assign([0.0,0.0,0.0,0.0],'x','y','z','power')  

class Trajectory(FxnBlock):
    def __init__(self,name, flows):
        super().__init__(name, flows, ['Env','DOF', 'Dir', 'Force_GR'])
        self.assoc_modes({'crash':[0, 100000], 'lost':[0.0, 50000]})
    def behavior(self, time):
        self.DOF.vertvel = max(min(-2+2*self.DOF.uppwr, 2), -2)
        self.Force_GR.value =self.DOF.vertvel
        self.DOF.planvel=self.DOF.planpwr
        if self.DOF.vertvel>1.5 or self.DOF.vertvel<-1:
            self.add_fault('crash')
            self.Env.elev=0.0
        if self.DOF.planvel>1.5 or self.DOF.planvel<0.5:
            self.add_fault('lost')
            self.x=0.0
        else:
            self.x=1.0

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

class ViewEnvironment(FxnBlock):
    def __init__(self, name, flows):
        super().__init__(name, flows, ['Env'])
        self.failrate=1e-5
        self.assoc_modes({'poorview':[0.2, 10000]})
        
class Drone(Model):
    def __init__(self, params=Parameter(), modelparams=ModelParam(), valparams={}):
        super().__init__(params, modelparams, valparams)
        self.params=params
        #add flows to the model
        self.add_flow('Force_ST', {'support':1.0})
        self.add_flow('Force_Lin', {'support':1.0})
        self.add_flow('Force_GR' , {'value':0.0})
        self.add_flow('Force_LG', {'value':0.0})
        self.add_flow('EE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEmot', {'rate':1.0, 'effort':1.0})
        self.add_flow('EEctl', {'rate':1.0, 'effort':1.0})
        self.add_flow('Ctl1', {'forward':1.0, 'upward':1.0})
        self.add_flow('DOFs', {'vertvel':1.0, 'planvel':1.0, 'planpwr':1.0, 'uppwr':1.0})
        self.add_flow('Env1', {'x':0.0,'y':0.0,'elev':50.0} )
        self.add_flow('Dir1', {"x":1.0, "y":0.0, "z":0.0, "power":1.0})
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