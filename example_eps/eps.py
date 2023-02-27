# -*- coding: utf-8 -*-
"""
EPS Model 
This electrical power system model showcases how fmdtools can be used for purely static 
propogation models (where the dynamic states are not a concern). This EPS system was
previously provided in the IBFM fault modelling toolkit (see: https://github.com/DesignEngrLab/IBFM ) 
and other references--this implementation follows the simple_eps model in IBFM.
    
The main purpose of this system is to supply power to optical, mechanical, and heat loads.
In this model, we represent the failure behavior of the system at a high level
using solely the functions of the system.

Further information about this system (data, more detailed models) is presented
at: https://c3.nasa.gov/dashlink/projects/3/
"""
from fmdtools.modeldef.block import FxnBlock, Mode
from fmdtools.modeldef.model import Model, ModelParam
from fmdtools.modeldef.common import Parameter, State
from fmdtools.modeldef.flow import Flow

class ImportEEModes(Mode):
    faultparams={'low_v':(1e-5, 100), 
                 'high_v':(5e-6, 100), 
                 'no_v':(1e-5, 300)}
class ImportEE(FxnBlock):
    _init_m = ImportEEModes
    def __init__(self,name,flows):
        """ Static model representation is the same as the dynamic model respresntation, exept in this case 
        there is no opportunity vector. Thus the self.assoc_modes function takes a dictionary of modes with 
        just the vector of failure distribution and results cost. e.g. {'modename':[rate, cost]}.
        
        Also note that this model sets up the probability model differently--instead of specifying an overall failure rate
        for the function, one instead specifies an individual rate for eaach mode.
        
        Both representations can be used--this just shows this representation.
        """
        super().__init__(name,flows,['EEout'])
    def behavior(self,time):
        if self.m.has_fault('no_v'):        self.EEout.s.effort=0.0 
        elif self.m.has_fault('high_v'):    self.EEout.s.effort=2.0 
        elif self.m.has_fault('low_v'):     self.EEout.s.effort=0.5
        else:                               self.EEout.s.effort=1.0
class ImportSigModes(Mode):
    faultparams={'partial_signal':(1e-5, 750), 
                 'no_signal':(1e-6, 750)}
class ImportSig(FxnBlock):
    _init_m = ImportSigModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['Sigout'])
    def behavior(self,time):
        if  self.m.has_fault('partial_signal'):     self.Sigout.s.value=0.5 
        elif self.m.has_fault('no_signal'):         self.Sigout.s.value=0.0 
        else:                                       self.Sigout.s.value = 1.0

class StoreEEModes(Mode):
    faultparams={'low_storage':(5e-6, 2000),
                 'no_storage':(5e-6, 2000)}
class StoreEE(FxnBlock):
    _init_m=StoreEEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['EEin','EEout'])
    def condfaults(self,time):
        if self.EEout.s.effort*self.EEout.s.rate>=4.0:   self.m.add_fault('no_storage')
        elif self.EEout.s.effort*self.EEout.s.rate>=2.0: self.m.add_fault('low_storage') 
    def behavior(self,time):
        if      self.m.has_fault('no_storage'):   self.EEout.s.effort=0.0; self.EEin.s.rate=1.0
        elif    self.m.has_fault('low_storage'):  
            self.EEout.s.effort=1.0
            self.EEin.s.rate=self.EEout.s.rate
        else:
            self.EEout.s.effort=self.EEin.s.effort
            self.EEin.s.rate=self.EEout.s.rate
        
class SupplyEEModes(Mode):
    faultparams = {'adverse_resist':    (2e-6,  400), 
                   'minor_overload':    (1e-5, 400), 
                   'major_overload':    (3e-6,  400),
                   'short':             (1e-7, 400), 
                   'open_circuit':      (5e-8,  200)}
class SupplyEE(FxnBlock):
    _init_m=SupplyEEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['EEin','EEout','Heatout'])
    def condfaults(self,time):
        if self.EEout.s.rate > 2.0:        self.m.add_fault('short')
        elif self.EEout.s.rate >1.0:       self.m.add_fault('open_circuit')
    def behavior(self, time):
        if self.m.has_fault('open_circuit'):
            self.EEout.s.effort=0.0
            self.EEin.s.rate=1.0
        elif self.m.has_fault('short'):
            self.EEout.s.effort=self.EEin.s.effort*4.0
            self.EEin.s.rate=4.0
        elif self.m.has_fault('major_overload'):
            self.EEout.s.effort = self.EEin.s.effort+1.0
            self.Heatout.s.effort=2.0
        elif self.m.has_fault('minor_overload'):
            self.EEout.s.effort = 4.0
            self.Heatout.s.effort=4.0
        elif self.m.has_fault('adverse_resist'):
            self.EEout.s.effort = self.EEin.s.effort - 1.0
        else:
            self.EEout.s.effort = self.EEin.s.effort
            self.Heatout.s.effort = 1.0
            
class DistEEModes(Mode):
    faultparams = {'adverse_resist':(1e-5, 1500),
                   'poor_alloc':    (2e-5,500), 
                   'short':         (2e-5,1500), 
                   'open_circuit':  (3e-5,1500)}
    
class DistEE(FxnBlock):
    _init_m=DistEEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['Sigin','EEin','EEoutM','EEoutH','EEoutO'])
    def condfaults(self,time):
        if max(self.EEoutM.s.rate,self.EEoutH.s.rate,self.EEoutO.s.rate) > 2.0:     self.m.add_fault('short')
    def behavior(self,time):
        if self.m.has_fault('short'):
            self.EEin.s.rate = self.EEin.s.effort*4.0
            self.EEoutM.s.effort = 0.0
            self.EEoutH.s.effort = 0.0
            self.EEoutO.s.effort = 0.0
        elif self.m.has_fault('open_circuit') or self.Sigin.s.value <= 0.0:
            self.EEin.s.rate = 0.0
            self.EEoutM.s.effort = 0.0
            self.EEoutH.s.effort = 0.0
            self.EEoutO.s.effort = 0.0
        elif self.m.has_fault('poor_alloc') or self.m.has_fault('adverse_resist') or self.Sigin.s.value<1.0:
            self.EEoutM.s.effort = self.EEin.s.effort - 1.0
            self.EEoutH.s.effort = self.EEin.s.effort - 1.0
            self.EEoutO.s.effort = self.EEin.s.effort - 1.0
            self.EEin.s.rate = max(self.EEoutM.s.rate,self.EEoutH.s.rate,self.EEoutO.s.rate)
        else:
            self.EEoutM.s.effort = self.EEin.s.effort
            self.EEoutH.s.effort = self.EEin.s.effort
            self.EEoutO.s.effort = self.EEin.s.effort
            self.EEin.s.rate = max(self.EEoutM.s.rate,self.EEoutH.s.rate,self.EEoutO.s.rate)

class ExportHEModes(Mode):
    faultparams={'hot_sink':           (1e-5, 500), 
                 'ineffective_sink':   (0.5e-5,1000)}
class ExportHE(FxnBlock):
    _init_m = ExportHEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['HE_in'])
    def behavior(self,time):
        if self.m.has_fault('ineffective_sink'):  self.HE_in.s.rate=4.0
        elif self.m.has_fault('hot_sink'):        self.HE_in.s.rate=2.0
        else:                                   self.HE_in.s.rate=1.0
                 
        
class ExportME(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['ME_in'])
    def behavior(self,time):
        self.ME_in.s.rate = self.ME_in.s.effort
        
                
class ExportOE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['OE_in'])
    def behavior(self,time):
        self.OE_in.s.rate = self.OE_in.s.effort

class EEtoMEModes(Mode):
    faultparams={'high_torque':     (1e-4,200),
                 'low_torque':      (1e-4,200),
                 'toohigh_torque':  (5e-5,200),
                 'open_circuit':    (5e-5,200), 
                 'short':           (5e-5,200)}
class EEtoME(FxnBlock):
    _init_m = EEtoMEModes
    def __init__(self,name,flows):
        super().__init__(name, flows,['EE_in','ME_out','HE_out'])
    def behavior(self, time):
        if self.m.has_fault('high_torque'):
            self.HE_out.s.effort = self.EE_in.s.effort + 1.0
            self.ME_out.s.effort = self.EE_in.s.effort + 1.0
            self.EE_in.s.rate =1.0/(self.ME_out.s.rate+0.001) -1.0
        elif self.m.has_fault('low_torque'):
            self.HE_out.s.effort = self.EE_in.s.effort - 1.0
            self.ME_out.s.effort = self.EE_in.s.effort - 1.0
            self.EE_in.s.rate =1.0/(self.ME_out.s.rate+0.001) -1.0
        elif self.m.has_fault('toohigh_torque'):
            self.HE_out.s.effort = 4.0
            self.ME_out.s.effort = 4.0
            self.EE_in.s.rate = 4.0
        elif self.m.has_fault('open_circuit'):
            self.HE_out.s.effort = 0.0
            self.ME_out.s.effort = 0.0
            self.ME_out.s.rate = 0.0
            self.EE_in.s.rate = 0.0
        elif self.m.has_fault('short'):
            self.EE_in.s.rate = self.EE_in.s.effort * 4.0
            self.HE_out.s.effort = self.EE_in.s.effort 
            self.ME_out.s.effort = 0.0
            self.ME_out.s.rate = 0.0
        else:
            self.HE_out.s.effort = self.EE_in.s.effort
            self.ME_out.s.effort = self.EE_in.s.effort
            self.ME_out.s.rate =self.EE_in.s.effort
            self.EE_in.s.rate = self.EE_in.s.effort
class EEtoHEModes(Mode):
    faultparams={'low_heat':        (2e-6,200), 
                 'high_heat':       (1e-7,200), 
                 'toohigh_heat':    (5e-7,200), 
                 'open_circuit':    (1e-7,200)}
class EEtoHE(FxnBlock):
    _init_m=EEtoHEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['EE_in', 'HE_out'])
    def cond_faults(self, time):
        if self.EE_in.s.effort > 2.0: self.m.add_fault('open_circuit')
        elif self.EE_in.s.effort >1.0: self.m.add_fault('low_heat')
    def behavior(self, time):
        if self.m.has_fault('open_circuit'):
            self.HE_out.s.effort = 0.0
            self.EE_in.s.rate = 0.0
        elif self.m.has_fault('low_heat'):
            self.HE_out.s.effort = self.EE_in.s.effort -1.0
            self.EE_in.s.rate = self.EE_in.s.effort
        elif self.m.has_fault('high_heat'):
            self.HE_out.s.effort = self.EE_in.s.effort +1.0
            self.EE_in.s.rate = self.EE_in.s.effort+1.0
        elif self.m.has_fault('toohigh_heat'):
            self.HE_out.s.effort = 4.0
            self.EE_in.s.rate = 4.0
        else:
            self.HE_out.s.effort = self.EE_in.s.effort
            self.EE_in.s.rate = self.EE_in.s.effort
        
class EEtoOEModes(Mode):
    faultparams = {'optical_resist':    (5e-7, 70),
                   'burnt_out':         (2e-6, 100)}
class EEtoOE(FxnBlock):
    _init_m = EEtoOEModes
    def __init__(self,name,flows):
        super().__init__(name,flows,['EE_in','OE_out', 'HE_out'])
    def cond_faults(self,time):
        if self.EE_in.s.effort >= 2.0: self.m.add_fault('burnt_out')
    def behavior(self,time):
        if self.m.has_fault('burnt_out'):
            self.EE_in.s.rate = 0.0
            self.HE_out.s.effort = 0.0
            self.OE_out.s.effort = 0.0
        elif self.m.has_fault('optical_resist'):
            self.EE_in.s.rate = self.EE_in.s.effort - 1.0
            self.HE_out.s.effort = self.EE_in.s.effort - 1.0
            self.OE_out.s.effort = self.EE_in.s.effort - 1.0
        else:
            self.EE_in.s.rate = self.EE_in.s.effort 
            self.HE_out.s.effort = self.EE_in.s.effort 
            self.OE_out.s.effort =  self.EE_in.s.effort 

class GenericState(State):
    rate:   float = 1.0
    effort: float = 1.0
class GenericFlow(Flow):
    _init_s = GenericState

class SigState(State):
    value:  float=1.0
class Signal(Flow):
    _init_s = SigState

class EPS(Model):
    def __init__(self, params=Parameter(), modelparams=ModelParam(times=(0,1)),valparams={}):
        """
        The Model superclass uses a static model representation by default if
        there are no parameters for times, phases, etc.
        """
        super().__init__(params=params, modelparams=modelparams,valparams={})
        
        self.add_flow('EE_1', GenericFlow)
        self.add_flow('EE_2', GenericFlow)
        self.add_flow('EE_3', GenericFlow)
        self.add_flow('EE_M', GenericFlow)
        self.add_flow('EE_O', GenericFlow)
        self.add_flow('EE_H', GenericFlow)
        self.add_flow('ME', GenericFlow)
        self.add_flow('OE', GenericFlow)
        self.add_flow('HE', GenericFlow)
        self.add_flow('waste_HE_1',GenericFlow)
        self.add_flow('waste_HE_O', GenericFlow)
        self.add_flow('waste_HE_M', GenericFlow)
        self.add_flow('Sig_In', Signal)
        
        self.add_fxn('Import_EE',['EE_1'],fclass=ImportEE)
        self.add_fxn('Supply_EE',['EE_1', 'EE_2','waste_HE_1'],fclass=SupplyEE)
        self.add_fxn('Store_EE',['EE_2', 'EE_3'],fclass=StoreEE)
        self.add_fxn('Import_Signal',['Sig_In'],fclass=ImportSig)
        self.add_fxn('Distribute_EE',['Sig_In', 'EE_3', 'EE_M', 'EE_H', 'EE_O'],fclass=DistEE)
        self.add_fxn('EE_to_ME', ['EE_M', 'ME', 'waste_HE_M'], fclass=EEtoME)
        self.add_fxn('EE_to_OE', ['EE_O', 'OE', 'waste_HE_O'], fclass=EEtoOE)
        self.add_fxn('EE_to_HE', ['EE_H','HE'], fclass = EEtoHE)
        self.add_fxn('Export_ME', ['ME'], fclass = ExportME)
        self.add_fxn('Export_HE', ['HE'], fclass = ExportHE)
        self.add_fxn('Export_OE', ['OE'], fclass = ExportOE)
        self.add_fxn('Export_waste_H1', ['waste_HE_1'], fclass = ExportHE)
        self.add_fxn('Export_waste_HO', ['waste_HE_O'], fclass = ExportHE)
        self.add_fxn('Export_waste_HM', ['waste_HE_M'], fclass = ExportHE)
        
        self.build_model()
    def find_classification(self, scen, mdlhists):
        
        outflows = ['HE','ME', 'OE']
        
        qualfunc = [[-90.,-80.,-70.,-85.,-100.],
            [-80., -50., -20, -15, -100.],
            [-70., -20.,  0., -20., -100.],
            [-85., -10, -20., -50.,-110.],
            [-100., -100., -100.,-110.,-110.]]
        
        flowcost = -5*sum([qualfunc[discrep(self.flows[fl].s.effort)][discrep(self.flows[fl].s.rate)] for fl in outflows])
        
        repcost= self.calc_repaircost()
        cost = repcost+flowcost
        
        rate = scen['properties']['rate'] 
        return {'rate': rate, 'cost': cost, 'expected cost': 24*365*5*rate*cost}

def discrep(value):
    if      value <= 0.0:   return 0
    elif    value<=0.5:     return 1
    elif    value<=1.0:     return 2
    elif    value<=2.0:     return 3
    else:                   return 4 
    
if __name__ == '__main__':
    import fmdtools.faultsim.propagate as propagate
    import fmdtools.resultdisp as rd

    mdl= EPS()
    
    resgraph, mdlhists = propagate.one_fault(mdl, 'Distribute_EE', 'short', desired_result="bipartite")
    
    rd.graph.show(mdl.bipartite, gtype='bipartite')
    #endclasses, mdlhists = propagate.single_faults(mdl)

    #resgraph, mdlhists = propagate.one_fault(mdl, 'EE_to_ME', 'toohigh_torque', desired_result="bipartite")
    rd.graph.show(resgraph)


    #endclasses, mdlhists = propagate.single_faults(mdl)
    reshists, diffs, summary = rd.process.hists(mdlhists)

    sumtable = rd.tabulate.summary(summary)


    degtimemap = rd.process.avg_degtime_heatmap(reshists)

    rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=degtimemap)
    rd.graph.show(resgraph,heatmap=degtimemap)

    
    
    
        