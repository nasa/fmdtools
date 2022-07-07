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
import sys, os
sys.path.insert(0, os.path.join('..'))
from fmdtools.modeldef import FxnBlock, Model

class ImportEE(FxnBlock):
    def __init__(self,name,flows):
        """ Static model representation is the same as the dynamic model respresntation, exept in this case 
        there is no opportunity vector. Thus the self.assoc_modes function takes a dictionary of modes with 
        just the vector of failure distribution and results cost. e.g. {'modename':[rate, cost]}.
        
        Also note that this model sets up the probability model differently--instead of specifying an overall failure rate
        for the function, one instead specifies an individual rate for eaach mode.
        
        Both representations can be used--this just shows this representation.
        """
        super().__init__(name,flows,['EEout'])
        self.assoc_modes({'low_v':[1e-5, 100], 'high_v':[5e-6, 100], 'no_v':[1e-5,300]})
    def behavior(self,time):
        if self.has_fault('no_v'):      self.EEout.effort=0.0 
        elif self.has_fault('high_v'):  self.EEout.effort=2.0 
        elif self.has_fault('low_v'):   self.EEout.effort=0.5
        else:                           self.EEout.effort=1.0
        
class ImportSig(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['Sigout'])
        self.assoc_modes({'partial_signal':[1e-5, 750], 'no_signal':[1e-6, 750]})
    def behavior(self,time):
        if  self.has_fault('partial_signal'):   self.Sigout.value=0.5 
        elif    self.has_fault('no_signal'):    self.Sigout.value=0.0 
        else:                                   self.Sigout.value = 1.0
        
class StoreEE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['EEin','EEout'])
        self.assoc_modes({'low_storage':[5e-6,2000], 'no_storage':[5e-6, 2000]})
    def condfaults(self,time):
        if self.EEout.effort*self.EEout.rate>=4.0:   self.add_fault('no_storage')
        elif self.EEout.effort*self.EEout.rate>=2.0: self.add_fault('low_storage') 
    def behavior(self,time):
        if      self.has_fault('no_storage'):   self.EEout.effort=0.0; self.EEin.rate=1.0
        elif    self.has_fault('low_storage'):  
            self.EEout.effort=1.0
            self.EEin.rate=self.EEout.rate
        else:
            self.EEout.effort=self.EEin.effort
            self.EEin.rate=self.EEout.rate
        
        
class SupplyEE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['EEin','EEout','Heatout'])
        self.assoc_modes({'adverse_resist':[2e-6,400], 'minor_overload':[1e-5, 400], 'major_overload':[3e-6, 400],\
                          'short':[1e-7,400], 'open_circuit':[5e-8, 200]})
    def condfaults(self,time):
        if self.EEout.rate > 2.0:        self.add_fault('short')
        elif self.EEout.rate >1.0:       self.add_fault('open_circuit')
    def behavior(self, time):
        if self.has_fault('open_circuit'):
            self.EEout.effort=0.0
            self.EEin.rate=1.0
        elif self.has_fault('short'):
            self.EEout.effort=self.EEin.effort*4.0
            self.EEin.rate=4.0
        elif self.has_fault('major_overload'):
            self.EEout.effort = self.EEin.effort+1.0
            self.Heatout.effort=2.0
        elif self.has_fault('minor_overload'):
            self.EEout.effort = 4.0
            self.Heatout.effort=4.0
        elif self.has_fault('adverse_resist'):
            self.EEout.effort = self.EEin.effort - 1.0
        else:
            self.EEout.effort = self.EEin.effort
            self.Heatout.effort = 1.0
            
            
class DistEE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['Sigin','EEin','EEoutM','EEoutH','EEoutO'])
        self.assoc_modes({'adverse_resist':[1e-5,1500],'poor_alloc':[2e-5,500], 'short':[2e-5,1500], 'open_circuit':[3e-5,1500]})
    def condfaults(self,time):
        if max(self.EEoutM.rate,self.EEoutH.rate,self.EEoutO.rate) > 2.0:     self.add_fault('short')
    def behavior(self,time):
        if self.has_fault('short'):
            self.EEin.rate = self.EEin.effort*4.0
            self.EEoutM.effort = 0.0
            self.EEoutH.effort = 0.0
            self.EEoutO.effort = 0.0
        elif self.has_fault('open_circuit') or self.Sigin.value <= 0.0:
            self.EEin.rate = 0.0
            self.EEoutM.effort = 0.0
            self.EEoutH.effort = 0.0
            self.EEoutO.effort = 0.0
        elif self.has_fault('poor_alloc') or self.has_fault('adverse_resist') or self.Sigin.value<1.0:
            self.EEoutM.effort = self.EEin.effort - 1.0
            self.EEoutH.effort = self.EEin.effort - 1.0
            self.EEoutO.effort = self.EEin.effort - 1.0
            self.EEin.rate = max(self.EEoutM.rate,self.EEoutH.rate,self.EEoutO.rate)
        else:
            self.EEoutM.effort = self.EEin.effort
            self.EEoutH.effort = self.EEin.effort
            self.EEoutO.effort = self.EEin.effort
            self.EEin.rate = max(self.EEoutM.rate,self.EEoutH.rate,self.EEoutO.rate)
        
class ExportHE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['HE_in'])
        self.assoc_modes({'hot_sink':[1e-5,500], 'ineffective_sink':[0.5e-5,1000]})
    def behavior(self,time):
        if self.has_fault('ineffective_sink'):  self.HE_in.rate=4.0
        elif self.has_fault('hot_sink'):        self.HE_in.rate=2.0
        else:                                   self.HE_in.rate=1.0
                 
        
class ExportME(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['ME_in'])
    def behavior(self,time):
        self.ME_in.rate = self.ME_in.effort
        
                
class ExportOE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['OE_in'])
    def behavior(self,time):
        self.OE_in.rate = self.OE_in.effort
        
        
class EEtoME(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name, flows,['EE_in','ME_out','HE_out'])
        self.assoc_modes({'high_torque':[1e-4,200],'low_torque':[1e-4,200],'toohigh_torque':[5e-5,200],\
                          'open_circuit':[5e-5,200], 'short':[5e-5,200]})
    def behavior(self, time):
        if self.has_fault('high_torque'):
            self.HE_out.effort = self.EE_in.effort + 1.0
            self.ME_out.effort = self.EE_in.effort + 1.0
            self.EE_in.rate =1.0/(self.ME_out.rate+0.001) -1.0
        elif self.has_fault('low_torque'):
            self.HE_out.effort = self.EE_in.effort - 1.0
            self.ME_out.effort = self.EE_in.effort - 1.0
            self.EE_in.rate =1.0/(self.ME_out.rate+0.001) -1.0
        elif self.has_fault('toohigh_torque'):
            self.HE_out.effort = 4.0
            self.ME_out.effort = 4.0
            self.EE_in.rate = 4.0
        elif self.has_fault('open_circuit'):
            self.HE_out.effort = 0.0
            self.ME_out.effort = 0.0
            self.ME_out.rate = 0.0
            self.EE_in.rate = 0.0
        elif self.has_fault('short'):
            self.EE_in.rate = self.EE_in.effort * 4.0
            self.HE_out.effort = self.EE_in.effort 
            self.ME_out.effort = 0.0
            self.ME_out.rate = 0.0
        else:
            self.HE_out.effort = self.EE_in.effort
            self.ME_out.effort = self.EE_in.effort
            self.ME_out.rate =self.EE_in.effort
            self.EE_in.rate = self.EE_in.effort
        
class EEtoHE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['EE_in', 'HE_out'])
        self.assoc_modes({'low_heat':[2e-6,200], 'high_heat':[1e-7,200], 'toohigh_heat':[5e-7,200], 'open_circuit':[1e-7,200]})
    def cond_faults(self, time):
        if self.EE_in.effort > 2.0: self.add_fault('open_circuit')
        elif self.EE_in.effort >1.0: self.add_fault('low_heat')
    def behavior(self, time):
        if self.has_fault('open_circuit'):
            self.HE_out.effort = 0.0
            self.EE_in.rate = 0.0
        elif self.has_fault('low_heat'):
            self.HE_out.effort = self.EE_in.effort -1.0
            self.EE_in.rate = self.EE_in.effort
        elif self.has_fault('high_heat'):
            self.HE_out.effort = self.EE_in.effort +1.0
            self.EE_in.rate = self.EE_in.effort+1.0
        elif self.has_fault('toohigh_heat'):
            self.HE_out.effort = 4.0
            self.EE_in.rate = 4.0
        else:
            self.HE_out.effort = self.EE_in.effort
            self.EE_in.rate = self.EE_in.effort
        
        
class EEtoOE(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['EE_in','OE_out', 'HE_out'])
        self.assoc_modes({'optical_resist':[5e-7,70],'burnt_out':[2e-6, 100]})
    def cond_faults(self,time):
        if self.EE_in.effort >= 2.0: self.add_fault('burnt_out')
    def behavior(self,time):
        if self.has_fault('burnt_out'):
            self.EE_in.rate = 0.0
            self.HE_out.effort = 0.0
            self.OE_out.effort = 0.0
        elif self.has_fault('optical_resist'):
            self.EE_in.rate = self.EE_in.effort - 1.0
            self.HE_out.effort = self.EE_in.effort - 1.0
            self.OE_out.effort = self.EE_in.effort - 1.0
        else:
            self.EE_in.rate = self.EE_in.effort 
            self.HE_out.effort = self.EE_in.effort 
            self.OE_out.effort =  self.EE_in.effort 

class EPS(Model):
    def __init__(self, params={}, modelparams={},valparams={}):
        """
        The Model superclass uses a static model representation by default if
        there are no parameters for times, phases, etc.
        """
        super().__init__(params=params)
        
        self.add_flow('EE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_2', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_3', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_M', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_O', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_H', {'rate':1.0, 'effort':1.0})
        self.add_flow('ME', {'rate':1.0, 'effort':1.0})
        self.add_flow('OE', {'rate':1.0, 'effort':1.0})
        self.add_flow('HE', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_1', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_O', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_M', {'rate':1.0, 'effort':1.0})
        self.add_flow('Sig_In', {'value':1.0})
        
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
        
        flowcost = -5*sum([qualfunc[discrep(self.flows[fl].effort)][discrep(self.flows[fl].rate)] for fl in outflows])
        
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
    rd.graph.show(mdl.bipartite, gtype='bipartite')
    endclasses, mdlhists = propagate.single_faults(mdl)

    endresults,resgraph, mdlhists = propagate.one_fault(mdl, 'EE_to_ME', 'toohigh_torque')
    rd.graph.show(resgraph)


    endclasses, mdlhists = propagate.single_faults(mdl)
    reshists, diffs, summary = rd.process.hists(mdlhists)

    sumtable = rd.tabulate.summary(summary)


    degtimemap = rd.process.avg_degtime_heatmap(reshists)

    rd.graph.show(mdl.bipartite,gtype='bipartite', heatmap=degtimemap)
    rd.graph.show(resgraph,heatmap=degtimemap)

    
    
    
        