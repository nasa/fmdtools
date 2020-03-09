# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:48:35 2020

@author: hulsed
"""

from fmdtools.modeldef import FxnBlock, Model


class ImportEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEout'],flows)
        self.failrate=1e-3
        self.assoc_modes({'low_v':[1.0,[1], 100], 'high_v':[0.5, [1], 100], 'no_v':[1.0,[1],300]})
    def behavior(self,time):
        if self.has_fault('no_v'):      self.EEout.effort=0.0 
        elif self.has_fault('high_v'):  self.EEout.effort=2.0 
        elif self.has_fault('low_v'):   self.EEout.effort=0.5
        else:                           self.EEout.effort=1.0
        
class ImportSig(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Sigout'],flows)
        self.failrate=1e-5
        self.assoc_modes({'partial_signal':[1.0,[1], 750], 'no_signal':[0.1, [1], 750]})
    def behavior(self,time):
        if  self.has_fault('partial_signal'):   self.Sigout.value=0.5 
        elif    self.has_fault('no_signal'):    self.Sigout.value=0.0 
        else:                                   self.Sigout = 1.0
        
class StoreEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEout'],flows)
        self.failrate=1e-4
        self.assoc_modes({'low_storage':[5,[1],2000], 'no_storage':[0.5,[1], 2000]})
    def behavior(self,time):
        if      self.has_fault('no_storage'):   self.EEout.effort=0.0
        elif    self.has_fault('low_storage'):  
            self.EEout.effort=1.0
            self.EEout.rate=0.5
        else:
            self.EEout.effort=self.EEin.effort
            self.EEout.rate=1.0
        
        
class SupplyEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEout','Heatout'],flows)
        self.failrate=1e-5
        self.assoc_modes({'adverse_resist':[5.0,[1],400], 'minor_overload':[1.0,[1], 400], 'major_overload':[0.5,[1], 400],\
                          'short':[0.01,[1],400], 'open_circuit':[0.004,[1], 200]})
    def condfaults(self,time):
        if self.EEout.rate > 2.0:        self.add_fault('short')
        elif self.EEout.rate >1.0:       self.add_fault('open_circuit')
    def behavior(self, time):
        if self.has_fault('open_circuit'):
            self.EEout.effort=0.0
            self.EEin.rate=1.0
        elif self.has_fault('short'):
            self.EEout.effort=self.EEin.effort*4.0
            self.EEin.rate=self.EEout.rate
        elif self.has_fault('major_overload'):
            self.EEout.effort = self.EEin.effort+1.0
            self.Heatout.effort=2.0
        elif self.has_fault('minor_overload'):
            self.EEout.effort = 4.0
            self.Heatout.effort=4.0
        elif self.has_fault('adverse_resist'):
            self.EEout.effort = self.EEin.effort - 1.0
        else:
            self.EEout.effort = 1.0
            self.Heatout.effort = 1.0
            
            
class DistEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Sigin','EEin','EEoutM','EEoutH','EEoutO'],flows)
        self.failrate=1e-4
        self.assoc_modes({'adverse_resist':[1,[1],1500],'poor_alloc':[100,[1],500], 'short':[5,[1],1500], 'open_circuit':[5,[1],1500]})
    def condfaults(self,time):
        if self.EEin.rate > 2.0:     self.add_fault('short')
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
    def __init__(self,flows):
        super().__init__(['HE_in'],flows)
        self.failrate=1e-5
        self.assoc_modes({'hot_sink':[1,[1],500], 'ineffective_sink':[0.5,[1],1000]})
    def behavior(self,time):
        if self.has_fault('ineffective_sink'):  self.HE_in.effort=4.0
        elif self.has_fault('hot_sink'):        self.HE_in.effort=2.0
        else:                                   self.HE_in.effort=1.0
                 
        
class ExportME(FxnBlock):
    def __init__(self,flows):
        super().__init__(['ME_in'],flows)
        self.failrate=0
    def behavior(self,time):
        self.Me_in.rate = self.Me_in.effort
        
                
class ExportOE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['OE_in'],flows)
        self.failrate=0
    def behavior(self,time):
        self.OE_in.rate = self.OE_in.effort
        
        
class EEtoME(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EE_in','ME_out','HE_out'], flows)
        self.failrate=1e-4
        self.assoc_modes({'high_torque':[1,[1],200],'low_torque':[1,[1],200],'toohigh_torque':[0.5,[1],200],\
                          'open_circuit':[0.5,[1],200], 'short':[0.5,[1],200]})
    def behavior(self, time):
        if self.has_fault('high_torque'):
            self.HE_out.rate = self.EE_in.effort + 1.0
            self.ME_out.effort = self.EE_in.effort + 1.0
            self.EE_in.rate =1.0/self.ME_out.rate -1.0
        elif self.has_fault('low_torque'):
            self.HE_out.rate = self.EE_in.effort - 1.0
            self.ME_out.effort = self.EE_in.effort - 1.0
            self.EE_in.rate =1.0/self.ME_out.rate -1.0
        elif self.has_fault('toohigh_torque'):
            self.HE_out.rate = 4.0
            self.ME_out.effort = 4.0
            self.EE_in.rate = 4.0
        elif self.has_fault('open_circuit'):
            self.HE_out.rate = 0.0
            self.ME_out.effort = 0.0
            self.ME_out.rate = 0.0
            self.EE_in.rate = 0.0
        elif self.has_fault('short'):
            self.EE_in.rate = self.EE_in.effort * 4.0
            self.HE_out.rate = self.EE_in.effort 
            self.ME_out.effort = 0.0
            self.ME_out.rate = 0.0
        else:
            self.HE_out.rate = self.EE_in.effort
            self.ME_out.effort = self.EE_in.effort
            self.ME_out.rate =self.EE_in.effort
            self.EE_in.rate = self.EE_in.effort
        
class EEtoHE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EE_in', 'HE_out'],flows)
        self.failrate=1e-5
        self.assoc_modes({'low_heat':[0.1, [1],200], 'high_heat':[1,[1],200], 'toohigh_heat':[5,[1],200], 'open_circuit':[0.001,[1],200]})
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
    def __init__(self,flows):
        super().__init__(['EE_in','OE_out', 'HE_out'],flows)
        self.failrate=1e-3
        self.assoc_modes({'optical_resist':[1,[1],70],'burnt_out':[1,[1], 100]})
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
    def __init__(self, params={}):
        super().__init__()
        
        self.params=params
        #Declare time range to run model over
        self.phases={'na':[1]}
        self.times=[1]
        self.tstep = 1 
        
        self.add_flow('EE_1', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_2', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_3', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_M', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_O', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('EE_H', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('ME', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('OE', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('HE', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_1', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_O', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('waste_HE_M', 'electricity', {'rate':1.0, 'effort':1.0})
        self.add_flow('Sig_In', 'signal', {'value':1.0})
        
        self.add_fxn('Import_EE',ImportEE,['EE_1'])
        self.add_fxn('Supply_EE',SupplyEE,['EE_1', 'EE_2','waste_HE_1'])
        self.add_fxn('Store_EE',StoreEE,['EE_2', 'EE_3'])
        self.add_fxn('Import_Signal',ImportSig,['Sig_In'])
        self.add_fxn('Distribute_EE',DistEE,['Sig_In', 'EE_3', 'EE_M', 'EE_H', 'EE_O'])
        self.add_fxn('EE_to_ME', EEtoME, ['EE_M', 'ME', 'waste_HE_M'])
        self.add_fxn('EE_to_OE', EEtoOE, ['EE_O', 'OE', 'waste_HE_O'])
        self.add_fxn('EE_to_HE', EEtoHE, ['EE_H','HE'])
        self.add_fxn('Export_ME', ExportME, ['ME'])
        self.add_fxn('Export_HE', ExportHE, ['HE'])
        self.add_fxn('Export_OE', ExportOE, ['OE'])
        self.add_fxn('Export_waste_H1', ExportHE, ['waste_HE_1'])
        self.add_fxn('Export_waste_HO', ExportHE, ['waste_HE_O'])
        self.add_fxn('Export_waste_HM', ExportHE, ['waste_HE_M'])
        
        self.construct_graph()
        
        