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
        if self.has_fault('no_v'):      self.EEout.voltage=0.0 
        elif self.has_fault('high_v'):  self.EEout.voltage=2.0 
        elif self.has_fault('low_v'):   self.EEout.voltage=0.5
        else:                           self.EEout.voltage=1.0
        
class StoreEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEout'],flows)
        self.failrate=1e-4
        self.assoc_modes({'low_storage':[5,[1],2000], 'no_storage':[0.5,[1], 2000]})
    def behavior(self,time):
        if      self.has_fault('no_storage'):   self.EEout.voltage=0.0
        elif    self.has_fault('low_storage'):  
            self.EEout.voltage=1.0
            self.EEout.current=0.5
        else:
            self.EEout.voltage=self.EEin.voltage
            self.EEout.current=1.0
        
        
class SupplyEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['EEin','EEout','Heatout'],flows)
        self.failrate=1e-5
        self.assoc_modes({'adverse_resist':[5.0,[1],400], 'minor_overload':[1.0,[1], 400], 'major_overload':[0.5,[1], 400],\
                          'short':[0.01,[1],400], 'open_circuit':[0.004,[1], 200]})
    def condfaults(self,time):
        if self.EEout.current > 2.0:        self.add_fault('short')
        elif self.EEout.current >1.0:       self.add_fault('open_circuit')
    def behavior(self, time):
        if self.has_fault('open_circuit'):
            self.EEout.voltage=0.0
            self.EEin.current=1.0
        elif self.has_fault('short'):
            self.EEout.voltage=self.EEin.voltage*4.0
            self.EEin.rate=self.EEout.rate
        elif self.has_fault('major_overload'):
            self.EEout.voltage = self.EEin.voltage+1.0
            self.Heatout.effort=2.0
        elif self.has_fault('minor_overload'):
            self.EEout.voltage = 4.0
            self.Heatout.effort=4.0
        elif self.has_fault('adverse_resist'):
            self.EEout.voltage = self.EEin.voltage - 1.0
        else:
            self.EEout.voltage = 1.0
            self.Heatout.effort = 1.0
            
            
class DistEE(FxnBlock):
    def __init__(self,flows):
        super().__init__(['Sigin','EEin','EEoutM','EEoutH','EEoutO'],flows)
        self.failrate=1e-4
        self.assoc_modes({'adverse_resist':[1,[1],1500],'poor_alloc':[100,[1],500], 'short':[5,[1],1500], 'open_circuit':[5,[1],1500]})
    def condfaults(self,time):
        if self.EEin.current > 2.0:     self.add_fault('short')
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
        
class ExportME(FxnBlock):
    def __init__(self,flows):
        
                
class ExportOE(FxnBlock):
    def __init__(self,flows):
        
        
class EEtoME(FxnBlock):
    def __init__(self,flows):
        
        
class EEtoHE(FxnBlock):
    def __init__(self,flows):
        
        
class EEtoOE(FxnBlock):
    def __init__(self,flows):
        