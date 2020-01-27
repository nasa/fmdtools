# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:21:51 2020

@author: Daniel Hulse
"""

from fmdtools.modeldef import *

class Place(FxnBlock):
    def __init__(self,flows):
        super().__init__(['People_Out_1', 'People_Out_2'],flows, {'Infected':100.0,'Susceptible':500.0,'Recovered':0.0})
        self.failrate=1e-5
        self.assoc_modes({'na':[1.0, [1,1,1], 1]})
    def behavior(self,time):
        Infect_rate = 10.0 * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered)
        Recover_Rate =  self.Infected / 1.25
        if time>self.time:
            self.Infected = self.Infected + 0.05* (Infect_rate - Recover_Rate)
            self.Susceptible = self.Susceptible - 0.05* Infect_rate
            self.Recovered = self.Recovered + 0.05* Recover_Rate
        
class DiseaseModel(Model):
    def __init__(self, params={}):
        super().__init__()
        
        self.times = [1,140]
        self.tstep = 1
        
        self.add_flow('People_campus_living', 'People', {'Infected':100,'Susceptible':100,'Recovered':100})
        self.add_flow('People_campus_downtown', 'People', {'Infected':100,'Susceptible':100,'Recovered':100})
        self.add_flow('People_downtown_living', 'People', {'Infected':100,'Susceptible':100,'Recovered':100})
        
        self.add_fxn('Campus',Place,['People_campus_living', 'People_campus_downtown'])
        self.add_fxn('Downtown',Place,['People_campus_downtown','People_downtown_living'])
        self.add_fxn('Living',Place,['People_campus_living', 'People_downtown_living'])
        
        
        self.construct_graph()
    
    def find_classification(self,resgraph, endfaults, endflows, scen, mdlhists):
        rate=1
        totcost=1
        expcost=1
        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost}