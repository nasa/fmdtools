# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:21:51 2020

@author: Daniel Hulse
"""

from fmdtools.modeldef import *


class Place(FxnBlock):
    def __init__(self,flows, params):
        population = params[0]['pop']
        self.extra = params[0]['extra']
        super().__init__(['Transport'],flows, {'Infected':population/10,'Susceptible':population*9/10,'Recovered':0.0})
        self.failrate=1e-5
        self.assoc_modes({'na':[1.0, [1,1,1], 1]})
    def behavior(self,time):
        Infect_rate = 10.0 * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
        Recover_Rate =  self.Infected / 1.25
        Leave_Rate = 0.5
        if time>self.time:
            self.Infected += 0.05* (Infect_rate - Recover_Rate)
            self.Susceptible -= 0.05* Infect_rate
            self.Recovered += 0.05* Recover_Rate
            if self.extra:
                self.Infected += 2
                self.Recovered -= 2
            # Arriving/Leaving
            self.Infected += self.Transport.In_I  - self.Transport.Out_I 
            self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
            self.Recovered += self.Transport.In_R - self.Transport.Out_R
            self.Transport.Stay_I  = self.Infected
            self.Transport.Stay_S  = self.Susceptible
            self.Transport.Stay_R  = self.Recovered
            

class Transit(FxnBlock):
    def __init__(self,flows):
        super().__init__(['T_Campus', 'T_Downtown', 'T_Living'],flows)
        self.failrate=1e-5
        self.assoc_modes({'na':[1.0, [1,1,1], 1]})
    def behavior(self,time):
        C_to_L = 0.1
        D_to_C = 0.1
        L_to_D = 0.1
        
        if time > self.time:
            self.T_Campus.Out_I = C_to_L * self.T_Campus.Stay_I
            self.T_Campus.Out_S = C_to_L * self.T_Campus.Stay_S
            self.T_Campus.Out_R = C_to_L * self.T_Campus.Stay_R 
            
            self.T_Downtown.Out_I = D_to_C * self.T_Downtown.Stay_I
            self.T_Downtown.Out_S = D_to_C * self.T_Downtown.Stay_S
            self.T_Downtown.Out_R  = D_to_C * self.T_Downtown.Stay_R 
            
            self.T_Living.Out_I = L_to_D * self.T_Living.Stay_I
            self.T_Living.Out_S = L_to_D * self.T_Living.Stay_S
            self.T_Living.Out_R  = L_to_D * self.T_Living.Stay_R          
            
            
            self.T_Downtown.In_I = self.T_Campus.Out_I
            self.T_Downtown.In_S = self.T_Campus.Out_S
            self.T_Downtown.In_R  = self.T_Campus.Out_R 
            
            self.T_Campus.In_I = self.T_Living.Out_I
            self.T_Campus.In_S = self.T_Living.Out_S
            self.T_Campus.In_R  = self.T_Living.Out_R  
            
            self.T_Living.In_I = self.T_Downtown.Out_I
            self.T_Living.In_S = self.T_Downtown.Out_S
            self.T_Living.In_R  =  self.T_Downtown.Out_R 
            
            
        
class DiseaseModel(Model):
    def __init__(self, params={}):
        super().__init__()
        
        self.times = [1,50]
        self.tstep = 1
        
        travel = {'In_I':0,'In_S':0,'In_R':0,'Out_I':0,'Out_S':0,'Out_R':0,'Stay_I':0,'Stay_S':0,'Stay_R':0}
        self.add_flow('Travel_Campus', 'People', travel)
        self.add_flow('Travel_Downtown', 'People', travel)
        self.add_flow('Travel_Living', 'People', travel)
        
        self.add_fxn('Campus',Place,['Travel_Campus'],{'pop':100.0,'extra':False})
        self.add_fxn('Downtown',Place,['Travel_Downtown'], {'pop':1000.0,'extra':True})
        self.add_fxn('Living',Place,['Travel_Living'], {'pop':100.0,'extra':False})
        self.add_fxn('Movement', Transit, ['Travel_Campus','Travel_Downtown','Travel_Living'])
        
        
        self.construct_graph()
    