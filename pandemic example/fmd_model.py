# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:25:00 2020

@author: zhang
"""

from fmdtools.modeldef import *
import numpy as np
from scipy.optimize import minimize
import csv

class Area(FxnBlock):
    def __init__(self,name,flows, params):
# total polulation
        population = params['pop']
#         self.extra = params[0]['extra']

# a is the contact rate after policy 2
# n is the increased number of medical staff per step in policy 1 
# v is the number of people who get vaccine per step in policy 2
# m is the default number of the medical staff 
# alpha is the threshold to triger policy 2 
# IR is the threshold  to triger policy 1 
# NMS is the  total number of the medical staff 
# a0 is the contact rate in normal state
        self.a_PL1 = params['a']
        self.n = params['n']
        self.v = params['v']
        self.m = params['m']
        self.alpha = params['alpha']
        self.IR = params['IR']
        
        self.NMS=self.m
        
        self.a0=0.2
        self.a = self.a0
        self.b=10
        
        super().__init__(name,flows,['Transport'], {'Infected':population*1/10,'Susceptible':population*9/10,'Recovered':0.0, 'In_R':0.0, 'Re_R':0.0})
        self.failrate=1e-5
        self.assoc_modes({'PL1':[1.0, [1,1,1], 1],'PL2':[1.0, [1,1,1], 1]})
                    
        self.N=population
        self.PL1=0
        self.PL2=0
        self.vc=0
    def condfaults(self,time):
        # policy 2: if infect rate bigger than IR, add m medical staff per day,  infectious time will drop from 1.25 to 1.25/2
        # policy 1: if infected people bigger than alpha, contact rate will drop from 10 to a , susceptible people will get vaccine,v people/day
        if time > self.time:
            if self.Infected/(self.N) > self.alpha: 
                self.add_fault('PL1');self.PL1=self.PL1+1
            else:
                if self.has_fault('PL1'): self.replace_fault('PL1','nom')
            if ( self.a * self.Susceptible * self.Infected / (self.N)) > self.IR:
                self.add_fault('PL2')
                self.PL2=self.PL2+1
            else:
                if self.has_fault('PL2'): self.replace_fault('PL2','nom')
            
    def behavior(self,time):
        if time>self.time:
            if self.has_fault('PL1') and self.has_fault('PL2'):
                self.a=self.a_PL1
                self.c=(self.NMS+self.m)/(self.m)
                self.v = 5
                NMS_increase = self.n
            elif self.has_fault('PL1'):
                self.a=self.a_PL1
                self.c = 1.0
                self.v = 5
                NMS_increase = 0.0
            elif self.has_fault('PL2'):
                self.a = self.a0
                self.c=(self.NMS+self.m)/(self.m)
                self.v = 0
                NMS_increase = self.n
            else:
                self.a = self.a0
                self.c = 1.0
                self.v = 0
                NMS_increase = 0.0
            self.In_R= self.a * self.Susceptible * self.Infected / self.N
            self.Re_R = self.c * self.Infected / self.b 
            
            
            self.NMS+=NMS_increase  
            self.Infected = self.Infected +  (self.In_R - self.Re_R)
            self.Susceptible = self.Susceptible - self.In_R - self.v
            self.Recovered = self.Recovered +  self.Re_R + self.v
            
            self.Infected += self.Transport.In_I  - self.Transport.Out_I 
            self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
            self.Recovered += self.Transport.In_R - self.Transport.Out_R
            self.Transport.Stay_I  = self.Infected
            self.Transport.Stay_S  = self.Susceptible
            self.Transport.Stay_R  = self.Recovered
                
class Transportation(FxnBlock):
    def __init__(self,name,flows):
        super().__init__(name,flows,['T_Schools', 'T_City', 'T_Suburbs'])
        self.failrate=1e-5
        self.assoc_modes({'na':[1.0, [1,1,1], 1]})
    def behavior(self,time):
        C_to_L = 0
        D_to_C = 0
        L_to_D = 0
        
        if time > self.time:
            self.T_Schools.Out_I = C_to_L * self.T_Schools.Stay_I
            self.T_Schools.Out_S = C_to_L * self.T_Schools.Stay_S
            self.T_Schools.Out_R = C_to_L * self.T_Schools.Stay_R 
            
            self.T_City.Out_I = D_to_C * self.T_City.Stay_I
            self.T_City.Out_S = D_to_C * self.T_City.Stay_S
            self.T_City.Out_R  = D_to_C * self.T_City.Stay_R 
            
            self.T_Suburbs.Out_I = L_to_D * self.T_Suburbs.Stay_I
            self.T_Suburbs.Out_S = L_to_D * self.T_Suburbs.Stay_S
            self.T_Suburbs.Out_R  = L_to_D * self.T_Suburbs.Stay_R          
            
            
            self.T_City.In_I = self.T_Schools.Out_I
            self.T_City.In_S = self.T_Schools.Out_S
            self.T_City.In_R  = self.T_Schools.Out_R 
            
            self.T_Schools.In_I = self.T_Suburbs.Out_I
            self.T_Schools.In_S = self.T_Suburbs.Out_S
            self.T_Schools.In_R  = self.T_Suburbs.Out_R  
            
            self.T_Suburbs.In_I = self.T_City.Out_I
            self.T_Suburbs.In_S = self.T_City.Out_S
            self.T_Suburbs.In_R  =  self.T_City.Out_R 
            
            
        
class PandemicModel(Model):
#     def __init__(self, x0, params={}):
    def __init__(self, params={'x0':[0.2 , 0 , 0 , 10 , 0 , 0 ]},\
                 modelparams={'times':[0,160], 'tstep':1}, valparams='all'):
        
        super().__init__(params=params, modelparams=modelparams, valparams=valparams)
        x0=self.params['x0']
        
        travel = {'In_I':0,'In_S':0,'In_R':0,'Out_I':0,'Out_S':0,'Out_R':0,'Stay_I':0,'Stay_S':0,'Stay_R':0}
        self.add_flow('Travel_Schools', travel)
        self.add_flow('Travel_City', travel)
        self.add_flow('Travel_Suburbs', travel)
        
#         x0 = np.array([2,3,5,10,0.15,2])
        params= {'pop':1000.0, 'a': x0[0] ,'n': x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5] }
        self.add_fxn('Schools',['Travel_Schools'],fclass= Area, fparams=params)
        self.add_fxn('City',['Travel_City'],fclass= Area, fparams=params)
        self.add_fxn('Suburbs',['Travel_Suburbs'],fclass= Area, fparams=params)
        self.add_fxn('Trasportation',['Travel_Schools','Travel_City','Travel_Suburbs'], fclass=Transportation)
        
        
        self.construct_graph()
    def find_classification(self, scen, mdlhists):
        # total number of medical staff
        n1 = self.fxns['Schools'].n
        n2 = self.fxns['City'].n
        n3 = self.fxns['Suburbs'].n
        totalN=n1+n2+n3
        # total number of recovered people
        r1= self.fxns['Schools'].Recovered
        r2= self.fxns['City'].Recovered
        r3= self.fxns['Suburbs'].Recovered
        totalR=r1+r2+r3
        # total number of Susceptible people
        s1= self.fxns['Schools'].Susceptible
        s2= self.fxns['City'].Susceptible
        s3= self.fxns['Suburbs'].Susceptible
        totalS=s1+r2+r3
        # total number of Infected people
        i1= self.fxns['Schools'].Infected
        i2= self.fxns['City'].Infected
        i3= self.fxns['Suburbs'].Infected
        totalI=i1+i2+i3
        
        # total number of vaccine people
        vc1= self.fxns['Schools'].vc
        vc2= self.fxns['City'].vc
        vc3= self.fxns['Suburbs'].vc
        totalVC= vc1+vc2+vc3
        
        PL11= self.fxns['Schools'].PL1
        PL12= self.fxns['City'].PL1
        PL13= self.fxns['Suburbs'].PL1
        
        PL21= self.fxns['Schools'].PL2
        PL22= self.fxns['City'].PL2
        PL23= self.fxns['Suburbs'].PL2
        
        a0 = self.fxns['Schools'].a0
        a1 = self.fxns['Schools'].a
        a2 = self.fxns['City'].a
        a3 = self.fxns['Suburbs'].a 
       #t_Schools = len([i for i in mdlhists['Schools']['faults'] if 'PL1' in i])
                 
        rate=1
        totcost=1
        expcost=1     
        N1= self.fxns['Schools'].N
        
        t=self.times[-1]
         # treatment fee for each people            
        H=10000
    # average expense for each people
        E=100
    # PL1 lasting time
        T1=PL11 
        T2=PL12
        T3=PL13
    # salary for each medical people per day
        Em=200
    # extra medical people total working time
        Tm1=totalN * (t-1+t-PL21)*PL21/2        
        Tm2=totalN * (t-1+t-PL22)*PL22/2  
        Tm3=totalN * (t-1+t-PL23)*PL23/2  
        
        totalcost1=(r1-vc1)*H + (a1-a0)* N1 * E * T1 + Em * Tm1
        totalcost2=(r1-vc1)*H + (a2-a0)* N1 * E * T2 + Em * Tm2
        totalcost3=(r1-vc1)*H + (a3-a0)* N1 * E * T3 + Em * Tm3
        
        total = totalcost1 + totalcost2 + totalcost3
        
#         return {'rate':rate, 'cost': totcost, 'expected cost': expcost, 'total number of medical staff': totalN , 'total recovery people': totalR}
        return {'total cost':total}