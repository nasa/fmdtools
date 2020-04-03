# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:25:00 2020

@author: zhang
"""

from fmdtools.modeldef import *
import numpy as np
from scipy.optimize import minimize
import csv

class Place(FxnBlock):
    def __init__(self,flows, params):
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
        
        super().__init__(['Transport'],flows, {'Infected':population*1/10,'Susceptible':population*9/10,'Recovered':0.0, 'In_R':0.0, 'Re_R':0.0})
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
                
class Transit(FxnBlock):
    def __init__(self,flows):
        super().__init__(['T_Campus', 'T_Downtown', 'T_Living'],flows)
        self.failrate=1e-5
        self.assoc_modes({'na':[1.0, [1,1,1], 1]})
    def behavior(self,time):
        C_to_L = 0
        D_to_C = 0
        L_to_D = 0
        
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
#     def __init__(self, x0, params={}):
    def __init__(self, x0):
        super().__init__()
        
        self.times = [1,160]
        self.tstep = 1
        
        travel = {'In_I':0,'In_S':0,'In_R':0,'Out_I':0,'Out_S':0,'Out_R':0,'Stay_I':0,'Stay_S':0,'Stay_R':0}
        self.add_flow('Travel_Campus', 'People', travel)
        self.add_flow('Travel_Downtown', 'People', travel)
        self.add_flow('Travel_Living', 'People', travel)
        
#         x0 = np.array([2,3,5,10,0.15,2])
        params= {'pop':1000.0, 'a': x0[0] ,'n': x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5] }
        self.add_fxn('Campus',['Travel_Campus'],fclass= Place, fparams=params)
        self.add_fxn('Downtown',['Travel_Downtown'],fclass= Place, fparams=params)
        self.add_fxn('Living',['Travel_Living'],fclass= Place, fparams=params)
        self.add_fxn('Movement',['Travel_Campus','Travel_Downtown','Travel_Living'], fclass=Transit)
        
        
        self.construct_graph()
    def find_classification(self,resgraph, endfaults, endflows, scen, mdlhists):
        # total number of medical staff
        n1 = self.fxns['Campus'].n
        n2 = self.fxns['Downtown'].n
        n3 = self.fxns['Living'].n
        totalN=n1+n2+n3
        # total number of recovered people
        r1= self.fxns['Campus'].Recovered
        r2= self.fxns['Downtown'].Recovered
        r3= self.fxns['Living'].Recovered
        totalR=r1+r2+r3
        # total number of Susceptible people
        s1= self.fxns['Campus'].Susceptible
        s2= self.fxns['Downtown'].Susceptible
        s3= self.fxns['Living'].Susceptible
        totalS=s1+r2+r3
        # total number of Infected people
        i1= self.fxns['Campus'].Infected
        i2= self.fxns['Downtown'].Infected
        i3= self.fxns['Living'].Infected
        totalI=i1+i2+i3
        
        # total number of vaccine people
        vc1= self.fxns['Campus'].vc
        vc2= self.fxns['Downtown'].vc
        vc3= self.fxns['Living'].vc
        totalVC= vc1+vc2+vc3
        
        PL11= self.fxns['Campus'].PL1
        PL12= self.fxns['Downtown'].PL1
        PL13= self.fxns['Living'].PL1
        
        PL21= self.fxns['Campus'].PL2
        PL22= self.fxns['Downtown'].PL2
        PL23= self.fxns['Living'].PL2
        
        a0 = self.fxns['Campus'].a0
        a1 = self.fxns['Campus'].a
        a2 = self.fxns['Downtown'].a
        a3 = self.fxns['Living'].a 
       #t_campus = len([i for i in mdlhists['Campus']['faults'] if 'PL1' in i])
                 
        rate=1
        totcost=1
        expcost=1     
        N1= self.fxns['Campus'].N
        
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