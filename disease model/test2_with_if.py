from modeldef import *

class Place(FxnBlock):
    def __init__(self,flows, params):
        population = params[0]['pop']
#         self.extra = params[0]['extra']

# a is the contact rate after policy 2
# n is the increased number of medical staff per step in policy 1 
# v is the number of people who get vaccine per step in policy 2
# m is the default number of the medical staff 
# alpha is the threshold to triger policy 2 
# IR is the threshold  to triger policy 1 
# NMS is the  total number of the medical staff 
        self.a = params[0]['a']
        self.n = params[0]['n']
        self.v = params[0]['v']
        self.m = params[0]['m']
        self.alpha = params[0]['alpha']
        self.IR = params[0]['IR']
        self.NMS=0
        
        super().__init__(['Transport'],flows, {'Infected':population*2/10,'Susceptible':population*8/10,'Recovered':0.0})
        self.failrate=1e-5
        self.assoc_modes({'PL1':[1.0, [1,1,1], 1],'PL2':[1.0, [1,1,1], 1]})
                    
        
    def condfaults(self,time):
        # policy 2: if infect rate bigger than IR, add m medical staff per day,  infectious time will drop from 1.25 to 1.25/2
        # policy 1: if infected people bigger than alpha, contact rate will drop from 10 to a , susceptible people will get vaccine,v people/day
        if self.Infected/(self.Susceptible + self.Infected + self.Recovered)> self.alpha: 
            self.add_fault('PL1')
        if ( self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)) > self.IR:
            self.add_fault('PL2')            
        
            
    def behavior(self,time):
        if self.has_fault('PL1') and self.has_fault('PL2'):
            self.NMS+=self.n 
            self.c=(self.NMS+self.m)/(self.m)
            Infect_rate = self.a * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
            Recover_Rate = self.c * self.Infected / (1.25/2)
            
            if time>self.time:
                self.Infected =self.Infected +  ( Infect_rate - Recover_Rate)
                self.Susceptible = self.Susceptible-  (Infect_rate - self.v)
                self.Recovered = self.Recovered +  ( Recover_Rate + self.v)
                
                self.Infected += self.Transport.In_I  - self.Transport.Out_I 
                self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
                self.Recovered += self.Transport.In_R - self.Transport.Out_R
                self.Transport.Stay_I  = self.Infected
                self.Transport.Stay_S  = self.Susceptible
                self.Transport.Stay_R  = self.Recovered
                
        elif self.has_fault('PL1'):
            Infect_rate = self.a * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
            Recover_Rate = self.Infected / (1.25) 
            
            if time>self.time:
                self.Infected = self.Infected +  ( Infect_rate - Recover_Rate)
                self.Susceptible = self.Susceptible-  (Infect_rate + self.v)
                self.Recovered = self.Recovered +  ( Recover_Rate + self.v)
                
                self.Infected += self.Transport.In_I  - self.Transport.Out_I 
                self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
                self.Recovered += self.Transport.In_R - self.Transport.Out_R
                self.Transport.Stay_I  = self.Infected
                self.Transport.Stay_S  = self.Susceptible
                self.Transport.Stay_R  = self.Recovered
        
        elif self.has_fault('PL2'):
            self.NMS+=self.n 
            self.c=(self.NMS+self.m)/(self.m)
            Infect_rate = 0.5 * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
            Recover_Rate = self.c * self.Infected / (1.25/2) 
            
            if time>self.time:
                self.Infected =self.Infected +  ( Infect_rate - Recover_Rate)
                self.Susceptible = self.Susceptible-  (Infect_rate)
                self.Recovered = self.Recovered +  ( Recover_Rate)
                
                self.Infected += self.Transport.In_I  - self.Transport.Out_I 
                self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
                self.Recovered += self.Transport.In_R - self.Transport.Out_R
                self.Transport.Stay_I  = self.Infected
                self.Transport.Stay_S  = self.Susceptible
                self.Transport.Stay_R  = self.Recovered
            
        else:
            Infect_rate= 0.5 * self.alpha * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
            Recover_Rate =  self.Infected / 1.25
            
            if time>self.time:
                self.Infected = self.Infected + (Infect_rate - Recover_Rate)
                self.Susceptible = self.Susceptible - Infect_rate
                self.Recovered = self.Recovered + Recover_Rate
                
                self.Infected += self.Transport.In_I  - self.Transport.Out_I 
                self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
                self.Recovered += self.Transport.In_R - self.Transport.Out_R
                self.Transport.Stay_I  = self.Infected
                self.Transport.Stay_S  = self.Susceptible
                self.Transport.Stay_R  = self.Recovered
        
# ---------------------------------------------------------   
#         if self.has_fault('PL1'):
              
#             # c=(n+m)/m
            
#             self.NMS+=self.n 
#             self.c=(self.NMS+self.m)/(self.m)
#             Infect_rate = self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
#             Recover_Rate = self.c * self.Infected / (1.25/2)    
            
#             if time>self.time:
#                 self.Infected =self.Infected + 0.0001* ( Infect_rate - Recover_Rate)
#                 self.Susceptible = self.Susceptible- 0.0001* (Infect_rate - self.v)
#                 self.Recovered = self.Recovered + 0.0001* ( Recover_Rate + self.v)
#         if self.has_fault('PL2'):
#             self.a = self.alpha
            
#             if time>self.time:
#                 self.Infected =self.Infected + 0.0001* ( Infect_rate - Recover_Rate)
#                 self.Susceptible = self.Susceptible- 0.0001* (Infect_rate - self.v)
#                 self.Recovered = self.Recovered + 0.0001* ( Recover_Rate + self.v)           
            
#         # nominal state    
#         Infect_rate= self.alpha * self.Susceptible * self.Infected / (self.Susceptible + self.Infected + self.Recovered + 0.001)
#         Recover_Rate =  self.Infected / 1.25
#         Leave_Rate = 0.5
       
#         if time>self.time:
#             self.Infected += 0.0001* (Infect_rate - Recover_Rate)
#             self.Susceptible -= 0.0001* Infect_rate
#             self.Recovered +=0.0001* Recover_Rate
# #             if self.extra:
# #                 self.Infected += 2
# #                 self.Recovered -= 2
#             # Arriving/Leaving
#             self.Infected += self.Transport.In_I  - self.Transport.Out_I 
#             self.Susceptible += self.Transport.In_S  - self.Transport.Out_S 
#             self.Recovered += self.Transport.In_R - self.Transport.Out_R
#             self.Transport.Stay_I  = self.Infected
#             self.Transport.Stay_S  = self.Susceptible
#             self.Transport.Stay_R  = self.Recovered
            
# ---------------------------------------------------------   
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
#     def __init__(self, x0, params={}):
    def __init__(self, x0):
        super().__init__()
        
        self.times = [1,60]
        self.tstep = 1
        
        travel = {'In_I':0,'In_S':0,'In_R':0,'Out_I':0,'Out_S':0,'Out_R':0,'Stay_I':0,'Stay_S':0,'Stay_R':0}
        self.add_flow('Travel_Campus', 'People', travel)
        self.add_flow('Travel_Downtown', 'People', travel)
        self.add_flow('Travel_Living', 'People', travel)
        
#         x0 = np.array([2,3,5,10,0.15,2])
        params= {'pop':15.0, 'a': x0[0] ,'n': x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5] }
        self.add_fxn('Campus',Place,['Travel_Campus'],params)
        self.add_fxn('Downtown',Place,['Travel_Downtown'], params)
        self.add_fxn('Living',Place,['Travel_Living'], params)
        self.add_fxn('Movement', Transit, ['Travel_Campus','Travel_Downtown','Travel_Living'])
        
        
        self.construct_graph()
    def find_classification(self,resgraph, endfaults, endflows, scen, mdlhists):
        
        n1 = self.fxns['Campus'].n
        n2 = self.fxns['Downtown'].n
        n3 = self.fxns['Living'].n
        totalN=n1+n2+n3
        
        r1= self.fxns['Campus'].Recovered
        r2= self.fxns['Campus'].Recovered
        r3= self.fxns['Campus'].Recovered
        totalR=r1+r2+r3
        
        
#         t_campus = len([i for i in mdlhists['Campus']['faults'] if 'PL1' in i])
                 
        rate=1
        totcost=1
        expcost=1            
        
        

        
        return {'rate':rate, 'cost': totcost, 'expected cost': expcost, 'total number of medical staff': totalN , 'total recovery people': totalR}
    