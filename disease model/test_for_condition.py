# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:02:21 2020

@author: zhang
"""

import numpy as np
import csv
from scipy.optimize import minimize
import matplotlib
from matplotlib import pyplot as plt 
from scipy.optimize import differential_evolution


a=0.2
b=10
    
t=160
#step=int(t*1/h)

def DiseaseModel(x0):
    
    
    S=list(range(0,t)); S[0]=900
    I=list(range(0,t)); I[0]=100
    R=list(range(0,t)); R[0]=0
    N=S[0]+I[0]+R[0]
        
    PL1=0
    PL2=0
    nom=0
    
    
    a2=x0[0]
    n=x0[1]
    v=x0[2]
    m=x0[3]
    alpha=x0[4]
    IR=x0[5]
    
    c=( m )/ m
#    print(a2,n,v,m,alpha,IR)
    b2=b
    
       
    infect_rate=list(range(0,t))
    recover_rate=list(range(0,t))
    
    infect_rate[0]= a * S[0] * I[0] / N
    recover_rate[0]= I[0]/b
    
    state=list(range(0,t))
    state[0]='nom'
    for i in range(1,t):
    # PL1 and PL2 both triggered
#        print('\n')
         
        if I[i-1]/N > alpha and infect_rate[i-1] > IR:
            c =( m+n )/ m
            m = m+n
            
            infect_rate[i]= a2 * (S[i-1]) * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b2
            
            S[i]= S[i-1] -  (infect_rate[i] ) - v
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i])
            R[i]= R[i-1] +  (recover_rate[i]) + v
                   
            PL1=PL1+1
            PL2=PL2+1
            state[i]='PL1&PL2'
#            print(i)
#            print(S[i])
#            print(I[i])
#            print(R[i])
            
    # PL1 triggered
        elif I[i-1]/N > alpha :
            infect_rate[i] = a2 * (S[i-1]) * I[i-1] / N
            recover_rate[i] = I[i-1]/b2
            
            S[i]= S[i-1] -  (infect_rate[i] ) - v
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i] )
            R[i]= R[i-1] +  (recover_rate[i]) + v
                        
            PL1=PL1+1
            state[i]='PL1'
#            print(i)
#            print(S[i])
#            print(I[i])
#            print(R[i])
    # PL2 triggered     
        elif  infect_rate[i-1] > IR : 
            c =( m+n )/ m
            m = m+n
            
            infect_rate[i]= a * (S[i-1]) * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b2
        
            S[i]= S[i-1] -  (infect_rate[i])
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i])
            R[i]= R[i-1] +  (recover_rate[i])
            
            PL2=PL2+1
            state[i]='PL2'
#            print(i)
#            print(S[i])
#            print(I[i])
#            print(R[i])
    #     nominal state
        else:
            infect_rate[i]= a * S[i-1] * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b
            
            S[i]= S[i-1] -  (infect_rate[i])
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i])
            R[i]= R[i-1] +  (recover_rate[i])
            
            nom=nom+1  
            state[i]='nom'
#            print(i)
#            print(S[i])
#            print(I[i])
#            print(R[i])
                
    H=500
    E=50
    T=PL1 
    Em=80
    Tm=PL2        
    
    totalcost=R[-1]*H + (10-x0[0])* 300 * E * T + m * Em * Tm
    return totalcost, S , I , R, PL1, PL2, nom,infect_rate,recover_rate, state

def objective(x0):
    totalcost,_,_,_,_,_,_,_,_,_ = DiseaseModel(x0)
    return totalcost
    
# # 'a': x0[0] ,'n':x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5]
#x0 = np.array([0.1 , 10 , 5 , 10 , 0.15 , 1 ])
   
x0 = [0.1 , 10 , 5 , 10 , 0.05 , 2 ]
result0=list(DiseaseModel(x0))
#print(result0[3])
x=list(range(0,t))

plt.plot(x,result0[1])
plt.xlabel('Time /days')
plt.ylabel('Number of S')
plt.show()

plt.plot(x,result0[2])
plt.xlabel('Time /days')
plt.ylabel('Number of I')
plt.show()

plt.plot(x,result0[3])
plt.xlabel('Time /days')
plt.ylabel('Number of R')
plt.show()

plt.plot(x,result0[7])
plt.xlabel('Time /days')
plt.ylabel('Infect rate')
plt.show()
#print(result0[7])

plt.plot(x,result0[8])
plt.xlabel('Time /days')
plt.ylabel('Recover rate')
plt.show()

print(result0[1][-1]+result0[2][-1]+result0[3][-1])
print('PL1:',result0[4],'PL2:',result0[5],'nom:',result0[6])

print('totalcost',result0[0])

#x0 = np.array([5,10,5,10,0.15,2])
#result0=DiseaseModell evolution to find the best a,n,v,m,alpha, IR
#bounds = [(4.(x0)
#print(result0)

# use differentia9, 5.1),(9.9, 10.1),(4.9, 5.1),(9.9, 10.1),(0.14, 0.16),(1.9, 2.1)]

#def XXX(x0):
#    totalXXX=x0[0]+x0[1]+x0[2]+x0[3]+x0[4]+x0[5]
#    return totalXXX
bounds = [(0, 0.2), (0, 10),(0, 5),(9, 11),(0, 0.2),(0, 10)]
#x0 = [0.1 , 10 , 5 , 10 , 0.15 , 2 ]
##
result = differential_evolution(objective, bounds, maxiter=1000)
##
##print(result.x, result.fun)
# # 'a': x0[0] ,'n':x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5]
# 
print('a=',result.x[0])
print('n=',result.x[1])
print('v=',result.x[2])
print('m=',result.x[3])
print('alpha=',result.x[4])
print('IR=',result.x[5])
print('cost=',result.fun)
print(result0[9])