# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 02:59:28 2020

@author: zhang
"""

import numpy as np
import csv
from scipy.optimize import minimize
import matplotlib
from matplotlib import pyplot as plt 
from scipy.optimize import differential_evolution

def DiseaseModel(x0):
    a=10
    b=1.25
    h=0.05
    
    t=7
    step=int(t*1/h)
    
    S=list(range(0,step)); S[0]=200
    I=list(range(0,step)); I[0]=800
    R=list(range(0,step)); R[0]=0
    N=S[0]+I[0]+R[0]
    
    PL1=0
    PL2=0
    
    a2=x0[0]
    n=x0[1]
    v=x0[2]
    m=x0[3]
    alpha=x0[4]
    IR=x0[5]
    
    b2=b/2
       
    for i in range(1,step):
    # PL1 and PL2 both triggered
        if S[i-1]/N > alpha and (h * (a * S[i-1] * I[i-1] / N - I[i-1]/b)) > IR:
            c =( m+n )/ m
            m = m+n
        
            S[i]= S[i-1] - h * (a2 * S[i-1] * I[i-1] / N - v )
            I[i]= I[i-1] + h * (a2 * S[i-1] * I[i-1] / N - I[i-1]/b2)
            R[i]= R[i-1] + h * (c * I[i-1]/b2 + v)
            if R[i]>N:
                R[i]=N
        
            PL1=PL1+1
            PL2=PL2+1
    # PL1 triggered
        elif S[i-1]/N > alpha :
            S[i]= S[i-1] - h * (a2 * S[i-1] * I[i-1] / N - v)
            I[i]= I[i-1] + h * (a2 * S[i-1] * I[i-1] / N - I[i-1]/b2 )
            R[i]= R[i-1] + h * (I[i-1]/b2 + v)
            if R[i]>N:
                R[i]=N
        
        
            PL1=PL1+1
    # PL2 triggered     
        elif h * (a * S[i-1] * I[i-1] / N - I[i-1]/b)> IR : 
            c =( m+n )/ m
            m = m+n
        
            S[i]= S[i-1] - h * (a * S[i-1] * I[i-1] / N)
            I[i]= I[i-1] + h * (a * S[i-1] * I[i-1] / N - I[i-1]/b2)
            R[i]= R[i-1] + h * (c * I[i-1]/b2)
            if R[i]>N:
                R[i]=N
        
            PL2=PL2+1
    #     nominal state
        else:
            S[i]= S[i-1] - h * (a * S[i-1] * I[i-1] / N)
            I[i]= I[i-1] + h * (a * S[i-1] * I[i-1] / N - I[i-1]/b)
            R[i]= R[i-1] + h * (I[i-1]/b)
            if R[i]>N:
                R[i]=N
                
    H=500
    E=50
    T=PL1 
    Em=80
    Tm=PL2        
    return R[-1]*H + (10-x0[0])* 300 * E * PL1 + m * Em * PL2

print(PL1,PL2)
# print(m)        
# print(R)        

x=list(range(0,step))
plt.plot(x,S)
plt.show()
plt.plot(x,I)
plt.show()
plt.plot(x,R)
plt.show()
