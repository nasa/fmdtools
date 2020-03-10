# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:51:38 2020

@author: zhang
"""

import numpy as np
import csv
from scipy.optimize import minimize
import matplotlib
from matplotlib import pyplot as plt 

# a=10;
# b=1.25;
# h=0.05;
# s=[10];
# I=[90];
# R=[0];
# nStep=140;
# N=[100];
# day=linspace(0,7,141);

# x = np.array([10])
# x[0]=1

# nominal state 
# How S changes: 𝑑𝑆/𝑑𝑡=−𝑎𝑆𝐼/𝑁
# How I changes: 𝑑𝐼/𝑑𝑡=𝑎𝑆𝐼/𝑁-𝐼/𝑏
# How R changes: 𝑑𝑅/𝑑𝑡=𝐼/𝑏
# 𝑁=𝑆+𝐼+𝑅

a=0.2
b=10

# time /day
t=160
# total step

#step=int(t*1/h)



# x=list(range(0,t))
# print(x,x[0],x[-1],)


S=list(range(0,t)); S[0]=900
I=list(range(0,t)); I[0]=100
R=list(range(0,t)); R[0]=0
N=S[0]+I[0]+R[0]


# print(S,S[0],S[9]); print(len(S)); print(s[-1])
# print(S,S[0],S[1])
# print(I,I[0],I[1])
# print(R,R[0])

for i in range(1,t):
#     print(i,S[i],I[i],R[i])
    S[i]= S[i-1] - (a * S[i-1] * I[i-1] / N) 
    I[i]= I[i-1] + (a * S[i-1] * I[i-1] / N - I[i-1]/b)
    R[i]= R[i-1] + (I[i-1]/b) 
#     print(i,S[i],I[i],R[i])
#     print()
print(I)
x=list(range(0,t))

plt.plot(x,S)
plt.xlabel('Time /days')
plt.ylabel('Number of S')
plt.show()

plt.plot(x,I)
plt.xlabel('Time /days')
plt.ylabel('Number of I')
plt.show()

plt.plot(x,R)
plt.xlabel('Time /days')
plt.ylabel('Number of R')
plt.show()
print(S[-1]+I[-1]+R[-1])