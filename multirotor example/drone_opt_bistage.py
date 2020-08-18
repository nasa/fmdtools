## This file has the two-stage optimization framework for the drone model.
# Stage 1 considers the nominal scenario, optimizing battery, rotor config and oper height,
# minimizing design and operational cost.
# Stage 2 considers simulated fault environment, optimizing resilience policy, minimizing failure cost.
# Overall, it is a combinatorial optimization problem, with upper level is MIP (Discrete battery and rotor choices;
# continuous height variable) and lower level is IP (Discrete resilience policy)
# Grid search brute force is used as optimization algo and handling constraints with Penalty method
# Because of using grid search, the Stage 1 height variable has been discritized to gain computational efficiency at
# the expense of accuracy.
import sys
sys.path.append('../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import GA.geneticalgorithm.geneticalgorithm as ga
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import pandas as pd
import numpy as np

from drone_mdl import *
from drone_opt import *
import time
import timeit

start = timeit.default_timer()
######################### Stage 1 optimization ####################################
# Initializing design variables and parameters
ULXbound=(slice(0, 3, 1), slice(0, 2, 1), slice(10, 122, 10))
#ULXRes = [0,0] # Default Res. Pol. in Upper level (nominal env) is o continue
# Approx Min and max feasible values for each cost models (Obtained from data analysis)
desC0 = [0, 300000]
operC0 = [-630000, -37171.5989]
resC0 = [171426.3, 55932536.24]
ulparams = (desC0, operC0, resC0)

# Defining Stage1 objective function
def ULf(X, *ulparams):
    xdes = [int(X[0]),int(X[1])]
    xoper = [X[2]]
    desC0, operC0, resC0 = ulparams
    desC = x_to_dcost(xdes)
    operC = x_to_ocost(xdes, xoper)
    #Normalizing obj function to avoid issue on magnitudes
    ndesC = (desC-desC0[0])/(desC0[1]-desC0[0])
    noperC =(operC[0]-operC0[0])/(operC0[1]-operC0[0])
    #Constraints validation: >0 or 1(Boolean) means violation and is penalized in Obj Func
    c_batlife = operC[1]
    if operC[2] == True:
        c_faults = 1
    else:
        c_faults = 0
    c_maxh = operC[3]
    #Penalizing obj function with upper level contraints
    if ((operC[1] > 0 or operC[2] == True) or (operC[3] > 0)):  # Infeasible design if any above constraints violated
        ULpen = c_batlife**2 + 1000*c_faults + c_maxh**2 # Exterior Penalty method
        #LLpen = 10000 # Giving a big penalty of lower level if upper level decision is infeasible
    else: # Calling lower level only if all the upper level contraints are feasible: Reducing redundant lower level iterations
        ULpen = 0

    #Penalized obj func.
    #LLpen =0
    ULobj = ndesC + noperC + ULpen #normalized design and oper cost with penalty value
    return ULobj

# Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
ULoptmodel = optimize.brute(ULf, ULXbound, args=ulparams, full_output=True, finish=optimize.fmin)
UL_xopt = np.around(ULoptmodel[0])
UL_fopt = np.around(ULoptmodel[1], decimals= 4)
xdes_opt = [int(UL_xopt[0]), int(UL_xopt[1])]
xoper_opt = [UL_xopt[2]]
desC_opt = x_to_dcost(xdes_opt)
operC_opt = x_to_ocost(xdes_opt, xoper_opt)

#################### Stage 2 optimization##############################################
# Defining Stage2 objective function
def LLf(ll_x, *llparams):
    xdes, xoper, resC0 = llparams
    xres = ll_x
    resC = x_to_rcost(xdes, xoper, xres)
    nresC = (resC-resC0[0])/(resC0[1]-resC0[0])
    LLobj =nresC
    return LLobj

# Defining the Stage2 optimization model (Using brute force exhaustive grid search algortihm)
def LLmodel(xdes, xoper, resC0):
    LLXbound = (slice(0, 3, 1), slice(0, 3, 1))
    llparams = (xdes, xoper, resC0)
    LLoptmodel = optimize.brute(LLf, LLXbound, args=llparams, full_output=True, finish=None)
    return LLoptmodel
#Running Stage 2 model for the upper level optimal solution
LL_opt = LLmodel(xdes_opt, xoper_opt, resC0)
LL_xopt = LL_opt[0]
LL_fopt = np.around(LL_opt[1], decimals= 4)
xres_opt = [int(LL_xopt[0]), int(LL_xopt[1])]
resC_opt = x_to_rcost(xdes_opt, xoper_opt, xres_opt)
print("#####################Two-Stage approach###############################")
print("Stage 1 optimal solution:")
print(UL_xopt)
print(UL_fopt)
print(desC_opt)
print(operC_opt)
print("#####################################################################")
print("Stage 2 optimal solution:")
print(LL_xopt)
print(LL_fopt)
print(resC_opt)

###################################################################################
stop = timeit.default_timer()
total_time = stop - start

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d hrs:%d mins:%d secs.\n" % (hours, mins, secs))

