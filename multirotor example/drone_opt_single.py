## This file has the single MOO optimization framework for the drone model.
# Overall, it is a combinatorial optimization problem.
# Grid search brute force is used as optimization algo and handling constraints with Penalty method
# Because of using grid search, the height variable has been discritized to gain computational efficiency at
# the expense of accuracy.
# We use the Weighted Tchebycheff method to get the Pareto frontier between (Design + Oper) Cost vs Failure Cost
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
# Initializing design variables and parameters
ULXbound=(slice(0, 3, 1), slice(0, 2, 1), slice(10, 122, 10), slice(0, 3, 1), slice(0, 3, 1))
#ULXRes = [0,0] # Default Res. Pol. in Upper level (nominal env) is o continue
# Approx Min and max feasible values for each cost models (Obtained from data analysis)
desC0 = [0, 300000]
operC0 = [-630000, -37171.5989]
resC0 = [171426.3, 55932536.24]

# Defining Stage1 objective function
def ULf(X, *ulparams):
    xdes = [int(X[0]),int(X[1])]
    xoper = [X[2]]
    xres = [int(X[3]), int(X[4])]
    desC0, operC0, resC0, w1, w2 = ulparams
    desC = x_to_dcost(xdes)
    operC = x_to_ocost(xdes, xoper)
    resC = x_to_rcost(xdes, xoper, xres)

    #Normalizing obj function to avoid issue on magnitudes
    ndesC = (desC-desC0[0])/(desC0[1]-desC0[0])
    noperC =(operC[0]-operC0[0])/(operC0[1]-operC0[0])
    nresC = (resC - resC0[0]) / (resC0[1] - resC0[0])
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

    #Penalized weighted Tchebycheff obj func.
    #The normalized utopia values (global optimal)
    u1 = 0.0018
    u2 = 0
    ULobj = max((w1*(ndesC + noperC) -u1), (w2*nresC- u2)) + ULpen #normalized cost with penalty value
    return ULobj

# Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
weights = pd.Series(np.arange(0, 1.1, 0.1)) # Weights on obj 1: Design + Oper Cost
obj1_w = []
obj2_w = []
for ix in range(len(weights)):
    w1 = weights.iloc[ix]  # Weights on obj 1: Design + Oper Cost
    w2 = 1 - w1 # Weights on obj 2: Failure Cost
    ulparams = (desC0, operC0, resC0, w1, w2)
    ULoptmodel = optimize.brute(ULf, ULXbound, args=ulparams, full_output=True, finish=optimize.fmin)
    UL_xopt = abs(np.around(ULoptmodel[0]))
    UL_fopt = np.around(ULoptmodel[1], decimals=4)
    xdes_opt = [int(UL_xopt[0]), int(UL_xopt[1])]
    xoper_opt = [UL_xopt[2]]
    xres_opt = [int(UL_xopt[3]), int(UL_xopt[4])]

    desC_opt = x_to_dcost(xdes_opt)
    operC_opt = x_to_ocost(xdes_opt, xoper_opt)
    resC_opt = x_to_rcost(xdes_opt, xoper_opt, xres_opt)
    # desC_opt = ([0])
    # operC_opt = ([11])
    # resC_opt = ([12])
    obj1_w.append(desC_opt+operC_opt[0])
    obj2_w.append(resC_opt)

    print("#####################Single-Stage MOO approach###############################")
    sys.stdout.write("Optimal solution at weights w1 = %.1f and w2 =%.1f\n" % (w1, w2))
    print(UL_xopt)
    print(UL_fopt)
    print(desC_opt)
    print(operC_opt)
    print(resC_opt)

#Ploting Pareto frontier
plt.plot(obj1_w,obj2_w,'*b')
plt.xlabel("f1: Design + Operation Cost")
plt.ylabel("f2: Failure Cost")
#plt.legend(('Des','Oper','Res'))
plt.title("Pareto Frontier")

###################################################################################
stop = timeit.default_timer()
total_time = stop - start

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d hrs:%d mins:%d secs.\n" % (hours, mins, secs))