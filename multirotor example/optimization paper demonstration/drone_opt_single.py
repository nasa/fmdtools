## This file has the single MOO optimization framework for the drone model.
# Overall, it is a combinatorial optimization problem.
# Grid search brute force is used as optimization algo and handling constraints with Penalty method
# Because of using grid search, the height variable has been discritized to gain computational efficiency at
# the expense of accuracy.
# We use the Weighted Tchebycheff method to get the Pareto frontier between (Design + Oper) Cost vs Failure Cost
import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
#import GA.geneticalgorithm.geneticalgorithm as ga
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
ULXbound=(slice(0, 3, 1), slice(0, 2, 1), slice(10, 122, 10), slice(0, 3, 1), slice(0, 3, 1)) #specifies ranges + iter num for each var
#ULXRes = [0,0] # Default Res. Pol. in Upper level (nominal env) is o continue
# Approx Min and max feasible values for each cost models (Obtained from data analysis)
desC0 = [0, 300000]
operC0 = [-630000, -37171.5989]
resC0 = [5245622.35, 310771934]
#The normalized utopia values (global optimal)
u1 = 0.0018
u2 = 0
normalize=True
loc='rural'

# Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
weights = pd.Series(np.arange(0, 1.1, 0.1)) # Weights on obj 1: Design + Oper Cost
obj1_w = []
obj2_w = []
for ix in range(len(weights)):
    w1 = weights.iloc[ix]  # Weights on obj 1: Design + Oper Cost
    w2 = 1 - w1 # Weights on obj 2: Failure Cost
    ulparams = (desC0, operC0, resC0,normalize, w1, w2,u1,u2,loc)
    ULoptmodel = optimize.brute(AAO_f, ULXbound, args=ulparams, full_output=True, finish=optimize.fmin)
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