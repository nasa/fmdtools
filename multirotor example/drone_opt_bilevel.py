## This file has the bi-level optimization framework for the drone model.
# Upper level considers the nominal scenario, optimizing battery, rotor config and oper height,
# minimizing design and operational cost.
# Lower level considers simulated fault environment, optimizing resilience policy, minimizing failure cost.
# Overall, it is a combinatorial optimization problem, with upper level is MIP (Discrete battery and rotor choices;
# continuous height variable) and lower level is IP (Discrete resilience policy)
# Grid search brute force is used as optimization algo and handling constraints with Penalty method
# Because of using grid search, the upper level height variable has been discritized to gain computational efficiency at
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

# Initializing design variables and parameters
ULXbound=(slice(0, 3, 1), slice(0, 2, 1), slice(10, 122, 10))
#ULXbound=np.array([[0,3],[0,2],[10,122]])
#ULXtype=np.array([['int'],['int'],['real']])
#ULXRes = [0,0] # Default Res. Pol. in Upper level (nominal env) is o continue
# Approx Min and max feasible values for each cost models (Obtained from data analysis)
desC0 = [0, 300000]
operC0 = [-630000, -37171.5989]
resC0 = [171426.3, 55932536.24]
ulparams = (desC0, operC0, resC0)
# Defining lower level objetive function
def LLf(ll_x, *llparams):
    xdes, xoper, resC0 = llparams
    xres = ll_x
    # xdes = [int(xdes[0]), int(xdes[1])]
    # xres = [int(ll_x[0]), int(ll_x[1])]
    resC = x_to_rcost(xdes, xoper, xres)
    nresC = (resC-resC0[0])/(resC0[1]-resC0[0])
    LLobj =nresC
    return LLobj

# Defining the lower level optimization model (Using brute force exhaustive grid search algortihm)
def LLmodel(xdes, xoper, resC0):
    LLXbound = (slice(0, 3, 1), slice(0, 3, 1))
    llparams = (xdes, xoper, resC0)
    LLoptmodel = optimize.brute(LLf, LLXbound, args=llparams, full_output=True, finish=None)
   #  xbat = pd.Series(xdes[0])
   #  xrot = pd.Series(xdes[1])
   #  xoper = pd.Series(xoper)
   #  xrespolbat = pd.Series([0, 1, 2, 3])
   #  xrespolrot = xrespolbat
   #  xrespolbat_grid = pd.concat([pd.concat([xrespolbat.repeat(xrespolrot.size)] * 1, ignore_index=True)],
   #                        ignore_index=True)
   #  xrespolrot_grid = pd.concat([pd.concat([xrespolrot] * (xrespolbat.size), ignore_index=True)],
   #                      ignore_index=True)
   #  xbat_grid = pd.concat([pd.concat([xbat.repeat(xrespolbat_grid.size)] * 1, ignore_index=True)],
   #                        ignore_index=True)
   #  xrot_grid = pd.concat([pd.concat([xrot.repeat(xrespolbat_grid.size)] * 1, ignore_index=True)],
   #                        ignore_index=True)
   #  xoper_grid = pd.concat([pd.concat([xoper.repeat(xrespolbat_grid.size)] * 1, ignore_index=True)],
   #                        ignore_index=True)
   #  col_names = ["ResPolBat", "ResPolRot","Bat","Rot","Height"]
   #  ll_x_mat = pd.concat([xrespolbat_grid, xrespolrot_grid, xbat_grid, xrot_grid, xoper_grid], axis=1, ignore_index=True)
   #  ll_x_mat = pd.DataFrame(ll_x_mat.values, columns=col_names)
   # # index = pd.Series(np.arange(0, xrespolbat_grid.size, 1))
   #  Cost_eval = pd.DataFrame(columns=["index", "resC"])
   #  for ix in range(len(ll_x_mat)):
   #      xres = [ll_x_mat.iloc[ix, 0], ll_x_mat.iloc[ix, 1]]
   #      xdes = [ll_x_mat.iloc[ix, 2], ll_x_mat.iloc[ix, 3]]
   #      xoper = [ll_x_mat.iloc[ix, 4]]
   #      LLobj = LLf(xres, xdes, xoper)
   #      Cost_eval.loc[ix, ['index']] = ix
   #      Cost_eval.loc[ix, ['resC']] = LLobj
   #
   #  Cost_eval_opt = Cost_eval[['resC']].min()
   #  ix_Cost_eval_opt= Cost_eval['resC']== Cost_eval_opt['resC']
   #  #ix_Cost_eval_opt = Cost_eval[['resC']].idxmin()
   #  ll_x_opt = [ll_x_mat[ix_Cost_eval_opt == True]]
    return LLoptmodel

# Defining upper level objective function
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
        LLpen = 10000 # Giving a big penalty of lower level if upper level decision is infeasible
    else: # Calling lower level only if all the upper level contraints are feasible: Reducing redundant lower level iterations
        ULpen = 0
        LL_opt = LLmodel(xdes, xoper, resC0)
        LL_res_opt = LL_opt[0]
        LL_obj_opt = LL_opt[1]
        LLpen = 1000*LL_obj_opt # with increasing penalty term, optimal design decision is provided with lower risk to failure
        #LLpen = LL_obj_opt**2

    #Penalized obj func.(both upper and lower level): Double Penalty method
    #LLpen =0
    ULobj = ndesC + noperC + ULpen + LLpen #normalized design and oper cost with penalty value
    return ULobj

# Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
ULoptmodel = optimize.brute(ULf, ULXbound, args=ulparams, full_output=True, finish=optimize.fmin)
UL_xopt = np.around(ULoptmodel[0])
UL_fopt = np.around(ULoptmodel[1], decimals= 4)
xdes_opt = [int(UL_xopt[0]), int(UL_xopt[1])]
xoper_opt = [UL_xopt[2]]
desC_opt = x_to_dcost(xdes_opt)
operC_opt = x_to_ocost(xdes_opt, xoper_opt)
#Running final lower level solution for the upper level optimal solution
LL_opt = LLmodel(xdes_opt, xoper_opt, resC0)
LL_xopt = LL_opt[0]
LL_fopt = np.around(LL_opt[1], decimals= 4)
xres_opt = [int(LL_xopt[0]), int(LL_xopt[1])]
resC_opt = x_to_rcost(xdes_opt, xoper_opt, xres_opt)
print("#####################Bi-level approach###############################")
print("Upper level optimal solution:")
print(UL_xopt)
print(UL_fopt)
print(desC_opt)
print(operC_opt)
print("#####################################################################")
print("Lower level optimal solution:")
print(LL_xopt)
print(LL_fopt)
print(resC_opt)

# GA model
# algorithm_param = {'max_num_iteration': 2,
#                    'population_size':100,
#                    'mutation_probability':0.1,
#                    'elit_ratio': 0.1,
#                    'crossover_probability': 0.5,
#                    'parents_portion': 0.3,
#                    'crossover_type':'uniform',
#                    'max_iteration_without_improv':None}
#
# model=ga.geneticalgorithm(function=ULf,dimension=3,variable_type_mixed=ULXtype,variable_boundaries=ULXbound, algorithm_parameters=algorithm_param)
#
# model.run()




