# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:08:05 2020

@author: Daniel Hulse
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time
import fmdtools.sim.propagate as propagate
import fmdtools.analyze as an
from tank_model import Tank
from fmdtools.sim.approach import SampleApproach
from tank_opt import *
import matplotlib.pyplot as plt

# Nominal Run - nothing happens

# do nothing (default): params={'faultpolicy':{(a-1,b-1,c-1):(0,0) for a,b,c in np.ndindex((3,3,3))}}
# try to counteract leaks directly in valve: params={{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}}
#mdl = Tank(params={'capacity':20,'turnup':0.0, 'faultpolicy':{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}})

#endresults, resgraph, mdlhist = propagate.nominal(mdl)

#an.plot.mdlhistvals(mdlhist)
#an.graph.show(resgraph)

EA(popsize=30, iters=30, mutations=10, crossovers=3, numselect=15, verbose='iters')

#pop = seedpop()

#newpop = crossover(pop, 2)


#endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Import_Water','Leak', time=3)

#an.plot.mdlhistvals(mdlhist, fault='Leak', time=3, fxnflowvals={'Store_Water':['level', 'net_flow', 'coolingbuffer'], 'Tank_Sig':['indicator'], 'Valve1_Sig':['action']}, legend=False)
#an.graph.show(resgraph,faultscen='Leak', time=3)
#an.plot.mdlhistvals(mdlhist, time=3)

#result = minimize(x_to_descost, [1000, 0.5], method='trust-constr', bounds =((10, 100),(0,1)))

#xres=[0 for i in range(0,54)]
#cost = x_to_rcost2(xres)

#result = differential_evolution(x_to_rcost2, [(0,1) for i in range(0,27)]+[(10,100) for i in range(0,27)], maxiter=20, popsize=10)




#pop, values, time = EA(iters=100, popsize=30, mutations=14, numselect=15, verbose='iters')
#result, llargs = bilevel_opt()

#result, llargs, fhist, thist = alternating_opt()

#result, llargs = bilevel_opt()
#plt.plot(llargs['thist'],llargs['fhist'])

#x_to_rcost(llargs['ll_optx'][0],llargs['ll_optx'][1])
#result['x']

#x_to_rcost(result[1]['ll_optx'][0],result[1]['ll_optx'][1], xdes=[420,1])