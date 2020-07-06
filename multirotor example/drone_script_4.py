# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:58:54 2020

@author: danie
"""

import sys
sys.path.append('../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
import time
from drone_opt import *


mdl = x_to_mdl([0,2,100,0,0])
params = mdl.params


endresults, resgraph, mdlhist = propagate.nominal(mdl)

rd.plot.mdlhistvals(mdlhist, fxnflowvals={'StoreEE':'soc'})

#plot_nomtraj(mdlhist, params)

#plot_xy(mdlhist, endresults)