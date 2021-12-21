# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:54:15 2020

@author: dhulse
"""
import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
import time
from drone_opt import *


# Design Model
xdes1 = [0, 1]
desC1 = x_to_dcost(xdes1)
print(desC1)

# Operational Model
xoper1 = [122] #in m or ft?
desO1 = x_to_ocost(xdes1, xoper1)
print(desO1)

#Resilience Model
xres1 = [0, 0]
desR1 = x_to_rcost(xdes1, xoper1, xres1)
print(desR1)

#all-in-one model
xdes1 = [3,2]
xoper1 = [65]
xres1 = [0,0]

a,b,c,d = x_to_ocost(xdes1, xoper1)

mdl = x_to_mdl([0,2,100,0,0])


endresults, resgraph, mdlhist = propagate.nominal(mdl)

rd.plot.mdlhistvals(mdlhist, fxnflowvals={'StoreEE':'soc'})