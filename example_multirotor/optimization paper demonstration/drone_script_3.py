# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:13:13 2020

@author: hulsed
"""

import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
import time

#scenlist=fp.listinitfaults(graph, mdl.times)
mdl = Drone()

#endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)

#rd.graph.show(mdl)

#rd.plot.mdlhistvals(mdlhist, fxnflowvals={'DOFs':['vertvel', 'uppwr', 'elev'], 'CtlDOF':['vel']}, units=['m/s','N','m','m/s'])

#rd.plot.mdlhist(mdlhist, fxnflows=['DOFs', 'CtlDOF'], units=['m/s','m/s','N','N','m','m','m','m/s', 'unitless'])

scenseq = {2: {'AffectDOF': ['LRctlbreak'],'StoreEE': ['nocharge']}, 5:{'ManageHealth': ['falseemland'], 'HoldPayload': ['break']}}

endresults_nom, resgraph, mdlhist =propagate.mult_fault(mdl, scenseq)

#rd.plot.mdlhistvals(mdlhist)

