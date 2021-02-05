# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:08:05 2020

@author: Daniel Hulse
"""

import sys
sys.path.append('../')
import numpy as np

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from tank_model import Tank
from fmdtools.modeldef import SampleApproach


# Nominal Run - nothing happens

# do nothing (default): params={'faultpolicy':{(a-1,b-1,c-1):(0,0) for a,b,c in np.ndindex((3,3,3))}}
# try to counteract leaks directly in valve: params={{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}}
mdl = Tank(params={'capacity':20,'turnup':0.0, 'faultpolicy':{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}})

endresults, resgraph, mdlhist = propagate.nominal(mdl)

rd.plot.mdlhistvals(mdlhist)
rd.graph.show(resgraph)




endresults, resgraph, mdlhist = propagate.one_fault(mdl,'Import_Water','Leak', time=3)

rd.plot.mdlhistvals(mdlhist, fault='Leak', time=3, fxnflowvals={'Store_Water':['level', 'net_flow', 'coolingbuffer'], 'Tank_Sig':['indicator'], 'Valve1_Sig':['action']}, legend=False)
rd.graph.show(resgraph,faultscen='Leak', time=3)
#rd.plot.mdlhistvals(mdlhist, time=3)