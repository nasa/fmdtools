# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:17:31 2020

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

params={'flightplan':{ 1:[0,0,50], 2:[100, 200, 50], 3:[100, 100, 85], 4:[-25, 150, 20],5:[75, 300, 20],6:[0, 300, 20], 7:[0,0,50], 8:[0,0,0] },
        'bat':'series-split',                           #'monolithic', 'series-split', 'paralel-split', 'split-both'
        'linearch':'quad',                              #quad, hex, oct
        'respolicy':{'bat':'emland','line':'emland'}}   #continue, to_home, to_nearest, emland
mdl = Drone(params=params)

# Design Model
batcostdict = {'monolithic':0, 'series-split':50000, 'paralel-split':50000, 'split-both':100000}
linecostdict = {'quad':0, 'hex':100000, 'oct':200000}
descost = batcostdict[params['bat']] + linecostdict[params['linearch']]

# Operations Model
endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)
opercost = endresults_nom['classification']['expected cost']

# Resilience Model

app = SampleApproach(mdl, faults='single-component', phases={'forward'})
endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
rescost = rd.process.totalcost(endclasses)