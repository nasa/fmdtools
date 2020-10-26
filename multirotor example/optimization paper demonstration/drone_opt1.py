# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:17:31 2020

@author: danie
"""

import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_mdl import *
import time

params={'start': [0.0,0.0, 10, 10], 'target': [0, 150, 160, 160], 'safe': [0, 50, 10, 10], # areas
        'loc':'rural',
        'flightplan':{ 1:[0,0,50], 2:[100, 200, 50], 3:[100, 100, 85], 4:[-25, 150, 20],5:[75, 300, 20],6:[0, 300, 20], 7:[0,0,50], 8:[0,0,0] },
        'bat':'series-split',                           #'monolithic', 'series-split', 'paralel-split', 'split-both'
        'linearch':'quad',                              #quad, hex, oct
        'respolicy':{'bat':'emland','line':'emland'}}   #continue, to_home, to_nearest, emland
mdl = Drone(params=params)

# Design Model
def calc_des(mdl):
    batcostdict = {'monolithic':0, 'series-split':50000, 'parallel-split':50000, 'split-both':100000}
    linecostdict = {'quad':0, 'hex':100000, 'oct':200000}
    descost = batcostdict[mdl.params['bat']] + linecostdict[mdl.params['linearch']]
    return descost
desC2 = calc_des(mdl)
def x_to_dcost(xdes):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    batcostdict = {'monolithic':0, 'series-split':50000, 'parallel-split':50000, 'split-both':100000}
    linecostdict = {'quad':0, 'hex':100000, 'oct':200000}
    #print(xdes[0])
    descost = batcostdict[bats[xdes[0]]] + linecostdict[linarchs[xdes[1]]]
    return descost
xdes1 = [0, 1]
desC1 = x_to_dcost(xdes1)
print(desC1)
# Operations Model
# Obj - flight time
# Constraints   - batteries stay above 20% (to avoid damage) #postitive means violation
#               - no faults at end of simulation # True means faults exists (violation)
#               - cannot fly above 122 m (400 ft) #Positive means violation ??
def calc_oper(mdl):
    endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)
    opercost = endresults_nom['classification']['expected cost']
    g_soc = 20 - mdlhist['functions']['StoreEE']['soc'][-1]
    g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist['flows']['DOFs']['elev']-122 if i>0])
    return opercost, g_soc, g_faults, g_max_height
def x_to_ocost(xdes, xoper):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(xoper[0], sq, start[0:2]+[0])
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':'to_home','line':'to_home'}, 'target':target,'safe':safe,'start':start, 'loc':'rural' }
    #params = {'bat': bats[xdes[0]], 'linearch': linarchs[xdes[1]], 'flightplan': fp,
    #           'target': target, 'safe': safe, 'start': start}
    mdl = Drone(params=params)
    return calc_oper(mdl)

xoper1 = [122] #in m or ft?
desO1 = x_to_ocost(xdes1, xoper1)
print(desO1)
# Resilience Model
def calc_res(mdl):
    app = SampleApproach(mdl, faults='single-component', phases={'forward'})
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost
def x_to_rcost(xdes, xoper, xres):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(xoper[0], sq, start[0:2]+[0])
    
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':respols[xres[0]],'line':respols[xres[1]]}, 'target':target,'safe':safe,'start':start, 'loc':'rural' }
    mdl = Drone(params=params)
    return calc_res(mdl)

xres1 = [0, 0]
desR1 = x_to_rcost(xdes1, xoper1, xres1)
print(desR1)
#creates model from design variables
def x_to_mdl(x):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(x[2], sq, start[0:2]+[0])
    
    params = {'bat':bats[x[0]], 'linearch':linarchs[x[1]], 'flightplan':fp, 'respolicy':{'bat':respols[x[3]],'line':respols[x[4]]}, 'target':target,'safe':safe,'start':start, 'loc':'rural' }
    mdl = Drone(params=params)
    return mdl
# all-in-one-model
def x_to_cost(x):
    mdl = x_to_mdl(x)
    dcost = calc_des(mdl)
    oper = calc_oper(mdl)
    rcost = calc_res(mdl)
    return dcost + oper[0] + rcost, oper[1:]



