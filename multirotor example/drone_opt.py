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
def x_to_dcost(xdes):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    batcostdict = {'monolithic':0, 'series-split':50000, 'parallel-split':50000, 'split-both':100000}
    linecostdict = {'quad':0, 'hex':100000, 'oct':200000}
    descost = batcostdict[bats[xdes[0]]] + linecostdict[linarchs[xdes[1]]]
    return descost

xdes1 = [0, 1]
desC1 = x_to_dcost(xdes1)
print(desC1)
# Operations Model
# Obj - flight time
# Constraints   - batteries stay above 20% (to avoid damage)
#               - no faults at end of simulation
#               - cannot fly above 122 m (400 ft)
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
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':'continue','line':'continue'}, 'target':target,'safe':safe,'start':start, 'loc':'rural'}
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
    
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':respols[xres[0]],'line':respols[xres[1]]}, 'target':target,'safe':safe,'start':start,'loc':'rural', }
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
    
    params = {'bat':bats[x[0]], 'linearch':linarchs[x[1]], 'flightplan':fp, 'respolicy':{'bat':respols[x[3]],'line':respols[x[4]]}, 'target':target,'safe':safe,'start':start,'loc':'rural', }
    mdl = Drone(params=params)
    return mdl
# all-in-one-model
def x_to_cost(x):
    mdl = x_to_mdl(x)
    dcost = calc_des(mdl)
    oper = calc_oper(mdl)
    rcost = calc_res(mdl)
    return dcost + oper[0] + rcost, oper[1:]



def plot_nomtraj(mdlhist, params, title='Trajectory'):
    xnom=mdlhist['flows']['DOFs']['x']
    ynom=mdlhist['flows']['DOFs']['y']
    znom=mdlhist['flows']['DOFs']['elev']
    
    time = mdlhist['time']
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    plt.show()

def plot_faulttraj(mdlhist, params):
    xnom=mdlhist['nominal']['flows']['DOFs']['x']
    ynom=mdlhist['nominal']['flows']['DOFs']['y']
    znom=mdlhist['nominal']['flows']['DOFs']['elev']
    #
    x=mdlhist['faulty']['flows']['DOFs']['x']
    y=mdlhist['faulty']['flows']['DOFs']['y']
    z=mdlhist['faulty']['flows']['DOFs']['elev']
    
    time = mdlhist['nominal']['time']
    
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)
    ax2.plot(x,y,z)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title('Fault response to RFpropbreak fault at t=20')
    ax2.legend(['Nominal Flightpath','Faulty Flighpath'], loc=4)
    #
    plt.show()
    
def plot_xy(mdlhist, endresults):
    xnom=mdlhist['flows']['DOFs']['x']
    ynom=mdlhist['flows']['DOFs']['y']
    znom=mdlhist['flows']['DOFs']['elev']
    plt.figure()
    plt.plot(xnom,ynom)
    
    
    xviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    yviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    xunviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    yunviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    
    plt.scatter(xviewed,yviewed, color='red')
    plt.scatter(xunviewed,yunviewed, color='grey')
    
    plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue')
    plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red')
    plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow')
    
    
    plt.show()


xdes1 = [3,2]
xoper1 = [65]
xres1 = [0,0]

a,b,c,d = x_to_ocost(xdes1, xoper1)

mdl = x_to_mdl([0,2,100,0,0])


endresults, resgraph, mdlhist = propagate.nominal(mdl)

rd.plot.mdlhistvals(mdlhist, fxnflowvals={'StoreEE':'soc'})








