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
        'flightplan':{ 1:[0,0,50], 2:[100, 200, 50], 3:[100, 100, 85], 4:[-25, 150, 20],5:[75, 300, 20],6:[0, 300, 20], 7:[0,0,50], 8:[0,0,0] },
        'bat':'series-split',                           #'monolithic', 'series-split', 'paralel-split', 'split-both'
        'linearch':'quad',                              #quad, hex, oct
        'respolicy':{'bat':'emland','line':'emland'}}   #continue, to_home, to_nearest, emland
mdl = Drone(params=params)

# Design Model
def calc_des(mdl):
    batcostdict = {'monolithic':0, 'series-split':50000, 'paralel-split':50000, 'split-both':100000}
    linecostdict = {'quad':0, 'hex':100000, 'oct':200000}
    descost = batcostdict[mdl.params['bat']] + linecostdict[mdl.params['linearch']]
    return descost

# Operations Model
def calc_oper(mdl):
    endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)
    opercost = endresults_nom['classification']['expected cost']
    return opercost

# Resilience Model
def calc_res(mdl):
    app = SampleApproach(mdl, faults='single-component', phases={'forward'})
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost

#creates model from design variables
def x_to_mdl(x):
    bats = ['monolithic', 'series-split', 'paralel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(x[2], sq, start[0:2]+[0])
    
    params = {'bat':bats[x[0]], 'linearch':linarchs[x[1]], 'flightplan':fp, 'respolicy':{'bat':respols[x[3]],'line':respols[x[4]]}, 'target':target,'safe':safe,'start':start }
    mdl = Drone(params=params)
    return mdl

x=[1,1,100,1,1]
mdl = x_to_mdl(x)

endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)

#descost = calc_des(mdl)
#opercost = calc_oper(mdl)
#rescost = calc_res(mdl)


fp = mdl.params['flightplan']

fig1 = plt.figure()

ax2 = fig1.add_subplot(111, projection='3d')

xnom=mdlhist['flows']['DOFs']['x']
ynom=mdlhist['flows']['DOFs']['y']
znom=mdlhist['flows']['DOFs']['elev']
time = mdlhist['time']


for goal,loc in fp.items():
    ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
    ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
    ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
ax2.plot(xnom,ynom,znom)
fig2 = plt.figure()

x_traj = [a[0] for k,a in fp.items()]
y_traj = [a[1] for k,a in fp.items()]

plt.plot(x_traj, y_traj)
plt.plot(mdlhist['flows']['DOFs']['x'], mdlhist['flows']['DOFs']['y'])

xviewed = [x for (x,y),view in endresults_nom['classification']['viewed'].items() if view!='unviewed']
yviewed = [y for (x,y),view in endresults_nom['classification']['viewed'].items() if view!='unviewed']
xunviewed = [x for (x,y),view in endresults_nom['classification']['viewed'].items() if view=='unviewed']
yunviewed = [y for (x,y),view in endresults_nom['classification']['viewed'].items() if view=='unviewed']

plt.scatter(xviewed,yviewed, color='red')
plt.scatter(xunviewed,yunviewed, color='grey')

plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue')
plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red')
plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow')

