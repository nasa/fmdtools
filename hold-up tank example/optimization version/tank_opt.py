# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:01:45 2021

@author: dhulse
"""

import sys
sys.path.append('../')
import numpy as np
import scipy as sp
import time
import itertools
import random
from scipy.optimize import minimize

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from tank_model import Tank
from fmdtools.modeldef import SampleApproach

params={'capacity':20, # size of the tank (10 - 100)
        'turnup':0.0,  # amount the pump can be "turned up" (0 - 1)
        'faultpolicy':{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}} #state-action pairs for resilience policy: what to given fault signals


def x_to_descost(xdes):
    return (xdes[0]-10)*1000 + (xdes[0]-10)**2*1000   + xdes[1]**2*10000

def x_to_rcost(xres1,xres2, xdes=[20,1]):
    fp = {(a-1,b-1,c-1):(xres1[i],xres2[i]) for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))}
    mdl=Tank(params={'capacity':xdes[0],'turnup':xdes[1],'faultpolicy':fp})
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost


def x_to_totcost(xdes, xres1, xres2):
    do_cost = x_to_descost(xdes)
    rescost = x_to_rcost(xres1, xres2, xdes=xdes)
    return do_cost, rescost
def x_to_totcost2(xdes, xres1, xres2):
    do_cost = x_to_descost(xdes)
    rescost = x_to_rcost(xres1, xres2, xdes=xdes)
    return do_cost + rescost
def x_to_totcost3(xdes, xres1, xres2): # total cost with crude penalty function
    do_cost = x_to_descost(xdes)
    rescost = x_to_rcost(xres1, xres2, xdes=xdes)
    pen = 0
    if xdes[0]<10: pen+=1e5*(10-xdes)**2
    if xdes[0]>100: pen+=1e5*(100-xdes)**2
    if xdes[1]<0: pen+=1e5*(xdes[1])**2
    if xdes[1]>1: pen+=1e5*(1-xdes[1])**2
    return do_cost + rescost + pen

def lower_level(xdes, args):
    do_cost = x_to_descost(xdes) 
    bestsol, rcost, time = EA(args=args, xdes=xdes)
    return do_cost + rcost

def bilevel_opt():
    xdes = [20, 1]
    args = {'seed':seedpop(), 'll_opt':1e6, 'll_optx':[]}
    result = minimize(lower_level, xdes, method='Powell', bounds =((10, 100),(0,1)), callback=callbackF1, args = args, options={'direc':[[0,1],[1,0]], 'disp':True})
    return result, args

def alternating_opt():
    xdes = np.array([20, 1])
    args = {'seed':seedpop(), 'll_opt':1e6, 'll_optx':[]}
    newmin = 100000000
    lastmin = 1000000001
    bestsol = np.zeros((2,27))
    last_run = False
    for n in range(10):
        result = minimize(x_to_totcost2, [np.round(xdes[0],1), np.round(xdes[1],1)], method='Powell', callback=callbackF1, args = (bestsol[0],bestsol[1]), options={'direc':[[0,1],[1,0]], 'disp':True})
        #result = minimize(x_to_totcost2, xdes, method='Powell', callback=callbackF1,  args = (bestsol[0],bestsol[1]), options={'disp':True,'ftol': 0.000001})
        # doesn't really work: trust-constr, SLSQP, Nelder-Mead (doesn't respect bounds), COBYLA (a bit better, but converges poorly), 
        # powell does okay but I'm not sure if it's actually searching the x-direction
        xdes = result['x']
        bestsol, rcost, time = EA(args=args, popsize=100, mutations=20,numselect=30, iters=50, xdes = xdes)
        lastmin = newmin; newmin = x_to_descost(xdes) + rcost
        print(n, newmin, lastmin-newmin)
        if lastmin - newmin <0.1: 
            if last_run:    break
            else:           last_run = True
        else:               last_run = False
    return result, args


def callbackF(Xdes, result):
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(result['nit'], Xdes[0], Xdes[1], result['fun']))
def callbackF1(Xdes):
    print(Xdes)

def EA(popsize=10, iters=10, mutations=3, numselect=5, args={}, xdes=[20,1]):
    starttime = time.time()
    if args: pop=np.concatenate((args['seed'],seedpop(), randpop(popsize-3)))
    else:    pop=np.concatenate((seedpop(), randpop(popsize-3)))
    values = np.array([x_to_rcost(x[0],x[1], xdes=xdes) for x in pop])
    for i in range(iters):
        goodpop, goodvals = select(pop, values, numselect)
        newpop =  np.concatenate((randpop(popsize-len(goodvals)-mutations), mutepop(goodpop, mutations)))
        newvals = np.array([x_to_rcost(x[0],x[1], xdes=xdes) for x in newpop])
        pop, values = np.concatenate((goodpop, newpop)), np.concatenate((goodvals, newvals))        
    minind = np.argmin(values)
    if args: args['seed'] = goodpop; args['ll_opt']= values[minind];  args['ll_optx']= pop[minind];
    return pop[minind], values[minind], time.time() - starttime
def randpop(popsize):
    return np.array([[[random.randint(-1,1) for a in range(0,27)],[random.randint(-1,1) for a in range(0,27)]] for i in range(0,popsize)])
def seedpop():
    donothing = np.zeros((2,27))
    adjustup = np.ones((2,27))
    adjustdown = -np.ones((2,27))
    return np.array([donothing, adjustup, adjustdown])
def mutepop(goodpop, mutations):
    to_mutate = np.random.choice([i for i in range(len(goodpop))], size=mutations, replace=False)
    return np.array([permute(solution) for solution in goodpop[to_mutate]])
def permute(solution):
    solution[random.randint(0,1)][random.randint(0,26)]=random.randint(-1,1)
    return solution
def select(solutions, values, numselect):
    selection = np.argsort(values)[0:numselect]
    return solutions[selection], values[selection]

def time_rcost():
    starttime = time.time()
    mdl=Tank()
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return time.time() - starttime





