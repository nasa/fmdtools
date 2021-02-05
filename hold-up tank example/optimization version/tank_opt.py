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

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from tank_model import Tank
from fmdtools.modeldef import SampleApproach

params={'capacity':20, # size of the tank
        'turnup':0.0,  # amount the pump can be "turned up"
        'faultpolicy':{(a-1,b-1,c-1):(-(a-1),-(c-1)) for a,b,c in np.ndindex((3,3,3))}} #state-action pairs for resilience policy: what to given fault signals


def x_to_descost(xdes):
    return (xdes[0]-10)*1000 + xdes[1]**2*10000

def x_to_ocost(xdes):
    # is this even necessary?
    #mdl=Tank(params={'capacity':20,'turnup':1.0,'faultpolicy':{(a-1,b-1,c-1):(0,0) for a,b,c in np.ndindex((3,3,3))}})
    #endresults, resgraph, mdlhist = propagate.nominal(mdl)
    return (xdes[0]-10)**2*100  + xdes[1]**2*10000

def x_to_rcost(xdes,xres):
    fp = {(a-1,b-1,c-1):(xres[i],xres[i+27]) for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))}
    mdl=Tank(params={'capacity':xdes[0],'turnup':xdes[1],'faultpolicy':fp})
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost

def x_to_rcost2(xres):
    fp = {(a-1,b-1,c-1):(xres[i],xres[i+27]) for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))}
    mdl=Tank(params={'capacity':20,'turnup':1,'faultpolicy':fp})
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost
def x_to_rcost3(xres1,xres2):
    fp = {(a-1,b-1,c-1):(xres1,xres2) for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))}
    mdl=Tank(params={'capacity':20,'turnup':1,'faultpolicy':fp})
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return rescost

def EA(popsize=10, iters=10):
    starttime = time.time()
    pop = seedpop() + randpop(popsize-3)
    
    for i in range(iters):
        values = [x_to_rcost(x[0],x[1]) for x in pop]
        goodpop, goodvals = select(pop, values)
        
    
    return results, opt_hist, len(Xvals), time.time() - starttime
def randpop(popsize):
    return np.array([[[random.randint(-1,1) for a in range(0,27)],[random.randint(-1,1) for a in range(0,27)]] for i in range(0,popsize)])
def seedpop():
    donothing = np.zeros((2,27))
    adjustup = np.ones((2,27))
    adjustdown = -np.ones((2,27))
    return np.array([donothing, adjustup, adjustdown])
def permute(solution):
    solution[[random.randint(0,27)][random.randint(0,1)]]=random.randint(-1,1)
    return solution
def select(solutions, values, quantile=0.5):
    selection = [np.quantile(solutions,quantile)>solutions]
    return solutions[selection], values[selection]

def brute_search(): 
    #uh oh--the space of the lower level problem is way too big for an exhaustive search!
    # 3 actions, 54 states = 3^54 possible combinations--far too many to even enumerate in a list!!!
    starttime = time.time()
    Xvals = [ e for e in itertools.product(*(range(-1,2,1) for x in range(0,54)))]
    results = dict(); opt_hist = []
    for X in Xvals:
        rescost=x_to_rcost2(X)
        if not opt_hist:                     opt_hist= [[rescost, X]]
        elif rescost < opt_hist[-1][0]:     opt_hist.append([rescost, X])
            
    return results, opt_hist, len(Xvals), time.time() - starttime





