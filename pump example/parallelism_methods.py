# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:23:14 2021

@author: dhulse
"""

import sys
paths = sys.path
if paths[1]!='../':
    sys.path=[sys.path[0]] + ['../'] + paths

from ex_pump import * 
from fmdtools.modeldef import SampleApproach
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd

import time
import pickle

import multiprocessing as mp
import multiprocess as ms

from pathos.pools import ParallelPool, ProcessPool, SerialPool, ThreadPool

""" Delay parameter use-case described in notebook"""
def delay_test(delays =  [i for i in range(0,100,10)]):
    pool = mp.Pool(4)
    resultslist = pool.map(one_delay_helper, delays)
    return resultslist
def one_delay_helper(delay):
    mdl = Pump({'cost': {'repair', 'water'}, 'delay': delay, 'units': 'hrs'})
    endresults,resgraph, mdlhists = propagate.one_fault(mdl, 'ExportWater', 'block')
    return endresults

def run_model():
    mdl = Pump()
    endresults, resgraph, mdlhist=propagate.nominal(mdl)
    return endresults

def parallel_mc(iters=10):
    pool = mp.Pool(4)

    future_res = [pool.apply_async(run_model) for _ in range(iters)]
    res = [f.get() for f in future_res]

    return res

def parallel_mc2(iters=10):
    pool = mp.Pool(4)
    
    
    models = [Pump() for i in range(iters)]
    result_list = pool.map(propagate.nominal, models)
    return result_list

def one_fault_helper(args):
    mdl = Pump()
    endresults,resgraph, mdlhists = propagate.one_fault(mdl, args[0], args[1])
    return endresults,resgraph, mdlhists

def parallel_mc3():
    pool = mp.Pool(4)
    mdl = Pump()
    inputlist = [(fxn,fm) for fxn in mdl.fxns for fm in mdl.fxns[fxn].faultmodes.keys()]
    resultslist = pool.map(one_fault_helper, inputlist)
    return resultslist

def compare_pools(mdl, app, pools, staged=False, track=False):
    starttime = time.time()
    endclasses, mdlhists = propagate.approach(mdl,app, pool=False, staged = staged, track=track)
    exectime_single = time.time() - starttime
    print("single-thread exec time: "+str(exectime_single))
    
    for pool in pools:
        starttime = time.time()
        endclasses, mdlhists = propagate.approach(mdl,app, pool=pools[pool], staged = staged, track=track)
        exectime_par = time.time() - starttime
        print(pool+" exec time: "+str(exectime_par))

if __name__=='__main__':
    a=1