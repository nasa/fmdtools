# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:23:14 2021

@author: dhulse
"""

import sys, os
sys.path.append(os.path.join('..'))

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

def compare_pools(mdl, app, pools, staged=False, track=False, verbose= True, track_times='all'):
    exectimes = {}
    starttime = time.time()
    endclasses, mdlhists = propagate.approach(mdl,app, pool=False, staged = staged, track=track, showprogress=False, track_times=track_times)
    exectime_single = time.time() - starttime
    if verbose: print("single-thread exec time: "+str(exectime_single))
    exectimes['single'] = exectime_single
    
    for pool in pools:
        starttime = time.time()
        endclasses, mdlhists = propagate.approach(mdl,app, pool=pools[pool], staged = staged, track=track, showprogress=False, track_times=track_times)
        exectime_par = time.time() - starttime
        if verbose: print(pool+" exec time: "+str(exectime_par))
        exectimes[pool] = exectime_par
    return exectimes


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

def instantiate_pools(cores):
    from pathos.pools import ParallelPool, ProcessPool, SerialPool, ThreadPool
    from parallelism_methods import compare_pools
    return  {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores)} #, 'Ray': RayPool(cores) }


if __name__=='__main__':
    mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
    app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':3})
    
    cores = 4
    
    print("STAGED + SOME TRACKING")
    compare_pools(mdl,app,instantiate_pools(cores), staged=True, track={'flows':{'EE_1':'all', 'Wat_1':['pressure', 'flowrate']}})
    print("STAGED + FULL MODEL TRACKING")
    compare_pools(mdl,app,instantiate_pools(cores), staged=True, track='all')
    print("STAGED + FLOW TRACKING")
    compare_pools(mdl,app,instantiate_pools(cores), staged=True, track='flows')
    print("STAGED + FUNCTION TRACKING")
    compare_pools(mdl,app,instantiate_pools(cores), staged=True, track='functions')
    print("STAGED + NO TRACKING")
    compare_pools(mdl,app,instantiate_pools(cores), staged=True, track='none')