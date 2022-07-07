# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:23:14 2021

@author: dhulse
"""

import sys, os
sys.path.insert(0, os.path.join('..'))

from ex_pump import * 
from fmdtools.modeldef import SampleApproach
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd

import time
import pickle

import multiprocessing as mp
import multiprocess as ms

from pathos.pools import ParallelPool, ProcessPool, SerialPool, ThreadPool


def delay_test(delays =  [i for i in range(0,100,10)]):
    """ Delay parameter use-case described in notebook--here we are using pool.map to manually
    simulate the model several times over parameters we want to run."""
    pool = mp.Pool(4)
    resultslist = pool.map(one_delay_helper, delays)
    return resultslist
def one_delay_helper(delay):
    """This helper function is used by pool.map to generate output over the given 
    input delays"""
    mdl = Pump({'cost': {'repair', 'water'}, 'delay': delay, 'units': 'hrs'})
    endresults,resgraph, mdlhists = propagate.one_fault(mdl, 'ExportWater', 'block')
    return endresults

def compare_pools(mdl, app, pools, staged=False, track=False, verbose= True, track_times='all'):
    """
    Method to compare the performance of different process pools on a given model/fault sampling approach

    Parameters
    ----------
    mdl : Model
        fmdtools model
    app : SampleApproach
        SampleApproach of fault scenarios to simulate the model over using propagate.approach
    pools : dict
        Dictionary of parallel/process pools {'poolname':pool} to compare. Pools must have a map function.
    staged : bool, optional
        Whether to use staged execution on the model. The default is False.
    track : str/bool, optional
        Parameters to track (e.g. 'functions','flows','all'). The default is False.
    verbose : bool, optional
        Whether to output execution time. The default is True.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]

    Returns
    -------
    exectimes : dict
        Dictionary of execution times for each of the different provided pools
    """
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
    """run for performance evaluation of model with asyncronous execution"""
    pool = mp.Pool(4)
    future_res = [pool.apply_async(run_model) for _ in range(iters)]
    res = [f.get() for f in future_res]
    return res

def parallel_mc2(iters=10):
    """run for performance evaluation of model with map function"""
    pool = mp.Pool(4)
    models = [Pump() for i in range(iters)]
    result_list = pool.map(propagate.nominal, models)
    return result_list

def one_fault_helper(args):
    """Helper function for parallel_mc3 to run in pool.map"""
    mdl = Pump()
    endresults,resgraph, mdlhists = propagate.one_fault(mdl, args[0], args[1])
    return endresults,resgraph, mdlhists

def parallel_mc3():
    """run for performance evaluation of model single fault simulation"""
    pool = mp.Pool(4)
    mdl = Pump()
    inputlist = [(fxn,fm) for fxn in mdl.fxns for fm in mdl.fxns[fxn].faultmodes.keys()]
    resultslist = pool.map(one_fault_helper, inputlist)
    return resultslist

def instantiate_pools(cores):
    """Used to instantiate multiprocessing pools for comparison"""
    from pathos.pools import ParallelPool, ProcessPool, SerialPool, ThreadPool
    from parallelism_methods import compare_pools
    return  {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores)} #, 'Ray': RayPool(cores) }


if __name__=='__main__':
    mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,4], 'on':[5, 49], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
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