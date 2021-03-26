# -*- coding: utf-8 -*-
"""
pickle tests
Created on Wed Mar 24 16:19:59 2021

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
#IE = ImportEE([{'current':1.0, 'voltage':1.0}])
#mdl = Pump()


#mdl.flows["EE_1"]._attributes

#check_pickleability(IE)

#check_pickleability(mdl)
#check_model_pickleability(mdl)


#pickle.dump( IE, open( "fxn_save.p", "wb" ) )
#IE_loaded = pickle.load( open( "fxn_save.p", "rb" ) )

#pickle.dump( mdl, open( "mdl_save.p", "wb" ) )
#mdl_loaded = pickle.load( open( "mdl_save.p", "rb" ) )

## Compare simulation results
#endresults, resgraph, mdlhist=propagate.nominal(mdl, track=True)
#endresults_loaded, resgraph_loaded, mdlhist_loaded=propagate.nominal(mdl_loaded, track=True)
#plot graph
#rd.graph.show(resgraph)
#rd.graph.show(resgraph_loaded)
#plot the flows over time
#rd.plot.mdlhistvals(mdlhist, 'Nominal')
#rd.plot.mdlhistvals(mdlhist_loaded, 'Nominal')
# both plots give equivalent outputs!!


## attempting parallelism?
# Note: Presently these work when used in the console but not in a script?!?!?

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
    resultslist = parallel_mc3()
    
    print("--MANY SIMULATIONS--")
    mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
    app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':10})
    
    cores = 4
    pools = {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores) }
    
    print("STAGED + MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=True, track=True)
    print("STAGED + NO MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=True, track=False)
    print("NOT STAGED + MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=False, track=True)
    print("NOT STAGED + NO MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=False, track=False)
    
    print("--FEW SIMULATIONS--")
    mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
    app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':1})
    
    cores = 4
    pools = {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores) }
    
    print("STAGED + MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=True, track=True)
    print("STAGED + NO MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=True, track=False)
    print("NOT STAGED + MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=False, track=True)
    print("NOT STAGED + NO MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=False, track=False)
        
    
    # Note: ParallelPool seems slower, unclear whether processpool or threadpool is faster
    
    # Note: Parallelism works as expected when there is nothing to output
    #  (specifically, pathos ProcessPool is ~1/3  and ParallelPool is ~1/2 the time on 4 cores/nodes)
    # Ergo, to get parallelism to work, either the histories returned need to either be ~simpler~
    # or the metrics should be minimal



