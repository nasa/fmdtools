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
# from ray.util.multiprocessing import Pool as RayPool (need to figure out how to use Ray)
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
 
from parallelism_methods import compare_pools

if __name__=='__main__':
    #resultslist = parallel_mc3()
    
    print("--MANY SIMULATIONS--")
    mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
    app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':3})
    
    cores = 4
    pools = {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores)} #, 'Ray': RayPool(cores) }
    
    
    #print("STAGED + SOME TRACKING")
    #compare_pools(mdl,app,pools, staged=True, track={'flows':{'EE_1':'all', 'Wat_1':['pressure', 'flowrate']}})
    
    
    print("STAGED + FULL MODEL TRACKING")
    compare_pools(mdl,app,pools, staged=True, track='all')
    print("STAGED + FLOW TRACKING")
    compare_pools(mdl,app,pools, staged=True, track='flows')
    print("STAGED + FUNCTION TRACKING")
    compare_pools(mdl,app,pools, staged=True, track='functions')
    print("STAGED + NO TRACKING")
    compare_pools(mdl,app,pools, staged=True, track='none')
    
    # print("--FEW SIMULATIONS--")
    # mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,500]}, 'times':[0,20, 500], 'tstep':1})
    # app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':1})
    
    # cores = 4
    # pools = {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores) }
    
    # print("STAGED + MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=True, track=True)
    # print("STAGED + NO MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=True, track=False)
    # print("NOT STAGED + MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=False, track=True)
    # print("NOT STAGED + NO MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=False, track=False)
    
    # print("--FEW SHORT SIMULATIONS--")
    # mdl=Pump(params={'cost':{'repair'}, 'delay':10}, modelparams = {'phases':{'start':[0,5], 'on':[5, 50], 'end':[50,55]}, 'times':[0,20, 55], 'tstep':1})
    # app = SampleApproach(mdl,jointfaults={'faults':1},defaultsamp={'samp':'evenspacing','numpts':1})
    
    # cores = 4
    # pools = {'multiprocessing':mp.Pool(cores), 'ProcessPool':ProcessPool(nodes=cores), 'ParallelPool': ParallelPool(nodes=cores), 'ThreadPool':ThreadPool(nodes=cores), 'multiprocess':ms.Pool(cores) }
    
    # print("STAGED + MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=True, track=True)
    # print("STAGED + NO MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=True, track=False)
    # print("NOT STAGED + MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=False, track=True)
    # print("NOT STAGED + NO MODEL TRACKING")
    # compare_pools(mdl,app,pools, staged=False, track=False)
        
    
    # Note: ParallelPool seems slower, unclear whether processpool or threadpool is faster
    
    # Note: Parallelism works as expected when there is nothing to output
    #  (specifically, pathos ProcessPool is ~1/3  and ParallelPool is ~1/2 the time on 4 cores/nodes)
    # Ergo, to get parallelism to work, either the histories returned need to either be ~simpler~
    # or the metrics should be minimal



