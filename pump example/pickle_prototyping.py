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

import multiprocessing as mp

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
 
if __name__=='__main__':
    resultslist = parallel_mc3()
    mdl=Pump()
    app = SampleApproach(mdl,jointfaults={'faults':5},defaultsamp={'samp':'evenspacing','numpts':20})
    
    starttime = time.time()
    endclasses, mdlhists = propagate.approach(mdl,app, parallel=False)
    exectime_single = time.time() - starttime
    print("single-thread exec time: "+str(exectime_single))
    
    # test parallel execution
    starttime = time.time()
    endclasses, mdlhists = propagate.approach(mdl,app, parallel=True, poolsize=3)
    exectime_par = time.time() - starttime
    print("parallel exec time: "+str(exectime_par))
    
    

#mdlhists,endclasses = propagate.single_faults(mdl)

#res_list = parallel_mc2(iters=1)

#res = parallel_mc(iters=1)



