#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods for comparing parallelism in the Pump example.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from examples.pump.ex_pump import Pump, PumpParam
import fmdtools.sim.propagate as propagate

import time
import multiprocessing as mp
import multiprocess as ms


def close_pool(pool):
    """Close a pool process."""
    pool.close()
    pool.terminate()


def delay_test(delays=[i for i in range(0, 100, 10)]):
    """
    Delay parameter use-case described in notebook.

    Uses pool.map to manually simulate the model several times over parameters we want
    to run.
    """
    pool = mp.Pool(4)
    resultslist = pool.map(one_delay_helper, delays)
    close_pool(pool)
    return resultslist


def one_delay_helper(delay):
    """Generate output over the given input delays."""
    mdl = Pump(p=PumpParam(delay=delay))
    endclasses, mdlhists = propagate.one_fault(mdl, 'export_water', 'block')
    return endclasses


def compare_pools(mdl, app, pools, staged=False, track=False, verbose=True,
                  track_times='all'):
    """
    Compare the performance for given a model and sampling approach.

    Parameters
    ----------
    mdl : Model
        fmdtools model
    app : FaultSample
        SampleApproach of fault scenarios to simulate the model over using
        propagate.fault_sample
    pools : dict
        Dictionary of parallel/process pools {'poolname':pool} to compare. Pools must
        have a map function.
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
            ('times', [t1, ... tn])--only includes times defined in the vector
            [t1 ... tn]

    Returns
    -------
    exectimes : dict
        Dictionary of execution times for each of the different provided pools
    """
    exectimes = {}
    starttime = time.time()
    endclasses, mdlhists = propagate.fault_sample(mdl, app, pool=False, staged=staged,
                                                  showprogress=False,
                                                  track_times=track_times,
                                                  desired_result={})
    exectime_single = time.time() - starttime
    if verbose:
        print("single-thread exec time: "+str(exectime_single))
    exectimes['single'] = exectime_single

    for pool in pools:
        starttime = time.time()
        loc_kwargs = dict(pool=pools[pool],
                          staged=staged,
                          showprogress=False,
                          track_times=track_times,
                          desired_result={},
                          close_pool=False)
        endclasses, mdlhists = propagate.fault_sample(mdl, app, **loc_kwargs)
        exectime_par = time.time() - starttime
        if verbose:
            print(pool+" exec time: "+str(exectime_par))
        exectimes[pool] = exectime_par
    return exectimes


def run_model():
    mdl = Pump()
    endclasses, mdlhist=propagate.nominal(mdl)
    return endclasses

def parallel_mc(iters=10):
    """run for performance evaluation of model with asyncronous execution"""
    pool = mp.Pool(4)
    future_res = [pool.apply_async(run_model) for _ in range(iters)]
    res = [f.get() for f in future_res]
    close_pool(pool)
    return res


def parallel_mc2(iters=10):
    """Run to evaluate performance of model with map function."""
    pool = mp.Pool(4)
    models = [Pump() for i in range(iters)]
    result_list = pool.map(propagate.nominal, models)
    close_pool(pool)
    return result_list


def one_fault_helper(args):
    """Helper function for parallel_mc3 to run in pool.map."""
    mdl = Pump()
    endclasses, mdlhists = propagate.one_fault(mdl, args[0], args[1])
    return endclasses, mdlhists


def parallel_mc3():
    """Run for performance evaluation of model single fault simulation."""
    pool = mp.Pool(4)
    mdl = Pump()
    inputlist = [(fxn, fm) for fxn in mdl.fxns
                 for fm in mdl.fxns[fxn].m.faultmodes.keys()]
    resultslist = pool.map(one_fault_helper, inputlist)
    close_pool(pool)
    return resultslist


def instantiate_pools(cores, with_pathos=False):
    """Instantiate multiprocessing pools for comparison."""
    if with_pathos:
        from pathos.pools import ParallelPool, ProcessPool, SerialPool, ThreadPool
        return {'multiprocessing': mp.Pool(cores),
                'ProcessPool': ProcessPool(nodes=cores),
                'ParallelPool': ParallelPool(nodes=cores),
                'ThreadPool': ThreadPool(nodes=cores),
                'multiprocess': ms.Pool(cores)}
    else:
        return {'multiprocessing': mp.Pool(cores),
                'multiprocess': ms.Pool(cores)}


def terminate_pools(pools):
    """End pools after comparing them."""
    for pool in pools.values():
        pool.terminate()


if __name__ == '__main__':
    mdl = Pump(sp=dict(phases=(('start', 0, 4),
                               ('on', 5, 49),
                               ('end', 50, 500)),
                       end_time=500))
    from fmdtools.sim.sample import FaultDomain, FaultSample
    fd = FaultDomain(mdl)
    fd.add_all()
    fs = FaultSample(fd)
    fs.add_fault_phases('start', 'end', args=(3,))

    cores = 4
    pools = instantiate_pools(cores)

    print("STAGED + FULL MODEL TRACKING")
    mdl=Pump(sp = dict(phases=(('start',0,4),('on',5, 49),('end',50,500)), end_time=500),
             track='all')
    compare_pools(mdl, fs, pools, staged=True)
    print("NOT STAGED + FULL MODEL TRACKING")
    compare_pools(mdl, fs, pools, staged=False)

    print("STAGED + SOME TRACKING")
    mdl=Pump(sp = dict(phases=(('start',0,4),('on',5, 49),('end',50,500)), end_time=500),
             track={'flows': {'ee_1': 'all', 'wat_2': ['pressure', 'flowrate']}})
    compare_pools(mdl, fs, pools, staged=True)



    print("STAGED + FLOW TRACKING")
    mdl=Pump(sp = dict(phases=(('start',0,4),('on',5, 49),('end',50,500)), end_time=500),
             track='flows')
    compare_pools(mdl, fs, pools, staged=True)

    print("STAGED + FUNCTION TRACKING")
    mdl=Pump(sp = dict(phases=(('start',0,4),('on',5, 49),('end',50,500)), end_time=500),
             track='fxns')
    compare_pools(mdl, fs, pools, staged=True)

    print("STAGED + NO TRACKING")
    mdl=Pump(sp = dict(phases=(('start',0,4),('on',5, 49),('end',50,500)), end_time=500),
             track='none')
    compare_pools(mdl, fs, pools, staged=True)
    print("NOT STAGED + NO TRACKING")
    compare_pools(mdl, fs, pools, staged=False)
    terminate_pools(pools)
