# -*- coding: utf-8 -*-
"""
Functions to propagate faults through a user-defined fault model.

Main Methods:

- :func:`nominal()`: Runs the model over time in the nominal scenario.
- :func:`one_fault()`:  Runs one fault in the model at a specified time.
- :func:`sequence()`: Runs arbitrary scenario of fault modes at specified times.
- :func:`single_faults()`: Creates and propagates a list of failure scenarios in a model
  over given model times.
- :func:`fault_sample`: Injects and propagates faults defined by a FaultSample.
- :func:`parameter_sample`: Simulates a model over a range of parameters defined by a
  ParameterSample
- :func:`nested_sample`: Injects and propagates faults in the model defined by a
  given SampleApproach over a range of parameters defined by a ParameterSample.

Shared Method Parameters:

- :data:`sim_kwargs`: Simulation keyword arguments.
- :data:`run_kwargs`: Run keyword arguments.
- :data:`mult_kwargs`: Multi-scenario keyword arguments

Private Methods:

- :func:`list_init_faults()`: Creates a list of single-fault scenarios for the graph,
  given the modes set up in the fault model
- :func:`prop_one_scen()`: Runs a fault scenario in the model over time
- :func:`save_helper()`: Helper function for inline results saving.
- :func:`unpack _res_list()`: Helper function for unpacking results
- :func:`exec_nom_par`: Helper function for executing nominal scenarios in parallel
- :func:`exec_nom_helper`: Helper function for executing nominal scenarios
- :func:`nom_helper`: Helper function for initial run of nominal scenario
- :func:`scenlist_helper`: Helper function for `approach`
- :func:`exec_scen_par`:  Helper function for executing the scenario in parallel
- :func:`exec_scen`: Executes a scenario and generates results and classifications given
  a model and nominal model history
- :func:`check_hist_memory`: Checks if the memory will be exhausted given the size of
  the mdlhist and number of scenarios
- :func:`check_mdl_memory`: Raises exception if model size is too large.
- :func:`check_overwrite`: Checks if file can be overwritten
- :func:`check_end_condition`: Helper function for `prop_one_scen` to end simulation
  earlier.
- :func:`get_result`: Helper function for `prop_one_scen` to get result at specific
  timestep.
- :func:`get_endclass_vars`: Helper function for `get_result`

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

from fmdtools.define.base import get_var, t_key
from fmdtools.sim.sample import SampleApproach
from fmdtools.sim.scenario import Sequence, Scenario, SingleFaultScenario
from fmdtools.analyze.common import create_indiv_filename, file_check
from fmdtools.analyze.result import Result
from fmdtools.analyze.history import History
from fmdtools.analyze.phases import from_hist

import numpy as np
import copy
import tqdm
import os

# DEFAULT ARGUMENTS
sim_kwargs = {'desired_result': 'endclass',
              'staged': False,
              'cut_hist': True,
              'run_stochastic': False,
              'use_end_condition': True,
              'warn_faults': True}
"""
Simulation keyword arguments.

Parameters
----------
desired_result : dict/str/list
    Desired quantities to return in the first argument.
    Options are:

    - 'endclass': a dict returned by find_classification (default)
    - 'endfaults': a dict of returned fault modes and their propagation, e.g., ::

        {'endfaults':faultdict, 'faultprops':faultpropdict}

    - 'graph'/'flowgraph'/etc: a networkx graph of the model with fault modes
      superimposed
    - 'fxnname.varname': variable values to get
    - a list of the above arguments (for multiple at the end)
    - a dict of lists (for multiple over time), e.g., ::

        {time:[varnames,... 'endclass']}

    The default is 'all'.

run_stochastic : bool
    Whether to run stochastic behaviors or use default values. Default is False.
    Can set as 'track_pdf' to calculate/track the probability densities of random
    states over time.
staged : bool, optional
    Whether to inject the faults in a copy of the nominal model at the fault time
    (True) or instantiate a new model for the fault (False). Setting to True
    roughly halves execution time. The default is False.
warn_faults : bool
    Whether to produce a warning when faults occur in a nominal sim.
"""


def unpack_sim_kwargs(**kwargs):
    """Unpack :data:`sim_kwargs` parameters for :func:`prop_one_scen`."""
    return (kwargs.get(k, v) for k, v in sim_kwargs.items())


def pack_sim_kwargs(**kwargs):
    """Create :data:`sim_kwargs` for :func:`prop_one_scen`."""
    return {k: kwargs.get(k, v) for k, v in sim_kwargs.items()}


run_kwargs = {'save_args': {},
              'mdl_kwargs': {},
              'protect': True}
"""
Run keyword arguments.

Parameters
----------
protect : bool
    Whether or not to protect the model object via copying.
    Options:

    - True (default): re-instances the model so that multiple simulations can
      be run successively without causing problems
    - False : Thus, the model object that is returned can be modified and
      analyzed if needed

save_args : dict (optional)
    Dictionary specifying if/how to save results. Default is {}, which doesn't
    save anything.
    Has structure ::
        {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv}

    where mdlhistargs and endclassargs are dictionaries of save arguments, e.g.,::

    {'filename':'filename.npz', 'filetype':'npz', 'overwrite':True}

    and indiv is an (optional) bool specifying whether to save results individually
    (in a folder) or as a monolythic file.

mdl_kwargs: dict (optional)
    Parameter dictionary to be instantiated in the model prior to simulation.
    Has structure ::
        {"p": Parameter, "sp":SimParam, "track":track}

    Parameter dictionaries do not need to be complete (if incomplete).
"""


def pack_run_kwargs(**kwargs):
    """Create subset of run kwargs for :func:`nom_helper` and :data:`run_kwarg`."""
    return {k: copy.deepcopy(kwargs.get(k, v)) for k, v in run_kwargs.items()}


mult_kwargs = {'max_mem': 2e9,
               'showprogress': True,
               'pool': False,
               'close_pool': True}


"""
Multi-scenario keyword arguments.

Parameters
----------
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    max_mem : int
        Max memory (warns the user when memory is above threshold)
"""


def pack_mult_kwargs(**kwargs):
    """Create subset of mult kwargs."""
    return {k: kwargs.get(k, v) for k, v in mult_kwargs.items()}


def unpack_mult_kwargs(kwargs):
    """Unpack the mult kwarg parameters for the :func:`parameter_sample`."""
    return (kwargs.pop(k, v) for k, v in mult_kwargs.items())

# FAULT PROPAGATION


def nominal(mdl, **kwargs):
    """
    Run the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Simulable
        Model of the system
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        dict of result corresponding to desired_result, e.g. ::

            Result({'endclass': endclasses,
                    'endfaults': endfaults,
                    'varname': var,
                    t: {'endclass': endclasses...} ...})

    nomhist : History
        A History dict with a history of modelstates
    """
    result, mdlhist, _, mdl, t_end = nom_helper(mdl, None, cut_hist=True, **kwargs)
    save_helper(kwargs.get('save_args', {}), result, mdlhist)
    return result, mdlhist


def save_helper(save_args, endclass, mdlhist, indiv_id='', result_id=''):
    """
    Save results (helper function).

    Parameters
    ----------
    save_args : dict
        Dict with structure ::

            {'mdlhists': mdlhistargs,
             'endclass': endclassargs,
             'indiv': individual_saving}

        where mdlhistargs and endclassargs are dictionaries of arguments to Result.save
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and individual_saving is a bool (True/False)
    endclass : dict
        dict of end-state classifications (from simulation)
    mdlhist : dict
        dict of model histories (from simulation)
    """
    if 'mdlhists' in save_args:
        save_args['mdlhist'] = save_args.pop('mdlhists')
    if 'endclasses' in save_args:
        save_args['endclass'] = save_args.pop('endclasses')
    for save_arg in save_args:
        if save_arg not in {'mdlhist', 'endclass', 'indiv'}:
            raise Exception("Invalid key in save_args: "+save_arg)
    if save_args.get('indiv', False) and indiv_id:
        if 'endclass' in save_args:
            newfilename = create_indiv_filename(save_args['endclass']['filename'],
                                                indiv_id, splitchar="/")
            endclass.save(**{**save_args['endclass'], 'filename': newfilename},
                          result_id=result_id)
        if 'mdlhist' in save_args:
            newfilename = create_indiv_filename(save_args['mdlhist']['filename'],
                                                indiv_id, splitchar="/")
            mdlhist.save(**{**save_args['mdlhist'], 'filename': newfilename},
                         result_id=result_id)
    elif not save_args.get('indiv', False) and not indiv_id:
        if 'mdlhist' in save_args:
            mdlhist.save(**save_args['mdlhist'])
        if 'endclass' in save_args:
            endclass.save(**save_args['endclass'])


def parameter_sample(mdl, ps, **kwargs):
    """
    Simulate a set of nominal scenarios through a model.

    Useful for exploring/understanding the sets of parameters where the system will run
    nominally and/or fail.

    Parameters
    ----------
    mdl : Simulable
        Model to simulate
    ps: ParameterSample
        Parameter Sample defining the nominal scenarios to run the system over.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    nomresults: Result
        Result dict of result corresponding to desired result {'scenname': result}
    nomhists : History
        History of model histories, with structure {'scenname': mdlhist}
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    check_overwrite(kwargs['save_args'])
    kwargs['max_mem'], showprogress, pool, close_p = unpack_mult_kwargs(kwargs)
    num_scens = ps.num_scenarios()
    kwargs['num_scens'] = num_scens

    if pool:
        check_mdl_memory(mdl, num_scens, max_mem=kwargs['max_mem'])
        inputs = [(mdl, sc, name, kwargs) for name, sc in ps.named_scenarios().items()]
        res_list = list(tqdm.tqdm(pool.imap(exec_nom_par, inputs),
                                  total=len(inputs),
                                  disable=not (showprogress),
                                  desc="SCENARIOS COMPLETE"))
        n_results, n_mdlhists = unpack_res_list(ps.scenarios(), res_list)
    else:
        scennames = ps.scen_names()
        n_results = Result.fromkeys(scennames)
        n_mdlhists = History.fromkeys(scennames)
        for scenname, scen in tqdm.tqdm(ps.named_scenarios().items(),
                                        disable=not (showprogress),
                                        desc="SCENARIOS COMPLETE"):
            loc_kwargs = {**kwargs, 'use_end_condition': False}
            res, hist = exec_nom_helper(mdl, scen, scenname, **loc_kwargs)
            n_results[scenname], n_mdlhists[scenname] = res, hist
    save_helper(kwargs['save_args'], n_results, n_mdlhists)
    close_pool({'pool': pool, 'close_pool': close_p})
    return n_results.flatten(), n_mdlhists.flatten()


def unpack_res_list(scenlist, res_list):
    """Create result/history from output list."""
    results = Result()
    mdlhists = History()
    results.data = {scen.name: res_list[i][0] for i, scen in enumerate(scenlist)}
    mdlhists.data = {scen.name: res_list[i][1] for i, scen in enumerate(scenlist)}
    return results, mdlhists


def exec_nom_par(arg):
    """Execute a nominal scenario (helper function/interface for parallel pools)."""
    endclass, mdlhist = exec_nom_helper(arg[0], arg[1], arg[2],
                                        **{**arg[3], 'use_end_condition': False})
    return endclass, mdlhist


def exec_nom_helper(mdl, scen, name, mdl_kwargs={}, **kwargs):
    """Execute a nominal scenario (helper function)."""
    mdl_kwargs = {'p': {**mdl_kwargs.get('p', {}), **scen.p},
                  'sp': {**mdl_kwargs.get('sp', {}), **scen.sp},
                  'r': {**mdl_kwargs.get('r', {}), **scen.r}}
    mdl_run = mdl.new(**mdl_kwargs)
    result, mdlhist, _, t_end = prop_one_scen(mdl_run, scen, **kwargs)
    check_hist_memory(mdlhist, kwargs['num_scens'], max_mem=kwargs['max_mem'])
    save_helper(kwargs['save_args'], result, mdlhist, name, name)
    return result, mdlhist


def one_fault(mdl, *fxnfault, time=0, **kwargs):
    """
    Run one fault in the model at a specified time.

    Parameters
    ----------
    mdl : Simulable
        The model to inject the fault in.
    *fxnfault : str
        Has options:
        - 'fxnname', 'faultmode' when a Model is provided, or
        - 'faultmode' when a Block/Function is provided
    time : float, optional
        Time to inject fault. Must be in the range of model times
        (i.e. in range(0, end, mdl.sp.dt)). The default is 0.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        Result dict of result corresponding to desired_result, with structure,::

        Result({'nominal': nomresult, 'faulty': faultyresult})

    mdlhists : History
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'faulty': faulthist}
    """
    if len(fxnfault) == 2:
        fxnname, fault = fxnfault
    elif len(fxnfault) == 3:
        fxnname, fault, time = fxnfault
    else:
        fxnname, fault = mdl.name, fxnfault[0]
    seq = Sequence(faultseq={time: {fxnname: [fault]}})

    scen = SingleFaultScenario(sequence=seq,
                               fault=fault,
                               function=fxnname,
                               time=time,
                               rate=mdl.get_scen_rate(fxnname, fault, time))
    result, mdlhists = sequence(mdl, scen=scen, **kwargs)
    return result.flatten(), mdlhists.flatten()


def sequence(mdl, seq={}, faultseq={}, disturbances={}, scen={}, rate=np.nan,
             include_nominal=True, **kwargs):
    """
    Run a sequence of faults and disturbances in the model at given times.

    Parameters
    ----------
    mdl : Simulable
        The model to inject the fault in.
    seq : dict
        Scenario dict defining the scenario
        {time:{`faults`:faults, `disturbances`:disturbances}}
    faultseq : dict
        Dict of times and modes defining the fault scenario {time:{fxns: [modes]},}
    disturbances : dict
        Dict of times and states defining the disturbances in the scenario::

        {time: {path.to.state: stateval}}

    scen : Scenario, optional
        Scenario dictionary, if already constructed (for external calls)
    rate : float, optional
        Input rate for the sequence (must be calculated elsewhere)
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others

    Returns
    -------
    result: Result
        Result dict of result corresponding to desired_result, with structure,::

        Result({'nominal': nomresult, 'faulty': faultyresult})

    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time with
        structure: {'nominal': nomhist, 'faulty': faulthist}
    """
    sim_kwarg = pack_sim_kwargs(**kwargs)
    run_kwarg = pack_run_kwargs(**kwargs)

    if not scen:
        if not seq:
            seq = Sequence(faultseq=faultseq, disturbances=disturbances)
        scen = Scenario(sequence=seq,
                        rate=rate,
                        name='faulty',
                        times=tuple([*seq.keys()]))

    n_outs = nom_helper(mdl,
                        [min(scen.sequence)],
                        **{**sim_kwarg, 'use_end_condition': False},
                        **run_kwarg)
    nomresult, nomhist, nomscen, mdls, t_end_nom = n_outs

    mdl_f = [*mdls.values()][0]

    result, faulthist, _, t_end = prop_one_scen(mdl_f,
                                                scen,
                                                **sim_kwarg,
                                                nomhist=nomhist,
                                                nomresult=nomresult)
    if include_nominal:
        nomhist.cut(t_end_nom)
        mdlhists = History(nominal=nomhist, faulty=faulthist)
    save_helper(kwargs.get('save_args', {}), result, mdlhists)
    return result.flatten(), mdlhists.flatten()


def nom_helper(mdl, ctimes, protect=True, save_args={}, mdl_kwargs={}, scen={},
               warn_faults=True, **kwargs):
    """
    Run initial run of nominal scenario.

    Parameters
    ----------
    mdl : Simulable (object or class)
        Model of the system
    ctimes : float/list
        Times to copy the nominal model from
    protect : bool
        Whether or not to protect the model object via copying.
        Options:

        - True (default): re-instances the model so that multiple simulations can
        be run successively without causing problems
        - False : Thus, the model object that is returned can be modified and
        analyzed if needed
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't
        save anything.
        Has structure ::

            {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv}
    mdl_kwargs : dict, optional
        Model arameter dictionary.
        Has structure ::

            {"p": Parameter, "sp":SimParam, "track":track}
    scen : scenario, optional
        Scenario to use. The default is {}.
    warn_faults : bool
        choose whether to display a warning message if faults are identified during
        nominal runs. Default is True.
    **kwargs : kwargs
        :data:`sim_kwargs` simulation options for :func:`prop_one_scen`

    Returns
    -------
    result : Result
        results dict from nominal sim
    nommdlhist : History
        result history from nominal sim
    nomscen : dict
        nominal scenario dict
    mdls : list
        Models from copy time(s) ctimes
    t_end_nom : float
        Nominal simulation end time
    """
    staged = kwargs.get('staged', False)
    check_overwrite(save_args)
    # run model nominally, get relevant results
    if isinstance(mdl, type):
        mdl = mdl(**mdl_kwargs)
    elif protect or mdl_kwargs:
        mdl = mdl.new(**mdl_kwargs)

    if not scen:
        nomscen = Scenario(sequence=Sequence(disturbances=kwargs.get('disturbances',
                                                                     {})))
    else:
        nomscen = scen

    if staged:
        if type(ctimes) in [float, int]:
            ctimes = [ctimes]
        else:
            ctimes = ctimes
    else:
        ctimes = []

    result, nommdlhist, mdls, t_end_nom = prop_one_scen(mdl,
                                                        nomscen,
                                                        ctimes=ctimes,
                                                        **kwargs)

    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults) and warn_faults:
        print("Faults found during the nominal run " + str(endfaults))

    if not staged:
        mdls = {0: mdl.new(**mdl_kwargs)}

    return result, nommdlhist, nomscen, mdls, t_end_nom


def fault_sample(mdl, fs, include_nominal=True, get_phasemap=False, **kwargs):
    """
    Injects and propagates faults in the model defined by a FaultSample/SampleApproach.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == 'main':
            results, mdlhists = fault_sample(mdl, fs)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in.
    fs : FaultSample/SampleApproach
        FaultSample used to define the list of faults and sample time for the model.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    get_phasemap : bool, optional
        Whether to regenerate the FaultSample using new phase information.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    results : Result
        A Result dictionary with results desired from each scenario corresponding to
        desired_result over the set of scenarios.
    mdlhists : History
        A History dictionary with the tracked scenario (including the nominal)
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    n_outs = nom_helper(mdl,
                        fs.times(),
                        **{**kwargs, 'use_end_condition': False})
    nomresult, nomhist, nomscen, c_mdl, t_end_nom = n_outs
    scenlist = fs.scenarios()

    results, mdlhists = scenlist_helper(mdl,
                                        scenlist,
                                        c_mdl,
                                        **kwargs,
                                        nomhist=nomhist,
                                        nomresult=nomresult)

    if include_nominal:
        process_nominal(mdlhists, nomhist, results, nomresult, t_end_nom, **kwargs)
    save_helper(kwargs['save_args'], results, mdlhists)
    close_pool(kwargs)
    return results.flatten(), mdlhists.flatten()


#  pool=pool, close_pool=False, showprogress=False,

def fault_sample_from(mdl, faultdomains={}, faultsamples={}, get_phasemap=True,
                      scen={}, include_nominal=False, **kwargs):
    """
    Create and simulate a fault_sample from the given arguments.

    Use to generate and sample from phases in the same simulation.

    Parameters
    ----------
    mdl : Simulable
        Model to simulate
    faultdomains : dict
        Dict of arguments to SampleApproach.add_faultdomains
    faultsamples : dict
        Dict of arguments to SampleApproach.add_faultsamples
        FaultSamples to add to othe SampleApproach and their arguments.
        Has structure::
        {'fs_name': (*args, **kwargs)}
        where args and kwargs are arguments/kwargs to SampleApproach.add_faultsamples.
    get_phasemap : bool, optional
        Whether to generate the FaultSample from the phasemap. The default is True.
    scen : scenario, optional
        Scenario to use as nominal. The default is {}.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is False.
    **kwargs : kwargs
        kwargs to simulate over

    Returns
    -------
    res : Result
        A Result dictionary with results desired from each scenario corresponding to
        desired_result over the set of scenarios.
    hist : History
        A History dictionary with the tracked scenario (including the nominal)
    app : SampleApproach
        Generated SampleApproach
    """
    run_kwarg = pack_run_kwargs(**kwargs)
    sim_kwarg = pack_sim_kwargs(**kwargs)
    mult_kwarg = pack_mult_kwargs(**kwargs)
    loc_kwargs = {**sim_kwarg, **run_kwarg, 'staged': False}
    if not scen:
        mdl = mdl.new(**run_kwarg.get('mdl_kwargs', {}))
        _, nomhist, _, _, t_end = nom_helper(mdl, [], **loc_kwargs)
    else:
        mdl = mdl.new(p=scen.p, sp=scen.sp, r=scen.r)
        _, nomhist, _, t_end, = prop_one_scen(mdl, scen, **loc_kwargs)
    app = gen_sampleapproach(mdl, faultdomains, faultsamples, get_phasemap, nomhist)
    res, hist = fault_sample(mdl, app, **sim_kwarg, **run_kwargs, **mult_kwarg,
                             include_nominal=include_nominal)
    return res, hist, app


def process_nominal(mdlhists, nomhist, results, nomresult, t_end_nom, **kwargs):
    """Add/save nominal hists/result to overall hist/result."""
    nomhist.cut(t_end_nom)
    mdlhists['nominal'] = nomhist
    results['nominal'] = nomresult
    save_helper(kwargs.get('save_args', {}),
                nomresult,
                mdlhists['nominal'],
                indiv_id=str(len(results)-1),
                result_id='nominal')


def single_faults(mdl, times=[0.0], include_nominal=True, **kwargs):
    """
    Create and propagates a list of failure scenarios in a model.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == 'main':
            results, mdlhists = single_faults(mdl)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in
    times : list
        List of times to inject the single faults in. Default is [1.0]
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is True.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options

    Returns
    -------
    results: Result
        Result dict of result corresponding to desired_result {scenname:result}
    mdlhists : History
        History dict with the history of all tracked model states for each scenario
        (including the nominal)
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    n_outs = nom_helper(mdl,
                        times,
                        **{**kwargs, 'use_end_condition': False})
    nomresult, nomhist, nomscen, c_mdl, t_end_nom = n_outs

    scenlist = list_init_faults(mdl, times)
    results, mdlhists = scenlist_helper(mdl,
                                        scenlist,
                                        c_mdl,
                                        **kwargs,
                                        nomhist=nomhist,
                                        nomresult=nomresult)
    if include_nominal:
        process_nominal(mdlhists, nomhist, results, nomresult, t_end_nom, **kwargs)
    save_helper(kwargs['save_args'], results, mdlhists)
    close_pool(kwargs)
    return results.flatten(), mdlhists.flatten()


def scenlist_helper(mdl, scenlist, c_mdl, **kwargs):
    max_mem, showprogress, pool, close_p = unpack_mult_kwargs(kwargs)
    staged = kwargs.get('staged', False)
    mem, mem_profile = kwargs['nomhist'].get_memory()
    if mem * len(scenlist) > max_mem:
        raise Exception("Model history will be too large: "
                        + str(mem) + " > " + str(max_mem))
    results = Result()
    mdlhists = History()
    if pool:
        check_mdl_memory(mdl, len(scenlist), max_mem=max_mem)
        if staged:
            inputs = [(c_mdl[scen.time], scen, kwargs, str(i))
                      for i, scen in enumerate(scenlist)]
        else:
            inputs = [(c_mdl[0], scen, kwargs, str(i))
                      for i, scen in enumerate(scenlist)]
        res_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs),
                                  total=len(inputs),
                                  disable=not (showprogress),
                                  desc="SCENARIOS COMPLETE"))
        results, mdlhists = unpack_res_list(scenlist, res_list)
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist,
                                           disable=not (showprogress),
                                           desc="SCENARIOS COMPLETE")):
            name = scen.name
            if staged:
                mdl_i = c_mdl[scen.time].copy()
            else:
                mdl_i = c_mdl[0].new()
            ec, mh, t_end = exec_scen(mdl_i, scen, indiv_id=str(i), **kwargs)
            results[name], mdlhists[name] = ec, mh
    return results, mdlhists


def close_pool(kwargs):
    """Close pool to avoid memory problems."""
    if kwargs.get('pool', False) and kwargs.get('close_pool', True):
        kwargs['pool'].close()
        kwargs['pool'].terminate()
        kwargs['pool'].join()


def exec_scen_par(args):
    """Execute scenario (parallel execution helper function for pool.map)."""
    return exec_scen(args[0].copy(), args[1], **args[2], indiv_id=args[3])


def exec_scen(mdl, scen, save_args={}, indiv_id='', **kwargs):
    """
    Executes a scenario and generates results and classifications given a model and
    nominal model history.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in
    scen : scenario
        scenario used to define time and faults where the fault is to be injected
    save_args : dict
        Save dictionary to use in save_helper defining when/how to save the dictionary
    indiv_id : str
        ID str to insert into the file name (if saving individually)
    **kwargs : kwargs
        :data:`sim_kwargs` for :func:`prop_one_scen`
    """
    result, mdlhist, _, t_end, = prop_one_scen(mdl, scen, **kwargs)
    save_helper(save_args, result, mdlhist, indiv_id=indiv_id, result_id=str(scen.name))
    return result, mdlhist, t_end


def check_hist_memory(mdlhist, nscens, max_mem=2e9):
    """Check if the memory will be exhausted by the hist over the scenarios."""
    mem_total, mem_profile = mdlhist.get_memory()
    total_memory = int(mem_total) * int(nscens)
    if total_memory > max_mem:
        raise Exception("Mdlhist has size: " + str(mem_total)
                        + " bytes. With " + str(nscens)
                        + " scenarios, it is expected that this run will pass the"
                        + " user-defined max_mem=" + str(max_mem)
                        + " byte limit by a factor of: "+str(total_memory/max_mem)
                        + ". To avoid, use the track= option to track less information"
                        + " in the mdlhist")


def check_mdl_memory(mdl, nscens, max_mem=2e9):
    """Check if memory will be exhausted by the model over the scenarios."""
    mem_total, mem_profile = mdl.get_memory()
    total_memory = int(mem_total) * int(nscens)
    if total_memory > max_mem:
        raise Exception("Model has size: " + str(mem_total)
                        + " bytes. With "+str(nscens)
                        + " scenarios, it is expected that this run will pass the"
                        + " user-defined max_mem="+str(max_mem)
                        + " byte limit by a factor of: "+str(total_memory/max_mem)
                        + ". To avoid, increase mem_total, reduce the size of the model"
                        + "or number of scenarios, or run outside a parallel pool")


def check_overwrite(save_args):
    for arg, args in save_args.items():
        if arg != 'indiv':
            filename = args['filename']
            if args.get('filename', False):
                file_check(filename, args.get('overwrite', False))
            if save_args.get('indiv', False):
                last_split_index = filename.rfind(".")
                foldername = filename[:last_split_index]
                if not os.path.exists(foldername):
                    os.makedirs(foldername)


def nested_sample(mdl, ps, get_phasemap=False, faultdomains={}, faultsamples={},
                  include_nominal=True, **kwargs):
    """
    Simulate a set of fault scenarios within a ParameterSample.

    NOTE: When calling in a script/module using parallel=True, execute using the
    protection statement ::

        if __name__ == "main":
            results, mdlhists = nested_sample(mdl, ps)

    Otherwise, the method will keep spawning parallel processes.
    See multiprocessing documentation.

    Parameters
    ----------
    mdl : Simulable
        Model Object to use in the simulation.
    ps : ParameterSample
        Parameter Scenario defining the parameters the model will be run over
    get_phasemap : Bool/List/Dict, optional
        Whether to use nominal simulation phasemap to set up the SampleApproach.
    faultdomains : dict
        Dict of arguments to SampleApproach.add_faultdomains
    faultsamples : dict
        Dict of arguments to SampleApproach.add_faultsamples
        FaultSamples to add to othe SampleApproach and their arguments.
        Has structure::
        {'fs_name': (*args, **kwargs)}
        where args and kwargs are arguments/kwargs to SampleApproach.add_faultsamples.
    include_nominal : bool, optional
        Whether to return nominal hists/results back. Default is False.
    **kwargs : kwargs
        Additional keyword arguments, may include:

        - :data:`sim_kwargs` : kwargs
              Simulation options for :func:`prop_one_scen`.
        - :data:`run_kwargs` : kwargs
              Run options for :func:`nom_helper` and others.
        - :data:`mult_kwargs` : kwargs
              Multi-scenario options
        - app_args : mdl_kwargs
              Keyword arguments for the SampleApproach.
              See sim.sample.SampleApproach documentation.

    Returns
    -------
    nested_results : Result
        A nested Result dictionary with the desired results of each scenario.
    nested_mdlhists : History
        A nested History dictionary with the history of all model states for each
        scenario
    apps : dict
        A dictionary of the SampleApproaches generated corresponding to each nominal
        scenario with structure {'nomscen1': app1}
    """
    save_args = kwargs.get('save_args', {})
    check_overwrite(save_args)
    max_mem, showprogress, pool, close_p = unpack_mult_kwargs(kwargs)
    sim_kwarg = pack_sim_kwargs(**kwargs)
    run_kwargs_nest = pack_run_kwargs(**kwargs)
    scennames = ps.scen_names()
    nest_hist = History.fromkeys(scennames)
    nest_res = Result.fromkeys(scennames)
    apps = dict.fromkeys(scennames)
    for scenname, scen in tqdm.tqdm(ps.named_scenarios().items(),
                                    disable=not (showprogress),
                                    desc="NESTED SCENARIOS COMPLETE"):
        loc_kwarg = {**sim_kwarg, 'scen': scen, 'close_pool': False,
                     'get_phasemap': get_phasemap, 'showprogress': False,
                     'include_nominal': include_nominal}
        res, hist, app = fault_sample_from(mdl, faultdomains, faultsamples, **loc_kwarg)
        save_helper(save_args, res, hist, indiv_id=scenname, result_id=scenname)
        nest_res[scenname] = res
        nest_hist[scenname] = hist
        apps[scenname] = app
    save_helper(save_args, nest_res, nest_hist)
    close_pool({'pool': pool, 'close_pool': close_p})
    return nest_res.flatten(), nest_hist.flatten(), apps


def gen_sampleapproach(mdl, faultdomains={}, faultsamples={},
                       get_phasemap=False, nomhist={}):
    """Generate a SampleApproach from faultdomain and faultsample arguments."""
    if get_phasemap:
        pm = from_hist(nomhist)
    else:
        pm = {}
    app = SampleApproach(mdl, phasemaps=pm)
    app.add_faultdomains(**faultdomains)
    app.add_faultsamples(**faultsamples)
    return app


def list_init_faults(mdl, times):
    """
    Create list of single-fault scenarios for the given Model mode information.

    Parameters
    ----------
    mdl : Simulable/Function
        Simulable
    times
        list with list of times in (start_time, end_time)

    Returns
    -------
    faultlist : list
        A list of SingleFaultScenario
    """
    faultlist = []
    fxns = mdl.get_fxns()
    for time in times:
        for fxnname, fxn in fxns.items():
            if hasattr(fxn, 'm'):
                faultmodes = fxn.m.faultmodes
            else:
                faultmodes = {}
            for mode in faultmodes:
                rate = mdl.get_scen_rate(fxnname, mode, time)
                newscen = SingleFaultScenario(sequence={time:
                                                        {'faults': {fxnname: mode}}},
                                              function=fxnname,
                                              fault=mode,
                                              rate=rate,
                                              time=time,
                                              name=fxnname+'_'+mode+'_'+t_key(time))
                faultlist.append(newscen)
    return faultlist


def check_end_condition(mdl, use_end_condition, t):
    """
    Check if the end condition of the simulation has been met.

    Parameters
    ----------
    mdl : Simulable
        Model with or without a given end condition and simparam
    use_end_condition : bool
        Whether to use the end condition
    t : float
        time.

    Returns
    -------
    end_condition : bool
        Whether to end the simulation.
    """
    if use_end_condition and mdl.sp.end_condition:
        end_condition = get_var(mdl, mdl.sp.end_condition)
        if end_condition(t):
            return True
        else:
            return False
    else:
        return False


def prop_one_scen(mdl, scen, ctimes=[], nomhist={}, nomresult={}, **kwargs):
    """
    Simulate a single scenario in the model over time.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in.
    scen : Scenario
        The Scenario to run.
    ctimes : list, optional
        List of times to copy the model (for use in staged execution).
        The default is [].
    nomhist : dict, optional
        Model history dictionary from previous runs, for use in creating the new
        mdlhist. The default is {}.
    nomresult : dict, optional
        Nominal result dictionary (to compare with current if desired)
    **kwargs : kwargs
        simulation options, see :data:`sim_kwargs`
    Returns
    -------
    result: Result
        dict of result corresponding to desired_result.
    mdlhist : dict
        A dictionary with a history of modelstates.
    c_mdl : dict
        A dictionary of models at each time given in ctimes with structure {time:model}
    t_end: float
        Last sim time
    """
    desired_result, staged, cut_hist, run_stochastic, use_end_condition, warn_faults = unpack_sim_kwargs(**kwargs)
    # if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    if staged:
        start_time = scen.time
    else:
        start_time = 0
    timerange = mdl.sp.get_timerange(start_time)
    # check if sequence is out of timerange
    for t in scen['sequence']:
        if t not in timerange:
            raise Exception("t="+str(t)+" from sequence not in timerange: "
                            + str(timerange))
    shift = mdl.sp.get_shift(start_time)
    mdl.init_time_hist()
    # run model through the time range defined in the object
    c_mdl = dict.fromkeys(ctimes)
    result = Result()
    for t_ind, t in enumerate(timerange):
        # inject fault when it occurs, track defined flow states and graph
        try:
            if t in ctimes:
                c_mdl[t] = mdl.copy()
            if t in scen['sequence']:
                fxnfaults = scen['sequence'][t].get('faults', {})
                disturbances = scen['sequence'][t].get('disturbances', {})
            else:
                fxnfaults = {}
                disturbances = {}
            try:
                mdl.propagate(t, fxnfaults, disturbances, run_stochastic=run_stochastic)
            except Exception as e:
                raise Exception("Error in scenario " + str(scen)) from e

            mdl.log_hist(t_ind, t, shift)

            if type(desired_result) is dict:
                if "all" in desired_result:
                    des_res = desired_result['all']
                    nom_res = nomresult
                elif t in desired_result:
                    des_res = desired_result[t]
                    nom_res = nomresult.get(t_key(t))
                else:
                    des_res = False
                if des_res:
                    result[t_key(t)] = get_result(scen, mdl, des_res, nomhist,
                                                  nom_res, time=t)
            if check_end_condition(mdl, use_end_condition, t):
                break
        except:
            print("Error at t=" + str(t) + ' in scenario ' + str(scen))
            raise
            break
    if cut_hist:
        mdl.h.cut(t_ind + shift)
    if type(desired_result) is dict and 'end' in desired_result:
        result['end'] = get_result(scen, mdl, desired_result['end'], nomhist, nomresult,
                                   time=t)
    elif (type(desired_result) is not dict
          or all([type(k) is str for k in desired_result])):
        result.update(get_result(scen, mdl, desired_result, nomhist, nomresult, time=t))

    if None in c_mdl.values():
        raise Exception("Sample times" + str(ctimes)
                        + " go beyond simulation time " + str(t))
    return result, mdl.h, c_mdl, t_ind + shift


def get_result(scen, mdl, desired_result, nomhist={}, nomresult={}, time=0.0):
    """Get the desired_result specified from the model."""
    mdlhist = mdl.h
    desired_result = copy.deepcopy(desired_result)
    if type(desired_result) is str:
        desired_result = {desired_result: None}
    elif type(desired_result) in [list, set]:
        des_res = desired_result
        desired_result = {str(k): k for k in des_res if type(k) is not str}
        desired_result.update({k: None for k in des_res if type(k) is str})
    result = Result()
    if not nomhist:
        nomhist = mdlhist
    elif len(nomhist['time']) != len(mdlhist['time']):
        nomhist = nomhist.cut(start_ind=len(nomhist['time']) - len(mdlhist['time']),
                              newcopy=True)
    ec_des_res = [x.split('.')[1] for x in desired_result
                  if 'endclass' in x and x != 'endclass']
    if 'endclass' in desired_result or ec_des_res:
        mdlhists = History()
        mdlhists['faulty'] = mdlhist
        mdlhists['nominal'] = nomhist
        endclass = Result(**mdl.find_classification(scen, mdlhists))
        if ec_des_res:
            result['endclass'] = Result({k: v for k, v in endclass.items()
                                         if k in ec_des_res})
            for k in ec_des_res:
                desired_result.pop('endclass.'+k)
        elif type(desired_result['endclass']) is dict:
            result['endclass'] = Result({k: v for k, v in endclass.items()
                                         if k in desired_result['endclass']})
            desired_result.pop('endclass')
        else:
            result['endclass'] = endclass
            desired_result.pop('endclass')
    if 'endfaults' in desired_result:
        result['endfaults'], result['faultprops'] = mdl.return_faultmodes()
        desired_result.pop('endfaults')

    graphs_to_get = [g for g in desired_result
                     if type(g) is str and (g.startswith('graph')
                                            or g.startswith('Graph'))]
    for g in graphs_to_get:
        arg = desired_result.pop(g)
        if isinstance(arg, tuple):
            Gclass, kwargs = arg
        else:
            Gclass = False
            kwargs = {}

        if '.' in g:
            strs = g.split(".")
            obj = get_var(mdl, strs[1:])
        else:
            obj = mdl
        rgraph = obj.as_modelgraph(time=time, **kwargs)

        if nomresult and g in nomresult:
            rgraph.set_resgraph(nomresult[g])
        elif nomresult:
            rgraph.set_resgraph(nomresult)
        else:
            rgraph.set_resgraph()
        result[g] = rgraph

    if desired_result:
        if 'vars' in desired_result:
            result['vars'] = {}
            get_endclass_vars(mdl, desired_result['vars'], result['vars'])
        else:
            get_endclass_vars(mdl, desired_result, result)
    return result


def get_endclass_vars(mdl, desired_result, result):
    """
    Get variables in the model corresponding to the provided desired_result dictionary.

    Appends them to provided result.

    Parameters
    ----------
    mdl : Simulable
        fmdtools Simulation (e.g., Model)
    desired_result : dict
        dictionary arguments for desired_result
    result : Result
        Result to append results to.
    """
    if type(desired_result) is str:
        vars_to_get = [desired_result]
    else:
        vars_to_get = [d for d in desired_result
                       if type(d) not in [int, float]]
    var_result = mdl.get_vars(*vars_to_get, trunc_tuple=False)
    for i, var in enumerate(vars_to_get):
        result[var] = var_result[i]
