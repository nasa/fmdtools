# -*- coding: utf-8 -*-
"""
Description: functions to propagate faults through a user-defined fault model.

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
- :func:`unpack _res_list`: Helper function for unpacking results
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
- :func:`phases_from_hist`: Helper function for `nested_approach`
- :func:`check_end_condition`: Helper function for `prop_one_scen` to end simulation
  earlier.
- :func:`get_result`: Helper function for `prop_one_scen` to get result at specific
  timestep.
- :func:`get_endclass_vars`: Helper function for `get_result`
"""
# File name: propagate.py
# Author: Daniel Hulse
# Created: December 2019

import numpy as np
import copy
import tqdm
import dill
import os
from fmdtools.define.common import get_var, t_key
from fmdtools.sim.sample import SampleApproach
from fmdtools.sim.scenario import Sequence, Scenario, SingleFaultScenario
from fmdtools.analyze.result import Result, History,  create_indiv_filename, file_check
from fmdtools.analyze.graph import graph_factory
from fmdtools.analyze.phases import from_hist

# DEFAULT ARGUMENTS
sim_kwargs = {'desired_result': 'endclass',
              'track': 'default',
              'track_times': 'all',
              'staged': False,
              'run_stochastic': False,
              'use_end_condition': True}
"""
Simulation keyword arguments.

Parameters
----------
desired_result : dict/str/list
    Desired quantities to return in the first argument.
    Options are:

    - 'endclass': a dict returned by find_classification (default)
    - 'endfaults': a dict of returned fault modes and their propagation, e.g. ::

        {'endfaults':faultdict, 'faultprops':faultpropdict}

    - 'graph'/'flowgraph'/etc: a networkx graph of the model with fault modes
      superimposed
    - 'fxnname.varname': variable values to get
    - a list of the above arguments (for multiple at the end)
    - a dict of lists (for multiple over time), e.g. ::

        {time:[varnames,... 'endclass']}

track : str, optional
    Which model states to track over time (overwrites mdl.default_track).
    Default is 'default'
    Options:

    - 'default'
    - 'all'
    - 'none'
    - or a dict of form ::

        {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}

    The default is 'all'.
track_times : str/tuple
    Defines what times to include in the history.
    Options are:

    - 'all'--all simulated times
    - ('interval', n)--includes every nth time in the history
    - ('times', [t1, ... tn])--only includes times defined in the
      vector [t1 ... tn]

run_stochastic : bool
    Whether to run stochastic behaviors or use default values. Default is False.
    Can set as 'track_pdf' to calculate/track the probability densities of random
    states over time.
staged : bool, optional
    Whether to inject the faults in a copy of the nominal model at the fault time
    (True) or instantiate a new model for the fault (False). Setting to True
    roughly halves execution time. The default is False.
"""


def unpack_sim_kwargs(**kwargs):
    """Unpacks :data:`sim_kwargs` parameters for :func:`prop_one_scen`"""
    return (kwargs.get(k, v) for k, v in sim_kwargs.items())


def pack_sim_kwargs(**kwargs):
    """Creates :data:`sim_kwargs` for :func:`prop_one_scen`"""
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

    where mdlhistargs and endclassargs are dictionaries of save arguments, e.g.::

    {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True}

    and indiv is an (optional) bool specifying whether to save results individually
    (in a folder) or as a monolythic file.

mdl_kwargs: dict (optional)
    Parameter dictionary to be instantiated in the model prior to simulation.
    Has structure ::

        {"p":Parameter, "sp":SimParam, "track":track}

    Parameter dictionaries do not need to be complete (if incomplete).
"""


def pack_run_kwargs(**kwargs):
    """Creates subset of run kwargs for :func:`nom_helper` and :data:`run_kwarg`"""
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


def unpack_mult_kwargs(kwargs):
    """Unpack the mult kwarg parameters for the :func:`parameter_sample`."""
    return (kwargs.pop(k, v) for k, v in mult_kwargs.items())

# FAULT PROPAGATION


def nominal(mdl, **kwargs):
    """
    Runs the model over time in the nominal scenario.

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
    if kwargs.get('protect', False):
        mdl.reset()
    save_helper(kwargs.get('save_args', {}), result, mdlhist)
    return result, mdlhist


def save_helper(save_args, endclass, mdlhist, indiv_id='', result_id=''):
    """
    Helper function for inline results saving.

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
    get_endclass : bool
        Whether to return endclasses from mdl.find_classification. Default is True.
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

    scennames = ps.scen_names()
    n_results = Result.fromkeys(scennames)
    n_mdlhists = History.fromkeys(scennames)

    if pool:
        check_mdl_memory(mdl, num_scens, max_mem=kwargs['max_mem'])
        inputs = [(mdl, sc, name, kwargs) for name, sc in ps.named_scenarios.items()]
        res_list = list(tqdm.tqdm(pool.map(exec_nom_par, inputs),
                                  total=len(inputs),
                                  disable=not (showprogress),
                                  desc="SCENARIOS COMPLETE"))
        n_results, n_mdlhists = unpack_res_list(scennames, res_list)
    else:
        for scenname, scen in tqdm.tqdm(ps.named_scenarios.items(),
                                        disable=not (showprogress),
                                        desc="SCENARIOS COMPLETE"):
            loc_kwargs = {**kwargs, 'use_end_condition': False}
            n_results[scenname], n_mdlhists[scenname] = exec_nom_helper(mdl, scen,
                                                                        scenname,
                                                                        **loc_kwargs)
    save_helper(kwargs['save_args'], n_results, n_mdlhists)
    close_pool(kwargs)
    return n_results.flatten(), n_mdlhists.flatten()


def unpack_res_list(scenlist, res_list):
    results = Result()
    mdlhists = History()
    results.data = {scen.name: res_list[i][0] for i, scen in enumerate(scenlist)}
    mdlhists.data = {scen.name: res_list[i][1] for i, scen in enumerate(scenlist)}
    return results, mdlhists


def exec_nom_par(arg):
    endclass, mdlhist = exec_nom_helper(arg[0], arg[1], arg[2],
                                        **{**arg[3], 'use_end_condition': False})
    return endclass, mdlhist


def exec_nom_helper(mdl, scen, name, **kwargs):
    """Helper function for executing nominal scenarios"""
    mdl = mdl.new_with_params(p=scen.p, sp=scen.sp, r=scen.r)
    result, mdlhist, _, t_end = prop_one_scen(mdl, scen, **kwargs)
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
        - 'faultmode' when a Block/FxnBlock is provided
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


def sequence(mdl, seq={}, faultseq={}, disturbances={}, scen={}, rate=np.NaN, **kwargs):
    """
    Runs a sequence of faults and disturbances in the model at given times.

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

    mdl = [*mdls.values()][0]

    result, faulthist, _, t_end = prop_one_scen(mdl,
                                                scen,
                                                **sim_kwarg,
                                                nomhist=nomhist,
                                                nomresult=nomresult)
    nomhist.cut(t_end_nom)
    mdlhists = History(nominal=nomhist, faulty=faulthist)
    if kwargs.get('protect', False):
        mdl.reset()
    save_helper(kwargs.get('save_args', {}), result, mdlhists)
    return result.flatten(), mdlhists.flatten()


def nom_helper(mdl, ctimes, protect=True, save_args={}, mdl_kwargs={}, scen={},
               **kwargs):
    """
    Helper function for initial run of nominal scenario.

    Parameters
    ----------
    mdl : Simulable (object or class)
        Model of the system
    time : float/list
        Times to copy the nominal model from
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
        mdl = mdl.new_with_params(**mdl_kwargs)

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
    if any(endfaults):
        print("Faults found during the nominal run " + str(endfaults))

    # mdl.reset()
    if not staged:
        mdls = {0: mdl.new_with_params(**mdl_kwargs)}

    return result, nommdlhist, nomscen, mdls, t_end_nom


def fault_sample(mdl, fs,  **kwargs):
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

    nomhist.cut(t_end_nom)
    mdlhists['nominal'] = nomhist
    results['nominal'] = nomresult
    save_helper(kwargs.get('save_args', {}),
                nomresult,
                mdlhists['nominal'],
                indiv_id=str(len(results)-1),
                result_id='nominal')
    save_helper(kwargs['save_args'], results, mdlhists)
    close_pool(kwargs)
    return results.flatten(), mdlhists.flatten()


def single_faults(mdl, **kwargs):
    """
    Creates and propagates a list of failure scenarios in a model.

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
                        mdl.sp.times,
                        **{**kwargs, 'use_end_condition': False})
    nomresult, nomhist, nomscen, c_mdl, t_end_nom = n_outs

    scenlist = list_init_faults(mdl)
    results, mdlhists = scenlist_helper(mdl,
                                        scenlist,
                                        c_mdl,
                                        **kwargs,
                                        nomhist=nomhist,
                                        nomresult=nomresult)
    nomhist.cut(t_end_nom)
    mdlhists['nominal'] = nomhist
    results['nominal'] = nomresult
    save_helper(kwargs.get('save_args', {}),
                nomresult, mdlhists['nominal'],
                indiv_id=str(len(results)-1),
                result_id='nominal')
    save_helper(kwargs['save_args'], results, mdlhists)
    close_pool(kwargs)
    return results.flatten(), mdlhists.flatten()


def scenlist_helper(mdl, scenlist, c_mdl, **kwargs):
    # nomhist, track, track_times, desired_result, run_stochastic, save_args
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
            inputs = [(c_mdl[scen.time], scen, kwargs,  str(i))
                      for i, scen in enumerate(scenlist)]
        else:
            inputs = [(c_mdl[0], scen,  kwargs, str(i))
                      for i, scen in enumerate(scenlist)]
        res_list = list(tqdm.tqdm(pool.map(exec_scen_par, inputs),
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
                mdl_i = copy_staged(c_mdl[scen.time])
            else:
                mdl_i = c_mdl[0].new_with_params()
            ec, mh, t_end = exec_scen(mdl_i, scen, indiv_id=str(i), **kwargs)
            results[name], mdlhists[name] = ec, mh
    return results, mdlhists


def copy_staged(mdl):
    """
    Copies the model when used in staged execution.

    Parameters
    ----------
    mdl : Simulable
        Model to copy.

    Returns
    -------
    mdl : Simulable
        Copy of the model with corresponding history
    """
    if 'time' in mdl.h:
        ctime = np.copy(mdl.h.time)
        mdl = mdl.copy()
        mdl.h.time = ctime
    else:
        mdl = mdl.copy()
    return mdl


def close_pool(kwargs):
    """Closes pool to avoid memory problems"""
    if kwargs.get('pool', False) and kwargs.get('close_pool', True):
        kwargs['pool'].close()
        kwargs['pool'].join()


def exec_scen_par(args):
    """Helper function for executing the scenario in parallel"""
    mdl_in = args[0]
    if args[2].get('staged', False):
        mdl_out = copy_staged(mdl_in)
    else:
        mdl_out = mdl_in.new_with_params()
    return exec_scen(mdl_out, args[1], **args[2], indiv_id=args[3])


def exec_scen(mdl, scen, save_args={}, indiv_id='', **kwargs):
    """
    Executes a scenario and generates results and classifications given a model and
    nominal model history.

    Parameters
    ----------
    mdl : Simulable
        The model to inject faults in.
    scen : scenario
        scenario used to define time and faults where the fault is to be injected
    nomhist : History
        history of results in the nominal model run
    save_args : dict
        Save dictionary to use in save_helper defining when/how to save the dictionary
    indiv_id : str
        ID str to insert into the file name (if saving individually)
    **kwargs : kwargs
        :data:`sim_kwargs` for :func:`prop_one_scen`.
    """
    result, mdlhist, _, t_end,  = prop_one_scen(mdl, scen, **kwargs)
    save_helper(save_args, result, mdlhist, indiv_id=indiv_id, result_id=str(scen.name))
    return result, mdlhist, t_end


def check_hist_memory(mdlhist, nscens, max_mem=2e9):
    """Checks if the memory will be exhausted given the size of the mdlhist and number
    of scenarios"""
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


def nested_sample(mdl, ps, get_phasemap=False, faultdomains={}, faultsamples={}, **kwargs):
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
        FaultDomains to add to the SampleApproach and their arguments.
        Has structure: {'fd_name': (*args, **kwargs)}, where args and kwargs are
        arguments/kwargs to SampleApproach.add_faultdomain.
    faultsamples : dict
        FaultSamples to add t othe SampleApproach and their arguments.
        Has structure: {'fs_name': (*args, **kwargs)}, where args and kwargs are
        arguments/kwargs to SampleApproach.add_faultsample.
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
              See sim.approach.SampleApproach documentation.

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
    save_app = save_args.pop("apps", False)
    max_mem, showprogress, pool, close_p = unpack_mult_kwargs(kwargs)
    sim_kwarg = pack_sim_kwargs(**kwargs)
    run_kwargs_nest = pack_run_kwargs(**kwargs)
    scennames = ps.scen_names()
    num_params = ps.num_scenarios()
    nest_hist = History.fromkeys(scennames)
    nest_res = Result.fromkeys(scennames)
    apps = dict.fromkeys(scennames)
    for scenname, scen in tqdm.tqdm(ps.named_scenarios.items(),
                                    disable=not (showprogress),
                                    desc="NESTED SCENARIOS COMPLETE"):
        mdl = mdl.new_with_params(p=scen.p, sp=scen.sp, r=scen.r)
        loc_kwargs = {**sim_kwarg, 'staged': False}
        _, nomhist, _, t_end,  = prop_one_scen(mdl, scen, **loc_kwargs)

        if get_phasemap:
            pm = from_hist(nomhist)
        else:
            pm = {}
        app = SampleApproach(mdl, phasemaps=pm)
        check_hist_memory(nomhist, num_params*app.num_scenarios(), max_mem=max_mem)

        loc_kwargs = {**sim_kwarg, 'p': scen.p, 'r': scen.r}
        res, hist = fault_sample(mdl, app, pool=pool, close_pool=False,
                                 showprogress=False, **loc_kwargs)
        save_helper(save_args, res, hist, indiv_id=scenname, result_id=scenname)

        nest_res[scenname] = res
        nest_hist[scenname] = hist
        apps[scenname] = app
    save_helper(save_args, nest_res, nest_hist)
    if save_app:
        with open(save_app['filename'], 'wb') as file_handle:
            dill.dump(apps, file_handle)
    close_pool(kwargs)
    return nest_res.flatten(), nest_hist.flatten(), apps


def list_init_faults(mdl):
    """
    Creates a list of single-fault scenarios for the Model, given the modes set up
    in the fault model.

    Parameters
    ----------
    mdl : Simulable/FxnBlock
        Simulable with list of times in mdl.sp.times

    Returns
    -------
    faultlist : list
        A list of SingleFaultScenario
    """
    faultlist = []
    trange = mdl.sp.times[-1]-mdl.sp.times[0] + 1.0
    fxns = mdl.get_fxns()
    for time in mdl.sp.times:
        for fxnname, fxn in fxns.items():
            fm = fxn.m
            for mode in fm.faultmodes:
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


def init_histrange(mdl, start_time, staged, track, track_times):
    """
    Determines the timerange the model will be simulated over and initializes
    the history

    Parameters
    ----------
    mdl : Simulable
        Model (with times)
    start_time : float
        Time to start the history over
    staged : bool
        Whether the simulation will be staged.
    track : dict
        Tracking dictionary.
    track_times : dict/list
        Specific tracking times to include in the history

    Returns
    -------
    mdlhist : History
        initialized model history
    histrange : array
        times to record history over
    timerange : array
        times to simulate the model over (which may be a subset of histrange)
    shift : int
        Time index to shift the history by.
    """
    if staged:
        timerange = np.arange(start_time, mdl.sp.times[-1] + mdl.sp.dt, mdl.sp.dt)
        prevtimerange = np.arange(mdl.sp.times[0], start_time, mdl.sp.dt)
        if track_times == "all":
            shift = len(prevtimerange)
        elif track_times[0] == 'interval':
            shift = len(prevtimerange[0:len(prevtimerange):track_times[1]])
        elif track_times[0] == 'times':
            shift = 0
    else:
        timerange = np.arange(mdl.sp.times[0], mdl.sp.times[-1] + mdl.sp.dt, mdl.sp.dt)
        shift = 0

    if track_times == "all":
        histrange = timerange
    elif track_times[0] == 'interval':
        histrange = timerange[0:len(timerange):track_times[1]]
    elif track_times[0] == 'times':
        histrange = track_times[1]

    mdlhist = mdl.create_hist(histrange, track)
    if 'time' not in mdlhist:
        mdlhist.init_att('time',
                         timerange[0],
                         timerange=timerange,
                         track='all',
                         dtype=float)

    return mdlhist, histrange, timerange, shift


def check_end_condition(mdl, use_end_condition, t):
    """
    Checks if the end condition of the simulate has been met.

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


def prop_one_scen(mdl, scen, ctimes=[], nomhist={}, nomresult={}, cut_hist=True,
                  **kwargs):
    """
    Runs a fault scenario in the model over time

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
        mdlhist.
    nomhist : dict, optional
        Model history dictionary from the nominal state, for use in
        Model.find_classification and for initializing model history in staged execution
        option. The default is {}.
    nomresult : dict, optional
        Nominal result dictionary (to compare with current if desired)
    cut_hist : bool
        Whether to cut the model history to a given size. The default is True
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
    desired_result, track, track_times, staged, run_stochastic, use_end_condition = unpack_sim_kwargs(**kwargs)
    # if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    mdlhist, histrange, timerange, shift = init_histrange(mdl,
                                                          scen.time,
                                                          staged,
                                                          track,
                                                          track_times)
    # run model through the time range defined in the object
    c_mdl = dict.fromkeys(ctimes)
    result = Result()
    for t_ind, t in enumerate(timerange):
        # inject fault when it occurs, track defined flow states and graph
        try:
            if t in ctimes: 
                c_mdl[t] = mdl.copy()
                if 'time' in mdl.h:
                    c_mdl[t].h['time'] = np.copy(mdl.h.time)
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

            if track_times:
                if track_times == 'all':
                    t_ind_rec = t_ind + shift
                elif track_times[0] == 'interval':
                    t_ind_rec = t_ind//track_times[1]+shift
                elif track_times[0] == 'times':
                    t_ind_rec = track_times[1].index(t)
                else:
                    raise Exception("Invalid argument, track_times=" + str(track_times))
                mdlhist.log(mdl, t_ind_rec, time=t)

            if type(desired_result) == dict: 
                if "all" in desired_result:
                    des_res = desired_result['all']
                    nom_res = nomresult
                elif t in desired_result:
                    des_res = desired_result[t]
                    nom_res = nomresult.get(t)
                else:
                    des_res = False
                if des_res:
                    result[t] = get_result(scen,
                                           mdl,
                                           des_res,
                                           mdlhist,
                                           nomhist,
                                           nom_res,
                                           time=t)
            if check_end_condition(mdl, use_end_condition, t):
                break
        except:
            print("Error at t=" + str(t) + ' in scenario ' + str(scen))
            raise
            break
    if cut_hist:
        mdlhist.cut(t_ind + shift)
    if type(desired_result) == dict and 'end' in desired_result:
        result['end'] = get_result(scen,
                                   mdl,
                                   desired_result['end'],
                                   mdlhist, nomhist,
                                   nomresult,
                                   time=t)
    else:
        result.update(get_result(scen,
                                 mdl,
                                 desired_result,
                                 mdlhist,
                                 nomhist,
                                 nomresult,
                                 time=t))
    # if len(result)==1: result = [*result.values()][0]
    if None in c_mdl.values():
        raise Exception("Sample times" + str(ctimes)
                        + " go beyond simulation time " + str(t))
    return result, mdlhist, c_mdl, t_ind + shift


def get_result(scen, mdl, desired_result, mdlhist={}, nomhist={}, nomresult={},
               time=0.0):
    desired_result = copy.deepcopy(desired_result)
    if type(desired_result) == str:
        desired_result = {desired_result: None}
    elif type(desired_result) in [list, set]:
        des_res = desired_result
        desired_result = {str(k): k for k in des_res if type(k) != str}
        desired_result.update({k: None for k in des_res if type(k) == str})
    result = Result()
    if not nomhist:
        nomhist = mdlhist
    elif len(nomhist['time']) != len(mdlhist['time']):
        nomhist = nomhist.cut(start_ind=len(nomhist['time']) - len(mdlhist['time']),
                              newcopy=True)
    if 'endclass' in desired_result:
        mdlhists = History()
        mdlhists['faulty'] = mdlhist
        mdlhists['nominal'] = nomhist
        endclass = Result(**mdl.find_classification(scen, mdlhists))
        if type(desired_result['endclass']) == dict:
            result['endclass'] = {k: v for k, v in endclass
                                  if k in desired_result['endclass']}
        else:
            result['endclass'] = endclass
        desired_result.pop('endclass')
    if 'endfaults' in desired_result:
        result['endfaults'], result['faultprops'] = mdl.return_faultmodes()
        desired_result.pop('endfaults')

    graphs_to_get = [g for g in desired_result
                     if type(g) == str and (g.startswith('graph')
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

        if Gclass:
            rgraph = Gclass(obj, time=time, **kwargs)
        else:
            rgraph = graph_factory(obj, time=time, **kwargs)

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
    Gets variables in the model corresponding to the provided desired_result dictionary
    argument and appends them to result.

    Parameters
    ----------
    mdl : Simulable
        fmdtools Simulation (e.g., Model)
    desired_result : dict
        dictionary arguments for desired_result
    result : Result
        Result to append results to.
    """
    if type(desired_result) == str:
        vars_to_get = [desired_result]
    else:
        vars_to_get = [d for d in desired_result
                       if type(d) not in [int, float]]
    var_result = mdl.get_vars(*vars_to_get, trunc_tuple=False)
    for i, var in enumerate(vars_to_get):
        result[var] = var_result[i]
