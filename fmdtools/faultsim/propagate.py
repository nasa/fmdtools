# -*- coding: utf-8 -*-
"""
Description: functions to propagate faults through a user-defined fault model

Main Methods:
    - :func:`nominal()`:            Runs the model over time in the nominal scenario.
    - :func:`one_fault()`:          Runs one fault in the model at a specified time.
    - :func:`mult_fault()`:         Runs arbitrary scenario of fault modes at specified times
    - :func:`single_faults()`:       Creates and propagates a list of failure scenarios in a model over given model times
    - :func:`approach`:             Injects and propagates faults in the model defined by a given sample approach.
    - :func:`nominal_approach`:     Simulates a model over a range of parameters defined by a nominal approach.
    - :func:`nested_approach`:      Injects and propagates faults in the model defined by a given sample approach over a range of parameters defined by a nominal approach. 
    
Shared Method Parameters:
    - :data:`sim_kwargs`:           Simulation keyword arguments.
    - :data:`run_kwargs`:           Run keyword arguments.
    - :data:`mult_kwargs`:         Multi-scenario keyword arguments

Private Methods:
    - :func:`list_init_faults()`:   Creates a list of single-fault scenarios for the graph, given the modes set up in the fault model
    - :func:`prop_one_scen()`:      Runs a fault scenario in the model over time
    - :func:`propagate()`:          Injects and propagates faults through the graph at one time-step
    - :func:`prop_time()`:          Propagates faults through model graph.
    - :func:`update_mdlhist()`:     Updates the model history at a given time.
        - :func:`update_flowhist()`:Updates the flows in the model history at t_ind
        - :func:`update_fxnhist()`: Updates the functions (faults and states) in the model history at t_ind
        - :func:`update_blockhist()`:Updates the blocks in the model history at t_ind
    - :func:`init_mdlhist()`:       Initializes the model history over a given timerange
        - :func:`init_flowhist()`:  Initializes the flow history flowhist of the model mdl over the time range timerange
        - :func:`init_fxnhist()`:   Initializes the function state history fxnhist of the model mdl over the time range timerange
        - :func:`init_blockhist()`: Initializes the block state history fxnhist of the model mdl over the time range timerange
    - :func:`cut_mdlhist()`:        Cuts a given model history only to the time simulated
        - :func:`cut_hist()`:       Recursively cuts the given individual (flow or function) history at ind.
    - :func:`save_helper()`:        Helper function for inline results saving.
"""
#File name: propagate.py
#Author: Daniel Hulse
#Created: December 2019

import numpy as np
import copy
import fmdtools.resultdisp.process as proc
import tqdm
import dill
import warnings
import sys,os
from fmdtools.modeldef.approach import SampleApproach
from fmdtools.modeldef.model import ModelParam
from recordclass import asdict

##DEFAULT ARGUMENTS
sim_kwargs= {'desired_result':'endclass',
             'track': 'all',
             'track_times':'all',
             'staged':False,
             'run_stochastic':False}
"""
Simulation keyword arguments.

Parameters
----------
    desired_result: dict/str/list
        Desired quantities to return in the first argument. 
        Options are:
            - 'endclass': a dict returned by find_classification (default)
            - 'endfaults': a dict of returned fault modes and their propagation {'endfaults':faultdict, 'faultprops':faultpropdict}
            - 'normal'/'bipartite'/'typegraph': a networkx graph of the model with fault modes superimposed
            - 'fxnname.varname': variable values to get
            - a list of the above arguments (for multiple at the end)
            - a dict of lists (for multiple over time), e.g. {time:[varnames,... 'endclass']}
    track : str, optional
        Which model states to track over time
        Options:
            - 'functions'
            - 'flows' 
            - 'all'
            - 'none'
            - 'valparams' (model states specified in mdl.valparams), 
            - or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    track_times : str/tuple
        Defines what times to include in the history. 
        Options are:
            - 'all'--all simulated times
            - ('interval', n)--includes every nth time in the history
            - ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
        Can set as 'track_pdf' to calculate/track the probability densities of random states over time.
    staged : bool, optional
        Whether to inject the faults in a copy of the nominal model at the fault time (True) or 
        instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
"""
def unpack_sim_kwargs(**kwargs):
    """Unpacks :data:`sim_kwargs` parameters for :func:`prop_one_scen`"""
    return (kwargs.get(k,v) for k,v in sim_kwargs.items())
def pack_sim_kwargs(**kwargs):
    """Creates :data:`sim_kwargs` for :func:`prop_one_scen`"""
    return {k:kwargs.get(k,v) for k,v in sim_kwargs.items()}

run_kwargs = {'save_args':{},
              'new_params':{},
              'protect':True}
"""
Run keyword arguments.

Parameters
----------
    protect : bool
        Whether or not to protect the model object via copying.
        Options:
            - True (default): re-instances the model so that multiple simulations can be run successively without causing problems
            - False : Thus, the model object that is returned can be modified and analyzed if needed
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything.
        Has structure: 
            - {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv},
            - where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
            - (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
            - and indiv is an (optional) bool specifying whether to save results individually (in a folder)
            or as a monolythic file
    new_params: dict (optional)
        Parameter dictionary to be instantiated in the model prior to simulation. Has structure:
            - {"params":params, "modelparams":modelparams, "valparams":valparams}
        Parameter dictionaries do not need to be complete (if incomplete).
"""
def pack_run_kwargs(**kwargs):
    """Creates subset of run kwargs for :func:`nom_helper` and :data:`run_kwarg`"""
    return {k:copy.deepcopy(kwargs.get(k,v)) for k,v in run_kwargs.items()}
mult_kwargs = {'max_mem':2e9,
               'showprogress': True,
               'pool': False}
"""
Multi-scenario keyword arguments.

Parameters
----------
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages. Pathos is recommended.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    max_mem : int
        Max memory (warns the user when memory is above threshold)
"""
def unpack_mult_kwargs(kwargs):
    """Unpacks the mult kwarg parameters for the :func:`approach`"""
    return (kwargs.pop(k,v) for k,v in mult_kwargs.items())

## FAULT PROPAGATION
def nominal(mdl, **kwargs):
    """
    Runs the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Model
        Model of the system
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
    Returns
    -------
    result:
        dict of result corresponding to desired result {'endclass':endclasses, 'endfaults': endfaults, 'varname': var, t: {'endclass':endclasses...} ...}
    nomhist : Dict
        A dictionary with a history of modelstates
    """
    result, mdlhist, _, mdl, t_end = nom_helper(mdl, None, cut_hist=True, **kwargs)
    if kwargs.get('protect', False): mdl.reset()
    save_helper(kwargs.get('save_args',{}), result, mdlhist)
    return result, mdlhist

def save_helper(save_args, endclass, mdlhist, indiv_id='', result_id=''):
    """
    Helper function for inline results saving.

    Parameters
    ----------
    save_args : dict
        Dict with structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':individual_saving},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and individual_saving is a bool (True/False) 
    endclass : dict
        dict of end-state classifications (from simulation)
    mdlhist : dict
        dict of model histories (from simulation)
    """
    if 'mdlhists' in save_args:     save_args['mdlhist'] = save_args.pop('mdlhists')
    if 'endclasses' in save_args:   save_args['endclass'] = save_args.pop('endclasses')
    for save_arg in save_args:
        if save_arg not in {'mdlhist', 'endclass', 'indiv'}: raise Exception("Invalid key in save_args: "+save_arg)
    if save_args.get('indiv', False) and indiv_id:
        if 'endclass' in save_args:
            newfilename = proc.create_indiv_filename(save_args['endclass']['filename'], indiv_id, splitchar="/")
            proc.save_result(endclass, **{**save_args['endclass'], 'filename':newfilename}, result_id=result_id)
        if 'mdlhist' in save_args:
            newfilename = proc.create_indiv_filename(save_args['mdlhist']['filename'], indiv_id, splitchar="/")
            proc.save_result(mdlhist, **{**save_args['mdlhist'], 'filename':newfilename}, result_id=result_id)
    elif not save_args.get('indiv', False) and not indiv_id:
        if 'mdlhist' in save_args:     proc.save_result(mdlhist, **save_args['mdlhist'])
        if 'endclass' in save_args:     proc.save_result(endclass, **save_args['endclass'])
    
def update_params(params, new_params):
    """
    Updates a dictionary with the given keyword arguments

    Parameters
    ----------
    params : dict
        Parameter dictionary
    new_params : dict
        New arguments to add/update in the parameter dictionary

    Returns
    -------
    params : dict
        Updated parameter dictionary
    """
    params = copy.deepcopy(params)
    new_params = copy.deepcopy(new_params)
    for kwarg in new_params: 
        if new_params.get(kwarg, None)!=None: params[kwarg]=new_params[kwarg]
    return params

def new_mdl(mdl, paramdict):
    return mdl.__class__(*new_mdl_params(mdl,paramdict))
def new_mdl_params(mdl,paramdict):
    """
    Creates parameter inputs for a new model. Used for exploring parameter ranges and seeding models.

    Parameters
    ----------
    mdl : Model
        fmdtools simulation model 
    paramdict : Dict
        Dict of parameters to update with structure params/modelparams/valparams to update
        e.g. {'params':{'param1': 1.0}}

    Returns
    -------
    params : Parameter
        Updated param 
    modelparams : ModelParam
        Updated modelparam 
    valparams : dict
        Updated valparam dictionary
    """
    params = mdl.params.copy_with_vals(**paramdict.get('params', {}))
    modelparams = mdl.modelparams.copy_with_vals(**paramdict.get('modelparams', {}))
    valparams = update_params(mdl.valparams, paramdict.get('valparams', {}))
    return params, modelparams, valparams


def nominal_approach(mdl,nomapp, **kwargs):
    """
    Simulates a set of nominal scenarios through a model. Useful to understand
    the sets of parameters where the system will run nominally and/or lead to 
    a fault.

    Parameters
    ----------
    mdl : Model
        Model to simulate
    nomapp : NominalApproach
        Nominal Approach defining the nominal scenarios to run the system over.
    get_endclass : bool
        Whether to return endclasses from mdl.find_classification. Default is True.
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
            - :data:`mult_kwargs` : kwargs
                Multi-scenario options for :func:`approach` and others
    Returns
    -------
    nomresults:
        dict of result corresponding to desired result {'scenname':return}
    nomhists : Dict
        Dictionary of model histories, with structure {'scenname':mdlhist}
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    check_overwrite(kwargs['save_args'] )
    kwargs['max_mem'], showprogress, pool = unpack_mult_kwargs(kwargs)
    kwargs['num_scens']=nomapp.num_scenarios
    n_mdlhists, n_results = dict.fromkeys(nomapp.scenarios), dict.fromkeys(nomapp.scenarios)
    if pool:
        check_mdl_memory(mdl, nomapp.num_scenarios, max_mem=kwargs['max_mem'])
        inputs = [(mdl, scen, name, kwargs) for name, scen in nomapp.scenarios.items()]
        res_list = list(tqdm.tqdm(pool.imap(exec_nom_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        n_results, n_mdlhists = unpack_res_list([*nomapp.scenarios.values()], res_list)
    else:
        for scenname, scen in tqdm.tqdm(nomapp.scenarios.items(), disable=not(showprogress), desc="SCENARIOS COMPLETE"):
            n_results[scenname], n_mdlhists[scenname]= exec_nom_helper(mdl, scen, scenname, **kwargs)
    save_helper(kwargs['save_args'] , n_results, n_mdlhists)
    return n_results, n_mdlhists
def unpack_res_list(scenlist, res_list):
    results = { scen['properties']['name']: res_list[i][0] for i, scen in enumerate(scenlist)}
    mdlhists = {scen['properties']['name']: res_list[i][1] for i, scen in enumerate(scenlist)}
    return results, mdlhists

def exec_nom_par(arg):
    endclass, mdlhist = exec_nom_helper(arg[0], arg[1], arg[2], **arg[3])
    return endclass, mdlhist
def exec_nom_helper(mdl, scen, name, **kwargs):
    """Helper function for executing nominal scenarios"""
    mdl = new_mdl(mdl,scen['properties'])
    result, mdlhist, _, t_end =prop_one_scen(mdl, scen, **kwargs)
    check_hist_memory(mdlhist,kwargs['num_scens'], max_mem=kwargs['max_mem'])
    save_helper(kwargs['save_args'], result, mdlhist, name, name)
    return result, mdlhist

def one_fault(mdl, fxnname, faultmode, time=1, **kwargs):
    """
    Runs one fault in the model at a specified time.

    Parameters
    ----------
    mdl : Model
        The model to inject the fault in.
    fxnname : str
        Name of the function with the faultmode
    faultmode : str
        Name of the faultmode
    time : float, optional
        Time to inject fault. Must be in the range of model times (i.e. in range(0, end, mdl.modelparams.dt)). The default is 0.
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
    Returns
    -------
    result:
        dict of result corresponding to desired result {'endclass':endclasses, 'endfaults': endfaults, 'varname': var, t: {'endclass':endclasses...} ...}
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time with structure: {'nominal':nomhist, 'faulty':faulthist}
    """
    faultseq = {time:{fxnname:faultmode}}
    disturbances={}
    scen= create_single_fault_scen(mdl, fxnname, faultmode, time)
    result, mdlhists = mult_fault(mdl, faultseq, disturbances, scen=scen, **kwargs)
    return result, mdlhists
def create_single_fault_scen(mdl, fxnname, faultmode, time):
    scen=construct_nomscen(mdl) #note: this is a shallow copy, so don't define it earlier
    scen['sequence'][time]={'faults':{fxnname:faultmode}}
    scen['properties']['type']='single fault'
    scen['properties']['function']=fxnname
    scen['properties']['fault']=faultmode
    fxn = mdl.fxns[fxnname]
    fm= fxn.m
    if not fm.faultmodes.get(faultmode, False) or fm.faultmodes[faultmode]=='synth': 
        scen['properties']['rate'] = 1/len(fm.faultmodes)
    else:
        #if hasattr(fxn, 'c') and faultmode in fxn.c.faultmodes:
        #    fxn = fxn.c.components[fxn.c.faultmodes[faultmode]]
        #    faultmode = faultmode[len(fxn.name):]
        #elif faultmode in fxn.actfaultmodes:
        #    fxn = fxn.actions[fxn.actfaultmodes[faultmode]]
        #    faultmode = faultmode[len(fxn.name):]
        if fm.faultmodes[faultmode].probtype=='rate':
            scen['properties']['rate']=fm.failrate*fm.faultmodes[faultmode]['dist']*eq_units(fm.faultmodes[faultmode]['units'], mdl.modelparams.units)*(mdl.modelparams.times[-1]-mdl.modelparams.times[0]) # this rate is on a per-simulation basis
        elif fm.faultmodes[faultmode].get('probtype','')=='prob':
            scen['properties']['rate'] = fm.failrate*fm.faultmodes[faultmode]['dist'] 
    scen['properties']['time']=time
    return  scen

def mult_fault(mdl, faultseq, disturbances, scen={}, rate=np.NaN, **kwargs):
    """
    Runs a sequence of faults in the model at given times.

    Parameters
    ----------
    mdl : Model
        The model to inject the fault in.
    faultseq : dict
        Dict of times and modes defining the fault scenario {time:{fxns: [modes]},}
    disturbances : dict
        Dict of times and modes defining the disturbances in the scenario {time:{fxns: [modes]},}
    scen : dict, optional
        Scenario dictionary (for external calls)
    rate : float, optional
        Input rate for the sequence (must be calculated elsewhere)
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
    Returns
    -------
    result:
        dict of result corresponding to desired result {'endclass':endclasses, 'endfaults': endfaults, 'varname': var, t: {'endclass':endclasses...} ...}
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time with structure: {'nominal':nomhist, 'faulty':faulthist}
    """
    sim_kwarg = pack_sim_kwargs(**kwargs)
    nomresult , nomhist, nomscen, mdls, t_end_nom = nom_helper(mdl, [min(faultseq)], **sim_kwarg, use_end_condition=False)
    mdl = [*mdls.values()][0]
    if not scen: scen = create_faultseq_scen(mdl, rate, faultseq=faultseq, disturbances=disturbances)
    result, faulthist, _, t_end = prop_one_scen(mdl, scen, **sim_kwarg, nomhist=nomhist, nomresult=nomresult)
    nomhist = cut_mdlhist(nomhist, t_end_nom)
    mdlhists = {'nominal':nomhist, 'faulty':faulthist}
    if kwargs.get('protect', False): mdl.reset()
    save_helper(kwargs.get('save_args',{}), result, mdlhists)
    return result, mdlhists
def create_faultseq_scen(mdl, rate, sequence={}, faultseq={}, disturbances={}):
    scen=construct_nomscen(mdl) #note: this is a shallow copy, so don't define it earlier
    times = {*faultseq, *disturbances}
    if not sequence: scen['sequence']= {t:{'faults':faultseq.get(t, {}), 'disturbances': disturbances.get(t, {})} for t in times}
    else:            scen['sequence']=sequence
    scen['properties']['type']='sequence'
    scen['properties']['sequence']=scen['sequence']
    scen['properties']['rate']=rate # this rate is on a per-simulation basis
    scen['properties']['time']=list(times)
    return scen

def nom_helper(mdl, ctimes, protect=True, save_args={}, new_params={}, scen={}, use_end_condition=None, **kwargs):
    """
    Helper function for initial run of nominal scenario.

    Parameters
    ----------
    mdl : Model
        Model of the system
    time : float/list
        Times to copy the nominal model from
    **kwargs : kwargs
        :data:`sim_kwargs` simulation options for :func:`prop_one_scen

    Returns
    -------
    result : dict
        results dict from nominal sim
    nommdlhist : dict
        result history from nominal sim
    nomscen : dict
        nominal scenario dict
    mdls : list
        Models from copy time(s) ctimes
    t_end_nom : float
        Nominal simulation end time
    """
    staged = kwargs.get('staged',False)
    check_overwrite(save_args)
    #run model nominally, get relevant results
    if protect or new_params:   mdl = new_mdl(mdl,new_params)
    if not scen:    nomscen=construct_nomscen(mdl)
    else:           nomscen=create_faultseq_scen(mdl, scen, rate=1.0)
    if staged:  
        if type(ctimes) in [float, int]:ctimes=[ctimes]
        else:                           ctimes=ctimes
    else:                               ctimes=[]
    if use_end_condition==True and hasattr(mdl, "end_condition"): mdl.use_end_condition=True
    elif use_end_condition==False:                                mdl.use_end_condition=False
    result, nommdlhist, mdls, t_end_nom = prop_one_scen(mdl, nomscen, ctimes = ctimes, **kwargs)
    
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    
    mdl.reset()
    if not staged:  mdls = {0:new_mdl(mdl, {})}
    return result, nommdlhist, nomscen, mdls, t_end_nom

def approach(mdl, app,  **kwargs):
    """
    Injects and propagates faults in the model defined by a given sample approach

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    app : sampleapproach
        SampleApproach used to define the list of faults and sample time for the model.
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
            - :data:`mult_kwargs` : kwargs
                Multi-scenario options for :func:`approach` and others
    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    nomresult, nomhist, nomscen, c_mdl, t_end_nom = nom_helper(mdl, copy.copy(app.times), **kwargs, use_end_condition=False)
    scenlist = app.scenlist
    results, mdlhists = scenlist_helper(mdl, scenlist, c_mdl, **kwargs, nomhist=nomhist, nomresult=nomresult)
    mdlhists['nominal'] = cut_mdlhist(nomhist, t_end_nom)
    results['nominal'] = nomresult
    save_helper(kwargs.get('save_args',{}), nomresult, mdlhists['nominal'], indiv_id=str(len(results)-1),result_id='nominal')
    save_helper(kwargs['save_args'], results, mdlhists)
    return results, mdlhists

def single_faults(mdl, **kwargs):
    """
    Creates and propagates a list of failure scenarios in a model.
    
    NOTE: When calling in a script using parallel=True, keep the script in the if statement:
        "if __name__=='main':
            endclasses, mdlhists = single_faults(mdl)"
        Otherwise, the method will keep spawning parallel processes. See multiprocessing documentation.

    Parameters
    ----------
    mdl : model
        The model to inject faults in
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
            - :data:`mult_kwargs` : kwargs
                Multi-scenario options for :func:`approach` and others
    Returns
    -------
    results:
        dict of result corresponding to desired result {scenname:result}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    kwargs.update(pack_run_kwargs(**kwargs))
    nomresult, nomhist, nomscen, c_mdl, t_end_nom = nom_helper(mdl, mdl.modelparams.times,**kwargs, use_end_condition=False)
    
    scenlist = list_init_faults(mdl)
    results, mdlhists = scenlist_helper(mdl, scenlist, c_mdl, **kwargs, nomhist=nomhist, nomresult=nomresult)
    mdlhists['nominal'] = cut_mdlhist(nomhist, t_end_nom)
    results['nominal'] = nomresult
    save_helper(kwargs.get('save_args',{}), nomresult, mdlhists['nominal'], indiv_id=str(len(results)-1),result_id='nominal')
    save_helper(kwargs['save_args'], results, mdlhists)
    return results, mdlhists

def scenlist_helper(mdl, scenlist, c_mdl, **kwargs):
    #nomhist, track, track_times, desired_result, run_stochastic, save_args
    max_mem, showprogress, pool = unpack_mult_kwargs(kwargs)
    staged = kwargs.get('staged',False)
    check_hist_memory(kwargs['nomhist'],len(scenlist), max_mem=max_mem)
    results = {}
    mdlhists = {}
    if pool:
        check_mdl_memory(mdl, len(scenlist), max_mem=max_mem)
        if staged:  
            inputs = [(c_mdl[scen['properties']['time']], scen, kwargs,  str(i)) for i, scen in enumerate(scenlist)]
        else:       
            inputs = [(mdl, scen,  kwargs, str(i)) for i, scen in enumerate(scenlist)]
        res_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        results, mdlhists = unpack_res_list(scenlist, res_list)
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist, disable=not(showprogress), desc="SCENARIOS COMPLETE")):
            name = scen['properties']['name']
            if staged:  mdl_i = c_mdl[scen['properties']['time']]
            else:       mdl_i = mdl
            ec, mh, t_end = exec_scen(mdl_i, scen, indiv_id=str(i), **kwargs)
            results[name],mdlhists[name] = ec, mh
    return results, mdlhists
def exec_scen_par(args):
    """Helper function for executing the scenario in parallel"""
    return exec_scen(args[0], args[1], **args[2], indiv_id=args[3])
def exec_scen(mdl, scen, save_args={}, indiv_id='', **kwargs):
    """ 
    Executes a scenario and generates results and classifications given a model and nominal model history
    
     Parameters
    ----------
    mdl : model
        The model to inject faults in.
    scen : scenario
        scenario used to define time and faults where the fault is to be injected
    nomhist:
        history of results in the nominal model run
    save_args : dict
        Save dictionary to use in save_helper defining when/how to save the dictionary
    indiv_id : str
        ID str to insert into the file name (if saving individually)
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
    """
    if kwargs.get('staged',False):  mdl = mdl.copy(); 
    else:                           mdl = new_mdl(mdl, {})
    result, mdlhist, _, t_end,  =prop_one_scen(mdl, scen, **kwargs)
    save_helper(save_args, result, mdlhist, indiv_id=indiv_id, result_id=str(scen['properties']['name']))
    return result, mdlhist, t_end

def check_hist_memory(mdlhist, nscens, max_mem=2e9):
    """Checks if the memory will be exhausted given the size of the mdlhist and number of scenarios"""
    mem_total, mem_profile = proc.get_hist_memory(mdlhist)
    total_memory = int(mem_total) * int(nscens)
    if total_memory > max_mem:
        raise Exception("Mdlhist has size: "+str(mem_total)+" bytes. With "+str(nscens)+" scenarios, it is expected that this run will pass the user-defined max_mem="+str(max_mem)+\
                        " byte limit by a factor of: "+str(total_memory/max_mem)+". To avoid, use the track= option to track less information in the mdlhist")
def check_mdl_memory(mdl, nscens, max_mem=2e9):
    mem_total, mem_profile = mdl.get_memory()
    total_memory = int(mem_total) * int(nscens)
    if total_memory > max_mem:
        raise Exception("Model has size: "+str(mem_total)+" bytes. With "+str(nscens)+" scenarios, it is expected that this run will pass the user-defined max_mem="+str(max_mem)+\
                        " byte limit by a factor of: "+str(total_memory/max_mem)+". To avoid, increase mem_total, reduce the size of the model or number of scenarios, or run outside a parallel pool")

def check_overwrite(save_args):
    for arg, args in save_args.items():
        if arg!='indiv':
            filename = args['filename']
            if args.get('filename', False):  proc.file_check(filename, args.get('overwrite', False))
            if save_args.get('indiv', False):           
                last_split_index = filename.rfind(".")
                foldername = filename[:last_split_index]
                if not os.path.exists(foldername): os.makedirs(foldername)

def nested_approach(mdl, nomapp, get_phases = False, **kwargs):
    """
    Simulates a set of fault modes within a set of nominal scenarios defined by a nominal approach.

    Parameters
    ----------
    mdl : Model
        Model Object to use in the simulation.
    nomapp : NominalApproach
        NominalApproach defining the nominal situations the model will be run over
    get_phases : Bool/List/Dict, optional
        Whether and how to use nominal simulation phases to set up the SampleApproach. The default is False.
        - If True, all phases from the nominal simulation are passed to SampleApproach()
        - If a list ['Fxn1', 'Fxn2' etc.], only the phases from the listed functions will be passed.
        - If a dict {'Fxn1':'phase1'}, only the phase 'phase1' in the function 'Fxn1' will be passed.
    **kwargs : kwargs
        Additional keyword arguments, may include:
            - :data:`sim_kwargs` : kwargs
                Simulation options for :func:`prop_one_scen
            - :data:`run_kwargs` : kwargs
                Run options for :func:`nom_helper` and others
            - :data:`mult_kwargs` : kwargs
                Multi-scenario options for :func:`approach` and others
            - **app_args : new_params
                Keyword arguments for the SampleApproach. See modeldef.SampleApproach documentation.

    Returns
    -------
    nested_results : dict
        A nested dictionary with the desired results of each scenario run with structure {'nomscen1':results, 'nomscen2':results}
    nested_mdlhists : dict
        A nested dictionary with the history of all model states for each scenario with structure {'nomscen1':mdlhists, 'nomscen2':mdlhists}
    apps : dict
        A dictionary of the SampleApproaches generated corresponding to each nominal scenario with structure {'nomscen1':app1}
    """
    save_args = kwargs.get('save_args', {})
    check_overwrite(save_args)
    save_app = save_args.pop("apps", False)
    max_mem, showprogress, pool = unpack_mult_kwargs(kwargs)
    sim_kwarg = pack_sim_kwargs(**kwargs)
    run_kwargs_nest = pack_run_kwargs(**kwargs)
    app_args = {k:v for k,v in kwargs.items() if k not in [*sim_kwarg,*run_kwargs_nest, 'max_mem', 'showprogress', 'pool']}
    
    nest_mdlhists = dict.fromkeys(nomapp.scenarios)
    nest_results = dict.fromkeys(nomapp.scenarios)
    apps = dict.fromkeys(nomapp.scenarios)
    for scenname, scen in tqdm.tqdm(nomapp.scenarios.items(), disable=not(showprogress), desc="NESTED SCENARIOS COMPLETE"):
        mdl = new_mdl(mdl,scen['properties'])
        _, nomhist, _, t_end,  = prop_one_scen(mdl, scen, **{**sim_kwarg, 'staged':False})
        if get_phases:
            app_args.update({'phases':phases_from_hist(get_phases, t_end, nomhist)})
        app = SampleApproach(mdl,**app_args)
        apps[scenname]=app
        check_hist_memory(nomhist,len(app.scenlist)*nomapp.num_scenarios, max_mem=max_mem)
        
        nest_results[scenname], nest_mdlhists[scenname] = approach(mdl, app, pool=pool, showprogress=False, **sim_kwarg, **scen['properties'])
        save_helper(save_args, nest_results[scenname], nest_mdlhists[scenname], indiv_id=scenname, result_id=scenname)
    save_helper(save_args, nest_results, nest_mdlhists)
    if save_app:
        with open(save_app['filename'], 'wb') as file_handle:
            dill.dump(apps, file_handle)
    return nest_results, nest_mdlhists, apps

def phases_from_hist(get_phases, t_end, nomhist):
    if get_phases=='global':      phases={'global':[0,t_end]}
    else:
        phases, modephases = proc.modephases(nomhist)
        if type(get_phases)==list:      phases= {fxnname:phases[fxnname] for fxnname in get_phases}
        elif type(get_phases)==dict:    phases= {phase:phases[fxnname][phase] for fxnname,phase in get_phases.items()}
    return phases

def construct_nomscen(mdl):
    """
    Creates a nominal scenario nomscen given a graph object g by setting all function modes to nominal.

    Parameters
    ----------
    mdl : Model

    Returns
    -------
    nomscen : scen
    """
    nomscen={'sequence':{}, 'properties':{}}
    nomscen['properties']['time']=0.0
    nomscen['properties']['rate']=1.0
    nomscen['properties']['type']='nominal'
    return nomscen


def eq_units(rateunit, timeunit):
    """Provides conversion factor for from rateunit (str) to timeunit (str)
    Options for units are: 'sec', 'min', 'hr', 'day', 'wk', 'month', and 'year' """
    factors = {'sec':1, 'min':60,'hr':360,'day':8640,'wk':604800,'month':2592000,'year':31556952}
    return factors[timeunit]/factors[rateunit]

def list_init_faults(mdl):
    """
    Creates a list of single-fault scenarios for the graph, given the modes set up in the fault model

    Parameters
    ----------
    mdl : Model
        Model with list of times in mdl.modelparams.times

    Returns
    -------
    faultlist : list
        A list of fault scenarios, where a scenario is defined as: {faults:{functions:faultmodes}, properties:{(changes depending scenario type)} }
    """
    faultlist=[]
    trange = mdl.modelparams.times[-1]-mdl.modelparams.times[0] + 1.0
    for time in mdl.modelparams.times:
        for fxnname, fxn in mdl.fxns.items():
            fm=fxn.m
            for mode in fm.faultmodes:
                nomscen=construct_nomscen(mdl)
                newscen=nomscen.copy()
                newscen['sequence']={time:{'faults':{fxnname:mode}}}
                if fm.faultmodes[mode]['probtype']=='rate':
                    rate=fm.failrate*fm.faultmodes[mode]['dist']*eq_units(fm.faultmodes[mode]['units'], mdl.modelparams.units)*trange # this rate is on a per-simulation basis
                elif fm.faultmodes[mode]['probtype']=='prob':
                    rate = fm.failrate*fm.faultmodes[mode]['dist']
                newscen['properties']={'type': 'single-fault', 'function': fxnname, 'fault': mode, 'rate': rate, 'time': time, 'name': fxnname+' '+mode+', t='+str(time)}
                faultlist.append(newscen)
    return faultlist

def prop_one_scen(mdl, scen, ctimes=[], nomhist={}, nomresult={}, prevhist={}, cut_hist=True, **kwargs):
    """
    Runs a fault scenario in the model over time

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    scen : Dict
        The fault scenario to run. Has structure: {'faults':{fxn:fault}, 'properties':{rate, time, name, etc}}
    ctimes : list, optional
        List of times to copy the model (for use in staged execution). The default is [].
    nomhist : dict, optional
        Model history dictionary from previous runs, for use in creating the new mdlhist.
    nomhist : dict, optional
        Model history dictionary from the nominal state, for use in Model.find_classification 
        and for initializing model history in staged execution option. The default is {}.
    nomresult : dict, optional
        Nominal result dictionary (to compare with current if desired)
    cut_hist : bool
        Whether to cut the model history to a given size. The default is True
    **kwargs : kwargs
        simulation options, see :data:`sim_kwargs` 
    Returns
    -------
    result:
        dict of result corresponding to desired result {'endclass':endclasses, 'endfaults': endfaults, 'varname': var, t: {'endclass':endclasses...} ...}
    mdlhist : dict
        A dictionary with a history of modelstates.
    c_mdl : dict
        A dictionary of models at each time given in ctimes with structure {time:model}
    t_end: float
        Last sim time 
    """
    desired_result, track, track_times, staged, run_stochastic = unpack_sim_kwargs(**kwargs)
    #if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    if staged:
        timerange=np.arange(scen['properties']['time'], mdl.modelparams.times[-1]+mdl.modelparams.dt, mdl.modelparams.dt)
        prevtimerange = np.arange(mdl.modelparams.times[0], scen['properties']['time'], mdl.modelparams.dt)
        if track_times == "all":            
            histrange = timerange
            shift = len(prevtimerange)
        elif track_times[0]=='interval':    
            histrange = timerange[0:len(timerange):track_times[1]]
            shift = len(prevtimerange[0:len(prevtimerange):track_times[1]])
        elif track_times[0]=='times':       
            histrange = track_times[1]
            shift=0
        if prevhist:    mdlhist = proc.copy_hist(prevhist)
        elif nomhist:   mdlhist = proc.copy_hist(nomhist)
        else:           mdlhist = init_mdlhist(mdl, histrange, track=track, run_stochastic=run_stochastic)
    else: 
        timerange = np.arange(mdl.modelparams.times[0], mdl.modelparams.times[-1]+mdl.modelparams.dt, mdl.modelparams.dt)
        if track_times == "all":            histrange = timerange
        elif track_times[0]=='interval':    histrange = timerange[0:len(timerange):track_times[1]]
        elif track_times[0]=='times':       histrange = track_times[1]
        shift = 0
        mdlhist = init_mdlhist(mdl, histrange, track=track, run_stochastic=run_stochastic)
    
    # run model through the time range defined in the object
    c_mdl=dict.fromkeys(ctimes); result={}
    flowstates=dict.fromkeys(mdl.staticflows)
    for t_ind, t in enumerate(timerange):
       # inject fault when it occurs, track defined flow states and graph
       try:
           if t in ctimes: c_mdl[t]=mdl.copy()
           if t in scen['sequence']: 
               fxnfaults = scen['sequence'][t].get('faults',{})
               disturbances = scen['sequence'][t].get('disturbances', {})
           else: fxnfaults, disturbances = {}, {}
           flowstates = propagate(mdl, t, fxnfaults, disturbances, flowstates, run_stochastic=run_stochastic)
           if track_times:
               if track_times=='all':           t_ind_rec = t_ind+shift
               elif track_times[0]=='interval': t_ind_rec = t_ind//track_times[1]+shift
               elif track_times[0]=='times':    t_ind_rec = track_times[1].index(t)
               else: raise Exception("Invalid argument, track_times="+str(track_times))
               update_mdlhist(mdl, mdlhist, t_ind_rec, track=track)
           if type(desired_result)==dict: 
               if "all" in desired_result: 
                   result[t] = get_result(scen,mdl,desired_result['all'], mdlhist,nomhist, nomresult)
               if t in desired_result:
                   result[t] = get_result(scen,mdl,desired_result[t], mdlhist,nomhist, nomresult.get(t, {}))
                   #desired_result.pop(t)
           if (mdl.modelparams.use_end_condition and hasattr(mdl, 'end_condition')):
               if mdl.end_condition(t):
                   break
       except:
            print("Error at t="+str(t)+' in scenario '+str(scen))
            raise
            break
    if cut_hist: cut_mdlhist(mdlhist, t_ind+shift)
    if type(desired_result)==dict and 'end' in desired_result: 
        result['end'] = get_result(scen,mdl,desired_result['end'],mdlhist,nomhist, nomresult)
    else:                       
        result.update(get_result(scen,mdl,desired_result,mdlhist,nomhist, nomresult))
    if len(result)==1: result = [*result.values()][0]
    if None in c_mdl.values(): raise Exception("Approach times"+str(ctimes)+" go beyond simulation time "+str(t))
    return  result, mdlhist, c_mdl, t_ind+shift
def get_result(scen, mdl, desired_result, mdlhist={}, nomhist={}, nomresult={}):
    desired_result = copy.deepcopy(desired_result)
    if type(desired_result)==str:               desired_result = {desired_result:None}
    elif type(desired_result) in [list, set]:   desired_result = dict.fromkeys(desired_result)
    result={}
    if not nomhist: nomhist=mdlhist
    if 'endclass' in desired_result:   
        mdlhists={'faulty':mdlhist, 'nominal':nomhist}
        endclass = mdl.find_classification(scen, mdlhists)
        endclass.update(mdl.find_sub_classifications(scen, mdlhists))
        if type(desired_result['endclass'])==dict: 
            result['endclass'] = {k:v for k,v in endclass if k in desired_result['endclass']}
        else: result['endclass']=endclass
        desired_result.pop('endclass')
    if 'endfaults' in desired_result:  
        result['endfaults'], result['faultprops'] = mdl.return_faultmodes()
        desired_result.pop('endfaults')
    for gtype in ["normal","bipartite", "typegraph", "component", *mdl.flows]:
        if gtype in desired_result:
            if gtype in ["normal","bipartite", "typegraph", "component"]:
                rgraph = mdl.return_stategraph(gtype)
                proctype=gtype
            elif gtype in mdl.flows:
                rgraph = mdl.flows[gtype].return_stategraph(**desired_result[gtype])
                proctype="bipartite"
            
            if nomresult and type(nomresult)==dict:     result[gtype] = proc.resultsgraph(rgraph, nomresult[gtype], proctype)
            elif nomresult:                             result[gtype] = proc.resultsgraph(rgraph, nomresult, proctype)
            else:           result[gtype] = proc.resultsgraph(rgraph, rgraph, proctype)
            desired_result.pop(gtype)
    if desired_result:
        if 'vars' in desired_result:
            result['vars']={}
            get_endclass_vars(mdl,desired_result['vars'], result['vars'])
        else:                           
            get_endclass_vars(mdl,desired_result, result)
    return result
    
def get_endclass_vars(mdl, desired_result, result):
    if type(desired_result)==str:   vars_to_get = [desired_result]
    else:                           vars_to_get = [d for d in desired_result if type(d) not in [int,float]]
    var_result = mdl.get_vars(*vars_to_get, trunc_tuple=False)
    for i, var in enumerate(vars_to_get):
        result[var]=var_result[i]
    

def propagate(mdl, time, fxnfaults={}, disturbances={}, flowstates={}, run_stochastic=False):
    """
    Injects and propagates faults through the graph at one time-step

    Parameters
    ----------
    mdl : model
        The model to propagate the fault in
    time : float
        The current timestep.
    fxnfaults : dict
        Faults to inject during this propagation step. With structure {'function':['fault1', 'fault2'...]}
    disturbances : 
        Variables to change during this propagation step. With structure {'function.var1':value}
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
        Can set as 'track_pdf' to calculate/track the probability densities of random states over time.
    Returns
    -------
    flowstates : dict
        States of the model at the current time-step.
    """
    #Step 0: Update model states with disturbances
    mdl.set_vars(**disturbances)
    
    #Step 1: Run Dynamic Propagation Methods in Order Specified and Inject Faults if Applicable
    for fxnname in mdl.dynamicfxns.union(fxnfaults.keys()):
        fxn=mdl.fxns[fxnname]
        faults = fxnfaults.get(fxnname, [])
        if type(faults)!=list: faults=[faults]
        fxn.updatefxn('dynamic', faults=faults, time=time, run_stochastic=run_stochastic)
        
    #Step 2: Run Static Propagation Methods
    flowstates = prop_time(mdl, time, flowstates, run_stochastic=run_stochastic)
    return flowstates
def prop_time(mdl, time, flowstates={}, run_stochastic=False):
    """
    Propagates faults through model graph.

    Parameters
    ----------
    mdl : model
        Model to propagate faults in
    time : float
        Current time-step.
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
        Can set as 'track_pdf' to calculate/track the probability densities of random states over time.

    Returns
    -------
    flowstates : dict
        States of each flow in the model after propagation
    """
    #set up history of flows to see if any has changed
    activefxns=mdl.staticfxns.copy()
    nextfxns=set()
    if not flowstates: 
        flowstates=dict.fromkeys(mdl.staticflows)
        for flowname in mdl.staticflows:
            flowstates[flowname]=mdl.flows[flowname].return_states()
    n=0
    while activefxns:
        flows_to_check = {*mdl.staticflows}
        for fxnname in list(activefxns).copy():
            #Update functions with new values, check to see if new faults or states
            oldstates, oldfaults = mdl.fxns[fxnname].return_states()
            mdl.fxns[fxnname].updatefxn('static', time=time, run_stochastic=run_stochastic)
            if mdl.fxns[fxnname].has_new_states(oldstates, oldfaults): nextfxns.update([fxnname])
            
            #Check to see what flows now have new values and add connected functions (done for each because of communications potential)
            for flowname in mdl.fxns[fxnname].flows:
                if flowname in flows_to_check:
                    if flowstates[flowname]!=mdl.flows[flowname].return_states():
                        nextfxns.update(set([n for n in mdl.bipartite.neighbors(flowname) if n in mdl.staticfxns]))
                        flows_to_check.remove(flowname)
        # check remaining flows that have not been checked already
        for flowname in flows_to_check:
            if flowstates[flowname]!=mdl.flows[flowname].return_states():
                nextfxns.update(set([n for n in mdl.bipartite.neighbors(flowname) if n in mdl.staticfxns]))
        # update flowstates
        for flowname in mdl.staticflows:
            flowstates[flowname]=mdl.flows[flowname].return_states()
        activefxns=nextfxns.copy()
        nextfxns.clear()
        n+=1
        if n>1000: #break if this is going for too long
            print("Undesired looping in function")
            print(time)
            print(fxnname)
            print(activefxns)
            break
    return flowstates

#update_mdlhist
# find a way to make faster (e.g. by automatically getting values by reference)
def update_mdlhist(mdl, mdlhist, t_ind, track = 'all'):
    """
    Updates the model history at a given time.

    Parameters
    ----------
    mdl : model
        Model at the timestep
    mdlhist : dict
        History of model states (a dict with a vector of each state)
    t_ind : float
        The time to update the model history at.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    """
    if track == 'valparams':  track = mdl.valparams
    if  'flows' in track:     update_flowhist(mdl, mdlhist, t_ind)
    if 'functions' in track:  update_fxnhist(mdl, mdlhist, t_ind)
    if track == 'all':      
        update_flowhist(mdl, mdlhist, t_ind)
        update_fxnhist(mdl, mdlhist, t_ind)
def update_flowhist(mdl, mdlhist, t_ind):
    """ Updates the flows in the model history at t_ind 
    
    Parameters
    ----------
    mdl : model
        the Model object
    mdlhist : dict
        dictionary of model histories for functions/flows
    t_ind : int
        index to update the history at
    """
    for flowname in mdlhist["flows"]:
        atts=mdl.flows[flowname].status()
        update_dicthist(atts, mdlhist["flows"][flowname], t_ind)
def update_dicthist(current_dict, dicthist, t_ind):
    for att, hist in dicthist.items():
        if att in current_dict:
            if type(hist)==dict:    update_dicthist(current_dict[att], hist, t_ind)
            else:
                if t_ind >= len(hist):
                    raise Exception("Time beyond range of model history--check staged execution and simulation time settings (end condition, mdl.modelparams.times)")
                if not np.can_cast(type(current_dict[att]), type(hist[t_ind])):
                    raise Exception(str(att)+" changed type: "+str(type(hist[t_ind]))+" to "+str(type(current_dict[att]))+" at t_ind="+str(t_ind))
                try:    
                    if type(current_dict[att]) in [list, np.ndarray]:
                        hist[t_ind]=copy.deepcopy(current_dict[att])
                    else: 
                        hist[t_ind]=current_dict[att]
                except: 
                    print("Value too large to represent: "+att+"="+str(current_dict[att]))
                    raise
def update_fxnhist(mdl, mdlhist, t_ind):
    """ Updates the functions (faults, states, components, actions) in the model history at t_ind 
    
    Parameters
    ----------
    mdl : model
        the Model object
    mdlhist : dict
        dictionary of model histories for functions/flows
    t_ind : int
        index to update the history at
    """
    for fxnname in mdlhist["functions"]:
        fxn=mdl.fxns[fxnname]
        update_blockhist(fxnname, fxn, mdlhist['functions'][fxnname], t_ind)
        for compname, comp in getattr(fxn, 'c', {'components':{}})['components'].items():
            if compname in mdlhist['functions'][fxnname]:
                update_blockhist(compname, comp, mdlhist['functions'][fxnname][compname], t_ind)
        for actname, act in getattr(fxn, 'a', {'actions':{}})['actions'].items():
            if actname in mdlhist['functions'][fxnname]:
                update_blockhist(actname, act, mdlhist['functions'][fxnname][actname], t_ind)
        for flowname, flow in getattr(fxn, 'a', {'flows':{}})['flows'].items():
            if flowname in mdlhist['functions'][fxnname]:
                update_dicthist(flow.status(), mdlhist['functions'][fxnname][flowname], t_ind)
def update_blockhist(blockname, block, blockhist, t_ind):
    """ Updates the blocks (faults, states) in the model history at t_ind 
    
    Parameters
    ----------
    blockname : str
        Name of the block 
    block : Block
        Object for the block
    blockhist : dict
        Dictionary history of the given block
    t_ind : int
        index to update the history at
    """
    if block.type not in ['function', 'component', 'action', 'block']: raise Exception(blockname+" is not a block. Is it being overwritten?")
    states, faults = block.return_states()
    if 'faults' in blockhist:
        if type(blockhist["faults"]) == dict:
            for fault in blockhist["faults"]:
                if fault in faults: blockhist["faults"][fault][t_ind] = 1
        else:
            if len(faults) > 1: raise Exception("More than one fault present in "+blockname+"\n at t= "+str(t_ind)+"\n faults: "+str(faults)+"\n Is the mode representation nonexclusive?")
            else:               blockhist["faults"][t_ind]=[*faults, ''][0]
    update_dicthist(states,blockhist,t_ind)
    if 'probdens' in blockhist: blockhist['probdens'][t_ind]=block.probdens
def cut_mdlhist(mdlhist, ind, newcopy=False):
    """Cuts unsimulated values from end of array
    
    Parameters
    ----------
    mdlhist : dict
        dictionary of model histories for functions/flows
    ind : int
        index to cut the history at
        
    Returns
    -------
    mdlhist : dict
        The model history until the given index.
    """
    if newcopy: mdlhist = proc.copy_hist(mdlhist)
    if len(mdlhist['time'])>ind+1:
        mdlhist['time'] = mdlhist['time'][:ind+1]
        if 'flows' in mdlhist:
            for flowname, atts in mdlhist['flows'].items():
                mdlhist['flows'][flowname] = cut_hist(atts, ind)
        if 'functions' in mdlhist:
            for fxnname, atts in mdlhist['functions'].items():
                mdlhist['functions'][fxnname] = cut_hist(atts, ind)
    return mdlhist 
def cut_hist(hist, ind):
    """
    Recursively cuts the given individual (flow or function) history at ind.

    Parameters
    ----------
    hist : dict
        history to cut
    ind : int
        index to cut the history at

    Returns
    -------
    newhist : dict
        Cut history
    """
    newhist = {}
    for attname, vals in hist.items():
        if type(vals)==dict:            newhist[attname] = cut_hist(vals, ind)
        elif type(vals)==np.ndarray:    newhist[attname] = vals[:ind+1]
        else:                           newhist[attname] = vals[:ind+1]
    return newhist

def init_mdlhist(mdl, timerange, track = 'all', run_stochastic=False):
    """
    Initializes the model history over a given timerange

    Parameters
    ----------
    mdl : model
        the Model object
    timerange : array
        Numpy array of times to initialize in the dictionary.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    Returns
    -------
    mdlhist : dict
        A dictionary history of each model state over the given timerange.
    """
    mdlhist={}
    if track == 'valparams':        track = mdl.valparams
    if track=='functions':          mdlhist["functions"]=init_fxnhist(mdl, timerange, track='all', run_stochastic=run_stochastic) 
    elif track=='flows':            mdlhist["flows"]=init_flowhist(mdl, timerange, track='all')
    elif track == 'all':                       
        mdlhist["flows"]=init_flowhist(mdl, timerange)
        mdlhist["functions"]=init_fxnhist(mdl, timerange, run_stochastic=run_stochastic)
    elif type(track)==dict:
        if 'functions' in track:    mdlhist["functions"]=init_fxnhist(mdl, timerange, track=track, run_stochastic=run_stochastic)
        if 'flows' in track:         mdlhist["flows"]=init_flowhist(mdl, timerange, track=track)
    else:
        if not track in ['none','None']: raise Exception("Invalid track option: "+str(track))
    mdlhist["time"]=np.array([i for i in timerange])
    return mdlhist
def init_flowhist(mdl, timerange, track='all'):
    """ Initializes the flow history flowhist of the model mdl over the time range timerange
    
    Parameters
    ----------
    mdl : model
        the Model object
    timerange : array
        Numpy array of times to initialize in the dictionary.
    track : 'all' or dict, 'none'), optional
        Which model states to track over time, which can be given as 'all' or a 
        dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    Returns
    -------
    flowhist : dict
        A dictionary history of each recorded flow state over the given timerange.
    """
    flowhist={}
    flows_track = proc.get_sub_include("flows", track)
    for flowname, flow in mdl.flows.items():
        flow_track = proc.get_sub_include(flowname, flows_track)
        if flow_track:
            atts=flow.status()
            flowhist[flowname] = init_dicthist(atts, timerange, flow_track)
    return flowhist
def init_dicthist(start_dict, timerange, track="all", modelength=10):
    dicthist = {}
    for att, val in start_dict.items():
        if type(val)==dict: 
            sub_track = proc.get_sub_include(att, track)
            if sub_track: dicthist[att]=init_dicthist(val, timerange, sub_track)
        elif track=="all" or att in track:
            if att=="mode" or type(val)==str:     
                dicthist[att]= np.full([len(timerange)], val, dtype="U"+str(modelength))
            else:
                try:            dicthist[att] = np.full([len(timerange)], val)
                except:         dicthist[att] = np.empty((len(timerange),), dtype=object)  
    return dicthist
    
def init_fxnhist(mdl, timerange, track='all', run_stochastic=False):
    """Initializes the function state history fxnhist of the model mdl over the time range timerange
    
    Parameters
    ----------
    mdl : model
        the Model object
    timerange : array
        Numpy array of times to initialize in the dictionary.
    track : 'all' or dict, 'none'), optional
        Which model states to track over time, which can be given as 'all' or a 
        dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    Returns
    -------
    fxnhist : dict
        A dictionary history of each recorded function state over the given timerange.
    """
    fxnhist = {}
    functions_track = proc.get_sub_include("functions", track)
    for fxnname, fxn in mdl.fxns.items():
        fxn_track = proc.get_sub_include(fxnname, functions_track)
        if fxn_track:
            fxnhist[fxnname] = init_blockhist(fxnname, fxn, timerange, fxn_track, run_stochastic=run_stochastic)
            for compname, comp in getattr(fxn, 'c', {'components':{}})['components'].items():
                comp_track = proc.get_sub_include(compname, fxn_track)
                if comp_track: 
                    fxnhist[fxnname][compname]=init_blockhist(compname, comp, timerange, track=comp_track)
            for compname, comp in getattr(fxn, 'a', {'actions':{}})['actions'].items():
                comp_track = proc.get_sub_include(compname, fxn_track)
                if comp_track: 
                    fxnhist[fxnname][compname]=init_blockhist(compname, comp, timerange, track=comp_track)
            for flowname, flow in getattr(fxn, 'a', {'flows':{}})['flows'].items():
                flow_track = proc.get_sub_include(flow, fxn_track)
                if flow_track: fxnhist[fxnname][flowname] = init_dicthist(flow.status(), timerange, flow_track)
    return fxnhist
def init_blockhist(blockname, block, timerange, track='all', run_stochastic=False):
    """ 
    Instantiates the block hist (faults, states) over the given timerange
    
    Parameters
    ----------
    blockname : str
        Name of the block 
    block : Block
        Object for the block
    timerange : array
        Numpy array of times to initialize in the dictionary.
    track : 'all' or dict, 'none'), optional
        Which model states to track over time, which can be given as 'all' or a 
        dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    """
    states, faults = block.return_states()
    blockhist={}
    modelength = max([0]+[len(modename) for modename in block.m.opermodes+tuple(block.m.faultmodes.keys())])
    if track == 'all' or 'faults' in track:
        if block.m.faultmodes:
            if block.m.exclusive == False:   blockhist["faults"] = {faultmode:np.array([0 for i in timerange]) for faultmode in block.m.faultmodes} 
            elif block.m.exclusive == True:  blockhist["faults"]=np.full([len(timerange)], list(block.m.faultmodes.keys())[0], dtype="U"+str(modelength))
    blockhist.update(init_dicthist(states, timerange, track=track, modelength=modelength))               
    if run_stochastic=='track_pdf' and block.rngs: 
        blockhist['probdens'] = np.full([len(timerange)], block.return_probdens())
    return blockhist


    