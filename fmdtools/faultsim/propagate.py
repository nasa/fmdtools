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
from fmdtools.modeldef import SampleApproach

## FAULT PROPAGATION

def nominal(mdl, track='all', gtype='bipartite', track_times="all", protect=True, run_stochastic=False, save_args={}, **kwargs):
    """
    Runs the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Model
        Model of the system
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    gtype : TYPE, optional
        The type of graph to return ('bipartite'/'normal'/'typegraph'). The default is 'bipartite'.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    protect : bool
        Whether or not to protect the model object via copying
            True (default) - re-instances the model so that multiple simulations can be run successively without causing problems
            False - Thus, the model object that is returned can be modified and analyzed if needed
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True}) 
    **kwargs: kwargs (params, modelparams, and/or valparams)
        passing parameter dictionaries (e.g., params, modelparams, valparams) instantiates the model
        to be simulated with the given parameters. Parameter dictionaries do not 
        need to be complete (if incomplete)
    Returns
    -------
    endresult : Dict
        A dictionary summary of results at the end of the simulation with structure {faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
    resgraph : MultiGraph
        A networkx graph object with function faults and degraded flows as graph attributes
    mdlhist : Dict
        A dictionary with a history of modelstates
    """
    check_overwrite(save_args)
    if protect or kwargs:
        mdl = mdl.__class__(*new_mdl_params(mdl,kwargs))
    nomscen=construct_nomscen(mdl)
    scen=nomscen.copy()
    mdlhist, _, t_end = prop_one_scen(mdl, nomscen, track=track, staged=False, track_times = track_times, run_stochastic=run_stochastic)
    mdlhist = cut_mdlhist(mdlhist, t_end)
    
    resgraph = mdl.return_stategraph(gtype=gtype)   
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    
    endclass=mdl.find_classification(scen, {'nominal': mdlhist, 'faulty':mdlhist})
    endresult={'faults': endfaults, 'classification':endclass}
    
    if protect: mdl.reset()
    save_helper(save_args, endclass, mdlhist)
    return endresult, resgraph, mdlhist

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
    


def update_params(params, **kwargs):
    """
    Updates a dictionary with the given keyword arguments

    Parameters
    ----------
    params : dict
        Parameter dictionary
    **kwargs : kwargs
        New arguments to add/update in the parameter dictionary

    Returns
    -------
    params : dict
        Updated parameter dictionary
    """
    for kwarg in kwargs: 
        if kwargs.get(kwarg, None)!=None: params[kwarg]=kwargs[kwarg]
    return params
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
    params : dict
        Updated param dictionary
    modelparams : dict
        Updated modelparam dictionary
    valparams : dict
        Updated valparam dictionary
    """
    params = update_params(mdl.params, **paramdict.get('params', {}))
    modelparams = update_params(mdl.modelparams, **paramdict.get('modelparams', {}))
    valparams = update_params(mdl.valparams, **paramdict.get('valparams', {}))
    return params, modelparams, valparams

def nominal_approach(mdl,nomapp,track='all', showprogress=True, pool=False, track_times="all", run_stochastic=False, max_mem=2e9, save_args={}):
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
    track : str, optional
        States to track during simulation. The default is 'all'.
    showprogress : bool, optional
        Whether to display progress during simulation. The default is True.
    pool : Pool, optional
        Parallel pool (e.g. multiprocessing.Pool) to simulate with 
        (if using parallelism). The default is False.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
    max_mem : int
        Max memory (warns the user when memory is above threshold)
    Returns
    -------
    nomapp_endclasses : Dict
        Classifications of the set of scenarios, with structure {'scenname':classification}
    nomapp_mdlhists : Dict
        Dictionary of model histories, with structure {'scenname':mdlhist}
    """
    check_overwrite(save_args)
    nomapp_mdlhists = dict.fromkeys(nomapp.scenarios)
    nomapp_endclasses = dict.fromkeys(nomapp.scenarios)
    
    if pool:
        check_mdl_memory(mdl, nomapp.num_scenarios, max_mem=max_mem)
        inputs = [(mdl.__class__(*new_mdl_params(mdl,scen['properties'])), scen, track, track_times, run_stochastic, save_args, name, nomapp.num_scenarios, max_mem ) for name,scen in nomapp.scenarios.items()]
        a=1
        result_list = list(tqdm.tqdm(pool.imap(exec_nom_helper, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        nomapp_endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(nomapp.scenarios.values())}
        nomapp_mdlhists = { scen['properties']['name']:result_list[i][1] for i, scen in enumerate(nomapp.scenarios.values())}
    else:
        for scenname, scen in tqdm.tqdm(nomapp.scenarios.items(), disable=not(showprogress), desc="SCENARIOS COMPLETE"):
            mdl = mdl.__class__(*new_mdl_params(mdl,scen['properties']))
            nomapp_mdlhists[scenname], _, t_end = prop_one_scen(mdl, scen, track=track, staged=False, track_times=track_times, run_stochastic=run_stochastic)
            check_hist_memory(nomapp_mdlhists[scenname],nomapp.num_scenarios, max_mem=max_mem)
            _ = cut_mdlhist(nomapp_mdlhists[scenname], t_end)
            endfaults, endfaultprops = mdl.return_faultmodes()
            nomapp_endclasses[scenname]=mdl.find_classification(scen, {'nominal': nomapp_mdlhists[scenname], 'faulty':nomapp_mdlhists[scenname]})
            save_helper(save_args, nomapp_endclasses[scenname], nomapp_mdlhists[scenname], scenname, scenname)
    save_helper(save_args, nomapp_endclasses, nomapp_mdlhists)
    return nomapp_endclasses, nomapp_mdlhists
def exec_nom_helper(arg):
    """Helper function for executing nominal scenarios"""
    mdlhist, _, t_end =prop_one_scen(arg[0], arg[1], track=arg[2], staged=False, track_times=arg[3], run_stochastic=arg[4])
    mdlhist = cut_mdlhist(mdlhist, t_end)
    check_hist_memory(mdlhist,arg[7], max_mem=arg[8])
    endclass=arg[0].find_classification(arg[1], {'nominal': mdlhist, 'faulty':mdlhist})
    save_helper(arg[5], endclass, mdlhist, arg[6], arg[6])
    return endclass, mdlhist

def one_fault(mdl, fxnname, faultmode, time=1, track='all', staged=False, gtype = 'bipartite', track_times="all", protect=True, run_stochastic=False, save_args={}, **kwargs):
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
        Time to inject fault. Must be in the range of model times (i.e. in range(0, end, mdl.tstep)). The default is 0.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). The default is False.
    gtype : str, optional
        The graph type to return ('bipartite'/'normal'/'typegraph'). The default is 'bipartite'.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    protect : bool
        Whether or not to protect the model object via copying
            True (default) - re-instances the model so that multiple simulations can be run successively without causing problems
            False - Thus, the model object that is returned can be modified and analyzed if needed
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
    **kwargs: kwargs (params, modelparams, and/or valparams)
        passing parameter dictionaries (e.g., params, modelparams, valparams) instantiates the model
        to be simulated with the given parameters. Parameter dictionaries do not 
        need to be complete (if incomplete)
    Returns
    -------
    endresult : dict
        A dictionary summary of results at the end of the simulation with structure {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val}
    resgraph : networkx.classes.graph.Graph
        A graph object with function faults and degraded flows noted as attributes
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time.
    """
    check_overwrite(save_args)
    #run model nominally, get relevant results
    if protect or kwargs:
        mdl = mdl.__class__(*new_mdl_params(mdl,kwargs))
    nomscen=construct_nomscen(mdl)
    if staged:
        nommdlhist, mdls, t_end_nom = prop_one_scen(mdl, nomscen, track=track, staged=staged, ctimes=[time], track_times=track_times, run_stochastic=run_stochastic)
        nomresgraph = mdl.return_stategraph(gtype)
        endfaults, endfaultprops = mdl.return_faultmodes()
        if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
        mdl.reset()
        mdl = mdls[time]
    else:
        nommdlhist, _, t_end_nom = prop_one_scen(mdl, nomscen, track=track, staged=staged, track_times=track_times, run_stochastic=run_stochastic)
        nomresgraph = mdl.return_stategraph(gtype)
        endfaults, endfaultprops = mdl.return_faultmodes()
        if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
        mdl.reset()
        mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    #run with fault present, get relevant results
    scen=nomscen.copy() #note: this is a shallow copy, so don't define it earlier
    scen['faults'][fxnname]=faultmode
    scen['properties']['type']='single fault'
    scen['properties']['function']=fxnname
    scen['properties']['fault']=faultmode
    if not mdl.fxns[fxnname].faultmodes.get(faultmode, False) or mdl.fxns[fxnname].faultmodes[faultmode]=='synth': 
        scen['properties']['rate'] = 1/len(mdl.fxns[fxnname].faultmodes)
    else:
        if mdl.fxns[fxnname].faultmodes[faultmode].get('probtype', '')=='rate':
            scen['properties']['rate']=mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[faultmode]['dist']*eq_units(mdl.fxns[fxnname].faultmodes[faultmode]['units'], mdl.units)*(mdl.times[-1]-mdl.times[0]) # this rate is on a per-simulation basis
        elif mdl.fxns[fxnname].faultmodes[faultmode].get('probtype','')=='prob':
            scen['properties']['rate'] = mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[faultmode]['dist'] 
    scen['properties']['time']=time
    
    faultmdlhist, _, t_end = prop_one_scen(mdl, scen, track=track, staged=staged, prevhist=nommdlhist, track_times=track_times, run_stochastic=run_stochastic)
    faultmdlhist = cut_mdlhist(faultmdlhist, t_end)
    nommdlhist = cut_mdlhist(nommdlhist, t_end_nom)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = proc.graphflows(faultresgraph, nomresgraph, gtype)
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    endclass=mdl.find_classification(scen, mdlhists)
    resgraph = proc.resultsgraph(faultresgraph, nomresgraph, gtype=gtype) 
    
    endresult={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    if protect: mdl.reset()
    save_helper(save_args, endclass, mdlhists)
    return endresult,resgraph, mdlhists

def mult_fault(mdl, faultseq, track='all', rate=np.NaN, gtype='bipartite', track_times="all", protect=True, run_stochastic=False, save_args={}, **kwargs):
    """
    Runs one fault in the model at a specified time.

    Parameters
    ----------
    mdl : Model
        The model to inject the fault in.
    faultseq : dict
        Dict of times and modes defining the fault scenario {time:{fxns: [modes]},}
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    rate : float, optional
        Input rate for the sequence (must be calculated elsewhere)
    gtype : str, optional
        The graph type to return ('bipartite'/'normal'/'typegraph'). The default is 'bipartite'.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    protect : bool
        Whether or not to protect the model object via copying
            True (default) - re-instances the model so that multiple simulations can be run successively without causing problems
            False - Thus, the model object that is returned can be modified and analyzed if needed
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and indiv is an (optional) bool specifying whether to save results individually (in a folder)
        or as a monolythic file
    **kwargs: kwargs (params, modelparams, and/or valparams)
        passing parameter dictionaries (e.g., params, modelparams, valparams) instantiates the model
        to be simulated with the given parameters. Parameter dictionaries do not 
        need to be complete (if incomplete)
    Returns
    -------
    endresult : dict
        A dictionary summary of results at the end of the simulation with structure {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val}
    resgraph : networkx.classes.graph.Graph
        A graph object with function faults and degraded flows noted as attributes
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time.

    """
    check_overwrite(save_args)
    #run model nominally, get relevant results
    if protect or kwargs:
        mdl = mdl.__class__(*new_mdl_params(mdl,kwargs))
    nomscen=construct_nomscen(mdl)
    
    nommdlhist, _, t_end_nom = prop_one_scen(mdl, nomscen, track=track, staged=False, track_times=track_times, run_stochastic=run_stochastic)
    nomresgraph = mdl.return_stategraph(gtype)
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    mdl.reset()
    
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    #run with fault present, get relevant results
    scen=nomscen.copy() #note: this is a shallow copy, so don't define it earlier
    scen['faults']=list(faultseq.values())
    scen['properties']['type']='sequence'
    scen['properties']['sequence']=faultseq
    scen['properties']['rate']=rate # this rate is on a per-simulation basis
    scen['properties']['time']=list(faultseq.keys())
    
    faultmdlhist, _, t_end = prop_one_scen(mdl, scen, track=track, staged=False, prevhist=nommdlhist, track_times=track_times, run_stochastic=run_stochastic)
    faultmdlhist = cut_mdlhist(faultmdlhist, t_end)
    nommdlhist = cut_mdlhist(nommdlhist, t_end_nom)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = proc.graphflows(faultresgraph, nomresgraph, gtype)
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    endclass=mdl.find_classification(scen, mdlhists)
    resgraph = proc.resultsgraph(faultresgraph, nomresgraph, gtype=gtype) 
    
    endresult={'flows': endflows, 'faults': endfaults, 'classification':endclass} 
    if protect: mdl.reset()
    save_helper(save_args, endclass, mdlhists)
    return endresult,resgraph, mdlhists

def single_faults(mdl, staged=False, track='all', pool=False, showprogress=True, track_times="all", protect=True, run_stochastic=False, save_args={}, max_mem=2e9, **kwargs):
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
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages. multiprocessing is recommended.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    protect : bool
        Whether or not to protect the model object via copying
            True (default) - re-instances the model (safe)
            False - model is not re-instantiated (unsafe--do not use model afterwards)
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and indiv is an (optional) bool specifying whether to save results individually (in a folder)
        or as a monolythic file
    max_mem : int
        Max memory (warns the user when memory is above threshold)
    **kwargs: kwargs (params, modelparams, and/or valparams)
        passing parameter dictionaries (e.g., params, modelparams, valparams) instantiates the model
        to be simulated with the given parameters. Parameter dictionaries do not 
        need to be complete (if incomplete)

    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    check_overwrite(save_args)
    if protect or kwargs:
        mdl = mdl.__class__(*new_mdl_params(mdl,kwargs))
    scenlist=list_init_faults(mdl)
    nomscen=construct_nomscen(mdl)
    if staged:
        nomhist, c_mdl, t_end_nom = prop_one_scen(mdl, nomscen, track=track, ctimes=mdl.times, track_times=track_times, run_stochastic=run_stochastic)
    else:
        nomhist, c_mdl, t_end_nom = prop_one_scen(mdl, nomscen, track=track, track_times=track_times, run_stochastic=run_stochastic)
    nomresgraph = mdl.return_stategraph()
    endclass_nominal=mdl.find_classification(construct_nomscen(mdl), {'nominal': nomhist, 'faulty':nomhist})
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    mdl.reset()
    check_hist_memory(nomhist,len(scenlist), max_mem=max_mem)
    endclasses = {}
    mdlhists = {}
    if pool:
        check_mdl_memory(mdl, len(scenlist), max_mem=max_mem)
        if staged: inputs = [(c_mdl[scen['properties']['time']], scen, nomresgraph, nomhist, track, staged, track_times, run_stochastic, save_args, str(i)) for i, scen in enumerate(scenlist)]
        else: inputs = [(mdl, scen, nomresgraph, nomhist, track, staged, track_times, run_stochastic, save_args, str(i)) for i, scen in enumerate(scenlist)]
        result_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(scenlist)}
        mdlhists = { scen['properties']['name']: result_list[i][1] for i, scen in enumerate(scenlist)}
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist, disable=not(showprogress), desc="SCENARIOS COMPLETE")):
            if staged: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']], t_end = exec_scen(c_mdl[scen['properties']['time']], scen, nomresgraph,nomhist, track=track, staged=staged, track_times=track_times, run_stochastic=run_stochastic, save_args=save_args, indiv_id=str(i))
            else: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']], t_end = exec_scen(mdl, scen, nomresgraph,nomhist, track=track, staged=staged, track_times = track_times, run_stochastic=run_stochastic, save_args=save_args, indiv_id=str(i))
    mdlhists['nominal'] = cut_mdlhist(nomhist, t_end_nom)
    endclasses['nominal'] = endclass_nominal
    save_helper(save_args, endclasses['nominal'], mdlhists['nominal'], indiv_id=str(len(endclasses)-1),result_id='nominal')
    save_helper(save_args, endclasses, mdlhists)
    return endclasses, mdlhists

def approach(mdl, app, staged=False, track='all', pool=False, showprogress=True, track_times="all", protect=True, run_stochastic=False, save_args={}, max_mem=2e9, **kwargs):
    """
    Injects and propagates faults in the model defined by a given sample approach

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    app : sampleapproach
        SampleApproach used to define the list of faults and sample time for the model.
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages. Pathos is recommended.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    protect : bool
        Whether or not to protect the model object via copying
            True (default) - re-instances the model (safe)
            False - model is not re-instantiated (unsafe--do not use model afterwards)
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and indiv is an (optional) bool specifying whether to save results individually (in a folder)
        or as a monolythic file
    max_mem : int
        Max memory (warns the user when memory is above threshold)
    **kwargs: kwargs (params, modelparams, and/or valparams)
        passing parameter dictionaries (e.g., params, modelparams, valparams) instantiates the model
        to be simulated with the given parameters. Parameter dictionaries do not 
        need to be complete (if incomplete)
    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    check_overwrite(save_args)
    if protect or kwargs:
        mdl = mdl.__class__(*new_mdl_params(mdl,kwargs))
    if staged:
        nomhist, c_mdl, t_end_nom = prop_one_scen(mdl, app.create_nomscen(mdl), track=track, ctimes=copy.copy(app.times), track_times=track_times, run_stochastic=run_stochastic)
    else:
        nomhist, c_mdl, t_end_nom = prop_one_scen(mdl, app.create_nomscen(mdl), track=track, track_times=track_times, run_stochastic=run_stochastic)
    nomresgraph = mdl.return_stategraph()
    endfaults, endfaultprops = mdl.return_faultmodes()
    endclass_nominal=mdl.find_classification(construct_nomscen(mdl), {'nominal': nomhist, 'faulty':nomhist})
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    mdl.reset()
    
    check_hist_memory(nomhist,len(app.scenlist), max_mem=max_mem)
    
    endclasses, mdlhists = {}, {}
    scenlist = app.scenlist
    if pool:
        check_mdl_memory(mdl, len(app.scenlist), max_mem=max_mem)
        if staged: inputs = [(c_mdl[scen['properties']['time']], scen, nomresgraph, nomhist, track, staged, track_times, run_stochastic, save_args, str(i)) for i, scen in enumerate(scenlist)]
        else: inputs = [(mdl, scen, nomresgraph, nomhist, track, staged, track_times, run_stochastic, save_args, str(i)) for i, scen in enumerate(scenlist)]
        result_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(scenlist)}
        mdlhists = { scen['properties']['name']: result_list[i][1]for i, scen in enumerate(scenlist)}
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist, disable=not(showprogress), desc="SCENARIOS COMPLETE")):
            if staged: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']], t_end = exec_scen(c_mdl[scen['properties']['time']], scen, nomresgraph,nomhist, track=track, staged=staged, track_times=track_times, run_stochastic=run_stochastic, save_args=save_args, indiv_id=str(i))
            else: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']], t_end = exec_scen(mdl, scen, nomresgraph,nomhist, track=track, staged=staged, track_times = track_times, run_stochastic=run_stochastic, save_args=save_args, indiv_id=str(i))
    mdlhists['nominal'] = cut_mdlhist(nomhist, t_end_nom)
    endclasses['nominal'] = endclass_nominal
    save_helper(save_args, endclasses['nominal'], mdlhists['nominal'], indiv_id=str(len(endclasses)-1),result_id='nominal')
    save_helper(save_args, endclasses, mdlhists)
    return endclasses, mdlhists

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
            if args.get('filename', False): proc.file_check(args['filename'], args.get('overwrite', False))

def nested_approach(mdl, nomapp, staged=False, track='all', get_phases = False, showprogress=True, pool=False, track_times="all",run_stochastic=False, save_args={}, max_mem=2e9, **app_args):
    """
    Simulates a set of fault modes within a set of nominal scenarios defined by a nominal approach.

    Parameters
    ----------
    mdl : Model
        Model Object to use in the simulation.
    nomapp : NominalApproach
        NominalApproach defining the nominal situations the model will be run over
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    get_phases : Bool/List/Dict, optional
        Whether and how to use nominal simulation phases to set up the SampleApproach. The default is False.
        - If True, all phases from the nominal simulation are passed to SampleApproach()
        - If a list ['Fxn1', 'Fxn2' etc.], only the phases from the listed functions will be passed.
        - If a dict {'Fxn1':'phase1'}, only the phase 'phase1' in the function 'Fxn1' will be passed.
    pool : process pool, optional
        Process Pool Object from multiprocessing or pathos packages. Pathos is recommended.
        e.g. parallelpool = mp.pool(n) for n cores (multiprocessing)
        or parallelpool = ProcessPool(nodes=n) for n cores (pathos)
        If False, the set of scenarios is run serially. The default is False
    showprogress: bool, optional
        whether to show a progress bar during execution. default is true
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    save_args : dict (optional)
        Dictionary specifying if/how to save results. Default is {}, which doesn't save anything
        Has structure: {'mdlhists':mdlhistargs, 'endclass':endclassargs, 'indiv':indiv, 'apps':'filename.pkl'},
        where mdlhistargs and endclassargs are dictionaries of arguments to rd.process.save_result
        (i.e., {'filename':'filename.pkl', 'filetype':'pickle', 'overwrite':True})
        and indiv is an (optional) bool specifying whether to save results individually (in a folder)
        or as a monolythic file
        and filename.pkl is an optional file name to save the generated SampleApproaches to (if desired)
    max_mem : int
        Max memory (warns the user when memory is above threshold)
    **app_args : kwargs
        Keyword arguments for the SampleApproach. See modeldef.SampleApproach documentation.

    Returns
    -------
    nested_endclasses : dict
        A nested dictionary with the rate, cost, and expected cost of each scenario run with structure {'nomscen1':endclasses, 'nomscen2':mdlhists}
    nested_mdlhists : dict
        A nested dictionary with the history of all model states for each scenario with structure {'nomscen1':mdlhists, 'nomscen2':mdlhists}
    apps : dict
        A dictionary of the SampleApproaches generated corresponding to each nominal scenario with structure {'nomscen1':app1}
        
    """
    check_overwrite(save_args)
    nested_mdlhists = dict.fromkeys(nomapp.scenarios)
    nested_endclasses = dict.fromkeys(nomapp.scenarios)
    apps = dict.fromkeys(nomapp.scenarios)
    save_app = save_args.pop("apps", False)
    for scenname, scen in tqdm.tqdm(nomapp.scenarios.items(), disable=not(showprogress), desc="NESTED SCENARIOS COMPLETE"):
        mdl = mdl.__class__(*new_mdl_params(mdl,scen['properties']))
        nomhist, _, t_end = prop_one_scen(mdl, scen, track=track, staged=False, track_times=track_times, run_stochastic=run_stochastic)
        if get_phases:
            if get_phases=='global':      phases={'global':[0,t_end]}
            else:
                nomhist = cut_mdlhist(nomhist, t_end)
                phases, modephases = proc.modephases(nomhist)
                if type(get_phases)==list:      phases= {fxnname:phases[fxnname] for fxnname in get_phases}
                elif type(get_phases)==dict:    phases= {phase:phases[fxnname][phase] for fxnname,phase in get_phases.items()}
            app_args.update({'phases':phases})
        
        app = SampleApproach(mdl,**app_args)
        apps[scenname]=app
        
        check_hist_memory(nomhist,len(app.scenlist)*nomapp.num_scenarios, max_mem=max_mem)
        
        nested_endclasses[scenname], nested_mdlhists[scenname] = approach(mdl, app, staged=staged, track=track, pool=pool, showprogress=False, track_times=track_times, run_stochastic=run_stochastic, **scen['properties'])
        save_helper(save_args, nested_endclasses[scenname], nested_mdlhists[scenname], indiv_id=scenname, result_id=scenname)
    save_helper(save_args, nested_endclasses, nested_mdlhists)
    if save_app:
        with open(save_app['filename'], 'wb') as file_handle:
            dill.dump(apps, file_handle)
    return nested_endclasses, nested_mdlhists, apps

def exec_scen_par(args):
    """Helper function for executing the scenario in parallel"""
    return exec_scen(args[0], args[1], args[2], args[3], track=args[4], staged=args[5], track_times=args[6], run_stochastic=args[7], save_args=args[8], indiv_id=args[9])
def exec_scen(mdl, scen, nomresgraph,nomhist, track='all', staged = True, track_times="all", run_stochastic=False, save_args={}, indiv_id=''):
    """ 
    Executes a scenario and generates results and classifications given a model and nominal model history
    
     Parameters
    ----------
    mdl : model
        The model to inject faults in.
    scen : scenario
        scenario used to define time and faults where the fault is to be injected
    nomresgraph: 
        results graph of the nominal model run
    nomhist:
        history of results in the nominal model run
    c_mdl:
        the nominal model at the time to be executed in the scenarios (a dict keyed by times)
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    save_args : dict
        Save dictionary to use in save_helper defining when/how to save the dictionary
    indiv_id : str
        ID str to insert into the file name (if saving individually)
    """
    if staged:
        mdl = mdl.copy()
        mdlhist, _, t_end =prop_one_scen(mdl, scen, track=track, staged=True, prevhist=nomhist, track_times=track_times, run_stochastic=run_stochastic)
    else:
        mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
        mdlhist, _, t_end =prop_one_scen(mdl, scen, track=track, track_times=track_times, run_stochastic=run_stochastic)
    cut_mdlhist(mdlhist, t_end)
    endclass = mdl.find_classification(scen, {'nominal':nomhist, 'faulty':mdlhist})
    save_helper(save_args, endclass, mdlhist, indiv_id=indiv_id, result_id=str(scen['properties']['name']))
    return endclass, mdlhist, t_end

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
    nomscen={'faults':{},'properties':{}}
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
        Model with list of times in mdl.times

    Returns
    -------
    faultlist : list
        A list of fault scenarios, where a scenario is defined as: {faults:{functions:faultmodes}, properties:{(changes depending scenario type)} }
    """
    faultlist=[]
    trange = mdl.times[-1]-mdl.times[0] + 1.0
    for time in mdl.times:
        for fxnname, fxn in mdl.fxns.items():
            modes=fxn.faultmodes
            for mode in modes:
                nomscen=construct_nomscen(mdl)
                newscen=nomscen.copy()
                newscen['faults'][fxnname]=mode
                if mdl.fxns[fxnname].faultmodes[mode]['probtype']=='rate':
                    rate=mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[mode]['dist']*eq_units(mdl.fxns[fxnname].faultmodes[mode]['units'], mdl.units)*trange # this rate is on a per-simulation basis
                elif mdl.fxns[fxnname].faultmodes[mode]['probtype']=='prob':
                    rate = mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[mode]['dist']
                newscen['properties']={'type': 'single-fault', 'function': fxnname, 'fault': mode, 'rate': rate, 'time': time, 'name': fxnname+' '+mode+', t='+str(time)}
                faultlist.append(newscen)
    return faultlist
       
def prop_one_scen(mdl, scen, track='all', staged=False, ctimes=[], prevhist={}, track_times="all", run_stochastic=False):
    """
    Runs a fault scenario in the model over time

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    scen : Dict
        The fault scenario to run. Has structure: {'faults':{fxn:fault}, 'properties':{rate, time, name, etc}}
    track : str ('all', 'functions', 'flows', 'valparams', dict, 'none'), optional
        Which model states to track over time, which can be given as 'functions', 'flows', 
        'all', 'none', 'valparams' (model states specified in mdl.valparams),
        or a dict of form {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        The default is 'all'.
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    ctimes : list, optional
        List of times to copy the model (for use in staged execution). The default is [].
    prevhist : dict, optional
        The previous results hist (for used in staged execution). The default is {}.
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    Returns
    -------
    mdlhist : dict
        A dictionary with a history of modelstates.
    c_mdl : dict
        A dictionary of models at each time given in ctimes with structure {time:model}
    track_times : str/tuple
        Defines what times to include in the history. Options are:
            'all'--all simulated times
            ('interval', n)--includes every nth time in the history
            ('times', [t1, ... tn])--only includes times defined in the vector [t1 ... tn]
    """
    #if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    if staged:
        timerange=np.arange(scen['properties']['time'], mdl.times[-1]+mdl.tstep, mdl.tstep)
        prevtimerange = np.arange(mdl.times[0], scen['properties']['time'], mdl.tstep)
        if track_times == "all":            
            histrange = timerange
            shift = len(prevtimerange)
        elif track_times[0]=='interval':    
            histrange = timerange[0:len(timerange):track_times[1]]
            shift = len(prevtimerange[0:len(prevtimerange):track_times[1]])
        elif track_times[0]=='times':       
            histrange = track_times[1]
            shift=0
        if prevhist:    mdlhist = copy.deepcopy(prevhist)
        else:           mdlhist = init_mdlhist(mdl, histrange, track=track)
    else: 
        timerange = np.arange(mdl.times[0], mdl.times[-1]+mdl.tstep, mdl.tstep)
        if track_times == "all":            histrange = timerange
        elif track_times[0]=='interval':    histrange = timerange[0:len(timerange):track_times[1]]
        elif track_times[0]=='times':       histrange = track_times[1]
        shift = 0
        mdlhist = init_mdlhist(mdl, histrange, track=track)
    
    # run model through the time range defined in the object
    c_mdl=dict.fromkeys(ctimes)
    flowstates=dict.fromkeys(mdl.staticflows)
    if type(scen['properties']['time'])==list:    singletime=False
    else:                                         singletime=True
    for t_ind, t in enumerate(timerange):
       # inject fault when it occurs, track defined flow states and graph
       try:
           if t in ctimes: c_mdl[t]=mdl.copy()
           if singletime:
               if t==scen['properties']['time']:    flowstates = propagate(mdl, scen['faults'], t, flowstates, run_stochastic=run_stochastic)
               else:                                flowstates = propagate(mdl,{},t, flowstates, run_stochastic=run_stochastic)
           else:
               if t in scen['properties']['time']:
                   ind = scen['properties']['time'].index(t)
                   propagate(mdl, scen['faults'][ind], t, flowstates, run_stochastic=run_stochastic)
               else: propagate(mdl,{},t, flowstates, run_stochastic=run_stochastic)
           if track_times=='all':   update_mdlhist(mdl, mdlhist, t_ind+shift, track=track)
           elif track_times[0]=='interval':
               if t_ind%track_times[1]: update_mdlhist(mdl, mdlhist, t_ind//track_times[1]+shift, track=track)
           elif track_times[0]=='times':
               if t in track_times[1]: update_mdlhist(mdl, mdlhist, track_times[1].index(t), track=track)
           if (mdl.use_end_condition and hasattr(mdl, 'end_condition')):
               if mdl.end_condition(t):
                   break
       except:
            print("Error at t="+str(t)+' in scenario '+str(scen))
            raise
            break
    if None in c_mdl.values(): raise Exception("Approach times"+str(ctimes)+" go beyond simulation time "+str(t))
    return mdlhist, c_mdl, t_ind+shift

def propagate(mdl, initfaults, time, flowstates={}, run_stochastic=False):
    """
    Injects and propagates faults through the graph at one time-step

    Parameters
    ----------
    mdl : model
        The model to propagate the fault in
    initfaults : dict
        The faults to inject in the model with structure {fxn:fault}
    time : float
        The current timestep.
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.
    Returns
    -------
    flowstates : dict
        States of the model at the current time-step.
    """
    #Step 1: Run Dynamic Propagation Methods in Order Specified and Inject Faults if Applicable
    for fxnname in mdl.dynamicfxns.union(initfaults.keys()):
        fxn=mdl.fxns[fxnname]
        faults = initfaults.get(fxnname, [])
        if type(faults)==list: fxn.updatefxn('dynamic', faults=faults, time=time, run_stochastic=run_stochastic)
        else:                  fxn.updatefxn('dynamic', faults=[faults], time=time, run_stochastic=run_stochastic)
    #Step 2: Run Static Propagation Methods
    flowstates = prop_time(mdl, time, initfaults, flowstates, run_stochastic=run_stochastic)
    return flowstates
def prop_time(mdl, time, initfaults, flowstates={}, run_stochastic=False):
    """
    Propagates faults through model graph.

    Parameters
    ----------
    mdl : model
        Model to propagate faults in
    time : float
        Current time-step.
    initfaults : dict
        Faults to inject during this propagation step.
    run_stochastic : bool
        Whether to run stochastic behaviors or use default values for stochastic variables. Default is False.

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
            flowstates[flowname]=mdl.flows[flowname].status()
    n=0
    while activefxns:
        for fxnname in list(activefxns).copy():
            #Update functions with new values, check to see if new faults or states
            oldstates, oldfaults = mdl.fxns[fxnname].return_states()
            mdl.fxns[fxnname].updatefxn('static', time=time, run_stochastic=run_stochastic)
            newstates, newfaults = mdl.fxns[fxnname].return_states() 
            if oldstates != newstates or oldfaults != newfaults: nextfxns.update([fxnname])
        #Check to see what flows have new values and add connected functions
        for flowname in mdl.staticflows:
            if flowstates[flowname]!=mdl.flows[flowname].status():
                nextfxns.update(set([n for n in mdl.bipartite.neighbors(flowname) if n in mdl.staticfxns]))
            flowstates[flowname]=mdl.flows[flowname].status()
        activefxns=nextfxns.copy()
        nextfxns.clear()
        n+=1
        if n>1000: #break if this is going for too long
            print("Undesired looping in function")
            print(initfaults)
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
        for att, val in atts.items():
            if att in mdlhist['flows'][flowname]:
                try:
                    mdlhist["flows"][flowname][att][t_ind] = val
                except:
                    print("Value too large to represent: "+att+"="+str(val))
                    raise
                if not np.can_cast(type(val), type(mdlhist["flows"][flowname][att][t_ind])):
                    raise Exception(str(flowname)+" att "+str(att)+" changed type: "+str(type(mdlhist["flows"][flowname][att][t_ind]))+" to "+str(type(val))+" at t_ind="+str(t_ind))
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
        for comp_act in {*fxn.components, *fxn.actions}:
            if comp_act in mdlhist['functions'][fxnname]:
                update_blockhist(comp_act, getattr(fxn, comp_act), mdlhist['functions'][fxnname][comp_act], t_ind)
        for flowname, flow in fxn.internal_flows.items():
            if flowname in mdlhist['functions'][fxnname]:
                for att, val in flow.status().items():
                        mdlhist['functions'][fxnname][flowname][att][t_ind] = val
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
            else:               blockhist["faults"][t_ind]=faults.pop()
    for state, value in states.items():
        if state in blockhist:  
            blockhist[state][t_ind] = value
            if not np.can_cast(type(value), type(blockhist[state][t_ind])):
                raise Exception(str(blockname)+" state "+str(state)+" changed type: "+str(type(blockhist[state][t_ind]))+" to "+str(type(value))+" at t_ind="+str(t_ind))
    
def cut_mdlhist(mdlhist, ind):
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

def init_mdlhist(mdl, timerange, track = 'all'):
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
    if track=='functions':          mdlhist["functions"]=init_fxnhist(mdl, timerange, track='all') 
    elif track=='flows':            mdlhist["flows"]=init_flowhist(mdl, timerange, track='all')
    elif track == 'all':                       
        mdlhist["flows"]=init_flowhist(mdl, timerange)
        mdlhist["functions"]=init_fxnhist(mdl, timerange)
    elif type(track)==dict:
        if 'functions' in track:    mdlhist["functions"]=init_fxnhist(mdl, timerange, track=track)
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
    for flowname, flow in mdl.flows.items():
        if track=='all' or flowname in track['flows']:
            atts=flow.status()
            flowhist[flowname] = {}
            for att, val in atts.items():
                if track=='all' or track['flows'][flowname]=='all' or att in track['flows'][flowname]:
                    flowhist[flowname][att] = np.full([len(timerange)], val)
    return flowhist
def init_fxnhist(mdl, timerange, track='all'):
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
    for fxnname, fxn in mdl.fxns.items():
        if track=='all' or fxnname in track['functions']:
            fxnhist[fxnname] = init_blockhist(fxnname, fxn, timerange, track)
            for comp_act in {*fxn.components, *fxn.actions}:
                if track == 'all' or track['functions'][fxnname]=='all' or comp_act in track['functions'][fxnname]:
                    fxnhist[fxnname][comp_act]=init_blockhist(comp_act, getattr(fxn, comp_act), timerange, track='all')
            for flowname, flow in fxn.internal_flows.items():
                if track == 'all' or track['functions'][fxnname]=='all' or flowname in track['functions'][fxnname]:
                    fxnhist[fxnname][flowname] = {}
                    for att, val in flow.status().items():
                            fxnhist[fxnname][flowname][att] = np.full([len(timerange)], val)
    return fxnhist
def init_blockhist(blockname, block, timerange, track='all'):
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
    modelength = max([0]+[len(modename) for modename in block.opermodes+list(block.faultmodes.keys())])
    if track == 'all' or track['functions'][blockname]=='all' or 'faults' in track['functions'][blockname]:
        if block.faultmodes:
            if block.exclusive_faultmodes == False:   blockhist["faults"] = {faultmode:np.array([0 for i in timerange]) for faultmode in block.faultmodes} 
            elif block.exclusive_faultmodes == True:  blockhist["faults"]=np.full([len(timerange)], list(faults)[0], dtype="U"+str(modelength))
    for state, value in states.items():
        if track == 'all' or track['functions'][blockname]=='all' or state in track['functions'][blockname]:
            if state == 'mode': blockhist[state] = np.full([len(timerange)], value, dtype="U"+str(modelength))
            else:               blockhist[state] = np.full([len(timerange)], value)
    return blockhist


    