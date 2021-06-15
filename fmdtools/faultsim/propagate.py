# -*- coding: utf-8 -*-
"""
File name: propagate.py
Author: Daniel Hulse
Created: December 2019

Description: functions to propagate faults through a user-defined fault model

Main Methods:
    - nominal():            Runs the model over time in the nominal scenario.
    - one_fault():          Runs one fault in the model at a specified time.
    - mult_fault():         Runs arbitrary scenario of fault modes at specified times
    - singlefaults():       Creates and propagates a list of failure scenarios in a model over given model times
    - approach:             Injects and propagates faults in the model defined by a given sample approach.   
Private Methods:
    - list_init_faults():   Creates a list of single-fault scenarios for the graph, given the modes set up in the fault model
    - prop_one_scen():      Runs a fault scenario in the model over time
    - propagate():          Injects and propagates faults through the graph at one time-step
    - prop_time():          Propagates faults through model graph.
    - update_mdlhist():     Updates the model history at a given time.
        - update_flowhist():Updates the flows in the model history at t_ind
        - update_fxnhist(): Updates the functions (faults and states) in the model history at t_ind
    - init_mdlhist():       Initializes the model history over a given timerange
        - init_flowhist():  Initializes the flow history flowhist of the model mdl over the time range timerange
        - init_fxnhist():   Initializes the function state history fxnhist of the model mdl over the time range timerange
"""

import numpy as np
import copy
import fmdtools.resultdisp.process as proc
import tqdm

## FAULT PROPAGATION

def nominal(mdl, track='all', gtype='normal', track_times="all"):
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
        The type of graph to return (normal or bipartite). The default is 'normal'.

    Returns
    -------
    endresults : Dict
        A dictionary summary of results at the end of the simulation with structure {faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
    resgraph : MultiGraph
        A networkx graph object with function faults and degraded flows as graph attributes
    mdlhist : Dict
        A dictionary with a history of modelstates
    """
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    nomscen=construct_nomscen(mdl)
    scen=nomscen.copy()
    mdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=False, track_times = track_times)
    
    resgraph = mdl.return_stategraph(gtype=gtype)   
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    
    endclass=mdl.find_classification(scen, {'nominal': mdlhist, 'faulty':mdlhist})
    endresults={'faults': endfaults, 'classification':endclass}
    
    mdl.reset()
    return endresults, resgraph, mdlhist

def nominal_approach(mdl,nomapp,track='all', showprogress=True, pool=False, track_times="all"):
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

    Returns
    -------
    endclasses : Dict
        Classifications of the set of scenarios, with structure {'name':classification}
    mdlhists : Dict
        Dictionary of model histories, with structure {'name':mdlhist}

    """
    mdlhists = dict.fromkeys(nomapp.scenarios)
    endclasses = dict.fromkeys(nomapp.scenarios)
    if pool:
        inputs = [(mdl.__class__(scen['properties']['params'], mdl.modelparams, mdl.valparams), scen, track, track_times) for scen in nomapp.scenarios.values()]
        result_list = list(tqdm.tqdm(pool.imap(exec_nom_helper, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(nomapp.scenarios.values())}
        mdlhists = { scen['properties']['name']:result_list[i][1] for i, scen in enumerate(nomapp.scenarios.values())}
    else:
        for scenname, scen in tqdm.tqdm(nomapp.scenarios.items(), disable=not(showprogress), desc="SCENARIOS COMPLETE"):
            mdl = mdl.__class__(params=scen['properties']['params'], modelparams = mdl.modelparams, valparams=mdl.valparams)
            mdlhists[scenname], _ = prop_one_scen(mdl, scen, track=track, staged=False, track_times=track_times)
            endfaults, endfaultprops = mdl.return_faultmodes()
            endclasses[scenname]=mdl.find_classification(scen, {'nominal': mdlhists[scenname], 'faulty':mdlhists[scenname]})
    return endclasses, mdlhists
def exec_nom_helper(arg):
    mdlhist, _ =prop_one_scen(arg[0], arg[1], track=arg[2], staged=False, track_times=arg[3])
    endclass=arg[0].find_classification(arg[1], {'nominal': mdlhist, 'faulty':mdlhist})
    return endclass, mdlhist

def one_fault(mdl, fxnname, faultmode, time=1, track='all', staged=False, gtype = 'normal', track_times="all"):
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
        The graph type to return ('bipartite' or 'normal'). The default is 'normal'.

    Returns
    -------
    endresults : dict
        A dictionary summary of results at the end of the simulation with structure {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val}
    resgraph : networkx.classes.graph.Graph
        A graph object with function faults and degraded flows noted as attributes
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time.

    """
    #run model nominally, get relevant results
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    nomscen=construct_nomscen(mdl)
    if staged:
        nommdlhist, mdls = prop_one_scen(mdl, nomscen, track=track, staged=staged, ctimes=[time], track_times=track_times)
        nomresgraph = mdl.return_stategraph(gtype)
        endfaults, endfaultprops = mdl.return_faultmodes()
        if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
        mdl.reset()
        mdl = mdls[time]
    else:
        nommdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=staged, track_times=track_times)
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
    if mdl.fxns[fxnname].faultmodes[faultmode]['probtype']=='rate':
        scen['properties']['rate']=mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[faultmode]['dist']*eq_units(mdl.fxns[fxnname].faultmodes[faultmode]['units'], mdl.units)*(mdl.times[-1]-mdl.times[0]) # this rate is on a per-simulation basis
    elif mdl.fxns[fxnname].faultmodes[faultmode]['probtype']=='prob':
        scen['properties']['rate'] = mdl.fxns[fxnname].failrate*mdl.fxns[fxnname].faultmodes[faultmode]['dist']
    scen['properties']['time']=time
    
    faultmdlhist, _ = prop_one_scen(mdl, scen, track=track, staged=staged, prevhist=nommdlhist, track_times=track_times)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = proc.graphflows(faultresgraph, nomresgraph, gtype)
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    endclass=mdl.find_classification(scen, mdlhists)
    resgraph = proc.resultsgraph(faultresgraph, nomresgraph, gtype=gtype) 
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, mdlhists

def mult_fault(mdl, faultseq, track='all', rate=np.NaN, gtype='normal', track_times="all"):
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
        The graph type to return ('bipartite' or 'normal'). The default is 'normal'.

    Returns
    -------
    endresults : dict
        A dictionary summary of results at the end of the simulation with structure {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val}
    resgraph : networkx.classes.graph.Graph
        A graph object with function faults and degraded flows noted as attributes
    mdlhists : dict
        A dictionary of the states of the model of each fault scenario over time.

    """
    #run model nominally, get relevant results
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    nomscen=construct_nomscen(mdl)
    
    nommdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=False, track_times=track_times)
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
    
    faultmdlhist, _ = prop_one_scen(mdl, scen, track=track, staged=False, prevhist=nommdlhist, track_times=track_times)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = proc.graphflows(faultresgraph, nomresgraph, gtype)
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    endclass=mdl.find_classification(scen, mdlhists)
    resgraph = proc.resultsgraph(faultresgraph, nomresgraph, gtype=gtype) 
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, mdlhists

def single_faults(mdl, staged=False, track='all', pool=False, showprogress=True, track_times="all"):
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

    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """

    scenlist=list_init_faults(mdl)
    #run model nominally, get relevant results
    nomscen=construct_nomscen(mdl)
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    if staged:
        nomhist, c_mdl = prop_one_scen(mdl, nomscen, track=track, ctimes=mdl.times, track_times=track_times)
    else:
        nomhist, c_mdl = prop_one_scen(mdl, nomscen, track=track, track_times=track_times)
    nomresgraph = mdl.return_stategraph()
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    mdl.reset()
    
    endclasses = {}
    mdlhists = {}
    mdlhists['nominal'] = nomhist
    if pool:
        if staged: inputs = [(c_mdl[scen['properties']['time']], scen, nomresgraph, nomhist, track, staged, track_times) for scen in scenlist]
        else: inputs = [(mdl, scen, nomresgraph, nomhist, track, staged, track_times) for scen in scenlist]
        result_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(scenlist)}
        mdlhists = { scen['properties']['name']:result_list[i][1] for i, scen in enumerate(scenlist)}
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist, disable=not(showprogress), desc="SCENARIOS COMPLETE")):
            if staged: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']] = exec_scen(c_mdl[scen['properties']['time']], scen, nomresgraph,nomhist, track=track, staged=staged, track_times=track_times)
            else: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']] = exec_scen(mdl, scen, nomresgraph,nomhist, track=track, staged=staged, track_times = track_times)
            
    return endclasses, mdlhists

def approach(mdl, app, staged=False, track='all', pool=False, showprogress=True, track_times="all"):
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

    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
    if staged:
        nomhist, c_mdl = prop_one_scen(mdl, app.create_nomscen(mdl), track=track, ctimes=app.times, track_times=track_times)
    else:
        nomhist, c_mdl = prop_one_scen(mdl, app.create_nomscen(mdl), track=track, track_times=track_times)
    nomresgraph = mdl.return_stategraph()
    endfaults, endfaultprops = mdl.return_faultmodes()
    if any(endfaults): print("Faults found during the nominal run "+str(endfaults))
    mdl.reset()
    
    endclasses = {}
    mdlhists = {}
    mdlhists['nominal'] = nomhist
    scenlist = app.scenlist
    if pool:
        if staged: inputs = [(c_mdl[scen['properties']['time']], scen, nomresgraph, nomhist, track, staged, track_times) for scen in scenlist]
        else: inputs = [(mdl, scen, nomresgraph, nomhist, track, staged, track_times) for scen in scenlist]
        result_list = list(tqdm.tqdm(pool.imap(exec_scen_par, inputs), total=len(inputs), disable=not(showprogress), desc="SCENARIOS COMPLETE"))
        endclasses = { scen['properties']['name']:result_list[i][0] for i, scen in enumerate(scenlist)}
        mdlhists = { scen['properties']['name']:result_list[i][1] for i, scen in enumerate(scenlist)}
    else:
        for i, scen in enumerate(tqdm.tqdm(scenlist, disable=not(showprogress), desc="SCENARIOS COMPLETE")):
            if staged: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']] = exec_scen(c_mdl[scen['properties']['time']], scen, nomresgraph,nomhist, track=track, staged=staged, track_times=track_times)
            else: endclasses[scen['properties']['name']],mdlhists[scen['properties']['name']] = exec_scen(mdl, scen, nomresgraph,nomhist, track=track, staged=staged, track_times=track_times)
    return endclasses, mdlhists

def exec_scen_par(args):
    return exec_scen(args[0], args[1], args[2], args[3], track=args[4], staged=args[5], track_times=args[6])
def exec_scen(mdl, scen, nomresgraph,nomhist, track='all', staged = True, track_times="all"):
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
    """
    if staged:
        mdl = mdl.copy()
        mdlhist, _ =prop_one_scen(mdl, scen, track=track, staged=True, prevhist=nomhist, track_times=track_times)
    else:
        mdl = mdl.__class__(params=mdl.params, modelparams = mdl.modelparams, valparams=mdl.valparams)
        mdlhist, _ =prop_one_scen(mdl, scen, track=track, track_times=track_times)
    endclass = mdl.find_classification(scen, {'nominal':nomhist, 'faulty':mdlhist})
    return endclass, mdlhist 

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
       
def prop_one_scen(mdl, scen, track='all', staged=False, ctimes=[], prevhist={}, track_times="all"):
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

    Returns
    -------
    mdlhist : dict
        A dictionary with a history of modelstates.
    c_mdl : dict
        A dictionary of models at each time given in ctimes with structure {time:model}
    """
    #if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    if staged:
        timerange=np.arange(scen['properties']['time'], mdl.times[-1]+1, mdl.tstep)
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
        timerange = np.arange(mdl.times[0], mdl.times[-1]+1, mdl.tstep)
        if track_times == "all":            histrange = timerange
        elif track_times[0]=='interval':    histrange = timerange[0:len(timerange):track_times[1]]
        elif track_times[0]=='times':       histrange = track_times[0]
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
           if singletime:
               if t==scen['properties']['time']:    flowstates = propagate(mdl, scen['faults'], t, flowstates)
               else:                                flowstates = propagate(mdl,{},t, flowstates)
           else:
               if t in scen['properties']['time']:
                   ind = scen['properties']['time'].index(t)
                   propagate(mdl, scen['faults'][ind], t, flowstates)
               else: propagate(mdl,{},t, flowstates)
           if track_times=='all':   update_mdlhist(mdl, mdlhist, t_ind+shift, track=track)
           elif track_times[0]=='interval':
               if t_ind%track_times[1]: update_mdlhist(mdl, mdlhist, t_ind//track_times[1]+shift, track=track)
           elif track_times[0]=='times':
               if t in track_times[1]: update_mdlhist(mdl, mdlhist, track_times[1].index(t), track=track)
           if t in ctimes: c_mdl[t]=mdl.copy()
       except:
            print("Error at t="+str(t)+' in scenario '+str(scen))
            raise
            break
    return mdlhist, c_mdl

def propagate(mdl, initfaults, time, flowstates={}):
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

    Returns
    -------
    flowstates : dict
        States of the model at the current time-step.
    """
    #Step 1: Run Dynamic Propagation Methods in Order Specified and Inject Faults if Applicable
    for fxnname in mdl.dynamicfxns.union(initfaults.keys()):
        fxn=mdl.fxns[fxnname]
        faults = initfaults.get(fxnname, [])
        if type(faults)==list: fxn.updatefxn('dynamic', faults=faults, time=time)
        else:                  fxn.updatefxn('dynamic', faults=[faults], time=time)
    #Step 2: Run Static Propagation Methods
    flowstates = prop_time(mdl, time, initfaults, flowstates)
    return flowstates
def prop_time(mdl, time, initfaults, flowstates={}):
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
            mdl.fxns[fxnname].updatefxn('static', time=time)
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
    if track == 'valparams':        track = mdl.valparams
    if  'flows' in track:     update_flowhist(mdl, mdlhist, t_ind)
    if 'functions' in track:  update_fxnhist(mdl, mdlhist, t_ind)
    if track == 'all':      
        update_flowhist(mdl, mdlhist, t_ind)
        update_fxnhist(mdl, mdlhist, t_ind)
def update_flowhist(mdl, mdlhist, t_ind):
    """ Updates the flows in the model history at t_ind """
    for flowname in mdlhist["flows"]:
        atts=mdl.flows[flowname].status()
        for att, val in atts.items():
            if att in mdlhist['flows'][flowname]:
                try:
                    mdlhist["flows"][flowname][att][t_ind] = val
                except:
                    print("Value too large to represent: "+att+"="+str(val))
                    raise
def update_fxnhist(mdl, mdlhist, t_ind):
    """ Updates the functions (faults and states) in the model history at t_ind """
    for fxnname in mdlhist["functions"]:
        states, faults = mdl.fxns[fxnname].return_states()
        if 'faults' in mdlhist["functions"][fxnname]:
            if type(mdlhist["functions"][fxnname]["faults"]) == dict:
                for fault in mdlhist["functions"][fxnname]["faults"]:
                    if fault in faults: mdlhist["functions"][fxnname]["faults"][fault][t_ind] = 1
            else:
                if len(faults) > 1: raise Exception("More than one fault present in "+fxnname+"\n at t= "+str(t_ind)+"\n faults: "+str(faults)+"\n Is the mode representation nonexclusive?")
                else:               mdlhist["functions"][fxnname]["faults"][t_ind]=faults.pop()
        for state, value in states.items():
            if state in mdlhist["functions"][fxnname]:  mdlhist["functions"][fxnname][state][t_ind] = value 

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
    """ Initializes the flow history flowhist of the model mdl over the time range timerange"""
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
    """Initializes the function state history fxnhist of the model mdl over the time range timerange"""
    fxnhist = {}
    for fxnname, fxn in mdl.fxns.items():
        if track=='all' or fxnname in track['functions']:
            states, faults = fxn.return_states()
            fxnhist[fxnname]={}
            modelength = max([0]+[len(modename) for modename in fxn.opermodes+list(fxn.faultmodes.keys())])
            if track == 'all' or track['functions'][fxnname]=='all' or 'faults' in track['functions'][fxnname]:
                if fxn.faultmodes:
                    if fxn.exclusive_faultmodes == False:   fxnhist[fxnname]["faults"] = {faultmode:np.array([0 for i in timerange]) for faultmode in fxn.faultmodes} 
                    elif fxn.exclusive_faultmodes == True:  fxnhist[fxnname]["faults"]=np.full([len(timerange)], list(faults)[0], dtype="U"+str(modelength))
            for state, value in states.items():
                if track == 'all' or track['functions'][fxnname]=='all' or state in track['functions'][fxnname]:
                    if state == 'mode': fxnhist[fxnname][state] = np.full([len(timerange)], value, dtype="U"+str(modelength))
                    else:               fxnhist[fxnname][state] = np.full([len(timerange)], value)
    return fxnhist