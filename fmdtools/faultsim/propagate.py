# -*- coding: utf-8 -*-
"""
File name: propagate.py
Author: Daniel Hulse
Created: December 2019

Description: functions to propagate faults through a user-defined fault model

Main Methods:
    - nominal():            Runs the model over time in the nominal scenario.
    - one_fault():          Runs one fault in the model at a specified time.
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

## FAULT PROPAGATION

def nominal(mdl, track=True, gtype='normal'):
    """
    Runs the model over time in the nominal scenario.

    Parameters
    ----------
    mdl : Model
        Model of the system
    track : BOOL, optional
        Whether or not to track flows. The default is True.
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
    nomscen=construct_nomscen(mdl)
    scen=nomscen.copy()
    mdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=False)
    
    resgraph = mdl.return_stategraph(gtype=gtype)   
    endfaults, endfaultprops = mdl.return_faultmodes()
    endclass=mdl.find_classification(resgraph, endfaultprops, construct_nomscen(mdl), scen, {'nominal': mdlhist, 'faulty':mdlhist})
    
    endresults={'faults': endfaults, 'classification':endclass}
    
    mdl.reset()
    return endresults, resgraph, mdlhist

def one_fault(mdl, fxnname, faultmode, time=1, track=True, staged=False, gtype = 'normal'):
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
    track : bool, optional
        Whether to track model states over time. The default is True.
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
    nomscen=construct_nomscen(mdl)
    if staged:
        nommdlhist, mdls = prop_one_scen(mdl, nomscen, track=track, staged=staged, ctimes=[time])
        nomresgraph = mdl.return_stategraph(gtype)
        mdl.reset()
        mdl = mdls[time]
    else:
        nommdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=staged)
        nomresgraph = mdl.return_stategraph(gtype)
        mdl.reset()
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
    
    faultmdlhist, _ = prop_one_scen(mdl, scen, track=track, staged=staged, prevhist=nommdlhist)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = proc.graphflows(faultresgraph, nomresgraph, gtype)
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    endclass = mdl.find_classification(faultresgraph, endfaultprops, endflows, scen, mdlhists)
    resgraph = proc.resultsgraph(faultresgraph, nomresgraph, gtype=gtype) 
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, mdlhists

def single_faults(mdl, reuse=False, staged=False, track=True):
    """
    Creates and propagates a list of failure scenarios in a model

    Parameters
    ----------
    mdl : model
        The model to inject faults in
    reuse : bool, optional
        Whether to clear and re-use the same model over each run rather than copying (for less memory use). The default is False.
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : bool, optional
        Whether to track states over time. The default is True.

    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    if reuse and staged:
        print("invalid to use reuse and staged options at the same time. Using staged")
        reuse=False

    scenlist=list_init_faults(mdl)
    mdl.reset() #make sure the model is actually starting from the beginning
    #run model nominally, get relevant results
    nomscen=construct_nomscen(mdl)
    if staged:
        nomhist, c_mdl = prop_one_scen(mdl, nomscen, track=track, ctimes=mdl.times)
    else:
        nomhist, c_mdl = prop_one_scen(mdl, nomscen, track=track)
    nomresgraph = mdl.return_stategraph()
    mdl.reset()
    
    endclasses = {}
    mdlhists = {}
    mdlhists['nominal'] = nomhist
    for i, scen in enumerate(scenlist):
        #run model with fault scenario
        if staged:
            mdl=c_mdl[scen['properties']['time']].copy()
            mdlhists[scen['properties']['name']], _ =prop_one_scen(mdl, scen, track=track, staged=True, prevhist=nomhist)
        else:
            mdlhists[scen['properties']['name']], _ =prop_one_scen(mdl, scen, track=track)
        endfaults, endfaultprops = mdl.return_faultmodes()
        resgraph = mdl.return_stategraph()
        endflows = proc.graphflows(resgraph, nomresgraph)
        endclasses[scen['properties']['name']] = mdl.find_classification(resgraph, endfaultprops, endflows, scen, {'nominal':nomhist, 'faulty':mdlhists[scen['properties']['name']]})
        
        if reuse: mdl.reset()
        elif staged: _
        else: mdl = mdl.__class__(params=mdl.params)
    return endclasses, mdlhists

def approach(mdl, app, reuse=False, staged=False, track=True):
    """
    Injects and propagates faults in the model defined by a given sample approach

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    app : sampleapproach
        SampleApproach used to define the list of faults and sample time for the model.
    reuse : bool, optional
        Whether to clear and re-use the same model over each run rather than copying (for less memory use). The default is False.
    staged : bool, optional
        Whether to inject the fault in a copy of the nominal model at the fault time (True) or instantiate a new model for the fault (False). Setting to True roughly halves execution time. The default is False.
    track : bool, optional
        Whether to track states over time. The default is True.

    Returns
    -------
    endclasses : dict
        A dictionary with the rate, cost, and expected cost of each scenario run with structure {scenname:{expected cost, cost, rate}}
    mdlhists : dict
        A dictionary with the history of all model states for each scenario (including the nominal)
    """
    if reuse and staged:
        print("invalid to use reuse and staged options at the same time. Using staged")
        reuse=False
    mdl.reset()
    if staged:
        nomhist, c_mdl = prop_one_scen(mdl, app.create_nomscen(mdl), track=track, ctimes=app.times)
    else:
        nomhist, c_mdl = prop_one_scen(mdl, app.create_nomscen(mdl), track=track)
    nomresgraph = mdl.return_stategraph()
    mdl.reset()
    
    endclasses = {}
    mdlhists = {}
    mdlhists['nominal'] = nomhist
    for i, scen in enumerate(app.scenlist):
        #run model with fault scenario
        if staged:
            mdl=c_mdl[scen['properties']['time']].copy()
            mdlhists[scen['properties']['name']], _ =prop_one_scen(mdl, scen, track=track, staged=True, prevhist=nomhist)
        else:
            mdlhists[scen['properties']['name']], _ =prop_one_scen(mdl, scen, track=track)
        endfaults, endfaultprops = mdl.return_faultmodes()
        resgraph = mdl.return_stategraph()
        
        endflows = proc.graphflows(resgraph, nomresgraph) #TODO: supercede this with something in faultprop?
        endclasses[scen['properties']['name']] = mdl.find_classification(resgraph, endfaultprops, endflows, scen, {'nominal':nomhist, 'faulty':mdlhists[scen['properties']['name']]})
        
        if reuse: mdl.reset()
        elif staged: _
        else: mdl = mdl.__class__(params=mdl.params)
    return endclasses, mdlhists

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
       
def prop_one_scen(mdl, scen, track=True, staged=False, ctimes=[], prevhist={}):
    """
    Runs a fault scenario in the model over time

    Parameters
    ----------
    mdl : model
        The model to inject faults in.
    scen : Dict
        The fault scenario to run. Has structure: {'faults':{fxn:fault}, 'properties':{rate, time, name, etc}}
    track : bool, optional
        Whether to track states over time. The default is True.
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
        shift = len(np.arange(mdl.times[0], scen['properties']['time'], mdl.tstep))
        if track: 
            if prevhist:    mdlhist = copy.deepcopy(prevhist)
            else:           mdlhist = init_mdlhist(mdl, timerange)
    else: 
        timerange = np.arange(mdl.times[0], mdl.times[-1]+1, mdl.tstep)
        shift = 0
        if track:  mdlhist = init_mdlhist(mdl, timerange)
    if not track: mdlhist={}
    # run model through the time range defined in the object
    c_mdl=dict.fromkeys(ctimes)
    flowstates={}
    for t_ind, t in enumerate(timerange):
       # inject fault when it occurs, track defined flow states and graph 
        if t==scen['properties']['time']: flowstates = propagate(mdl, scen['faults'], t, flowstates)
        else: flowstates = propagate(mdl,[],t, flowstates)
        if track: update_mdlhist(mdl, mdlhist, t_ind+shift)
        if t in ctimes: c_mdl[t]=mdl.copy()
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
    flowstates : dict, optional
        States of the model at the previous time-step (if used). The default is {}.

    Returns
    -------
    flowstates : dict
        States of the model at the current time-step.
    """
    #set up history of flows to see if any has changed
    activefxns=mdl.timelyfxns.copy()
    nextfxns=set()
    #Step 1: Find out what the current value of the flows are (if not generated in the last iteration)
    if not flowstates:
        for flowname, flow in mdl.flows.items():
            flowstates[flowname]=flow.status()
    #Step 2: Inject faults if present
    if initfaults:
        flowstates = prop_time(mdl, activefxns, nextfxns, flowstates, time, initfaults)
    for fxnname in initfaults:
        fxn=mdl.fxns[fxnname]
        if type(initfaults[fxnname])==list: fxn.updatefxn(faults=initfaults[fxnname], time=time)
        else:                               fxn.updatefxn(faults=[initfaults[fxnname]], time=time)
        activefxns.update([fxnname])
    #Step 3: Propagate faults through graph
    flowstates = prop_time(mdl, activefxns, nextfxns, flowstates, time, initfaults)
    return flowstates
def prop_time(mdl, activefxns, nextfxns, flowstates, time, initfaults):
    """
    Propagates faults through model graph.

    Parameters
    ----------
    mdl : model
        Model to propagate faults in
    activefxns : set
        Set of functions that are active (must be checked, e.g. because a fault was injected)
    nextfxns : set
        Set of active functions for the next iteration.
    flowstates : dict
        States of each flow in the model.
    time : float
        Current time-step.
    initfaults : dict
        Faults to inject during this propagation step.

    Returns
    -------
    flowstates : dict
        States of each flow in the model after propagation
    """
    n=0
    while activefxns:
        for fxnname in list(activefxns).copy():
            #Update functions with new values, check to see if new faults or states
            oldstates, oldfaults = mdl.fxns[fxnname].return_states()
            mdl.fxns[fxnname].updatefxn(time=time)
            newstates, newfaults = mdl.fxns[fxnname].return_states() 
            if oldstates != newstates or oldfaults != newfaults: nextfxns.update([fxnname])
        #Check to see what flows have new values and add connected functions
        for flowname, flow in mdl.flows.items():
            if flowstates[flowname]!=flow.status():
                nextfxns.update(set([n for n in mdl.bipartite.neighbors(flowname)]))
            flowstates[flowname]=flow.status()
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
def update_mdlhist(mdl, mdlhist, t_ind):
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
    """
    update_flowhist(mdl, mdlhist, t_ind)
    update_fxnhist(mdl, mdlhist, t_ind)
def update_flowhist(mdl, mdlhist, t_ind):
    """ Updates the flows in the model history at t_ind """
    for flowname, flow in mdl.flows.items():
        atts=flow.status()
        for att, val in atts.items():
            mdlhist["flows"][flowname][att][t_ind] = val
def update_fxnhist(mdl, mdlhist, t_ind):
    """ Updates the functions (faults and states) in the model history at t_ind """
    for fxnname, fxn in mdl.fxns.items():
        states, faults = fxn.return_states()
        mdlhist["functions"][fxnname]["faults"][t_ind]=faults
        for state, value in states.items():
            mdlhist["functions"][fxnname][state][t_ind] = value 

def init_mdlhist(mdl, timerange):
    """
    Initializes the model history over a given timerange

    Parameters
    ----------
    mdl : model
        the Model object
    timerange : array
        Numpy array of times to initialize in the dictionary.

    Returns
    -------
    mdlhist : dict
        A dictionary history of each model state over the given timerange.
    """
    mdlhist={}
    mdlhist["flows"]=init_flowhist(mdl, timerange)
    mdlhist["functions"]=init_fxnhist(mdl, timerange)
    mdlhist["time"]=np.array([i for i in timerange])
    return mdlhist
def init_flowhist(mdl, timerange):
    """ Initializes the flow history flowhist of the model mdl over the time range timerange"""
    flowhist={}
    for flowname, flow in mdl.flows.items():
        atts=flow.status()
        flowhist[flowname] = {}
        for att, val in atts.items():
            flowhist[flowname][att] = np.full([len(timerange)], val)
    return flowhist
def init_fxnhist(mdl, timerange):
    """Initializes the function state history fxnhist of the model mdl over the time range timerange"""
    fxnhist = {}
    for fxnname, fxn in mdl.fxns.items():
        states, faults = fxn.return_states()
        fxnhist[fxnname]={}
        fxnhist[fxnname]["faults"]=[faults for i in timerange]
        for state, value in states.items():
            fxnhist[fxnname][state] = np.full([len(timerange)], value)
    return fxnhist