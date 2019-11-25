# -*- coding: utf-8 -*-
"""
File name: faultprop.py
Author: Daniel Hulse
Created: December 2018
Forked from the IBFM toolkit, original author Matthew McIntire

Description: functions to propagate faults through a user-defined fault model
"""
import numpy as np
import copy
import fmdkit.resultproc as rp
## FAULT PROPAGATION

#construct_nomscen
# creates a nominal scenario nomscen given a graph object g by setting all function modes to nominal
def construct_nomscen(mdl):
    nomscen={'faults':{},'properties':{}}
    for fxnname in mdl.fxns:
        nomscen['faults'][fxnname]='nom'
    nomscen['properties']['time']=0.0
    nomscen['properties']['type']='nominal'
    return nomscen

#run_nominal
# runs the model over time in the nominal scenario
# inputs:
#   - mdl, the python model module set up in mdl.py
#   - track, whether or not to track flows
#   - gtype, the type of graph to return
# outputs:
#   - endresults, a dictionary summary of results at the end of the simulation with structure
#    {faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
#   - resgraph, a graph object with function faults and degraded flows noted
#   - mdlhist, a dictionary with the history modelstates
def run_nominal(mdl, track=True, gtype='normal'):
    nomscen=construct_nomscen(mdl)
    scen=nomscen.copy()
    mdlhist, _ = prop_one_scen(mdl, nomscen, track=track, staged=False)
    
    resgraph = mdl.return_stategraph(gtype)   
    endfaults, endfaultprops = mdl.return_faultmodes()
    endclass=mdl.find_classification(resgraph, endfaultprops, {}, scen)
    
    endresults={'faults': endfaults, 'classification':endclass}
    
    mdl.reset()
    return endresults, resgraph, mdlhist

#proponefault
# runs the model given a single function and fault mode
# inputs:
#   - mdl, the python model module set up in mdl.py
#   - fxnname, the function the fault is initiated in
#   - faultmode, the mode to initiate
#   - time, the time when the mode is to be initiated
#   - track, the flows to track (a list of strings)
#   - gtrack, the times to snapshot the graph
# outputs:
#   - endresults, a dictionary summary of results at the end of the simulation with structure
#    {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
#   - resgraph, a graph object with function faults and degraded flows noted
#   - flowhist, a dictionary with the history of the flow over time
#   - graphhist, a dictionary of results graph objects over time with structure {time:graph}
def run_one_fault(mdl, fxnname, faultmode, time=0, track=True, staged=False, gtype = 'normal'):
    
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
    scen['properties']['rate']=mdl.fxns[fxnname].failrate 
    scen['properties']['time']=time
    
    faultmdlhist, _ = prop_one_scen(mdl, scen, track=track, staged=staged, prevhist=nommdlhist)
    faultresgraph = mdl.return_stategraph(gtype)
    
    #process model run
    endfaults, endfaultprops = mdl.return_faultmodes()
    endflows = rp.compare_graphflows(faultresgraph, nomresgraph, gtype) 
    
    endclass = mdl.find_classification(faultresgraph, endfaultprops, endflows, scen)
    # note: in the future, put this in rp package so it doesn't need to be imported
    if gtype=='normal': resgraph = rp.make_resultsgraph(faultresgraph, nomresgraph) 
    elif gtype=='bipartite': resgraph = rp.make_bipresultsgraph(faultresgraph, nomresgraph)
    
    mdlhists={'nominal':nommdlhist, 'faulty':faultmdlhist}
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, mdlhists

#list_init_faults
# creates a list of single-fault scenarios for the graph, given the modes set up in the fault model
# inputs: model graph, a vector of times for the scenarios to occur
# outputs: a list of fault scenarios, where a scenario is defined as:
#   {faults:{functions:faultmodes}, properties:{(changes depending scenario type)} }
def list_init_faults(mdl):
    faultlist=[]
    for time in mdl.times:
        for fxnname, fxn in mdl.fxns.items():
            modes=fxn.faultmodes
            
            for mode in modes:
                nomscen=construct_nomscen(mdl)
                newscen=nomscen.copy()
                newscen['faults'][fxnname]=mode
                rate=mdl.fxns[fxnname].failrate
                newscen['properties']={'type': 'single-fault', 'function': fxnname, 'fault': mode, 'rate': rate, 'time': time, 'name': fxnname+' '+mode+', t='+str(time)}
                faultlist.append(newscen)

    return faultlist

#proplist
# creates and propagates a list of failure scenarios in a model
# input: mdl, the module where the model was set up
# output: resultstab, a FMEA-style table of results
def run_list(mdl, reuse=False, staged=False, track=True):
    
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
        
        endflows = rp.compare_graphflows(resgraph, nomresgraph)
        endclasses[scen['properties']['name']] = mdl.find_classification(resgraph, endfaultprops, endflows, scen)
        
        if reuse: mdl.reset()
        elif staged: _
        else: mdl = mdl.__class__()
    return endclasses, mdlhists

#prop_one_scen
# runs a single fault scenario in the model over time
# inputs:
#   - mdl, the model object 
#   - scen, the fault scenario for a given model
#   - track, whether to track states over time
#   - staged, the starting time for the propagation
#   - ctimes, the time to copy the models
#   - prevhist, the previous results hist (if running staged)
# outputs:
#   - mdlhist, a dictionary with the history of the model states over time
#   - c_mdls, copies of the model object taken at each time listed in ctime
def prop_one_scen(mdl, scen, track=True, staged=False, ctimes=[], prevhist={}):
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
    nomscen=construct_nomscen(mdl)
    c_mdl=dict.fromkeys(ctimes)
    flowstates={}
    for t_ind, t in enumerate(timerange):
       # inject fault when it occurs, track defined flow states and graph
        if t==scen['properties']['time']: flowstates = propagate(mdl, scen['faults'], t, flowstates)
        else: flowstates = propagate(mdl,nomscen['faults'],t, flowstates)
        if track: update_mdlhist(mdl, mdlhist, t_ind+shift)
        if t in ctimes: c_mdl[t]=mdl.copy()
    return mdlhist, c_mdl

#propogate
# propagates faults through the graph at one time-step
# inputs:
#   g, the graph object of the model
#   initfaults, the faults (or lack of faults) to initiate in the model
#   time, the time propogation occurs at
#   flowstates, if generated in the last iteration
def propagate(mdl, initfaults, time, flowstates={}):
    #set up history of flows to see if any has changed
    n=0
    activefxns=mdl.timelyfxns.copy()
    nextfxns=set()
    #Step 1: Find out what the current value of the flows are (if not generated in the last iteration)
    if not flowstates:
        for flowname, flow in mdl.flows.items():
            flowstates[flowname]=flow.status()
    #Step 2: Inject faults if present     
    for fxnname in initfaults:
        if initfaults[fxnname]!='nom':
            fxn=mdl.fxns[fxnname]
            fxn.updatefxn(faults=[initfaults[fxnname]], time=time)
            activefxns.update([fxnname])
    #Step 3: Propagate faults through graph
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
            break
    return flowstates

#update_mdlhist
# find a way to make faster (e.g. by automatically getting values by reference)
def update_mdlhist(mdl, mdlhist, t_ind):
    update_flowhist(mdl, mdlhist, t_ind)
    update_fxnhist(mdl, mdlhist, t_ind)
def update_flowhist(mdl, mdlhist, t_ind):
    for flowname, flow in mdl.flows.items():
        atts=flow.status()
        for att, val in atts.items():
            mdlhist["flows"][flowname][att][t_ind] = val
def update_fxnhist(mdl, mdlhist, t_ind):
    for fxnname, fxn in mdl.fxns.items():
        states, faults = fxn.return_states()
        mdlhist["functions"][fxnname]["faults"][t_ind]=faults
        for state, value in states.items():
            mdlhist["functions"][fxnname][state][t_ind] = value 

#init_mdlhist
# initialize history of model
def init_mdlhist(mdl, timerange):
    mdlhist={}
    mdlhist["flows"]=init_flowhist(mdl, timerange)
    mdlhist["functions"]=init_fxnhist(mdl, timerange)
    mdlhist["time"]=np.array([i for i in timerange])
    return mdlhist
def init_flowhist(mdl, timerange):
    flowhist={}
    for flowname, flow in mdl.flows.items():
        atts=flow.status()
        flowhist[flowname] = {}
        for att, val in atts.items():
            flowhist[flowname][att] = np.full([len(timerange)], val)
    return flowhist
def init_fxnhist(mdl, timerange):
    fxnhist = {}
    for fxnname, fxn in mdl.fxns.items():
        states, faults = fxn.return_states()
        fxnhist[fxnname]={}
        fxnhist[fxnname]["faults"]=[faults for i in timerange]
        for state, value in states.items():
            fxnhist[fxnname][state] = np.full([len(timerange)], value)
    return fxnhist




        
            



    
    
    
    
