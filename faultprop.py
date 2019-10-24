# -*- coding: utf-8 -*-
"""
File name: faultprop.py
Author: Daniel Hulse
Created: December 2018
Forked from the IBFM toolkit, original author Matthew McIntire

Description: functions to propagate faults through a user-defined fault model
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from astropy.table import Table, Column


##PLOTTING AND RESULTS DISPLAY

#plotflowhist
# displays plots of a history of flow states over time
# inputs: 
#   - flowhist, the history of one or more flows over time stored in a dictionary with structure:
#       {nominal/faulty: {flow: {attribute: [values]}}}, where
#           - nominal/nominal keeps the history ifor both faulty and nominal flows
#           - flow is all flows that were tracked 
#           - attribute is the defined attributes of that flow (e.g. rate/effort/etc)
#           - values is a list of values that attribute takes over time
#   - fault, name of the fault that was injected (for the titles)
#   - time, the time in which the fault was initiated (so that time is displayed on the graph)
def plotflowhist(flowhist, fault='', time=0):
    flowhists={}
    if 'nominal' not in flowhist: flowhists['nominal']=flowhist
    else: flowhists=flowhist
    
    for flow in flowhists['nominal']:
        fig = plt.figure()
        plots=len(flowhists['nominal'][flow])
        fig.add_subplot(np.ceil((plots+1)/2),2,plots)
        plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])
        n=1
        for var in flowhists['nominal'][flow]:
            plt.subplot(np.ceil((plots+1)/2),2,n)
            n+=1
            if 'faulty' in flowhists:
                a, = plt.plot(flowhists['faulty'][flow][var], color='r')
                c = plt.axvline(x=time, color='k')
            b, =plt.plot(flowhists['nominal'][flow][var], color='b')
            plt.title(var)
        if 'faulty' in flowhists:
            plt.subplot(np.ceil((plots+1)/2),2,n)
            plt.legend([a,b],['faulty', 'nominal'])
        fig.suptitle('Dynamic Response of '+flow+' to fault'+' '+fault)
        plt.show()

#plotghist
# displays plots of the graph over time
# inputs:
#   - ghist, a dictionary of the history of the graph over time with structure:
#       {time: graphobject}, where
#           - time is the time where the snapshot of the graph was recorded
#           - graphobject is the snapshot of the graph at that time
#    - faultscen, the name of the fault scenario where this graph occured
def plotghist(ghist,faultscen=[]):
    for time, graph in ghist.items():
        showgraph(graph, faultscen, time)

#showgraph
# plots a single graph at a single time
# inputs:
#   - g, the graph object
#   - faultscen, the name of the fault scenario (for the title)
#   - time, the time of the fault scenario (also for the title)
def showgraph(g, faultscen=[], time=[]):
    labels=dict()
    for edge in g.edges:
        flows=list(g.get_edge_data(edge[0],edge[1]).keys())
        labels[edge[0],edge[1]]=flows
    
    pos=nx.shell_layout(g)
    #Add ability to label modes/values
    
    nx.draw_networkx(g,pos,node_size=2000,node_shape='s', node_color='g', \
                     width=3, font_weight='bold')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
    
    if list(g.nodes(data='status'))[0][1]:
        statuses=dict(g.nodes(data='status', default='Nominal'))
        faultnodes=[node for node,status in statuses.items() if status=='Faulty']
        
        degradednodes=[node for node,status in statuses.items() if status=='Degraded']
        
        nx.draw_networkx_nodes(g, pos, nodelist=degradednodes,node_color = 'y',\
                          node_shape='s',width=3, font_weight='bold', node_size = 2000)
        nx.draw_networkx_nodes(g, pos, nodelist=faultnodes,node_color = 'r',\
                          node_shape='s',width=3, font_weight='bold', node_size = 2000)
        faultflows,faultedges=findfaultflows(g)
        nx.draw_networkx_edges(g,pos,edgelist=faultedges.keys(), edge_color='r', width=2)

        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedges, font_color='r')
    
    if faultscen:
        plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    
    plt.show()

#printresult (maybe find a better name?)
# prints the results of a run in a nice FMEA-style table
# inputs:
#   - function: the function the mode occured in
#   - mode: the mode of the scenario
#   - time: the time the fault occured in
#   - endresult: the results dict given by the model after propagation
def printresult(function, mode, time, endresult):
    
    #FUNCTION  | MODE  | TIME  | FAULT EFFECTS | FLOW EFFECTS |  RATE  |  COST  |  EXP COST
    vals=  [[function],[mode],[time], [str(list(endresult['faults'].keys()))], \
            [str(list(endresult['flows'].keys()))] ,\
            [endresult['classification']['rate']], \
            [endresult['classification']['cost']],[endresult['classification']['expected cost']]]
    cnames=['Function', 'Mode', 'Time', 'End Faults', 'End Flow Effects', 'Rate', 'Cost', 'Expected Cost']
    t = Table(vals, names=cnames)
    return t

## FAULT PROPAGATION

#constructnomscen
# creates a nominal scenario nomscen given a graph object g by setting all function modes to nominal
def constructnomscen(mdl):
    nomscen={'faults':{},'properties':{}}
    for fxnname in mdl.fxns:
        nomscen['faults'][fxnname]='nom'
    nomscen['properties']['time']=0.0
    nomscen['properties']['type']='nominal'
    return nomscen

#runnominal
# runs the model over time in the nominal scenario
# inputs:
#   - mdl, the python model module set up in mdl.py
#   - track, the flows to track (a list of strings)
#   - gtrack, the times to snapshot the graph
# outputs:
#   - endresults, a dictionary summary of results at the end of the simulation with structure
#    {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
#   - resgraph, a graph object with function faults and degraded flows noted
#   - flowhist, a dictionary with the history of the flow over time
#   - graphhist, a dictionary of results graph objects over time with structure {time:graph}
def runnominal(mdl, track={}, gtrack={}):
    nomscen=constructnomscen(mdl)
    scen=nomscen.copy()
    flowhist, graphhist, _ =proponescen(mdl, scen, track, gtrack)
    
    resgraph = mdl.returnstategraph()   
    endfaults = mdl.returnfaultmodes()
    endflows={}
    endclass=mdl.findclassification(resgraph, endfaults, endflows, scen)
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}
    
    mdl.reset()
    return endresults, resgraph, flowhist, graphhist

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
def runonefault(mdl, fxnname, faultmode, time=0, track={}, gtrack={},graph={}, staged=False):
    
    #run model nominally, get relevant results
    nomscen=constructnomscen(mdl)
    if staged:
        nomflowhist, nomgraphhist, mdls = proponescen(mdl, nomscen, track, gtrack, ctimes=[time])
        nomresgraph = mdl.returnstategraph()
        mdl.reset()
        mdl = mdls[0]
    else:
        nomflowhist, nomgraphhist, _ = proponescen(mdl, nomscen, track, gtrack)
        nomresgraph = mdl.returnstategraph()
        mdl.reset()
    #run with fault present, get relevant results
    scen=nomscen.copy() #note: this is a shallow copy, so don't define it earlier
    scen['faults'][fxnname]=faultmode
    scen['properties']['type']='single fault'
    scen['properties']['function']=fxnname
    scen['properties']['fault']=faultmode
    scen['properties']['rate']=mdl.fxns[fxnname].faultmodes[faultmode]['rate']
    scen['properties']['time']=time
    
    faultflowhist, faultgraphhist, _ = proponescen(mdl, scen, track, gtrack, staged=staged, prevhist=nomflowhist, prevghist=nomgraphhist)
    faultresgraph = mdl.returnstategraph()
    
    #process model run
    endfaults = mdl.returnfaultmodes()
    endflows = comparegraphflows(faultresgraph, nomresgraph)
    endclass = mdl.findclassification(faultresgraph, endfaults, endflows, scen)
    resgraph = makeresultsgraph(faultresgraph, nomresgraph)
    resgraphhist = makeresultsgraphs(faultgraphhist, nomgraphhist)
    
    flowhist={'nominal':nomflowhist, 'faulty':faultflowhist}
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, flowhist, resgraphhist

#listinitfaults
# creates a list of single-fault scenarios for the graph, given the modes set up in the fault model
# inputs: model graph, a vector of times for the scenarios to occur
# outputs: a list of fault scenarios, where a scenario is defined as:
#   {faults:{functions:faultmodes}, properties:{(changes depending scenario type)} }
def listinitfaults(mdl):
    faultlist=[]
    for time in mdl.times:
        for fxnname, fxn in mdl.fxns.items():
            modes=fxn.faultmodes
            
            for mode in modes:
                nomscen=constructnomscen(mdl)
                newscen=nomscen.copy()
                newscen['faults'][fxnname]=mode
                rate=mdl.fxns[fxnname].faultmodes[mode]['rate']
                newscen['properties']={'type': 'single-fault', 'function': fxnname, 'fault': mode, 'rate': rate, 'time': time}
                faultlist.append(newscen)

    return faultlist

#proplist
# creates and propagates a list of failure scenarios in a model
# input: mdl, the module where the model was set up
# output: resultstab, a FMEA-style table of results
def runlist(mdl, reuse=False):

    scenlist=listinitfaults(mdl)
    resultsdict={} 
    
    numofscens=len(scenlist)
    
    fxns=np.zeros(numofscens, dtype='S25')
    modes=np.zeros(numofscens, dtype='S25')
    times=np.zeros(numofscens, dtype=int)
    floweffects=['']*numofscens
    faulteffects=['']*numofscens
    rates=np.zeros(numofscens, dtype=float)
    costs=np.zeros(numofscens, dtype=float)
    expcosts=np.zeros(numofscens, dtype=float)
    
    #run model nominally, get relevant results
    nomscen=constructnomscen(mdl)
    nomflowhist, nomgraphhist, c_mdl = proponescen(mdl, nomscen, {}, {})
    nomresgraph = mdl.returnstategraph()
    mdl.reset()
    
    for i, scen in enumerate(scenlist):
        #run model with fault scenario
        endresults, resgraph, _ =proponescen(mdl, scen, {}, {})
        endfaults = mdl.returnfaultmodes()
        resgraph = mdl.returnstategraph()
        
        endflows = comparegraphflows(resgraph, nomresgraph)
        endclass = mdl.findclassification(resgraph, endfaults, endflows, scen)
        if reuse: mdl.reset()
        else: mdl = mdl.__class__()
        #populate columns for results table
        fxns[i]=scen['properties']['function']
        modes[i]=scen['properties']['fault']
        times[i]=scen['properties']['time']
        floweffects[i] = str(endflows)
        faulteffects[i] = str(endfaults)        
        rates[i]=endclass['rate']
        costs[i]=endclass['cost']
        expcosts[i]=endclass['expected cost']
    #create results table
    vals=[fxns, modes, times, faulteffects, floweffects, rates, costs, expcosts]
    cnames=['Function', 'Mode', 'Time', 'End Faults', 'End Flow Effects', 'Rate', 'Cost', 'Expected Cost']
    resultstab = Table(vals, names=cnames)
    
    return resultstab

#proponescen
# runs a single fault scenario in the model over time
# inputs:
#   - mdl, the model object 
#   - scen, the fault scenario for a given model
#   - track, a list of flows to track
#   - gtrack, the times to take a snapshot of the graph 
#   - staged, the starting time for the propagation
#   - ctimes, the time to copy the models 
# outputs:
#   - flowhist, a dictionary with the history of the flow over time
#   - graphhist, a dictionary of results graph objects over time with structure {time:graph}
#   (note, this causes changes in the model of interest, also)
#   - c_mdls, copies of the model object taken at each time listed in ctime

def proponescen(mdl, scen, track={}, gtrack={}, staged=False, ctimes=[], prevhist={}, prevghist={}):
    #if staged, we want it to start a new run from the starting time of the scenario,
    # using a copy of the input model (which is the nominal run) at this time
    if staged:
        timerange=range(scen['properties']['time'], mdl.times[-1]+1)
        flowhist=copy.deepcopy(prevhist)
        graphhist=copy.deepcopy(prevghist)
    else: 
        timerange= range(mdl.times[0], mdl.times[-1]+1) 
        # initialize dict of tracked flows
        flowhist={}
        graphhist=dict.fromkeys(gtrack)
        if track:
            for flowname in track:
                    flowhist[flowname]=mdl.flows[flowname].status()
                    for var in flowhist[flowname]:
                        flowhist[flowname][var]=[{} for _ in timerange]
    # run model through the time range defined in the object
    nomscen=constructnomscen(mdl)
    c_mdl=[]
    for rtime in timerange:
       # inject fault when it occurs, track defined flow states and graph
        if rtime==scen['properties']['time']: propagate(mdl, scen['faults'], rtime)
        else: propagate(mdl,nomscen['faults'],rtime)
        if track:
            for flowname in track:
                for var in mdl.flows[flowname].status():
                    flowhist[flowname][var][rtime]=mdl.flows[flowname].status()[var]
        if rtime in gtrack: graphhist[rtime]=mdl.returnstategraph()
        if rtime in ctimes: c_mdl.append(mdl.copy())
    return flowhist, graphhist, c_mdl

#propogate
# propagates faults through the graph at one time-step
# inputs:
#   g, the graph object of the model
#   initfaults, the faults (or lack of faults) to initiate in the model
#   time, the time propogation occurs at
def propagate(mdl, initfaults, time):
    #set up history of flows to see if any has changed
    tests={}
    flowhist={}
    newflowhist={}
    #Step 1: Find out what the current value of the flows are
    for flowname, flow in mdl.flows.items():
        flowhist[flowname]=flow.status()
    #Step 2: Inject faults if present     
    for fxnname in initfaults:
        if initfaults[fxnname]!='nom':
            fxn=mdl.fxns[fxnname]
            fxn.updatefxn(faults=[initfaults[fxnname]], time=time)
    #Step 3: Propagate faults through graph
    n=0
    activefxns=set(mdl.fxns)
    nextfxns=set()
    while activefxns:
        for fxnname in list(activefxns).copy():
            #Update functions with new values
            fxn=mdl.fxns[fxnname]
            fxn.updatefxn(time=time)
        #Check to see what flows have new values and add connected functions
        for flowname, flow in mdl.flows.items():
            if flowhist[flowname]!=flow.status():
                nextfxns.update(set([n for n in mdl.bipartite.neighbors(flowname)]))
            flowhist[flowname]=flow.status()
        activefxns=nextfxns.copy()
        nextfxns.clear()
        n+=1
        
        if n>1000: #break if this is going for too long
            print("Undesired looping in function")
            print(initfaults)
            print(fxnname)
            break
    return

#makeresultsgraph
# creates a snapshot of the graph structure with model results superimposed
# inputs: g, the graph, and nomg, the graph in its nominal state
# outputs: rg, the graph snapshot
def makeresultsgraph(g, nomg):
    rg=g.copy() 
    for edge in g.edges:
        for flow in list(g.edges[edge].keys()):            
            if g.edges[edge][flow]!=nomg.edges[edge][flow]: status='Degraded'
            else:                               status='Nominal' 
            rg.edges[edge][flow]={'values':g.edges[edge][flow],'status':status}
    for node in g.nodes:        
        if g.nodes[node]['modes'].difference(['nom']): status='Faulty' 
        elif g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
        else: status='Nominal'
        rg.nodes[node]['status']=status
    return rg

def makeresultsgraphs(ghist, nomghist):
    rghist = dict.fromkeys(ghist.keys())
    for i,rg in rghist.items():
        rghist[i] = makeresultsgraph(ghist[i],nomghist[i])
    return rghist

#comparegraphflows
# extracts non-nominal flows by comparing the a results graph with a nominal results graph
# inputs:   g, a graph of results, with states of each flow in each provided
#           nomg, the same graph for the nominal system
# outputs:  endflows, a dictionary of degraded flows
# (maybe do this for values also???)
def comparegraphflows(g, nomg):
    endflows=dict()
    for edge in g.edges:
        flows=g.get_edge_data(edge[0],edge[1])
        nomflows=nomg.get_edge_data(edge[0],edge[1])
        for flow in flows:
            if flows[flow]!=nomflows[flow]: endflows[flow]=flows[flow]
    return endflows
#findfaultflows
# extracts non-nominal flow paths by comparing the graph with a nominal version of the graph
# inputs: g, the graph, and nomg, the graph in its nominal state
# outputs: 
#           -endflows, a dict of degraded flows
#           -endedges, a dict of degraded edges
def findfaultflows(g, nomg=[]):
    endflows=dict()
    endedges=dict()
    for edge in g.edges:
        flows=g.get_edge_data(edge[0],edge[1])
        flowedges=[]
        #if comparing a nominal with a non-nominal
        if nomg:
            nomflows=nomg.get_edge_data(edge[0],edge[1])
            for flow in flows:
                if flows[flow].status()!=nomflows[flow].status():
                    endflows[flow]=flows[flow].status()
                    flowedges=flowedges+[flow]
        #if results are already in the graph structure
        else:
            for flow in flows:
                if flows[flow]['status']=='Degraded':
                    endflows[flow]=flows[flow]['values']
                    flowedges=flowedges+[flow]
        if flowedges:
                endedges[edge]=flowedges    
    return endflows, endedges

        
            



    
    
    
    
