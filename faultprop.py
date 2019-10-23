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
                b, = plt.plot(flowhists['faulty'][flow][var], color='r')
                c = plt.axvline(x=time, color='k')
            a, =plt.plot(flowhists['nominal'][flow][var], color='b')
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
    for time in ghist:
        graph=ghist[time]
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
    flowhist, graphhist =proponescen(mdl, scen, track, gtrack)
    
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
def proponefault(mdl, fxnname, faultmode, time=0, track={}, gtrack={},graph={}):
    
    #run model nominally, get relevant results
    nomscen=constructnomscen(mdl)
    nomflowhist, nomgraphhist = proponescen(mdl, nomscen, track, gtrack)
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
    
    faultflowhist, faultgraphhist = proponescen(mdl, scen, track, gtrack)
    resgraph = mdl.returnstategraph()
    endfaults = mdl.returnfaultmodes()
    #process model run
    endflows = comparegraphflows(resgraph, nomresgraph)
    endclass=mdl.findclassification(resgraph, endfaults, endflows, scen)
    flowhist={'nominal':nomflowhist, 'faulty':faultflowhist}
    
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}  
    
    mdl.reset()
    return endresults,resgraph, flowhist, faultgraphhist

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
# output: resultsdict, a dictionary with the results (may be deprecated in the future?)
#         resultstab, a FMEA-style table of results
def proplist(mdl, reuse=False):

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
    
    
    
    for i, scen in enumerate(scenlist):
        if reuse: 
            endresults, resgraph, flowhist, graphhist=runonefault(mdl, scen)  
            mdl.reset()
        else: 
            endresults, resgraph, flowhist, graphhist=runonefault(mdl, scen)
            mdl = mdl.__class__()
        
        resultsdict[scen['properties']['function'],scen['properties']['fault'], scen['properties']['time']]=endresults
        
        fxns[i]=scen['properties']['function']
        modes[i]=scen['properties']['fault']
        times[i]=scen['properties']['time']
        floweffects[i] = str(endresults['flows'])
        faulteffects[i] = str(endresults['faults'])        
        rates[i]=endresults['classification']['rate']
        costs[i]=endresults['classification']['cost']
        expcosts[i]=endresults['classification']['expected cost']
    
    vals=[fxns, modes, times, faulteffects, floweffects, rates, costs, expcosts]
    cnames=['Function', 'Mode', 'Time', 'End Faults', 'End Flow Effects', 'Rate', 'Cost', 'Expected Cost']
    resultstab = Table(vals, names=cnames)
    mdl.reset()
    
    return resultsdict, resultstab

#classifyresults
# finds whether conditional faults have been added, flows are degraded, and how bad that is per the model definition
# inputs:
#   - mdl, the model module defined in mdl.py
#   - resgraph, the graph object with a particular result
#   - scen, the fault scenario for a given model
# outputs:
#   - endflows, a dictionary of degraded flows at t=end
#   - endfaults, a dictionary of faults present in the model at t=end
#   - endclass, a dict with the classification of the scenario, which includes rate, cost, expected cost
def classifyresults(mdl,resgraph, scen):
    endflows,endedges=findfaultflows(resgraph)
    endfaults =mdl.returnfaultmodes()
    endclass=mdl.findclassification(resgraph, endfaults, endflows, scen)
    return endflows, endfaults, endclass

#runonefault
# runs a single fault scenario in the model over time
# inputs:
#   - mdl, the model module defined in mdl.py
#   - scen, the fault scenario for a given model
#   - track, a list of flows to track
#   - gtrack, the times to take a snapshot of the graph 
# outputs:
#   - endresults, a dictionary summary of results at the end of the simulation with structure
#    {flows:{flow:attribute:value},faults:{function:{faults}}, classification:{rate:val, cost:val, expected cost: val} }
#   - resgraph, a graph object with function faults and degraded flows noted
#   - flowhist, a dictionary with the history of the flow over time
#   - graphhist, a dictionary of results graph objects over time with structure {time:graph}
def runonefault(mdl, scen, track={}, gtrack={}):
    nomscen=constructnomscen(mdl)
    nommdl = mdl.__class__()
    
    timerange=mdl.times
    flowhist={}
    graphhist={}
    time=scen['properties']['time']
    if track:
        for runtype in ['nominal','faulty']:
            flowhist[runtype]={}
            for flow in track:
                flowobj=mdl.flows[flow]
                flowhist[runtype][flow]=flowobj.status()
                for var in flowobj.status():
                    flowhist[runtype][flow][var]=[]
    
    for rtime in range(timerange[0], timerange[-1]+1):
        propagate(nommdl, nomscen['faults'], rtime)
        if rtime==time:
            propagate(mdl, scen['faults'], rtime)
        else:
            propagate(mdl,nomscen['faults'],rtime)
        if track:
            for flow in track:
                flowobj=mdl.flows[flow]
                nomflowobj=nommdl.flows[flow]
                for var in flowobj.status():
                    flowhist['nominal'][flow][var]=flowhist['nominal'][flow][var]+[nomflowobj.status()[var]]
                    flowhist['faulty'][flow][var]=flowhist['faulty'][flow][var]+[flowobj.status()[var]]
        if rtime in gtrack:
            rgraph=makeresultsgraph(mdl.graph,nommdl.graph)
            graphhist[rtime]=rgraph
            
    resgraph=makeresultsgraph(mdl.graph,nommdl.graph)        
    endflows, endfaults, endclass = classifyresults(mdl,resgraph, scen)
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}
    return endresults, resgraph, flowhist, graphhist

#proponescen
# runs a single fault scenario in the model over time
# inputs:
#   - mdl, the model object 
#   - scen, the fault scenario for a given model
#   - track, a list of flows to track
#   - gtrack, the times to take a snapshot of the graph 
# outputs:
#   - flowhist, a dictionary with the history of the flow over time
#   - graphhist, a dictionary of results graph objects over time with structure {time:graph}
#   (note, this causes changes in the model of interest, also)

def proponescen(mdl, scen, track={}, gtrack={}):
    
    timerange= range(mdl.times[0], mdl.times[-1]+1) 
    # initialize dict of tracked flows
    flowhist={}
    if track:
        for flowname in track:
                flowhist[flowname]=mdl.flows[flowname].status()
                for var in flowhist[flowname]:
                    flowhist[flowname][var]=[{} for _ in timerange]
    # run model through the time range defined in the object
    nomscen=constructnomscen(mdl)
    graphhist=[]
    for rtime in timerange:
       # inject fault when it occurs, track defined flow states and graph
        if rtime==scen['properties']['time']: propagate(mdl, scen['faults'], rtime)
        else: propagate(mdl,nomscen['faults'],rtime)
        if track:
            for flowname in track:
                for var in mdl.flows[flowname].status():
                    flowhist[flowname][var][rtime]=[mdl.flows[flowname].status()[var]]
        if rtime in gtrack:
            graphhist[rtime]=mdl.returnstategraph()
    return flowhist, graphhist

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
            flowobj=g.edges[edge][flow]
            nomflowobj=nomg.edges[edge][flow]
            
            if flowobj.status()!=nomflowobj.status():
                status='Degraded'
            else:
                status='Nominal'
            rg.edges[edge][flow]={'values':flowobj.status(),'status':status, 'obj':flowobj}
    for node in g.nodes:
        faults=findfault(node, g)
        rg.nodes[node]['faults']=faults
        fxn=getfxn(node, g)
        state, _ =fxn.returnstates()
        nomfxn=getfxn(node, nomg)
        nomstate, _ =nomfxn.returnstates()
        if faults: status='Faulty' 
        elif state!=nomstate: status='Degraded'
        else: status='Nominal'
        rg.nodes[node]['state']=state
        rg.nodes[node]['status']=status
    return rg

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

#USEFUL MISC FUNCTIONS

#findfaults
# generates a dict of faults present in each function endfaults given graph g
def findfaults(g):
    endfaults=dict()
    fxnnames=list(g.nodes)
    #extract list of faults present
    for fxnname in fxnnames:
        faults=findfault(fxnname, g)
        if len(faults) > 0:
            endfaults[fxnname]=faults
    return endfaults

#findfault
#find an individual fault in a given function
def findfault(fxnname, g):
    if 'faults' in g.nodes[fxnname]:
        faults=g.nodes[fxnname]['faults']
    else:
        fxn=getfxn(fxnname, g)
        faults=fxn.faults.copy()
        if faults.issuperset({'nom'}):
            faults.remove('nom')
        if faults.issuperset({'nominal'}):
            faults.remove('nominal')
    return faults
#getfxn
# gets the function object fxn in the model graph with the name fxnname
def getfxn(fxnname, graph):
    fxn=graph.nodes(data='obj')[fxnname]
    return fxn


#DEPRECATED???
#getflow
# gets the flow object flowobj in the model graph g with the name flowname
def getflow(flowname, g):
    for edge in g.edges:
        flows=g.get_edge_data(edge[0],edge[1])
        #flows=list(g.get_edge_data(edge[0],edge[1]).keys())
        for flow in flows:
            if flow==flowname:
                if type(flows[flow]) is dict:
                    flowobj=flows[flow]['obj']
                else:
                    flowobj=flows[flow]
    return flowobj
        
            



    
    
    
    
