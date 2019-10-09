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
    for flow in flowhist['faulty']:
        fig = plt.figure()
        plots=len(flowhist['faulty'][flow])
        fig.add_subplot(np.ceil((plots+1)/2),2,plots)
        plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])
        n=1
        for var in flowhist['faulty'][flow]:
            plt.subplot(np.ceil((plots+1)/2),2,n)
            n+=1
            a, =plt.plot(flowhist['faulty'][flow][var], color='r')
            b, =plt.plot(flowhist['nominal'][flow][var], color='b')
            c =plt.axvline(x=time, color='k')
            plt.title(var)
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
    
    faults=findfaults(g)   
    faultflows,faultedges=findfaultflows(g)
    nx.draw_networkx_nodes(g, pos, nodelist=faults.keys(),node_color = 'r',\
                          node_shape='s',width=3, font_weight='bold', node_size = 2000)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges.keys(), edge_color='r', width=2)
    
    nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
    
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
    
    #FUNCTION  | MODE  | TIME  | EFFECTS  |  RATE  |  COST  |  EXP COST
    vals=  [[function],[mode],[time],\
            [str(list(endresult['flows'].keys())+list(endresult['faults'].keys()))],\
            [endresult['classification']['rate']], \
            [endresult['classification']['cost']],[endresult['classification']['expected cost']]]
    cnames=['Function', 'Mode', 'Time', 'Effects', 'Rate', 'Cost', 'Expected Cost']
    t = Table(vals, names=cnames)
    return t

## FAULT PROPAGATION

#constructnomscen
# creates a nominal scenario nomscen given a graph object g by setting all function modes to nominal
def constructnomscen(g):
    fxnnames=list(g.nodes)
    nomscen={'faults':{},'properties':{}}
    for fxnname in fxnnames:
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
    graph=mdl.initialize()
    nomscen=constructnomscen(graph)
    scen=nomscen.copy()
    endresults, resgraph, flowhist, graphhist =runonefault(mdl, scen, track, gtrack)
    
    return endresults,resgraph, flowhist, graphhist

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
def proponefault(mdl, fxnname, faultmode, time=0, track={}, gtrack={}):
    graph=mdl.initialize()
    nomscen=constructnomscen(graph)
    scen=nomscen.copy()
    scen['faults'][fxnname]=faultmode
    scen['properties']['type']='single fault'
    scen['properties']['function']=fxnname
    scen['properties']['fault']=faultmode
    scen['properties']['rate']=getfaultprops(fxnname, faultmode, graph, prop='rate')
    scen['properties']['time']=time
    
    endresults, resgraph, flowhist, graphhist =runonefault(mdl, scen, track, gtrack)
    
    return endresults,resgraph, flowhist, graphhist

#listinitfaults
# creates a list of single-fault scenarios for the graph, given the modes set up in the fault model
# inputs: model graph, a vector of times for the scenarios to occur
# outputs: a list of fault scenarios, where a scenario is defined as:
#   {faults:{functions:faultmodes}, properties:{(changes depending scenario type)} }
def listinitfaults(g, times=[0]):
    faultlist=[]
    fxnnames=list(g.nodes)
        
    try:
        for time in times:
            for fxnname in fxnnames:
                fxn=getfxn(fxnname,g)
                modes=fxn.faultmodes
                
                for mode in modes:
                    nomscen=constructnomscen(g)
                    newscen=nomscen.copy()
                    newscen['faults'][fxnname]=mode
                    rate=getfaultprops(fxnname, mode, g, prop='rate')
                    newscen['properties']={'type': 'single-fault', 'function': fxnname, 'fault': mode, 'rate': rate, 'time': time}
                    faultlist.append(newscen)
    except: 
        print('Incomplete Function Definition, function: '+fxnname)
    return faultlist

#proplist
# creates and propagates a list of failure scenarios in a model
# input: mdl, the module where the model was set up
# output: resultsdict, a dictionary with the results (may be deprecated in the future?)
#         resultstab, a FMEA-style table of results
def proplist(mdl):
    
    graph=mdl.initialize()
    times=mdl.times
    scenlist=listinitfaults(graph, times)
    resultsdict={} 
    
    numofscens=len(scenlist)
    
    fxns=np.zeros(numofscens, dtype='S25')
    modes=np.zeros(numofscens, dtype='S25')
    times=np.zeros(numofscens, dtype=int)
    effects=['']*numofscens
    rates=np.zeros(numofscens, dtype=float)
    costs=np.zeros(numofscens, dtype=float)
    expcosts=np.zeros(numofscens, dtype=float)

    for i, scen in enumerate(scenlist):
        
        endresults, resgraph, flowhist, graphhist=runonefault(mdl, scen)
        
        resultsdict[scen['properties']['function'],scen['properties']['fault'], scen['properties']['time']]=endresults
        
        fxns[i]=scen['properties']['function']
        modes[i]=scen['properties']['fault']
        times[i]=scen['properties']['time']
        effects[i]=str(endresults['flows'])+str(endresults['faults'])        
        rates[i]=endresults['classification']['rate']
        costs[i]=endresults['classification']['cost']
        expcosts[i]=endresults['classification']['expected cost']
    
    vals=[fxns, modes, times, effects, rates, costs, expcosts]
    cnames=['Function', 'Mode', 'Time', 'Effects', 'Rate', 'Cost', 'Expected Cost']
    resultstab = Table(vals, names=cnames)
    
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
    endfaults=findfaults(resgraph)
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
    nomgraph=mdl.initialize()
    nomscen=constructnomscen(nomgraph)
    graph=mdl.initialize()
    timerange=mdl.times
    flowhist={}
    graphhist={}
    time=scen['properties']['time']
    if track:
        for runtype in ['nominal','faulty']:
            flowhist[runtype]={}
            for flow in track:
                flowobj=getflow(flow, graph)
                flowhist[runtype][flow]=flowobj.status()
                for var in flowobj.status():
                    flowhist[runtype][flow][var]=[]
    
    for rtime in range(timerange[0], timerange[-1]+1):
        propagate(nomgraph, nomscen['faults'], rtime)
        if rtime==time:
            propagate(graph, scen['faults'], rtime)
        else:
            propagate(graph,nomscen['faults'],rtime)
        if track:
            for flow in track:
                flowobj=getflow(flow, graph)
                nomflowobj=getflow(flow, nomgraph)
                for var in flowobj.status():
                    flowhist['nominal'][flow][var]=flowhist['nominal'][flow][var]+[nomflowobj.status()[var]]
                    flowhist['faulty'][flow][var]=flowhist['faulty'][flow][var]+[flowobj.status()[var]]
        if rtime in gtrack:
            rgraph=makeresultsgraph(graph,nomgraph)
            graphhist[rtime]=rgraph
            
    resgraph=makeresultsgraph(graph, nomgraph)        
    endflows, endfaults, endclass = classifyresults(mdl,resgraph, scen)
    endresults={'flows': endflows, 'faults': endfaults, 'classification':endclass}
    return endresults, resgraph, flowhist, graphhist

#propogate
# propagates faults through the graph at one time-step
# inputs:
#   g, the graph object of the model
#   initfaults, the faults (or lack of faults) to initiate in the model
#   time, the time propogation occurs at
def propagate(g, initfaults, time):
    fxnnames=list(g.nodes())
    activefxns=set(fxnnames)
    #set up history of flows to see if any has changed
    tests={}
    flowhist={}
    for fxnname in fxnnames:
        tests[fxnname]=0
        edges=list(g.in_edges(fxnname))+list(g.out_edges(fxnname))
        for big, end in edges:
            flows=g.edges[big,end]
            for flow in flows:
                flowhist[big, end,flow]=g.edges[big, end][flow].status()
                tests[fxnname]+=1
     #initialize fault           
    for fxnname in initfaults:
        if initfaults[fxnname]!='nom':
            fxn=getfxn(fxnname, g)
            fxn.updatefxn(faults=[initfaults[fxnname]], time=time)
    n=0
    while activefxns:
        funclist=list(activefxns).copy()
        for fxnname in funclist:
            fxn=getfxn(fxnname, g)
            fxn.updatefxn(time=time)
            test=0
            edges=list(g.in_edges(fxnname))+list(g.out_edges(fxnname))
            for big, end in edges:
                flows=g.edges[big,end]
                for flow in flows:
                    if g.edges[big, end][flow].status()!=flowhist[big, end, flow]:
                        activefxns.add(big)
                        activefxns.add(end)
                    else:
                        test+=1
                    flowhist[big, end, flow]=g.edges[big, end][flow].status()
                if test>=tests[fxnname]:
                    activefxns.discard(fxnname)
        n+=1
        if n>1000:
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
    return rg

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

#listfaultsprops
# gets the properties of a list of faults
# inputs:
#       - endfaults, a dictionary {function:fault} of the faults
#       - g, the model graph
#       - prop, the property to list (if not all)
# outputs:
#       - faultlist, a dict of properties for each fault 
def listfaultsprops(endfaults,g, prop='all'):
    faultlist=dict()
    for fxnname in endfaults:
        for faultname in endfaults[fxnname]:
            if prop==all:
                faultlist[fxnname+' '+faultname]=getfaultprops(fxnname,faultname,g)
            else:
                faultlist[fxnname+' '+faultname]=getfaultprops(fxnname,faultname,g, prop)
    return faultlist

#getfaultprops
# inputs:
#       -fxnname, the name of the function
#       -faultname, the name of the fault
#       -g, the model graph
#       -prop, the property to find
# outputs: faultprops, the properties of that fault in a dict
def getfaultprops(fxnname, faultname, g, prop='all'):
    fxn=getfxn(fxnname, g)
    if prop=='all':
        faultprops=fxn.faultmodes[faultname]
    else:
        faultprops=fxn.faultmodes[faultname][prop]
    return faultprops

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

#resetgraph
# resets the graph to nominal conditions (WARNING: UNTESTED & EXPERIMENTAL)
# may not work with graphs with components!
def resetgraph(g):
    #reset flows
    for edge in g.edges:
        flows=g.get_edge_data(edge[0],edge[1])
        for flowname in flows:
            flow=flows[flowname]['obj']
            flow.reset()
    #reset functions
    for fxnname in g.nodes:
        fxn=getfxn(fxnname, g)
        fxn.reset()
        
            



    
    
    
    
