# -*- coding: utf-8 -*-
"""
File Name: resultproc.py
Author: Daniel Hulse
Created: November 2018

Description: Results processing and plotting for single (or multiple) fault model runs
    
(module originally defined in faultprop.py--put here for organization)
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import copy
import pandas as pd

## PROCESSING RESULTS 
def compare_hists(mdlhists, returndiff=True):
    reshists={}
    diffs={}
    summaries={}
    nomhist = mdlhists.pop('nominal')
    for scenname, hist in mdlhists.items():
        reshists[scenname], diffs[scenname], summaries[scenname] = compare_hist(hist, nomhist=nomhist, returndiff=returndiff)
    return reshists, diffs, summaries
    
#compare_hist
#find non-nominal states over time
def compare_hist(mdlhist, nomhist={}, returndiff=True):
    if nomhist: mdlhist={'nominal':nomhist, 'faulty':mdlhist}
    reshist = {}
    reshist['time'] = mdlhist['nominal']['time']
    reshist['flowvals'], reshist['flows'], degflows, numdegflows, flowdiff = compare_flowhist(mdlhist, returndiff=returndiff)
    reshist['functions'], numfaults, degfxns, numdegfxns, fxndiff = compare_fxnhist(mdlhist, returndiff=returndiff)
    reshist['stats'] = {'degraded flows': numdegflows, 'degraded functions': numdegfxns, 'total faults': numfaults}
    summary = {'degraded functions': degfxns, 'degraded flows': degflows}
    diff = {**fxndiff, **flowdiff}
    return reshist, diff, summary
def compare_flowhist(mdlhist, returndiff=True):
    flowhist = {}
    summhist = {}
    degflows = []
    diff = {}
    for flowname in mdlhist['nominal']['flows']:
        flowhist[flowname]={}
        diff[flowname]={}
        for att in mdlhist['nominal']['flows'][flowname]:
            faulty  = mdlhist['faulty']['flows'][flowname][att]
            nominal = mdlhist['nominal']['flows'][flowname][att]
            flowhist[flowname][att] = 1* (faulty == nominal)
            if returndiff: diff[flowname][att] = nominal - faulty
        summhist[flowname] = np.prod(np.array(list(flowhist[flowname].values())), axis = 0)
        if 0 in summhist[flowname]: degflows+=[flowname]
    numdegflows = len(summhist) - np.sum(np.array(list(summhist.values())), axis=0)
    return flowhist, summhist, degflows, numdegflows, diff
def compare_fxnhist(mdlhist, returndiff=True):
    fxnhist = {}
    faulthist = {}
    deghist = {}
    degfxns = []
    diff = {}
    for fxnname in mdlhist['nominal']['functions']:
        fhist = copy.copy(mdlhist['faulty']['functions'][fxnname])
        del fhist['faults']
        fxnhist[fxnname] = {}
        diff[fxnname]={}
        for state in fhist:
            faulty  = mdlhist['faulty']['functions'][fxnname][state]
            nominal = mdlhist['nominal']['functions'][fxnname][state] 
            fxnhist[fxnname][state] = 1* (faulty == nominal)
            diff[fxnname][state] = nominal - faulty
        if fxnhist[fxnname]: status = np.prod(np.array(list(fxnhist[fxnname].values())), axis = 0) 
        else: status = np.ones(len(mdlhist['faulty']['functions'][fxnname]['faults']), dtype=int) #should empty be given 1 or nothing?
        fxnhist[fxnname]['faults']=mdlhist['faulty']['functions'][fxnname]['faults']
        faults = mdlhist['faulty']['functions'][fxnname]['faults']
        fxnhist[fxnname]['numfaults']=np.array(list(map(lambda f: len(f.difference(['nom'])), faults)))
        faulty = 1 - 1*(fxnhist[fxnname]['numfaults']>0)
        fxnhist[fxnname]['status'] = status*faulty
        faulthist[fxnname]=fxnhist[fxnname]['numfaults']
        deghist[fxnname] = fxnhist[fxnname]['status']
        if 0 in deghist[fxnname] or any(0 < faulthist[fxnname]): degfxns+=[fxnname]
    numfaults = np.sum(np.array(list(faulthist.values())), axis=0)
    numdegfxns   = len(deghist) - np.sum(np.array(list(deghist.values())), axis=0)
    return fxnhist, numfaults, degfxns, numdegfxns, diff
def make_heatmaps(reshist, diff):
    heatmaps = {'degtime':{},'maxdeg':{}, 'intdeg':{}, 'maxfaults':{}, 'intdiff':{}, 'maxdiff':{}}
    len_time = len(reshist['time'])
    for fxnname in reshist['functions'].keys():
        heatmaps['degtime'][fxnname]=1.0-sum(reshist['functions'][fxnname]['status'])/len_time
        heatmaps['maxfaults'][fxnname] = max(reshist['functions'][fxnname]['numfaults'])
        if diff[fxnname]:
            fxndiff =np.zeros(len(reshist['functions'][fxnname]['status']))
            for valname in diff[fxnname].keys():
                fxndiff = fxndiff + diff[fxnname][valname]   
            heatmaps['intdiff'][fxnname] = sum(fxndiff) /( len_time * len(diff[fxnname].keys()))
            heatmaps['maxdiff'][fxnname] = max(fxndiff) /( len_time * len(diff[fxnname].keys()))
    for flowname in reshist['flows'].keys():
        heatmaps['degtime'][flowname]=1.0 - sum(reshist['flows'][flowname])/len_time
        degraded=np.zeros(len(reshist['flows'][flowname]))
        flowdiff=np.zeros(len(reshist['flows'][flowname]))
        for valname in reshist['flowvals'][flowname].keys():
            degraded = degraded + reshist['flowvals'][flowname][valname]
            flowdiff = flowdiff + diff[flowname][valname]
        heatmaps['maxdeg'][flowname] = max(degraded)
        heatmaps['intdeg'][flowname] = sum(degraded)/len_time
        heatmaps['maxdiff'][flowname] = max(flowdiff) /( len_time * len(diff[flowname].keys()))
        heatmaps['intdiff'][flowname] = sum(flowdiff) /( len_time * len(diff[flowname].keys()))
    return heatmaps
def make_degtimemap(reshist):
    len_time = len(reshist['time'])
    degtimemap={}
    for fxnname in reshist['functions'].keys():
        degtimemap[fxnname]=1.0-sum(reshist['functions'][fxnname]['status'])/len_time
    for flowname in reshist['flows'].keys():
        degtimemap[flowname]=1.0 - sum(reshist['flows'][flowname])/len_time
    return degtimemap
def make_degtimemaps(reshists):
    degtimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        degtimemaps[reshist]=make_degtimemap(reshists[reshist])
    return degtimemaps
def make_avgdegtimeheatmap(reshists):
    degtimetable = pd.DataFrame(make_degtimemaps(reshists)).transpose()
    return degtimetable.mean().to_dict()
def make_expdegtimeheatmap(reshists, endclasses):
    degtimetable = pd.DataFrame(make_degtimemaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expdegtimetable = degtimetable.multiply(rates).transpose()
    return expdegtimetable.sum().to_dict()
def make_faultmap(reshist):
    heatmap={}
    for fxnname in reshist['functions'].keys():
        heatmap[fxnname] = max(reshist['functions'][fxnname]['numfaults'])
    return heatmap
def make_faultmaps(reshists):
    faulttimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        faulttimemaps[reshist]=make_faultmap(reshists[reshist])
    return faulttimemaps
def make_faultsheatmap(reshists):
    faulttable = pd.DataFrame(make_faultmaps(reshists)).transpose()
    return faulttable.mean().to_dict()
def make_expfaultsheatmap(reshists, endclasses):
    faulttable = pd.DataFrame(make_faultmaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expfaulttable = faulttable.multiply(rates).transpose()
    return expfaulttable.mean().to_dict()

#compare_graphflows
# extracts non-nominal flows by comparing the a results graph with a nominal results graph
# inputs:   g, a graph of results, with states of each flow in each provided
#           nomg, the same graph for the nominal system
# outputs:  endflows, a dictionary of degraded flows
# (maybe do this for values also???)
def compare_graphflows(g, nomg, gtype='normal'):
    endflows=dict()
    if gtype=='normal':
        for edge in g.edges:
            flows=g.get_edge_data(edge[0],edge[1])
            nomflows=nomg.get_edge_data(edge[0],edge[1])
            for flow in flows:
                if flows[flow]!=nomflows[flow]:
                    endflows[flow]={}
                    vals=flows[flow]
                    for val in vals:
                        if vals[val]!=nomflows[flow][val]: endflows[flow][val]=flows[flow][val]
    elif gtype=='bipartite':
        for node in g.nodes:
            if g.nodes[node]['bipartite']==1: #only flow states
                if g.nodes[node]['states']!=nomg.nodes[node]['states']:
                    endflows[node]={}
                    vals=g.nodes[node]['states']
                    for val in vals:
                        if vals[val]!=nomg.nodes[node]['states'][val]: endflows[node][val]=vals[val]     
    return endflows
#make_resultsgraph
# creates a snapshot of the graph structure with model results superimposed
# inputs: g, the graph, and nomg, the graph in its nominal state
# outputs: rg, the graph snapshot
def make_resultsgraph(g, nomg):
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
def make_bipresultsgraph(g, nomg):
    rg=g.copy() 
    for node in g.nodes:        
        if g.nodes[node]['bipartite']==0: #condition only checked for functions
            if g.nodes[node].get('modes').difference(['nom']): status='Faulty'
            else: status='Nominal'
        elif g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
        else: status='Nominal'
        rg.nodes[node]['status']=status
    return rg
def make_resultsgraphs(ghist, nomghist, gtype='normal'):
    rghist = dict.fromkeys(ghist.keys())
    for i,rg in rghist.items():
        if gtype=='normal': rghist[i] = make_resultsgraph(ghist[i],nomghist[i])
        elif  gtype=='bipartite': rghist[i] = make_bipresultsgraph(ghist[i],nomghist[i])
    return rghist

## MAKE TABLES
    
#makehisttable
# put history in a tabular format
def make_histtable(mdlhist):
    if "nominal" in mdlhist.keys(): mdlhist=mdlhist['faulty']
    if any(isinstance(i,dict) for i in mdlhist['flows'].values()):
        flowtable =  make_objtable(mdlhist, 'flows')
    else:
        flowtable = make_objtable(mdlhist, 'flowvals')
    fxntable  =  make_objtable(mdlhist, 'functions')
    timetable = pd.DataFrame()
    timetable['time', 't'] = mdlhist['time']
    timetable.reindex([('time', 't')], axis="columns")
    histtable = pd.concat([timetable, fxntable, flowtable], axis =1)
    index = pd.MultiIndex.from_tuples(histtable.columns)
    histtable = histtable.reindex(index, axis='columns')
    return histtable

def make_statstable(reshist):
    table = pd.DataFrame(reshist['stats'])
    table.insert(0, 'time', reshist['time'])
    return table
def make_degflowstable(reshist):
    table = pd.DataFrame(reshist['flows'])
    table.insert(0, 'time', reshist['time'])
    return table
def make_degflowvalstable(reshist):
    table = make_objtable(reshist, 'flowvals')
    table.insert(0, 'time', reshist['time'])
    return table
def make_degfxnstable(reshist):
    table = pd.DataFrame()
    for fxnname in reshist['functions']:
        table[fxnname]=reshist['functions'][fxnname]['status']
    table.insert(0, 'time', reshist['time'])
    return table
def make_deghisttable(reshist, withstats=False):
    fxnstable = make_degfxnstable(reshist)
    flowstable = pd.DataFrame(reshist['flows'])
    if withstats:
        statstable = pd.DataFrame(reshist['stats'])
        return pd.concat([fxnstable, flowstable, statstable], axis =1)
    else:
        return pd.concat([fxnstable, flowstable], axis =1)
def make_heatmapstable(heatmaps):
    table = pd.DataFrame(heatmaps)
    return table.transpose()
def make_simplefmea(endclasses):
    table = pd.DataFrame(endclasses)
    return table.transpose()
def make_phasefmea(endclasses, app):
    fmeadict = dict.fromkeys(app.scenids.keys())
    for modephase, ids in app.scenids.items():
        rate= sum([endclasses[scenid]['rate'] for scenid in ids])
        cost= np.mean([endclasses[scenid]['cost'] for scenid in ids])
        expcost= sum([endclasses[scenid]['expected cost'] for scenid in ids])
        fmeadict[modephase] = {'rate':rate, 'cost':cost, 'expected cost': expcost}
    table=pd.DataFrame(fmeadict)
    return table.transpose()
def make_summfmea(endclasses, app):
    fmeadict = dict()
    for modephase, ids in app.scenids.items():
        rate= sum([endclasses[scenid]['rate'] for scenid in ids])
        cost= np.mean([endclasses[scenid]['cost'] for scenid in ids])
        expcost= sum([endclasses[scenid]['expected cost'] for scenid in ids])
        if not fmeadict.get(modephase[0:2]): fmeadict[modephase[0:2]]= {'rate': 0.0, 'cost':0.0, 'expected cost':0.0}
        fmeadict[modephase[0:2]]['rate'] += rate
        fmeadict[modephase[0:2]]['cost'] += cost/len([1.0 for (fxn,mode,phase) in app.scenids if (fxn, mode)==modephase[0:2]])
        fmeadict[modephase[0:2]]['expected cost'] += expcost
    table=pd.DataFrame(fmeadict)
    return table.transpose()
def make_maptable(mapping):
    table = pd.DataFrame(mapping)
    return table.transpose()
def make_fullfmea(endclasses, summaries):
    degradedtable = pd.DataFrame(summaries)
    simplefmea=pd.DataFrame(endclasses)
    fulltable = pd.concat([degradedtable, simplefmea])
    return fulltable.transpose()
def make_resulttable(endresults, summary):
    table = pd.DataFrame(endresults['classification'], index=[0])
    table['degraded functions'] = [summary['degraded functions']]
    table['degraded flows'] = [summary['degraded flows']]
    return table
def make_dicttable(dictionary):
    return pd.DataFrame(dictionary, index=[0])
def make_samptimetable(sampletimes):
    table = pd.DataFrame()
    for phase, times in sampletimes.items():
        table[phase]= [str(list(times.keys()))]
    return table.transpose()
        

# make table of function OR flow value attributes - objtype = 'function' or 'flow'
def make_objtable(hist, objtype):
    df = pd.DataFrame()
    labels = []
    for fxn, atts in hist[objtype].items():
        for att, val in atts.items():
            label=(fxn, att)
            labels=labels+[label]
            df[label]=val
        if objtype =='functions':
            if hist[objtype][fxn].get('faults'):
                label=(fxn, 'faults')
                labels+=[label]
                df[label]=hist[objtype][fxn]['faults']
    index = pd.MultiIndex.from_tuples(labels)
    df = df.reindex(index, axis="columns")
    return df
    
# need to make degraded functions table
# also need to make table summary of just functions/flows degraded
def make_summarytable(summary):
    return pd.DataFrame.from_dict(summary, orient = 'index')

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
#   - objs, the functions/flows to plot
def plot_mdlhist(mdlhist, fault='', time=0, fxnflows=[]):
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    
    for objtype in ["flows", "functions"]:
        for fxnflow in mdlhists['nominal'][objtype]:
            if fxnflows: #if in the list 
                if fxnflow not in fxnflows: continue
            
            if objtype =="flows":
                nomhist=mdlhists['nominal']["flows"][fxnflow]
                if 'faulty' in mdlhists: hist = mdlhists['faulty']["flows"][fxnflow]
            elif objtype=="functions":
                nomhist=mdlhists['nominal']["functions"][fxnflow]
                del nomhist['faults']
                if 'faulty' in mdlhists: 
                    hist = mdlhists['faulty']["functions"][fxnflow]
                    del hist['faults']
            plots=len(nomhist)
            if plots:
                fig = plt.figure()
                fig.add_subplot(np.ceil((plots+1)/2),2,plots)
                plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])
                n=1
                for var in nomhist:
                    plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+var)
                    n+=1
                    if 'faulty' in mdlhists:
                        a, = plt.plot(times, hist[var], color='r')
                        c = plt.axvline(x=time, color='k')
                    b, =plt.plot(times, nomhist[var], color='b')
                    plt.title(var)
                if 'faulty' in mdlhists:
                    plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+'legend')
                    plt.legend([a,b],['faulty', 'nominal'])
                fig.suptitle('Dynamic Response of '+fxnflow+' to fault'+' '+fault)
                plt.show()
    
#plot_ghist
# displays plots of the graph over time
# inputs:
#   - ghist, a dictionary of the history of the graph over time with structure:
#       {time: graphobject}, where
#           - time is the time where the snapshot of the graph was recorded
#           - graphobject is the snapshot of the graph at that time
#    - faultscen, the name of the fault scenario where this graph occured
def plot_ghist(ghist,faultscen=[]):
    for time, graph in ghist.items():
        show_graph(graph, faultscen, time)

#show_graph
# plots a single graph at a single time
# inputs:
#   - g, the graph object
#   - faultscen, the name of the fault scenario (for the title)
#   - time, the time of the fault scenario (also for the title)
#   - showfaultprops, whether to list the faults occuring on functions and list degraded flows
#       #(only works well for relatively simple models)
def show_graph(g, faultscen=[], time=[], showfaultlabels=True, heatmap={}):
    edgeflows=dict()
    for edge in g.edges:
        flows=list(g.get_edge_data(edge[0],edge[1]).keys())
        edgeflows[edge[0],edge[1]]=''.join(flow for flow in flows)
    if heatmap:
        pos=nx.shell_layout(g)
        colors=[]
        for node in g.nodes():
            colors = colors +[heatmap.get(node,0.0)]
            nx.draw_networkx_edges(g,pos, width=2)
        nx.draw_networkx_nodes(g,pos,node_size=2000,node_shape='s', node_color=colors, \
                     cmap=plt.cm.coolwarm, alpha=0.7)
        nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows , font_weight='bold')
        labels={node:node for node in g.nodes} 
        nx.draw_networkx_labels(g, pos, labels=labels, font_weight='bold')
    elif not list(g.nodes(data='status'))[0][1]:    
        pos=nx.shell_layout(g)
        nx.draw_networkx(g,pos,node_size=2000,node_shape='s', node_color='g', \
                     width=3, font_weight='bold')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows)
    else:
        statuses=dict(g.nodes(data='status', default='Nominal'))
        faultnodes=[node for node,status in statuses.items() if status=='Faulty']
        degradednodes=[node for node,status in statuses.items() if status=='Degraded']
        faultedges = [edge for edge in g.edges if any([g.edges[edge][flow]['status']=='Degraded' for flow in g.edges[edge]])]
        faultflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if g.edges[edge][flow]['status']=='Degraded')]) for edge in faultedges}
        faults=dict(g.nodes(data='modes', default={'nom'}))
        faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
        plot_normgraph(g, edgeflows, faultnodes, degradednodes, faultflows, faultlabels, faultedges, faultflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[])
#same for bipartite graph     
def show_bipartite(g, scale=1, faultscen=[], time=[], showfaultlabels=True, heatmap={}, pos=[]):
    labels={node:node for node in g.nodes}
    if not pos: pos=nx.spring_layout(g)
    if heatmap:
        nodesize=scale*700
        fontsize=scale*6
        #nx.draw(g, pos, node_size=nodesize,node_color = 'k', alpha=0.3)
        colors = []
        for node in labels.keys():
            colors = colors + [heatmap.get(node, 0.0)]
        nx.draw(g, pos, node_color=colors, cmap=plt.cm.coolwarm, alpha=0.6, node_size=nodesize)
        nx.draw_networkx_labels(g, pos, labels=labels,font_size=fontsize, node_size=nodesize, font_weight='bold')
        if faultscen:
            plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
        plt.show()
    elif not list(g.nodes(data='status'))[0][1]: #just plots graph if no status information 
        nodesize=scale*700
        fontsize=scale*6
        nx.draw(g, pos, labels=labels,font_size=fontsize, node_size=nodesize,node_color = 'g', font_weight='bold')
        if faultscen:
            plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
        plt.show()
    else:                                      #plots graph with status information 
        statuses=dict(g.nodes(data='status', default='Nominal'))
        faultnodes=[node for node,status in statuses.items() if status=='Faulty']
        degradednodes=[node for node,status in statuses.items() if status=='Degraded']
        faults=dict(g.nodes(data='modes', default={'nom'}))
        faultlabels = {node:fault for node,fault in faults.items() if fault!={'nom'}}
        plot_bipgraph(g, labels, faultnodes, degradednodes, faultlabels,faultscen, time, showfaultlabels=True, scale=scale)

# plot_resultsgraph_from():
# plots a representation of the graph at a specific time given a results history
def plot_resultsgraph_from(mdl, reshist, time, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[]):
    [[t_ind,],] = np.where(reshist['time']==time)
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        degnodes = degfxns + degflows
        plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos=pos)
    elif gtype=='normal':
        g = mdl.graph.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale)
    return 0

# plot_resultsgraphs_from():
# iteratively plots a representation of the graph at a specific time given a results history
def plot_resultsgraphs_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[]):
    if times=='all':
        t_inds= [i for i in range(0,len(reshist['time']))]
    else:
        t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        pos=nx.spring_layout(g)
        for t_ind in t_inds:
            update_bipplot(t_ind, reshist, g, pos, faultscen=faultscen, showfaultlabels=showfaultlabels, scale=scale)
    elif gtype=='normal':
        g = mdl.graph.copy()
        pos=nx.shell_layout(g)
        for t_ind in t_inds:
            update_graphplot(t_ind, reshist, g, pos, faultscen=faultscen, showfaultlabels=showfaultlabels, scale=scale)
    return 0
# animate_resultsgraphs_from():
# plots and returns an animation of the model graph
# to view in spyder, make sure to set to display using: %matplotlib qt
# to save (or do anything useful)h, make sure ffmpeg is installed  https://www.wikihow.com/Install-FFmpeg-on-Windows
# use %matplotlib qt from spyder or %matplotlib notebook from jupyter
def animate_resultsgraphs_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, show=False, pos=[]):
    if times=='all':
        t_inds= [i for i in range(0,len(reshist['time']))]
    else:
        t_inds= [ np.where(reshist['time']==time)[0][0] for time in times]
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        if not pos: pos=nx.spring_layout(g)
        fig, ax = plt.subplots(figsize=(6,4))
        ani = matplotlib.animation.FuncAnimation(fig, update_bipplot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False))
        if show: plt.show()
    elif gtype=='normal':
        g = mdl.graph.copy()
        if not pos: pos=nx.shell_layout(g)
        fig, ax = plt.subplots(figsize=(6,4))
        ani = matplotlib.animation.FuncAnimation(fig, update_graphplot, frames=t_inds, fargs=(reshist, g, pos, faultscen, showfaultlabels, scale, False))
        if show: plt.show()
    return ani

def update_bipplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True):
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show)
def update_graphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True):
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show)

def plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True):
    if not pos: pos=nx.shell_layout(g)
    nx.draw_networkx(g,pos,node_size=2000,node_shape='s', node_color='g', \
                     width=3, font_weight='bold')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edgeflows)
    nx.draw_networkx_nodes(g, pos, nodelist=degfxns, node_color = 'y',\
                          node_shape='s',width=3, font_weight='bold', node_size = 2000)
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_color = 'r',\
                          node_shape='s',width=3, font_weight='bold', node_size = 2000)
    nx.draw_networkx_edges(g,pos,edgelist=faultedges, edge_color='r', width=2)
        
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=12, font_color='k')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=faultedgeflows, font_color='r')
    if faultscen:
        plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    if show: plt.show()
    return 0

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True):
    nodesize=scale*700
    fontsize=scale*6
    if not pos: pos=nx.spring_layout(g)
    
    nx.draw(g, pos, labels=labels,font_size=fontsize, node_size=nodesize, node_color = 'g', font_weight='bold')
    nx.draw_networkx_nodes(g, pos, nodelist=degnodes,node_color = 'y', node_size=nodesize, font_weight='bold')
    nx.draw_networkx_nodes(g, pos, nodelist=faultfxns,node_color = 'r', node_size=nodesize, font_weight='bold')
    if showfaultlabels:
        faultlabels_form = {node:''.join(['\n\n ',''.join(f+' ' for f in fault if f!='nom')]) for node,fault in faultlabels.items() if fault!={'nom'}}
        nx.draw_networkx_labels(g, pos, labels=faultlabels_form, font_size=fontsize, font_color='k')
    if faultscen:
            plt.title('Propagation of faults to '+faultscen+' at t='+str(time))
    if show: plt.show()
    return 0
def get_plotlabels(g, reshist, t_ind):
    labels={node:node for node in g.nodes}
    functions = reshist['functions'].keys()
    
    faultfxns = []
    degfxns = []
    degflows = []
    faultlabels = {}
    edgelabels=dict()
    for edge in g.edges:
        flows=list(g.get_edge_data(edge[0],edge[1]).keys())
        edgelabels[edge[0],edge[1]]=''.join(flow for flow in flows)
    for function in functions:
        if reshist['functions'][function]['numfaults'][t_ind]:
            faultfxns+=[function]
            faultlabels[function] = reshist['functions']['ImportEE']['faults'][t_ind].difference('nom')
        if not reshist['functions'][function]['status'][t_ind]:
            degfxns+=[function]
    flows = reshist['flows'].keys()
    for flow in flows:
        if not reshist['flows'][flow][t_ind]==1:
            degflows+=[flow] 
    faultedges = [edge for edge in g.edges if any([reshist['flows'][flow][t_ind]==0 for flow in g.edges[edge].keys()])]
    faultedgeflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if reshist['flows'][flow][t_ind]==0)]) for edge in faultedges}
    return labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgelabels