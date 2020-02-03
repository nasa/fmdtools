# -*- coding: utf-8 -*-
"""
File Name: resultproc.py
Author: Daniel Hulse
Created: November 2018

Description: Results processing and plotting for single (or multiple) fault model runs.

This module has functions for the following:
    - processing model histories (values over time) into fault information (e.g. when a function/flow went off-nominal)
        - compare_hists(), compare_hist(), make_heatmaps(), etc.
    - showing graph representations of the system with or without faults (using nx.graph), including animations over time (using matplotlib.animation)
        - make_resultsgraph(), show_graph(), plot_resultsgraph_from(), animate_resultsgraphs_from(), 
    - plotting system behaviors over time (using matplotlib)
        - plot_mdlhist()
    - plotting costs/rates of approach (using matplotlib)
        - plot_samplecost, plot_samplecosts
    - providing tables of the various states of the system over time (using pandas)
        - make_histtable(), make_deghisttable(), make_statstable(), etc
    - providing fmeas of faults (using pandas)
        - make_fullfmea(), make_simplefmea(), make_summfmea(), etc
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import copy
import pandas as pd

## PROCESSING RESULTS 
def compare_hists(mdlhists, returndiff=True):
    """
    Processes a model histories for each scenario into results histories by comparing the states over time in each scenario with the states in the nominal scenario.

    Parameters
    ----------
    mdlhists : dict
        A dictionary of model histories for each scenario (e.g. from run_list or run_approach)
    returndiff : bool, optional
        Whether to return diffs, a dict of the differences between the values of the states in the nominal scenario and fault scenario. The default is True.

    Returns
    -------
    reshists : dict
        A dictionary of the results histories of each scenario over time.
    diffs : dict
        The difference between the nominal and fault scenario states (if returndiff is true--otherwise returns empty)
    summaries : dict
        A dict with all degraded functions and degraded flows resulting from the fault scenarios.
    """
    reshists={}
    diffs={}
    summaries={}
    nomhist = mdlhists.pop('nominal')
    for scenname, hist in mdlhists.items():
        reshists[scenname], diffs[scenname], summaries[scenname] = compare_hist(hist, nomhist=nomhist, returndiff=returndiff)
    mdlhists['nominal']=nomhist
    return reshists, diffs, summaries
def compare_hist(mdlhist, nomhist={}, returndiff=True):
    """
    Compares model history with the nominal model history over time to make a history of degradation.

    Parameters
    ----------
    mdlhist : dict
        the model fault history or a dict of both the nominal and fault histories {'nominal':nomhist, 'faulty':mdlhist}
    nomhist : dict, optional
        The model history in the nominal scenario (if not provided in mdlhist) The default is {}.
    returndiff : bool, optional
        Whether to return diffs, a dict of the differences between the values of the states in the nominal scenario and fault scenario. The default is True.

    Returns
    -------
    reshist : dict
        The results history over time.
    diff : dict
        The difference between the nominal and fault scenario states (if returndiff is true--otherwise returns empty)
    summary : dict
        A dict with all degraded functions and degraded flows.
    """
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
    """ Compares the history of flow states in mdlhist over time."""
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
    """ Compares the history of function states in mdlhist over time."""
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

def compare_graphflows(g, nomg, gtype='normal'):
    """
    Extracts non-nominal flows by comparing the a results graph with a nominal results graph.

    Parameters
    ----------
    g : networkx graph
        The graph in the given fault scenario
    nomg : networkx graph
        The graph in the nominal fault scenario
    gtype : str, optional
        The type of graph to return ('normal' or 'bipartite') The default is 'normal'.

    Returns
    -------
    endflows : dict
        A dictionary of degraded flows.
    """
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

def make_resultsgraph(g, nomg):
    """
    Makes a graph of nominal/non-nominal states by comparing the nominal graph states with the non-nominal graph states

    Parameters
    ----------
    g : networkx Graph
        multgraph for the fault scenario where the functions are nodes and flows are edges and with 'faults' and 'states' attributes
    nomg : networkx Graph
        multgraph for the nominal scenario where the functions are nodes and flows are edges and with 'faults' and 'states' attributes

    Returns
    -------
    rg : networkx graph
        multgraph copy of g with 'status' attributes added for faulty/degraded functions/flows
    """
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
    """
    Makes a bipartite graph of nominal/non-nominal states by comparing the nominal graph states with the non-nominal graph states

    Parameters
    ----------
    g : networkx Graph
        bipartite graph for the fault scenario where the functions have 0 'bipartite' attributes and flows have 1 'bipartite' attribute
    nomg : networkx Graph
        bipartite graph for the nominal scenario where the functions have 0 'bipartite' attributes and flows have 1 'bipartite' attribute

    Returns
    -------
    rg : networkx graph
        bipartite copy of g with 'status' attributes added for faulty/degraded functions/flows/components
    """
    rg=g.copy() 
    for node in g.nodes:        
        if g.nodes[node]['bipartite']==0 or g.nodes[node].get('iscomponent', False): #condition only checked for functions
            if g.nodes[node].get('modes').difference(['nom']): status='Faulty'
            else: status='Nominal'
        elif g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
        else: status='Nominal'
        rg.nodes[node]['status']=status
    return rg
def make_resultsgraphs(ghist, nomghist, gtype='normal'):
    """
    Makes a dict history of results graphs given a dict history of the nominal and faulty graphs

    Parameters
    ----------
    ghist : dict
        dict history of the faulty graph
    nomghist : dict
        dict history of the nominal graph
    gtype : str, optional
        Type of graph provided/returned (bipartite, component, or normal). The default is 'normal'.

    Returns
    -------
    rghist : dict
        dict history of results graphs
    """
    rghist = dict.fromkeys(ghist.keys())
    for i,rg in rghist.items():
        if gtype=='normal': rghist[i] = make_resultsgraph(ghist[i],nomghist[i])
        elif  gtype=='bipartite' or gtype=='component': rghist[i] = make_bipresultsgraph(ghist[i],nomghist[i])
    return rghist


##HEATMAP FUNCTIONS
def make_heatmaps(reshist, diff):
    """
    Makes a dict of heatmaps given a results history and a history of the differences between nominal and faulty models.

    Parameters
    ----------
    reshist : dict
        The model results history (e.g. from compare_functionhist
    diff : dict
        The differences (e.g. from compare_functionhist(s))

    Returns
    -------
    heatmaps : dict
        A dict of heatmaps based on the results history, including:
            - degtime, the time the function/flow was degraded
            - maxdeg, the maximum degradation experienced by the function
            - intdeg, the integral of degradation of the function over the time interval
            - maxfaults, the maximum number of faults in the function
            - intdiff, the integral of the differences between function/flow states of the nominal and faulty model over time.
            - maxdiff, the maximum difference between function/flow states of the nominal and faulty model over time.
    """
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
    """ Makes a heatmap dictionary of degraded time for functions given a result history"""
    len_time = len(reshist['time'])
    degtimemap={}
    for fxnname in reshist['functions'].keys():
        degtimemap[fxnname]=1.0-sum(reshist['functions'][fxnname]['status'])/len_time
    for flowname in reshist['flows'].keys():
        degtimemap[flowname]=1.0 - sum(reshist['flows'][flowname])/len_time
    return degtimemap
def make_degtimemaps(reshists):
    """ Makes a dict of heatmap dictionaries of degraded time for functions given results histories"""
    degtimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        degtimemaps[reshist]=make_degtimemap(reshists[reshist])
    return degtimemaps
def make_avgdegtimeheatmap(reshists):
    """ Makes a heatmap dictionary of the average degraded heat time over a list of scenarios in the dict of results histories."""
    degtimetable = pd.DataFrame(make_degtimemaps(reshists)).transpose()
    return degtimetable.mean().to_dict()
def make_expdegtimeheatmap(reshists, endclasses):
    """ Makes a heatmap dictionary of the expected degraded heat time over a list of scenarios in the dict of results histories based on the rates in endclasses."""
    degtimetable = pd.DataFrame(make_degtimemaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expdegtimetable = degtimetable.multiply(rates).transpose()
    return expdegtimetable.sum().to_dict()
def make_faultmap(reshist):
    """ Makes a heatmap dictionary of faults given a results history."""
    heatmap={}
    for fxnname in reshist['functions'].keys():
        heatmap[fxnname] = max(reshist['functions'][fxnname]['numfaults'])
    return heatmap
def make_faultmaps(reshists):
    """ Makes dict of heatmaps dictionaries of resulting faults given a results history."""
    faulttimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        faulttimemaps[reshist]=make_faultmap(reshists[reshist])
    return faulttimemaps
def make_faultsheatmap(reshists):
    """Makes a heatmap dictionary of the average resulting faults over all scenarios"""
    faulttable = pd.DataFrame(make_faultmaps(reshists)).transpose()
    return faulttable.mean().to_dict()
def make_expfaultsheatmap(reshists, endclasses):
    """Makes a heatmap dictionary of the expected resulting faults over all scenarios"""
    faulttable = pd.DataFrame(make_faultmaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expfaulttable = faulttable.multiply(rates).transpose()
    return expfaulttable.mean().to_dict()


## MAKE TABLES
    
#makehisttable
# put history in a tabular format
def make_histtable(mdlhist):
    """ Returns formatted pandas dataframe of model history"""
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
def make_objtable(hist, objtype):
    """make table of function OR flow value attributes - objtype = 'function' or 'flow'"""
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
def make_statstable(reshist):
    """Makes a table of #of degraded flows, # of degraded functions, and # of total faults over time given a single result history"""
    table = pd.DataFrame(reshist['stats'])
    table.insert(0, 'time', reshist['time'])
    return table
def make_degflowstable(reshist):
    """Makes a table of flows over time, where 0 is degraded and 1 is nominal"""
    table = pd.DataFrame(reshist['flows'])
    table.insert(0, 'time', reshist['time'])
    return table
def make_degflowvalstable(reshist):
    """Makes a table of individual flow state values over time, where 0 is degraded and 1 is nominal"""
    table = make_objtable(reshist, 'flowvals')
    table.insert(0, 'time', reshist['time'])
    return table
def make_degfxnstable(reshist):
    """Makes a table showing which functions are degraded over time (0 for degraded, 1 for nominal)"""
    table = pd.DataFrame()
    for fxnname in reshist['functions']:
        table[fxnname]=reshist['functions'][fxnname]['status']
    table.insert(0, 'time', reshist['time'])
    return table
def make_deghisttable(reshist, withstats=False):
    """Makes a table of all funcitons and flows that are degraded over time. If withstats=True, the total # of each type degraded is provided in the last columns """
    fxnstable = make_degfxnstable(reshist)
    flowstable = pd.DataFrame(reshist['flows'])
    if withstats:
        statstable = pd.DataFrame(reshist['stats'])
        return pd.concat([fxnstable, flowstable, statstable], axis =1)
    else:
        return pd.concat([fxnstable, flowstable], axis =1)
def make_heatmapstable(heatmaps):
    """Makes a table of a heatmap dictionary"""
    table = pd.DataFrame(heatmaps)
    return table.transpose()
def make_simplefmea(endclasses):
    """Makes a simple fmea (rate, cost, expected cost) of the endclasses of a list of fault scenarios run"""
    table = pd.DataFrame(endclasses)
    return table.transpose()
def make_phasefmea(endclasses, app):
    """
    Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by phase.

    Parameters
    ----------
    endclasses : dict
        dict of endclasses of the simulation runs
    app : sampleapproach
        sample approach used for the underlying probability model of the set of scenarios run

    Returns
    -------
    table: dataframe
        table with cost, rate, and expected cost of each fault in each phase
    """
    fmeadict = dict.fromkeys(app.scenids.keys())
    for modephase, ids in app.scenids.items():
        rate= sum([endclasses[scenid]['rate'] for scenid in ids])
        cost= sum(np.array([endclasses[scenid]['cost'] for scenid in ids])*np.array(list(app.weights[modephase[0]][modephase[1]].values())))
        expcost= sum([endclasses[scenid]['expected cost'] for scenid in ids])
        fmeadict[modephase] = {'rate':rate, 'cost':cost, 'expected cost': expcost}
    table=pd.DataFrame(fmeadict)
    return table.transpose()
def find_costovertime(endclasses, app):
    """
    Makes a table of the total cost, rate, and expected cost of all faults over time

    Parameters
    ----------
    endclasses : dict
        dict with rate,cost, and expected cost for each injected scenario
    app : sampleapproach
        sample approach used to generate the list of scenarios

    Returns
    -------
    costovertime : dataframe
        pandas dataframe with the total cost, rate, and expected cost for the set of scenarios
    """
    costovertime={'cost':{time:0.0 for time in app.times}, 'rate':{time:0.0 for time in app.times}, 'expected cost':{time:0.0 for time in app.times}}
    for scen in app.scenlist:
        costovertime['cost'][scen['properties']['time']]+=endclasses[scen['properties']['name']]['cost']
        costovertime['rate'][scen['properties']['time']]+=endclasses[scen['properties']['name']]['rate']
        costovertime['expected cost'][scen['properties']['time']]+=endclasses[scen['properties']['name']]['expected cost'] 
    return pd.DataFrame.from_dict(costovertime)
        
def make_summfmea(endclasses, app):
    """
    Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by fault.

    Parameters
    ----------
    endclasses : dict
        dict of endclasses of the simulation runs
    app : sampleapproach
        sample approach used for the underlying probability model of the set of scenarios run

    Returns
    -------
    table: dataframe
        table with cost, rate, and expected cost of each fault (over all phases)
    """
    fmeadict = dict()
    for modephase, ids in app.scenids.items():
        rate= sum([endclasses[scenid]['rate'] for scenid in ids])
        cost= sum(np.array([endclasses[scenid]['cost'] for scenid in ids])*np.array(list(app.weights[modephase[0]][modephase[1]].values())))
        expcost= sum([endclasses[scenid]['expected cost'] for scenid in ids])
        if getattr(app, 'jointmodes', []):  index = str(modephase[0])
        else:                               index = modephase[0]
        if not fmeadict.get(modephase[0]): fmeadict[index]= {'rate': 0.0, 'cost':0.0, 'expected cost':0.0}
        fmeadict[index]['rate'] += rate
        fmeadict[index]['cost'] += cost/len([1.0 for (fxnmode,phase) in app.scenids if fxnmode==modephase[0]])
        fmeadict[index]['expected cost'] += expcost
    table=pd.DataFrame(fmeadict)
    return table.transpose()
def make_maptable(mapping):
    """Makes table of a generic map"""
    table = pd.DataFrame(mapping)
    return table.transpose()
def make_fullfmea(endclasses, summaries):
    """Makes full fmea table (degraded functions/flows, cost, rate, expected cost) of scenarios given endclasses dict (cost, rate, expected cost) and summaries dict (degraded functions, degraded flows)"""
    degradedtable = pd.DataFrame(summaries)
    simplefmea=pd.DataFrame(endclasses)
    fulltable = pd.concat([degradedtable, simplefmea])
    return fulltable.transpose()
def make_resulttable(endresults, summary):
    """Makes a table of results (degraded functions/flows, cost, rate, expected cost) of a single run"""
    table = pd.DataFrame(endresults['classification'], index=[0])
    table['degraded functions'] = [summary['degraded functions']]
    table['degraded flows'] = [summary['degraded flows']]
    return table
def make_dicttable(dictionary):
    """Makes table of a generic dictionary"""
    return pd.DataFrame(dictionary, index=[0])
def make_samptimetable(sampletimes):
    """Makes a table of the times sampled for each phase given a dict (i.e. app.sampletimes)"""
    table = pd.DataFrame()
    for phase, times in sampletimes.items():
        table[phase]= [str(list(times.keys()))]
    return table.transpose()
def make_summarytable(summary):
    """Makes a table of a summary dictionary from a given model run"""
    return pd.DataFrame.from_dict(summary, orient = 'index')

##PLOTTING AND RESULTS DISPLAY


def plot_samplecosts(app, endclasses, joint=False, title=""):
    """
    Plots the costs and rates of a set of faults injected over time according to the approach app

    Parameters
    ----------
    app : sampleapproach
        The sample approach used to run the list of faults
    endclasses : dict
        A dict of results for each of the scenarios.
    joint : bool, optional
        Whether to include joint fault scenarios. The default is False.
    """
    for fxnmode in app.list_modes(joint):
        if any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='fullint' and fm==fxnmode]):
            st='fullint'
        elif any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='quadrature' and fm==fxnmode]):
            st='quadrature'
        else: 
            st='std'
        plot_samplecost(app, endclasses, fxnmode, samptype=st, title="")
def plot_samplecost(app, endclasses, fxnmode, samptype='std', title=""):
    """
    Plots the sample cost and rate of a given fault over the injection times defined in the app sampleapproach

    Parameters
    ----------
    app : sampleapproach
        Sample approach defining the underlying samples to take and probability model of the list of scenarios.
    endclasses : dict
        A dict with the end classification of each fault (costs, etc)
    fxnmode : tuple
        tuple (or tuple of tuples) with structure ('function name', 'mode name') defining the fault mode
    samptype : str, optional
        The type of sample approach used:
            - 'std' for a single point for each interval
            - 'quadrature' for a set of points with weights defined by a quadrature
            - 'pruned piecewise-linear' for a set of points with weights defined by a pruned approach (from app.prune_scenarios())
            - 'fullint' for the full integral (sampling every possible time)
    """
    associated_scens=[]
    for phase in app.phases:
        associated_scens = associated_scens + app.scenids.get((fxnmode, phase), [])
    costs = np.array([endclasses[scen]['cost'] for scen in associated_scens])
    times = np.array([time  for phase, timemodes in app.sampletimes.items() if timemodes for time in timemodes if fxnmode in timemodes.get(time)] )  
    rates = np.array(list(app.rates_timeless[fxnmode].values()))
    
    tPlot, axes = plt.subplots(2, 1, sharey=False, gridspec_kw={'height_ratios': [3, 1]})
    phasetimes_start =[times[0] for phase, times in app.phases.items()]
    phasetimes_end =[times[1] for phase, times in app.phases.items()]
    ratetimes =[]
    ratesvect =[]
    phaselocs = []
    for (ind, phasetime) in enumerate(phasetimes_start):
        axes[0].axvline(phasetime, color="black")        
        phaselocs= phaselocs +[(phasetimes_end[ind]-phasetimes_start[ind])/2 + phasetimes_start[ind]]

        axes[1].axvline(phasetime, color="black") 
        ratetimes = ratetimes + [phasetimes_start[ind]] + [phasetimes_end[ind]]
        ratesvect = ratesvect + [rates[ind]] + [rates[ind]]
        #axes[1].text(middletime, 0.5*max(rates),  list(app.phases.keys())[ind], ha='center', backgroundcolor="white")
    #rate plots
    axes[1].set_xticks(phaselocs)
    axes[1].set_xticklabels(list(app.phases.keys()))
    
    axes[1].plot(ratetimes, ratesvect)
    axes[1].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    axes[1].set_ylim(0, np.max(ratesvect)*1.2 )
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Time")
    axes[1].grid()
    #cost plots
    axes[0].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    axes[0].set_ylim(0, 1.2*np.max(costs))
    if samptype=='fullint':
        axes[0].plot(times, costs, label="cost")
    else:
        if samptype=='quadrature' or samptype=='pruned piecewise-linear': 
            sizes =  1000*np.array([weight if weight !=1/len(timeweights) else 0.0 for phase, timeweights in app.weights[fxnmode].items() for time, weight in timeweights.items() if time in times])
            axes[0].scatter(times, costs,s=sizes, label="cost", alpha=0.5)
        axes[0].stem(times, costs, label="cost", markerfmt=",", use_line_collection=True)
    
    axes[0].set_ylabel("Cost")
    axes[0].grid()
    if title: axes[0].set_title(title)
    elif type(fxnmode[0])==tuple: axes[0].set_title("Cost function of "+str(fxnmode)+" over time")
    else:                       axes[0].set_title("Cost function of "+fxnmode[0]+": "+fxnmode[1]+" over time")
def plot_costovertime(endclasses, app, costtype='expected cost', timelabel='time'):
    """
    Plots the total cost or total expected cost of faults over time.

    Parameters
    ----------
    endclasses : dict
        dict with rate,cost, and expected cost for each injected scenario (e.g. from run_approach())
    app : sampleapproach
        sample approach used to generate the list of scenarios
    costtype : str, optional
        type of cost to plot ('cost', 'expected cost' or 'rate'). The default is 'expected cost'.
    """
    costovertime = find_costovertime(endclasses, app)
    plt.plot(list(costovertime.index), costovertime[costtype])
    plt.title('Total '+costtype+' of all faults over time.')
    plt.ylabel(costtype)
    plt.xlabel(timelabel)
    plt.grid()

def plot_mdlhist(mdlhist, fault='', time=0, fxnflows=[], returnfigs=False, legend=True, timelabel="time"):
    """
    Plots the states of a model over time given a history.

    Parameters
    ----------
    mdlhist : dict
        History of states over time. Can be just the scenario states or a dict of scenario states and nominal states per {'nominal':nomhist,'faulty':mdlhist}
    fault : str, optional
        Name of the fault (for the title). The default is ''.
    time : float, optional
        Time of fault injection. The default is 0.
    fxnflows : list, optional
        List of functions and flows to plot. The default is [], which returns all.
    returnfigs: bool, optional
        Whether to return the figure objects in a list. The default is False.
    legend: bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    figs =[]
    for objtype in ["flows", "functions"]:
        for fxnflow in mdlhists['nominal'][objtype]:
            if fxnflows: #if in the list 
                if fxnflow not in fxnflows: continue
            
            if objtype =="flows":
                nomhist=mdlhists['nominal']["flows"][fxnflow]
                if 'faulty' in mdlhists: hist = mdlhists['faulty']["flows"][fxnflow]
            elif objtype=="functions":
                nomhist=copy.deepcopy(mdlhists['nominal']["functions"][fxnflow])
                del nomhist['faults']
                if 'faulty' in mdlhists: 
                    hist = copy.deepcopy(mdlhists['faulty']["functions"][fxnflow])
                    del hist['faults']
            plots=len(nomhist)
            if plots:
                fig = plt.figure()
                figs = figs +[fig]
                if legend: fig.add_subplot(np.ceil((plots+1)/2),2,plots)
                else: fig.add_subplot(np.ceil((plots)/2),2,plots)
                
                plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])
                n=1
                for var in nomhist:
                    plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+var)
                    n+=1
                    if 'faulty' in mdlhists:
                        a, = plt.plot(times, hist[var], color='r')
                        c = plt.axvline(x=time, color='k')
                    b, =plt.plot(times, nomhist[var], ls='--', color='b')
                    plt.title(var)
                    plt.xlabel(timelabel)
                if 'faulty' in mdlhists:
                    fig.suptitle('Dynamic Response of '+fxnflow+' to fault'+' '+fault)
                    if legend:
                        ax_l = plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+'legend')
                        plt.legend([a,b],['faulty', 'nominal'], loc='center')
                        plt.box(on=None)
                        ax_l.get_xaxis().set_visible(False)
                        ax_l.get_yaxis().set_visible(False)
                plt.show()
    if returnfigs: return figs

def plot_mdlhistvals(mdlhist, fault='', time=0, fxnflowvals={}, cols=2, returnfig=False, legend=True, timelabel="time"):
    """
    Plots the states of a model over time given a history.

    Parameters
    ----------
    mdlhist : dict
        History of states over time. Can be just the scenario states or a dict of scenario states and nominal states per {'nominal':nomhist,'faulty':mdlhist}
    fault : str, optional
        Name of the fault (for the title). The default is ''.
    time : float, optional
        Time of fault injection. The default is 0.
    fxnflowsvals : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals]}. The default is {}, which returns all.
    cols: int, optional
        columns to use in the figure. The default is 2.
    returnfig: bool, optional
        Whether to return the figure. The default is False.
    legend: bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true
        
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    
    if fxnflowvals: num_plots = sum([len(val) for k,val in enumerate(fxnflowvals)])
    else: num_plots = sum([len(flow) for flow in mdlhist['nominal']['flows'].values()])+sum([len(f.keys())-1 for f in mdlhist['nominal']['functions'].values()])
    fig = plt.figure(figsize=(cols*3, 2*num_plots/cols))
    n=1
    
    for objtype in ["flows", "functions"]:
        for fxnflow in mdlhists['nominal'][objtype]:
            if fxnflowvals: #if in the list 
                if fxnflow not in fxnflowvals: continue
            
            if objtype =="flows":
                nomhist=mdlhists['nominal']["flows"][fxnflow]
                if 'faulty' in mdlhists: hist = mdlhists['faulty']["flows"][fxnflow]
            elif objtype=="functions":
                nomhist=copy.deepcopy(mdlhists['nominal']["functions"][fxnflow])
                del nomhist['faults']
                if 'faulty' in mdlhists: 
                    hist = copy.deepcopy(mdlhists['faulty']["functions"][fxnflow])
                    del hist['faults']

            for var in nomhist:
                if fxnflowvals: #if in the list of values
                    if var not in fxnflowvals[fxnflow]: continue
                if legend: plt.subplot(np.ceil((num_plots+1)/cols),cols,n, label=fxnflow+var)
                else: plt.subplot(np.ceil((num_plots)/cols),cols,n, label=fxnflow+var)
                n+=1
                if 'faulty' in mdlhists:
                    a, = plt.plot(times, hist[var], color='r')
                    c = plt.axvline(x=time, color='k')
                b, =plt.plot(times, nomhist[var], ls='--', color='b')
                plt.title(fxnflow+": "+var)
                plt.xlabel(timelabel)
    if 'faulty' in mdlhists:
        fig.suptitle('Dynamic Response of '+fxnflow+' to fault'+' '+fault)
        if legend:
            ax_l = plt.subplot(np.ceil((num_plots+1)/cols),cols,n, label=fxnflow+'legend')
            plt.legend([a,b],['faulty', 'nominal'], loc='center')
            plt.box(on=None)
            ax_l.get_xaxis().set_visible(False)
            ax_l.get_yaxis().set_visible(False)
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=0.93)
    if returnfig: return fig
    else: plt.show()

    
def plot_ghist(ghist,faultscen=[]):
    """
    Displays plots of the graph over time given a dict history of graph objects

    Parameters
    ----------
    ghist : dict
        A dictionary of the history of the graph over time with structure:
       {time: graphobject}, where
           - time is the time where the snapshot of the graph was recorded
           - graphobject is the snapshot of the graph at that time
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [] (no name).
    """
    for time, graph in ghist.items():
        show_graph(graph, faultscen, time)

def show_graph(g, faultscen=[], time=[], showfaultlabels=True, heatmap={}):
    """
    Plots a single graph object g.

    Parameters
    ----------
    g : networkx graph
        The multigraph to plot
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [].
    time : float, optional
        Time of fault injection. The default is [].
    showfaultlabels : bool, optional
        Whether or not to label the faults on the functions. The default is True.
    heatmap : dict, optional
        A heatmap dictionary to overlay on the plot. The default is {}.
    """
    edgeflows=dict()
    plt.figure()
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
    """
    Plots a bipartite graph object g.

    Parameters
    ----------
    g : networkx graph
        Bipartite graph object (e.g. mdl.bipartite)
    scale : float, optional
        Scale factor for node/label size. The default is 1.
    faultscen : str, optional
        Name of the fault scenario (for the title). The default is [].
    time : float, optional
        Time the fault was injected. The default is [].
    showfaultlabels : bool, optional
        Whether or not to label the faults on functions. The default is True.
    heatmap : dict, optional
        Heatmap dictionary to overlay on plot. The default is {}.
    pos : dict, optional
        Dictionary of node positions (to be consistent with other plots). The default is [].
    """
    labels={node:node for node in g.nodes}
    plt.figure()
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

def plot_resultsgraph_from(mdl, reshist, time, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[], retfig=False):
    """
    Plots a representation of the model graph at a specific time in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    time : float
        The time in the history to plot the graph at.
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    retfig:, bool, optional
        whether to return the figure and axis objects of the plot. The default is False.
    """
    [[t_ind,],] = np.where(reshist['time']==time)
    if gtype=='bipartite':
        g = mdl.bipartite.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        degnodes = degfxns + degflows
        
        fig_axis = plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos=pos, retfig=retfig)
    elif gtype=='normal':
        g = mdl.graph.copy()
        labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
        fig_axis= plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, retfig=retfig)
    if retfig: return fig_axis

def plot_resultsgraphs_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, pos=[]):
    """
    Plots a set of representations of the model graph at given times in the results history.

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
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

def animate_resultsgraphs_from(mdl, reshist, times, faultscen=[], gtype='bipartite', showfaultlabels=True, scale=1, show=False, pos=[]):
    """
    Creates an animation of the model graph using results at given times in the results history.
    To view, use %matplotlib qt from spyder or %matplotlib notebook from jupyter
    To save (or do anything useful)h, make sure ffmpeg is installed  https://www.wikihow.com/Install-FFmpeg-on-Windows

    Parameters
    ----------
    mdl : model
        The model the faults were run in.
    reshist : dict
        A dictionary of results (from compare_hists())
    times : list or 'all'
        The times in the history to plot the graph at. If 'all', plots them all
    faultscen : str, optional
        Name of the fault scenario. The default is [].
    gtype : str, optional
        The type of graph to plot (normal or bipartite). The default is 'bipartite'.
    showfaultlabels : bool, optional
        Whether or not to list faults on the plot. The default is True.
    scale : float, optional
        Scale factor for the node/label sizes. The default is 1.
    show : bool, optional
        Whether to show the plot at the end (may be redundant). The default is True.
    pos : dict, optional
        dict of node positions (if re-using positions). The default is [].
    """
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
    """Updates a bipartite graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    degnodes = degfxns + degflows
    plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen, time, showfaultlabels, scale, pos, show)
def update_graphplot(t_ind, reshist, g, pos, faultscen=[], showfaultlabels=True, scale=1, show=True):
    """Updates a normal graph plot at a given timestep t_ind given the result history reshist"""
    time = reshist['time'][t_ind]
    labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgeflows = get_plotlabels(g, reshist, t_ind)
    plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale, pos, show)

def plot_normgraph(g, labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, faultscen, time, showfaultlabels, edgeflows, scale=1, pos=[], show=True, retfig=False):
    """ Plots a standard graph. Used in other functions"""
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
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()

def plot_bipgraph(g, labels, faultfxns, degnodes, faultlabels, faultscen=[], time=0, showfaultlabels=True, scale=1, pos=[], show=True, retfig=False):
    """ Plots a bipartite graph. Used in other functions"""
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
    if retfig:
        return plt.gcf(), plt.gca()
    elif show: plt.show()
def get_plotlabels(g, reshist, t_ind):
    """
    Assigns labels to a graph g from reshist at time t so that it can be plotted

    Parameters
    ----------
    g : networkx graph
        The graph to get labels for
    reshist : dict
        The dict of results history over time (e.g. from compare_mdlhist)
    t_ind : float
        The time in reshist to update the graph at

    Returns
    -------
    labels : dict
        labels for the graph.
    faultfxns : dict
        functions with faults in them
    degfxns : dict
        functions that are degraded
    degflows : dict
        flows that are degraded
    faultlabels : dict
        names of each fault
    faultedges : dict
        edges with faults in them
    faultedgeflows : dict
        names of flows that are degraded on each edge
    edgelabels : dict
        labels of each edge
    """
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
            faultlabels[function] = reshist['functions'][function]['faults'][t_ind].difference('nom')
        if not reshist['functions'][function]['status'][t_ind]:
            degfxns+=[function]
    flows = reshist['flows'].keys()
    for flow in flows:
        if not reshist['flows'][flow][t_ind]==1:
            degflows+=[flow] 
    faultedges = [edge for edge in g.edges if any([reshist['flows'][flow][t_ind]==0 for flow in g.edges[edge].keys()])]
    faultedgeflows = {edge:''.join([' ',''.join(flow+' ' for flow in g.edges[edge] if reshist['flows'][flow][t_ind]==0)]) for edge in faultedges}
    return labels, faultfxns, degfxns, degflows, faultlabels, faultedges, faultedgeflows, edgelabels