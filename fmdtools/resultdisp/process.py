"""
File Name: resultdisp/process.py
Author: Daniel Hulse
Created: November 2019 (Refactored April 2020)

Description: Processes model results for visualization

Uses methods:
    - hists:                    Processes a model histories for each scenario into results histories by comparing the states over time in each scenario with the states in the nominal scenario.
    - hist:                     Compares model history with the nominal model history over time to make a history of degradation.
        - fxnhist:              Compares the history of function states in mdlhist over time.
        - flowhist:             Compares the history of flow states in mdlhist over time.
    - modephases:               Identifies the phases of operation for the system based on a mdlhist with a history of its modes
    - graphflows:               Extracts non-nominal flows by comparing the a results graph with a nominal results graph.
    - resultsgraph:         Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - resultsgraphs:        Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - totalcost:            Calculates the total host of a set of given end classifications
    - state_probabilities:  Calculates the probabilities of given end-state classifications given an endclasses dictionary
Also used for graph heatmaps, which use the results history to map results history statistics onto the graph, returning a dictonary with structure {fxn/flow: value}:         
    - heatmaps:            Makes a dict of heatmaps given a results history and a history of the differences between nominal and faulty models.
    - degtimemap:          Makes a heatmap dictionary of degraded time for functions given a result history
    - degtimemaps:         Makes a dict of heatmap dictionaries of degraded time for functions given results histories
    - avgdegtimeheatmap:   Makes a heatmap dictionary of the average degraded heat time over a list of scenarios in the dict of results histories.
    - expdegtimeheatmap:   Makes a heatmap dictionary of the expected degraded heat time over a list of scenarios in the dict of results histories based on the rates in endclasses.
    - faultmap:            Makes a heatmap dictionary of faults given a results history.
    - faultmaps:           Makes dict of heatmaps dictionaries of resulting faults given a results history.
    - faultsheatmap:       Makes a heatmap dictionary of the average resulting faults over all scenarios
    - expfaultsheatmap:    Makes a heatmap dictionary of the expected resulting faults over all scenarios
"""

import copy
import networkx as nx
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from fmdtools.faultsim.propagate import cut_mdlhist

def hists(mdlhists, returndiff=True):
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
    for scenname, history in mdlhists.items():
        reshists[scenname], diffs[scenname], summaries[scenname] = hist(history, nomhist=nomhist, returndiff=returndiff)
    mdlhists['nominal']=nomhist
    return reshists, diffs, summaries
def typehist(mdl, reshist):
    """
    Summarizes results history reshist over model classes

    Parameters
    ----------
    mdl : Model
        Model used in the simulation
    reshist : Dict
        Results history from rd.process.hist(mdlhist)

    Returns
    -------
    typehist : Dict
        Results history of flow types/function classes with structure: 
            {'functions':{'status':[],'faults':{fxn1:[], fxn2:[]},'numfaults':[]}, 'flows':[], 'flowvals'{'flow1':[], 'flow2':[]}}
    """
    typehist = {'flows':{}, 'flowvals':{}, 'functions':{}, 'time':reshist['time']}
    for flowtype in mdl.flowtypes():
        flows = mdl.flows_of_type(flowtype)
        typehist['flows'][flowtype] = np.prod([reshist['flows'][flow] for flow in flows], axis=0)
        typehist['flowvals'][flowtype] = flows
    
    for fxnclass in mdl.fxnclasses():
        fxns = mdl.fxns_of_class(fxnclass)
        typehist['functions'][fxnclass] = dict.fromkeys(['status', 'numfaults', 'faults'])
        typehist['functions'][fxnclass]['status'] = np.prod([reshist['functions'][fxn]['status'] for fxn in fxns], axis=0)
        typehist['functions'][fxnclass]['numfaults'] = np.sum([reshist['functions'][fxn]['numfaults'] for fxn in fxns], axis=0)
        typehist['functions'][fxnclass]['faults'] = {fxn:reshist['functions'][fxn]['numfaults'] for fxn in fxns}
    return typehist
    
def hist(mdlhist, nomhist={}, returndiff=True):
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
    if len(mdlhist['faulty']['time']) != len(mdlhist['nominal']['time']): 
           print("Faulty and nominal scenarios have different simulation times--cutting comparison to shared range.")
           mdlhist['nominal'] = cut_mdlhist(mdlhist['nominal'], len(mdlhist['faulty']['time'])-1)
           mdlhist['faulty'] = cut_mdlhist(mdlhist['faulty'], len(mdlhist['nominal']['time'])-1)
    reshist = {}
    reshist['time'] = mdlhist['nominal']['time']
    reshist['flowvals'], reshist['flows'], degflows, numdegflows, flowdiff = flowhist(mdlhist, returndiff=returndiff)
    reshist['functions'], numfaults, degfxns, numdegfxns, fxndiff = fxnhist(mdlhist, returndiff=returndiff)
    reshist['stats'] = {'degraded flows': numdegflows, 'degraded functions': numdegfxns, 'total faults': numfaults}
    summary = {'degraded functions': degfxns, 'degraded flows': degflows}
    diff = {**fxndiff, **flowdiff}
    return reshist, diff, summary
def flowhist(mdlhist, returndiff=True):
    """ Compares the history of flow states in mdlhist over time."""
    flowshist = {}
    summhist = {}
    degflows = []
    diff = {}
    for flowname in mdlhist['nominal']['flows']:
        flowshist[flowname]={}
        diff[flowname]={}
        for att in mdlhist['nominal']['flows'][flowname]:
            faulty  = mdlhist['faulty']['flows'][flowname][att]
            nominal = mdlhist['nominal']['flows'][flowname][att]
            flowshist[flowname][att] = 1* (faulty == nominal)
            if returndiff: diff[flowname][att] = nominal - faulty
        summhist[flowname] = np.prod(np.array(list(flowshist[flowname].values())), axis = 0)
        if 0 in summhist[flowname]: degflows+=[flowname]
    numdegflows = len(summhist) - np.sum(np.array(list(summhist.values())), axis=0)
    return flowshist, summhist, degflows, numdegflows, diff
def fxnhist(mdlhist, returndiff=True):
    """ Compares the history of function states in mdlhist over time."""
    fxnshist = {}
    faulthist = {}
    deghist = {}
    degfxns = []
    diff = {}
    for fxnname in mdlhist['nominal']['functions']:
        fhist = copy.copy(mdlhist['faulty']['functions'][fxnname])
        if any(fhist.get('faults', [])):  del fhist['faults']
        fxnshist[fxnname] = {}
        diff[fxnname]={}
        for state in fhist:
            faulty  = mdlhist['faulty']['functions'][fxnname][state]
            nominal = mdlhist['nominal']['functions'][fxnname][state] 
            fxnshist[fxnname][state] = 1* (faulty == nominal)
            if state=='mode': 
                diff[fxnname][state] = [int(nominal[i]==faulty[i]) for i,f in enumerate(nominal)]
            else:   diff[fxnname][state] = nominal - faulty
        if fxnshist[fxnname]: status = np.prod(np.array(list(fxnshist[fxnname].values())), axis = 0) 
        else: status = np.ones(len(mdlhist['faulty']['time']), dtype=int) #should empty be given 1 or nothing?
        fxnshist[fxnname]['faults']=mdlhist['faulty']['functions'][fxnname].get('faults', np.zeros(len(mdlhist['faulty']['time'])))
        faults = fxnshist[fxnname]['faults']
        if type(faults)==dict:              fxnshist[fxnname]['numfaults'] = np.sum([fhist for fhist in faults.values()], axis=0)
        elif type(faults[0])==np.float64:  fxnshist[fxnname]['numfaults'] = faults
        elif type(faults[0])==np.str_:     fxnshist[fxnname]['numfaults'] = np.array([int(f!='nom') for f in faults])
        else:   raise Exception("Invalid data type in "+fxnname+" hist: "+str(type(faults)))
        faulty = 1 - 1*(fxnshist[fxnname]['numfaults']>0)
        fxnshist[fxnname]['status'] = status*faulty
        faulthist[fxnname]=fxnshist[fxnname]['numfaults']
        deghist[fxnname] = fxnshist[fxnname]['status']
        if 0 in deghist[fxnname] or any(0 < faulthist[fxnname]): degfxns+=[fxnname]
    numfaults = np.sum(np.array(list(faulthist.values())), axis=0)
    numdegfxns   = len(deghist) - np.sum(np.array(list(deghist.values())), axis=0)
    return fxnshist, numfaults, degfxns, numdegfxns, diff
def modephases(mdlhist):
    """
    Identifies the phases of operation for the system based on its modes.

    Parameters
    ----------
    mdlhist : dict
        Model history from the nominal run

    Returns
    -------
    phases : dict
        Dictionary of distict phases that the system functions pass through, of the form: 
            {'fxn':{'phase1':[beg, end], phase2:[beg, end]}}
            where each phase is defined by its corresponding mode in the modelhist
            (numbered mode, mode1, mode2... for multiple modes)
    modephases : dict
        Dictionary of phases that the system passes through, of the form: {'fxn':{'mode1':{'phase1', 'phase2''}}}
    """
    modephases={}
    phases={}
    for fxn in mdlhist["functions"].keys():
        modehist = mdlhist["functions"][fxn].get('mode', [])
        if len(modehist)!=0:    
            modes = OrderedSet(modehist)
            modephases[fxn]=dict.fromkeys(modes)
            phases_unsorted = dict()
            for mode in modes:
                modeinds = [ind for ind,m in enumerate(modehist) if m==mode]
                startind = modeinds[0]
                phasenum = 0; phaseid=mode
                modephases[fxn][mode] = set()
                for i, ind in enumerate(modeinds):
                    if ind+1 not in modeinds:
                        phases_unsorted [phaseid] =[startind, ind]
                        modephases[fxn][mode].add(phaseid)
                        if i!=len(modeinds)-1: 
                            startind = modeinds[i+1]
                            phasenum+=1; phaseid=mode+str(phasenum)
            phases[fxn] = dict(sorted(phases_unsorted.items(), key = lambda item: item[1][0]))
    return phases, modephases

def graphflows(g, nomg, gtype='bipartite'):
    """
    Extracts non-nominal flows by comparing the a results graph with a nominal results graph.

    Parameters
    ----------
    g : networkx graph
        The graph in the given fault scenario
    nomg : networkx graph
        The graph in the nominal fault scenario
    gtype : str, optional
        The type of graph to return ('normal' or 'bipartite') The default is 'bipartite'.

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
def resultsgraph(g, nomg, gtype='bipartite'):
    """
    Makes a graph of nominal/non-nominal states by comparing the nominal graph states with the non-nominal graph states

    Parameters
    ----------
    g : networkx Graph
        graph for the fault scenario where the functions are nodes and flows are edges and with 'faults' and 'states' attributes
    nomg : networkx Graph
        graph for the nominal scenario where the functions are nodes and flows are edges and with 'faults' and 'states' attributes
    gtype : 'normal' or 'bipartite'
        whether the graph is a normal multgraph, or a bipartite graph. the default is 'bipartite'

    Returns
    -------
    rg : networkx graph
        copy of g with 'status' attributes added for faulty/degraded functions/flows
    """
    rg=g.copy()
    if gtype=='normal':
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
    elif gtype=='bipartite' or gtype=='component':
        for node in g.nodes:        
            if g.nodes[node]['bipartite']==0 or g.nodes[node].get('iscomponent', False): #condition only checked for functions
                if g.nodes[node].get('modes', {'nom'}).difference(['nom']): status='Faulty'
                elif g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
                else: status='Nominal'
            elif g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
            else: status='Nominal'
            rg.nodes[node]['status']=status
    elif gtype=='typegraph':
        for node in g.nodes:
            if g.nodes[node]['level']==2:
                if any({fxn for fxn, m in g.nodes[node]['modes'].items() if m not in [{'nom'},{}]}): status='Faulty'
                elif g.nodes[node]['states']!=nomg.nodes[node]['states']:   status='Degraded'
                else: status='Nominal'
            elif g.nodes[node]['level']==3:
                if g.nodes[node]['states']!=nomg.nodes[node]['states']: status='Degraded'
                else: status='Nominal'
            else: status='Nominal'
            rg.nodes[node]['status']=status
    return rg
def resultsgraphs(ghist, nomghist, gtype='bipartite'):
    """
    Makes a dict history of results graphs given a dict history of the nominal and faulty graphs

    Parameters
    ----------
    ghist : dict
        dict history of the faulty graph
    nomghist : dict
        dict history of the nominal graph
    gtype : str, optional
        Type of graph provided/returned (bipartite, component, or normal). The default is 'bipartite'.

    Returns
    -------
    rghist : dict
        dict history of results graphs
    """
    rghist = dict.fromkeys(ghist.keys())
    for i,rg in rghist.items():
        rghist[i] = resultsgraph(ghist[i],nomghist[i], gtype=gtype)
    return rghist

##HEATMAP FUNCTIONS
def heatmaps(reshist, diff):
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
def degtimemap(reshist):
    """ Makes a heatmap dictionary of degraded time for functions given a result history"""
    len_time = len(reshist['time'])
    degtimemap={}
    for fxnname in reshist['functions'].keys():
        degtimemap[fxnname]=1.0-sum(reshist['functions'][fxnname]['status'])/len_time
    for flowname in reshist['flows'].keys():
        degtimemap[flowname]=1.0 - sum(reshist['flows'][flowname])/len_time
    return degtimemap
def degtimemaps(reshists):
    """ Makes a dict of heatmap dictionaries of degraded time for functions given results histories"""
    degtimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        degtimemaps[reshist]=degtimemap(reshists[reshist])
    return degtimemaps
def avgdegtimeheatmap(reshists):
    """ Makes a heatmap dictionary of the average degraded heat time over a list of scenarios in the dict of results histories."""
    degtimetable = pd.DataFrame(degtimemaps(reshists)).transpose()
    return degtimetable.mean().to_dict()
def expdegtimeheatmap(reshists, endclasses):
    """ Makes a heatmap dictionary of the expected degraded heat time over a list of scenarios in the dict of results histories based on the rates in endclasses."""
    if 'nominal' in {*endclasses, *reshists}:
        if 'nominal' not in reshists:       endclasses=endclasses.copy(); endclasses.pop('nominal')
        elif 'nominal' not in endclasses:   reshists=reshists.copy(); reshists.pop('nominal')
    degtimetable = pd.DataFrame(degtimemaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expdegtimetable = degtimetable.multiply(rates).transpose()
    return expdegtimetable.sum().to_dict()
def faultmap(reshist):
    """ Makes a heatmap dictionary of faults given a results history."""
    heatmap={}
    for fxnname in reshist['functions'].keys():
        heatmap[fxnname] = max(reshist['functions'][fxnname]['numfaults'])
    return heatmap
def faultmaps(reshists):
    """ Makes dict of heatmaps dictionaries of resulting faults given a results history."""
    faulttimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        faulttimemaps[reshist]=faultmap(reshists[reshist])
    return faulttimemaps
def faultsheatmap(reshists):
    """Makes a heatmap dictionary of the average resulting faults over all scenarios"""
    faulttable = pd.DataFrame(faultmaps(reshists)).transpose()
    return faulttable.mean().to_dict()
def expfaultsheatmap(reshists, endclasses):
    """Makes a heatmap dictionary of the expected resulting faults over all scenarios"""
    if 'nominal' in {*endclasses, *reshists}:
        if 'nominal' not in reshists:       endclasses=endclasses.copy(); endclasses.pop('nominal')
        elif 'nominal' not in endclasses:   reshists=reshists.copy(); reshists.pop('nominal')
    faulttable = pd.DataFrame(faultmaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expfaulttable = faulttable.multiply(rates).transpose()
    return expfaulttable.mean().to_dict()
def totalcost(endclasses):
    """
    Tabulates the total expected cost of given endlcasses from a run.

    Parameters
    ----------
    endclasses : dict
        Dictionary of end-state classifications with 'expected cost' attributes

    Returns
    -------
    totalcost : Float
        The total expected cost of the scenarios.
    """
    return sum([e['expected cost'] for k,e in endclasses.items()])
def state_probabilities(endclasses):
    """
    Tabulates the probabilities of different states in endclasses.

    Parameters
    ----------
    endclasses : dict
        Dictionary of end-state classifications 'classification' and 'prob' attributes

    Returns
    -------
    probabilities : dict
        Dictionary of probabilities of different simulation classifications

    """
    classifications = set([props['classification'] for k,props in endclasses.items()])
    probabilities = dict.fromkeys(classifications)
    for classif in classifications:
        probabilities[classif] = sum([props['prob'] for k,props in endclasses.items() if classif==props['classification']])
    return probabilities
def nan_to_x(metric, x=0.0):
    """returns nan as zero if present, otherwise returns the number"""
    if np.isnan(metric):    return x
    else:                   return metric
def expected(endclasses, metric):
    """Calculates the expected value of a given metric in endclasses using the rate variable in endclasses"""
    return sum([e[metric]*e['rate'] for k,e in endclasses.items() if not np.isnan(e[metric])])
def average(endclasses, metric):
    """Calculates the average value of a given metric in endclasses"""
    return np.mean([e[metric] for k,e in endclasses.items() if not np.isnan(e[metric])])
def percent(endclasses, metric):
    """Calculates the percentage of a given indicator variable being True in endclasses"""
    return sum([int(bool(e[metric])) for k,e in endclasses.items() if not np.isnan(e[metric])])/len(endclasses)

def end_diff(endclasses, metric, nan_as=np.nan, as_ind=False, no_diff=False):
    """
    Calculates the difference between the nominal and fault scenarios for a set of endclasses

    Parameters
    ----------
    endclasses : dict
        endclass dictionary for the set {scen:endclass}, where endclass is a dict of metrics
    metric : str
        metric to calculate the difference of in the endclasses
    nan_as : float, optional
        How do deal with nans in the difference. The default is np.nan.
    as_ind : bool, optional
        Whether to return the difference as an indicator (1,-1,0) or real value. The default is False.
    no_diff : bool, optional
        Option for not computing the difference (but still performing the processing here). The default is False.
    Returns
    -------
    difference : dict
        dictionary of differences over the set of scenarios
    """
    endclasses=endclasses.copy()
    nomendclass = endclasses.pop('nominal')
    if not no_diff: 
        if as_ind:  difference = {scen:bool(nan_to_x(ec[metric], nan_as))-bool(nan_to_x(nomendclass[metric], nan_as)) for scen, ec in endclasses.items()}
        else:       difference = {scen:nan_to_x(nomendclass[metric], nan_as)-nan_to_x(ec[metric], nan_as) for scen, ec in endclasses.items()}
    else:           
        difference = {scen:nan_to_x(ec[metric], nan_as) for scen, ec in endclasses.items()}
        if as_ind: difference = {scen:np.sign(metric) for scen,metric in difference.items()}
    return difference
def overall_diff(endclasses, metric, nan_as=np.nan, as_ind=False, no_diff=False):
    return {scen:end_diff(endclass, metric, nan_as=nan_as, as_ind=as_ind, no_diff=no_diff) for scen, endclass in endclasses.items()}

def rate(endclasses, metric):
     """Calculates the rate of a given indicator variable being True in endclasses using the rate variable in endclasses"""
     return sum([int(bool(e[metric]))*e['rate'] for k,e in endclasses.items() if not np.isnan(e[metric])])
 
def bootstrap_confidence_interval(data, sample_size='data', num_samples=1000, interval=95, seed=False):
    """
    Calculates the bootstrap confidence interval for the mean (if data is float) 
    or proportion (if data is an indicator variable) of a set of data.

    Parameters
    ----------
    data : list/array/etc
        Iterable with the data. May be float (for mean) or indicator (for proportion)
    sample_size : int, optional
        Size of the sample used in the bootstrapping process. The default is 'sample', the size of the data.
        Specifying the sample size can reduce the computational time for large data, however using the size of the 
        data gives a narrower confidence interval.
    num_samples : int, optional
        Number of samples to bootstrap with. The default is 1000.
        Increase to decrease monte carlo error. 
    interval : int, optional
        Confidence interval to sample. The default is 95.
    seed : int, false
        Seed for numpy to use in the randomization
    Returns
    -------
    boot_mean: float
        bootstrap mean of the sample. should be equivalent to the sample mean (but not always)
    lb : float
        lower bound of the confidence interval
    ub : float
        upper bound of the confidence interval
    """
    tail_perc = (100-interval)/2
    if sample_size =='data': sample_size=len(data)
    if sample_size<10: print("Warning: sample size of "+str(sample_size)+" may be too small")
    if seed: np.random.seed(seed)
    boot_means = [np.mean(np.random.choice(data, sample_size, True)) for i in range(num_samples)]
    boot_mean = np.mean(boot_means)
    lower_bound = np.percentile(boot_means, tail_perc)
    upper_bound = np.percentile(boot_means, 100-tail_perc)
    return boot_mean, lower_bound, upper_bound
