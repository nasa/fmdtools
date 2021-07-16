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
    
    
