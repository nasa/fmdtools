"""
Description: Processes model results for visualization

Uses methods:
    - :func:`save_result`:              Saves a given result variable (endclasses or mdlhists) to a file filename. 
    - :func:`load_result`:              Loads a given (endclasses or mdlhists) results dictionary from a (pickle/csv/json) file.
    - :func:`load_results`:             Loads endclass/mdlhist results from a given folder 
    - :func:`flatten_hist`:             Recursively creates a flattenned (single-level dictionary) history of the given nested model history
    - :func:`renest_flattened_hist:     Re-nests a flattened history.  
    - :func:`hists`:                    Processes a model histories for each scenario into results histories by comparing the states over time in each scenario with the states in the nominal scenario.
    - :func:`hist`:                     Compares model history with the nominal model history over time to make a history of degradation.
        - :func:`fxnhist`:              Compares the history of function states in mdlhist over time.
        - :func:`flowhist`:             Compares the history of flow states in mdlhist over time.
    - :func:`modephases`:               Identifies the phases of operation for the system based on a mdlhist with a history of its modes
    - :func:`graphflows`:               Extracts non-nominal flows by comparing the a results graph with a nominal results graph.
    - :func:`resultsgraph`:         Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - :func:`resultsgraphs`:        Makes a dict history of results graphs given a dict history of the nominal and faulty graphs
    - :func:`totalcost`:            Calculates the total host of a set of given end classifications
    - :func:`state_probabilities`:  Calculates the probabilities of given end-state classifications given an endclasses dictionary
    - :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap. 
    - :func:`overall_diff`:         Calculates the difference between the nominal and fault scenarios for a set of nested endclasses
    - :func:`end_diff`:             Calculates the difference between the nominal and fault scenarios for a set of endclasses
    - :func:`percent`:              Calculates the percentage of a given indicator variable being True in endclasses
    - :func:`average`:              Calculates the average value of a given metric in endclasses
    - :func:`expected`:             Calculates the expected value of a given metric in endclasses using the rate variable in endclasses
    - :func:`rate`:                Calculates the rate of a given indicator variable being True in endclasses using the rate variable in endclasses
Also used for graph heatmaps, which use the results history to map results history statistics onto the graph, returning a dictonary with structure {fxn/flow: value}:         
    - :func:`heatmaps`:            Makes a dict of heatmaps given a results history and a history of the differences between nominal and faulty models.
    - :func:`degtime_heatmap`:          Makes a heatmap dictionary of degraded time for functions given a result history
    - :func:`degtime_heatmaps`:         Makes a dict of heatmap dictionaries of degraded time for functions given results histories
    - :func:`avg_degtime_heatmap`:   Makes a heatmap dictionary of the average degraded heat time over a list of scenarios in the dict of results histories.
    - :func:`exp_degtime_heatmap`:   Makes a heatmap dictionary of the expected degraded heat time over a list of scenarios in the dict of results histories based on the rates in endclasses.
    - :func:`fault_heatmap`:            Makes a heatmap dictionary of faults given a results history.
    - :func:`fault_heatmaps`:           Makes dict of heatmaps dictionaries of resulting faults given a results history.
    - :func:`faults_heatmap`:       Makes a heatmap dictionary of the average resulting faults over all scenarios
    - :func:`exp_faults_heatmap`:    Makes a heatmap dictionary of the expected resulting faults over all scenarios
"""
#File Name: resultdisp/process.py
#Author: Daniel Hulse
#Created: November 2019 (Refactored April 2020)

import copy
import numpy as np
import pandas as pd
import sys,os
from ordered_set import OrderedSet
from fmdtools.faultsim.propagate import cut_mdlhist
from scipy.stats import bootstrap

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
            if returndiff: get_diff(faulty, nominal, att, diff[flowname])
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
            if type(fhist[state])==dict:
                fxnshist[fxnname][state] = {}
                diff[fxnname][state]={}
                for substate in fhist[state]:
                    if substate!='faults':
                        get_diff_fxnhist(mdlhist['faulty']['functions'][fxnname][state][substate], mdlhist['nominal']['functions'][fxnname][state][substate], \
                                         diff[fxnname][state], fxnshist[fxnname][state], substate)
                if {'faults', 't_loc','mode'}.intersection(fhist[state]):
                    fxnshist[fxnname][state]['faults']= mdlhist['faulty']['functions'][fxnname][state].get('faults', np.zeros(len(mdlhist['faulty']['time'])))
                    fxnshist[fxnname][state]['numfaults'] = get_fault_hist(fxnshist[fxnname][state]['faults'], fxnname)
                fxnshist[fxnname][state]['status'] = get_status(len(mdlhist['faulty']['time']),fxnshist[fxnname][state])
            else:
                get_diff_fxnhist(mdlhist['faulty']['functions'][fxnname][state], mdlhist['nominal']['functions'][fxnname][state], \
                                 diff[fxnname], fxnshist[fxnname], state)
        fxnshist[fxnname]['faults']=mdlhist['faulty']['functions'][fxnname].get('faults', np.zeros(len(mdlhist['faulty']['time'])))
        fxnshist[fxnname]['numfaults'] = get_fault_hist(fxnshist[fxnname]['faults'], fxnname)
        fxnshist[fxnname]['status'] = get_status(len(mdlhist['faulty']['time']),fxnshist[fxnname])
        faulthist[fxnname]=fxnshist[fxnname]['numfaults']
        deghist[fxnname] = fxnshist[fxnname]['status']
        if 0 in deghist[fxnname] or any(0 < faulthist[fxnname]): degfxns+=[fxnname]
    numfaults = np.sum(np.array(list(faulthist.values())), axis=0)
    numdegfxns   = len(deghist) - np.sum(np.array(list(deghist.values())), axis=0)
    return fxnshist, numfaults, degfxns, numdegfxns, diff
def get_status(timelen, fhist=[]):
    stat=np.prod(np.array(list([i for j,i in fhist.items() if (type(i)!=dict and j not in ['faults', 'numfaults'])])), axis = 0)
    #if not stat:       stat = np.ones(timelen, dtype=int)
    return stat * (1 - 1*(fhist.get('numfaults', 0)>0)) 
def get_diff_fxnhist(faulty, nominal, diff, fxnhist, state):
    fxnhist[state] = 1* (faulty == nominal)
    get_diff(faulty, nominal, state, diff)
def get_diff(faulty, nominal, state, diff):
    if state=='mode' or faulty.dtype.type==np.str_: 
        diff[state] = [int(nominal[i]!=faulty[i]) for i,f in enumerate(nominal)]
    elif faulty.dtype.type==np.bool_:
        diff[state] = 1*nominal - 1*faulty
    else:   diff[state] = nominal - faulty
def get_fault_hist(faults, fxnname):
    if type(faults)==dict:             return np.sum([fhist for fhist in faults.values()], axis=0)
    elif type(faults[0])==np.float64:  return faults
    elif type(faults[0])==np.str_:     return np.array([int(f!='nom') for f in faults])
    else:   raise Exception("Invalid data type in "+fxnname+" hist: "+str(type(faults)))
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
def degtime_heatmap(reshist):
    """ Makes a heatmap dictionary of degraded time for functions given a result history"""
    len_time = len(reshist['time'])
    degtimemap={}
    for fxnname in reshist['functions'].keys():
        degtimemap[fxnname]=1.0-sum(reshist['functions'][fxnname]['status'])/len_time
    for flowname in reshist['flows'].keys():
        degtimemap[flowname]=1.0 - sum(reshist['flows'][flowname])/len_time
    return degtimemap
def degtime_heatmaps(reshists):
    """ Makes a dict of heatmap dictionaries of degraded time for functions given results histories"""
    degtimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        degtimemaps[reshist]=degtime_heatmap(reshists[reshist])
    return degtimemaps
def avg_degtime_heatmap(reshists):
    """ Makes a heatmap dictionary of the average degraded heat time over a list of scenarios in the dict of results histories."""
    degtimetable = pd.DataFrame(degtime_heatmaps(reshists)).transpose()
    return degtimetable.mean().to_dict()
def exp_degtime_heatmap(reshists, endclasses):
    """ Makes a heatmap dictionary of the expected degraded heat time over a list of scenarios in the dict of results histories based on the rates in endclasses."""
    if 'nominal' in {*endclasses, *reshists}:
        if 'nominal' not in reshists:       endclasses=endclasses.copy(); endclasses.pop('nominal')
        elif 'nominal' not in endclasses:   reshists=reshists.copy(); reshists.pop('nominal')
    degtimetable = pd.DataFrame(degtime_heatmaps(reshists))
    rates = list(pd.DataFrame(endclasses).transpose()['rate'])
    expdegtimetable = degtimetable.multiply(rates).transpose()
    return expdegtimetable.sum().to_dict()
def fault_heatmap(reshist):
    """ Makes a heatmap dictionary of faults given a results history."""
    heatmap={}
    for fxnname in reshist['functions'].keys():
        heatmap[fxnname] = max(reshist['functions'][fxnname]['numfaults'])
    return heatmap
def fault_heatmaps(reshists):
    """ Makes dict of heatmaps dictionaries of resulting faults given a results history."""
    faulttimemaps=dict.fromkeys(reshists.keys())
    for reshist in reshists:
        faulttimemaps[reshist]=fault_heatmap(reshists[reshist])
    return faulttimemaps
def faults_heatmap(reshists):
    """Makes a heatmap dictionary of the average resulting faults over all scenarios"""
    faulttable = pd.DataFrame(fault_heatmaps(reshists)).transpose()
    return faulttable.mean().to_dict()
def exp_faults_heatmap(reshists, endclasses):
    """Makes a heatmap dictionary of the expected resulting faults over all scenarios"""
    if 'nominal' in {*endclasses, *reshists}:
        if 'nominal' not in reshists:       endclasses=endclasses.copy(); endclasses.pop('nominal')
        elif 'nominal' not in endclasses:   reshists=reshists.copy(); reshists.pop('nominal')
    faulttable = pd.DataFrame(fault_heatmaps(reshists))
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
def average(endclasses, metric, empty_as='nan'):
    """Calculates the average value of a given metric in endclasses"""
    ecs = [e[metric] for k,e in endclasses.items() if not np.isnan(e[metric])]
    if len(ecs)>0 or empty_as=='nan':   return np.mean(ecs)
    else:                               return empty_as
def percent(endclasses, metric):
    """Calculates the percentage of a given indicator variable being True in endclasses"""
    return sum([int(bool(e[metric])) for k,e in endclasses.items() if not np.isnan(e[metric])])/(len(endclasses)+1e-16)
def rate(endclasses, metric):
     """Calculates the rate of a given indicator variable being True in endclasses using the rate variable in endclasses"""
     return sum([int(bool(e[metric]))*e['rate'] for k,e in endclasses.items() if not np.isnan(e[metric])])
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
        else:       difference = {scen:nan_to_x(ec[metric], nan_as)-nan_to_x(nomendclass[metric], nan_as) for scen, ec in endclasses.items()}
    else:           
        difference = {scen:nan_to_x(ec[metric], nan_as) for scen, ec in endclasses.items()}
        if as_ind: difference = {scen:np.sign(metric) for scen,metric in difference.items()}
    return difference
def overall_diff(nested_endclasses, metric, nan_as=np.nan, as_ind=False, no_diff=False):
    """
    Calculates the difference between the nominal and fault scenarios over a set of endclasses

    Parameters
    ----------
    nested_endclasses : dict
        Nested dict of endclasses from propogate.nested
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
    differences : dict
        nested dictionary of differences over the set of fault scenarios nested in nominal scenarios 
    """
    return {scen:end_diff(endclass, metric, nan_as=nan_as, as_ind=as_ind, no_diff=no_diff) for scen, endclass in nested_endclasses.items()}
def bootstrap_confidence_interval(data, method=np.mean, return_anyway=False, **kwargs):
    """
    Convenience wrapper for scipy.bootstrap. 

    Parameters
    ----------
    data : list/array/etc
        Iterable with the data. May be float (for mean) or indicator (for proportion)
    method : 
        numpy method to give scipy.bootstrap.
    return_anyway: bool
        Gives a dummy interval of (stat, stat) if no . Used for plotting
    Returns
    ----------
    statistic, lower bound, upper bound
    """
    if 'interval' in kwargs: kwargs['confidence_level']=kwargs.pop('interval')*0.01
    if data.count(data[0])!=len(data):
        bs = bootstrap([data], np.mean, **kwargs)
        return method(data), bs.confidence_interval.low, bs.confidence_interval.high
    elif return_anyway: return method(data), method(data), method(data)
    else: raise Exception("All data are the same!")

def file_check(filename, overwrite):
    """Check if files exists and whether to overwrite the file"""
    if os.path.exists(filename):
        if not overwrite: raise Exception("File already exists: "+filename)
        else:                   
            print("File already exists: "+filename+", writing anyway...")
            os.remove(filename)

def save_result(variable, filename, filetype="", overwrite=False, result_id=''):
    """
    Saves a given result variable (endclasses or mdlhists) to a file filename. 
    Files can be saved as pkl, csv, or json.

    Parameters
    ----------
    variable : dict
        Results dictionary (endclasses or mdlhists)
    filename : str
        File name for the file. Can be nested in a folder if desired.
    filetype : str, optional
        Optional specifier of file type (if not included in filename). The default is "".
    overwrite : bool, optional
        Whether to overwrite existing files with this name. The default is False.
    result_id : str, optional
        For individual results saving. Places an identifier for the result in the file. The default is ''.
    """
    import dill, json, csv
    file_check(filename, overwrite)
    
    if "/" in filename:
        last_split_index = filename.rfind("/")
        foldername = filename[:last_split_index]
        if not os.path.exists(foldername): os.makedirs(foldername)
    
    filetype = auto_filetype(filename, filetype)
    if filetype=='pickle':
        with open(filename, 'wb') as file_handle:
            if result_id: variable = {result_id:variable}
            dill.dump(variable, file_handle)
    elif filename[-4:]=='.csv': # add support for nested dict mdlhist using flatten_hist?
        variable = flatten_hist(variable)
        with open(filename, 'w', newline='') as file_handle:
            writer = csv.writer(file_handle)
            if result_id: writer.writerow([result_id])
            writer.writerow(variable.keys())
            if isinstance([*variable.values()][0], np.ndarray):
                writer.writerows(zip(*variable.values()))
            else:
                writer.writerow([*variable.values()])
    elif filename[-5:]=='.json':
        with open(filename, 'w', encoding='utf8') as file_handle:
            variable = flatten_hist(variable)
            new_variable = {}
            for key in variable:
                if isinstance(variable[key], np.ndarray):
                    new_variable[str(key)] =  [var.item() for var in variable[key]]
                else:
                    new_variable[str(key)] =  variable[key]
            if result_id: new_variable = {result_id:new_variable}
            strs = json.dumps(new_variable, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
            file_handle.write(str(strs))
    else:
        raise Exception("Invalid File Type")
    file_handle.close()
        
def load_result(filename, filetype="", renest_dict=True, indiv=False):
    """
    Loads a given (endclasses or mdlhists) results dictionary from a (pickle/csv/json) file.
    e.g. a file saved using process.save_result or save_args in propagate functions.

    Parameters
    ----------
    filename : str
        Name of the file.
    filetype : str, optional
        Use to specify a filetype for the file (if not included in the filename). The default is "".
    renest_dict : bool, optional
        Whether to return . The default is True.
    indiv : bool, optional
        Whether the result is an individual file (e.g. in a folder of results for a given simulation). 
        The default is False.

    Returns
    -------
    resultdict : dict
        Corresponding results dictionary from the file.
    """
    import dill, json, csv, pandas
    if not os.path.exists(filename): raise Exception("File does not exist: "+filename)
    filetype = auto_filetype(filename, filetype)
    if filetype=='pickle':
        with open(filename, 'rb') as file_handle:
            resultdict = dill.load(file_handle)
        file_handle.close()
    elif filetype=='csv': # add support for nested dict mdlhist using flatten_hist?
        if indiv:   resulttab = pandas.read_csv(filename, skiprows=1)
        else:           resulttab = pandas.read_csv(filename)
        resultdict = resulttab.to_dict("list")
        resultdict = clean_resultdict_keys(resultdict)
        for key in resultdict:
            if len(resultdict[key])==1 and isinstance(resultdict[key], list):
                resultdict[key] = resultdict[key][0]
            else: resultdict[key] = np.array(resultdict[key])             
        if renest_dict: resultdict = nest_flattened_hist(resultdict)
        if indiv: 
            scenname = [*pandas.read_csv(filename, nrows=0).columns][0]
            resultdict = {scenname: resultdict}
    elif filetype=='json':
        with open(filename, 'r', encoding='utf8') as file_handle:
            loadeddict = json.load(file_handle)
            if indiv:   
                key = [*loadeddict.keys()][0]
                loadeddict = loadeddict[key]
                loadeddict= {key+", "+innerkey:values for innerkey, values in loadeddict.items()}
                resultdict = clean_resultdict_keys(loadeddict)
            else:       resultdict = clean_resultdict_keys(loadeddict)
            
            if renest_dict: resultdict = nest_flattened_hist(resultdict)
        file_handle.close()
    else:
        raise Exception("Invalid File Type")
    return resultdict
def clean_resultdict_keys(resultdict_dirty):
    """
    Helper function for recreating results dictionary keys (tuples) from a dictionary loaded from a file (where keys are strings)
    (used in csv/json results)

    Parameters
    ----------
    resultdict_dirty : dict
        Results dictionary where keys are strings

    Returns
    -------
    resultdict : dict
        Results dictionary where keys are tuples
    """
    resultdict = {}
    for key in resultdict_dirty:
        newkey = tuple(key.replace("'","").replace("(","").replace(")","").split(", "))
        if any(['t=' in strs for strs in  newkey]):
            joinfirst = [ind for ind, strs in enumerate(newkey) if 't=' in strs][0] +1
        else:                           joinfirst=0
        if joinfirst==2:
            newkey = tuple([", ".join(newkey[:2])])+newkey[2:]
        elif joinfirst>2:
            nomscen = newkey[:joinfirst-2]
            faultscen = tuple([", ".join(newkey[joinfirst-2:joinfirst])])
            vals = newkey[joinfirst:]
            newkey = nomscen+faultscen+vals
        resultdict[newkey]=resultdict_dirty[key]
    return resultdict

def load_results(folder, filetype, renest_dict=True):
    """
    Loads endclass/mdlhist results from a given folder 
    (e.g., that have been saved from multi-scenario propagate methods with 'indiv':True)

    Parameters
    ----------
    folder : str
        Name of the folder. Must be in the current directory
    filetype : str
        Type of files in the folder ('pickle', 'csv', or 'json')
    renest_dict : bool, optional
        Whether to return result as a nested dict (as opposed to a flattenned dict). 
        The default is True.

    Returns
    -------
    resultdict : dict
        endclasses/mdlhists result dictionary reconstructed by the files in the folder.
    """
    files = os.listdir(folder)
    files_toread = []
    for file in files:
        read_filetype = auto_filetype(file)
        if read_filetype==filetype:
            files_toread.append(file)
    resultdict = {}
    for filename in files_toread:
        resultdict.update(load_result(folder+'/'+filename, filetype, renest_dict=renest_dict, indiv=True))
    return resultdict

def auto_filetype(filename, filetype=""):
    """Helper function that automatically determines the filetype (pickle, csv, or json) of a given filename"""
    if not filetype:
        if '.' not in filename: raise Exception("No file extension")
        if filename[-4:]=='.pkl':       filetype="pickle"
        elif filename[-4:]=='.csv':     filetype="csv"     
        elif filename[-5:]=='.json':    filetype="json"
        else: raise Exception("Invalid File Type in: "+filename+", ensure extension is pkl, csv, or json ")
    return filetype
def create_indiv_filename(filename, indiv_id, splitchar='_'):
    """Helper file that creates an individualized name for a file given the general filename and an individual id"""
    filename_parts = filename.split(".")
    filename_parts.insert(1,'.')
    filename_parts.insert(1,splitchar+indiv_id)   
    return "".join(filename_parts)

def get_hist_memory(hist):
    """
    Determines the memory usage of a given history and profiles by 

    Parameters
    ----------
    hist : TYPE
        DESCRIPTION.

    Returns
    -------
    mem_total : int
        Total memory usage of the history (in bytes)
    mem_profile : dict
        Memory usage of each construct of the model history (in bytes)
    """
    fhist = flatten_hist(hist)
    mem_total = 0
    mem_profile = dict.fromkeys(fhist.keys())
    for k,h in fhist.items():
        if np.issubdtype(h.dtype, np.number) or np.issubdtype(h.dtype, np.flexible) or np.issubdtype(h.dtype, np.bool_):
            mem=h.nbytes
        else:
            mem=0
            for entry in h:
                mem+=sys.getsizeof(entry)
        mem_total+=mem
        mem_profile[k]=mem
    return mem_total, mem_profile

def get_flat_hist_slice(flathist,t_ind=0):
    """
    Returns a dictionary of values from a given (flat) mdlhist at t_ind
    """
    slice_dict = dict.fromkeys(flathist)
    for key, arr in flathist.items():
        slice_dict[key]=flathist[key][t_ind]
    return slice_dict

def flatten_hist(hist, newhist = False, prevname=(), to_include='all'):
    """
    Recursively creates a flattenned history of the given nested model history

    Parameters
    ----------
    hist : dict
        Model history (e.g., from faultsim.propagate.nominal).
    newhist : dict, optional
        Flattened Model History (used when called recursively). The default is False.
    prevname : tuple, optional
        Current key of the flattened history (used when called recursively). The default is ().
    to_include : str/list/dict, optional
        What attributes to include in the dict. The default is 'all'. Can be of form
        - list e.g. ['att1', 'att2', 'att3'] to include the given attributes
        - dict e.g. fxnflowvals {'flow1':['att1', 'att2'], 'fxn1':'all', 'fxn2':['comp1':all, 'comp2':['att1']]}
        - str e.g. 'att1' for attribute 1 or 'all' for all attributes
    Returns
    -------
    newhist : dict
        Flattened model history of form: {(fxnflow, ..., attname):array(att)}
    """
    if newhist==False: newhist = dict()
    for att, val in hist.items():
        newname = prevname+tuple([att])
        if type(val)==dict: 
            if type(to_include)==list and att in to_include: new_to_include = 'all'
            elif type(to_include)==set and att in to_include: new_to_include = 'all'
            elif type(to_include)==dict and att in to_include: new_to_include = to_include[att]
            elif type(to_include)==str and att== to_include: new_to_include = 'all'
            elif to_include =='all': new_to_include='all'
            elif att in ['functions', 'flows']: new_to_include = to_include
            else: new_to_include= False
            if new_to_include: flatten_hist(val, newhist, newname, new_to_include)
        elif to_include=='all' or att in to_include: 
            if len(newname)==1: newhist[newname[0]] = val
            else:               newhist[newname] = val
    return newhist

def nest_flattened_hist(hists, prefix = ()):
    """
    Re-nests a flattened history.   

    Parameters
    ----------
    hists : dict
        Flattened Model history (e.g. from flatten_hist)
    """
    newhist = {}
    key_options = set([h[0] for h in hists.keys()])
    for key in key_options:
        if (key,) in hists:     newhist[key] = hists[(key,)]
        else:
            subdict = {histkey[1:]:val for histkey, val in hists.items() if key==histkey[0]}                       
            newhist[key] = nest_flattened_hist(subdict)
    return newhist
            

