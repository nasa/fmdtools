"""
File Name: resultdisp/tabulate.py
Author: Daniel Hulse
Created: November 2019 (Refactored April 2020)

Description: Translates simulation outputs to pandas tables for display, export, etc.

Uses methods:
    - hist:           Returns formatted pandas dataframe of model history
    - objtab:         Make table of function OR flow value attributes - objtype = 'function' or 'flow'
    - stats:          Makes a table of #of degraded flows, # of degraded functions, and # of total faults over time given a single result history
    - degflows:       Makes a  of flows over time, where 0 is degraded and 1 is nominal
    - degflowvals:    Makes a table of individual flow state values over time, where 0 is degraded and 1 is nominal
    - degfxns:        Makes a table showing which functions are degraded over time (0 for degraded, 1 for nominal)
    - deghist:        Makes a table of all funcitons and flows that are degraded over time. If withstats=True, the total # of each type degraded is provided in the last columns
    - heatmaps:       Makes a table of a heatmap dictionary
    - costovertime:   Makes a table of the total cost, rate, and expected cost of all faults over time
    - samptime:       Makes a table of the times sampled for each phase given a dict (i.e. app.sampletimes)
    - summary:        Makes a table of a summary dictionary from a given model run
    - result:         Makes a table of results (degraded functions/flows, cost, rate, expected cost) of a single run
    - dicttab:           Makes table of a generic dictionary
    - maptab:            Makes table of a generic map
Also used for FMEA-like tables:
    - simplefmea:          Makes a simple fmea (rate, cost, expected cost) of the endclasses of a list of fault scenarios run
    - phasefmea:           Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by phase.
    - summfmea:            Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by fault.
    - fullfmea:            Makes full fmea table (degraded functions/flows, cost, rate, expected cost) of scenarios given endclasses dict (cost, rate, expected cost) and summaries dict (degraded functions, degraded flows)
"""
import pandas as pd
import numpy as np

#makehisttable
# put history in a tabular format
def hist(mdlhist):
    """ Returns formatted pandas dataframe of model history"""
    if "nominal" in mdlhist.keys(): mdlhist=mdlhist['faulty']
    if any(isinstance(i,dict) for i in mdlhist['flows'].values()):
        flowtable =  objtab(mdlhist, 'flows')
    else:
        flowtable = objtab(mdlhist, 'flowvals')
    fxntable  =  objtab(mdlhist, 'functions')
    timetable = pd.DataFrame()
    timetable['time', 't'] = mdlhist['time']
    timetable.reindex([('time', 't')], axis="columns")
    histtable = pd.concat([timetable, fxntable, flowtable], axis =1)
    index = pd.MultiIndex.from_tuples(histtable.columns)
    histtable = histtable.reindex(index, axis='columns')
    return histtable
def objtab(hist, objtype):
    """make table of function OR flow value attributes - objtype = 'function' or 'flow'"""
    df = pd.DataFrame()
    labels = []
    for fxn, atts in hist[objtype].items():
        for att, val in atts.items():
            if att != 'faults':
                label=(fxn, att)
                labels=labels+[label]
                df[label]=val
        if objtype =='functions':
            faulthist = hist[objtype][fxn].get('faults', {})
            if type(faulthist)==dict:
                for fault in faulthist:
                    label=(fxn, fault+' fault')
                    labels+=[label]
                    df[label]=hist[objtype][fxn]['faults'][fault]
            elif len(faulthist)==1:
                label=(fxn, 'faults')
                labels+=[label]
                df[label]=hist[objtype][fxn]['faults']
                
    index = pd.MultiIndex.from_tuples(labels)
    df = df.reindex(index, axis="columns")
    return df
def stats(reshist):
    """Makes a table of #of degraded flows, # of degraded functions, and # of total faults over time given a single result history"""
    table = pd.DataFrame(reshist['stats'])
    table.insert(0, 'time', reshist['time'])
    return table
def degflows(reshist):
    """Makes a table of flows over time, where 0 is degraded and 1 is nominal"""
    table = pd.DataFrame(reshist['flows'])
    table.insert(0, 'time', reshist['time'])
    return table
def degflowvals(reshist):
    """Makes a table of individual flow state values over time, where 0 is degraded and 1 is nominal"""
    table = objtab(reshist, 'flowvals')
    table.insert(0, 'time', reshist['time'])
    return table
def degfxns(reshist):
    """Makes a table showing which functions are degraded over time (0 for degraded, 1 for nominal)"""
    table = pd.DataFrame()
    for fxnname in reshist['functions']:
        table[fxnname]=reshist['functions'][fxnname]['status']
    table.insert(0, 'time', reshist['time'])
    return table
def deghist(reshist, withstats=False):
    """Makes a table of all funcitons and flows that are degraded over time. If withstats=True, the total # of each type degraded is provided in the last columns """
    fxnstable = degfxns(reshist)
    flowstable = pd.DataFrame(reshist['flows'])
    if withstats:
        statstable = pd.DataFrame(reshist['stats'])
        return pd.concat([fxnstable, flowstable, statstable], axis =1)
    else:
        return pd.concat([fxnstable, flowstable], axis =1)
def heatmaps(heatmaps):
    """Makes a table of a heatmap dictionary"""
    table = pd.DataFrame(heatmaps)
    return table.transpose()
def costovertime(endclasses, app):
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
def samptime(sampletimes):
    """Makes a table of the times sampled for each phase given a dict (i.e. app.sampletimes)"""
    table = pd.DataFrame()
    for phase, times in sampletimes.items():
        table[phase]= [str(list(times.keys()))]
    return table.transpose()
def summary(summary):
    """Makes a table of a summary dictionary from a given model run"""
    return pd.DataFrame.from_dict(summary, orient = 'index')    
def result(endresults, summary):
    """Makes a table of results (degraded functions/flows, cost, rate, expected cost) of a single run"""
    table = pd.DataFrame(endresults['classification'], index=[0])
    table['degraded functions'] = [summary['degraded functions']]
    table['degraded flows'] = [summary['degraded flows']]
    return table

def dicttab(dictionary):
    """Makes table of a generic dictionary"""
    return pd.DataFrame(dictionary, index=[0])
def maptab(mapping):
    """Makes table of a generic map"""
    table = pd.DataFrame(mapping)
    return table.transpose()
    
##FMEA-like tables
def simplefmea(endclasses):
    """Makes a simple fmea (rate, cost, expected cost) of the endclasses of a list of fault scenarios run"""
    table = pd.DataFrame(endclasses)
    return table.transpose()
def phasefmea(endclasses, app):
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
def summfmea(endclasses, app):
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
def fullfmea(endclasses, summaries):
    """Makes full fmea table (degraded functions/flows, cost, rate, expected cost) of scenarios given endclasses dict (cost, rate, expected cost) and summaries dict (degraded functions, degraded flows)"""
    degradedtable = pd.DataFrame(summaries)
    simplefmea=pd.DataFrame(endclasses)
    fulltable = pd.concat([degradedtable, simplefmea])
    return fulltable.transpose()

