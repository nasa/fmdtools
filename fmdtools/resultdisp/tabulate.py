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
    - nominal_stats:  Makes a table of quantities of interest from endclasses from a nominal approach.
    - nested_stats:   Makes a table of quantities of interest from endclasses from a nested approach.
    - nominal_factor_comparison: Compares a metric for a given set of model parameters/factors over a set of nominal scenarios.
    - nested_factor_comparison: Compares a metric for a given set of model parameters/factors over a nested set of nominal and fault scenarios.
Also used for FMEA-like tables:
    - simplefmea:          Makes a simple fmea (rate, cost, expected cost) of the endclasses of a list of fault scenarios run
    - phasefmea:           Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by phase.
    - summfmea:            Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by fault.
    - fullfmea:            Makes full fmea table (degraded functions/flows, cost, rate, expected cost) of scenarios given endclasses dict (cost, rate, expected cost) and summaries dict (degraded functions, degraded flows)
"""
import pandas as pd
import numpy as np
from fmdtools.resultdisp.process import expected, average, percent, rate, overall_diff, nan_to_x, bootstrap_confidence_interval

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

def nominal_stats(nomapp, nomapp_endclasses, metrics='all', inputparams='from_range', scenarios='all'):
    """
    Makes a table of quantities of interest from endclasses.

    Parameters
    ----------
    nomapp : NominalApproach
        NominalApproach used to generate the simulation.
    nomapp_endclasses: dict
        End-state classifcations for the set of simulations from propagate.nominalapproach()
    metrics : 'all'/list, optional
        Metrics to show on the plot. The default is 'all'.
    inputparams : 'from_range'/'all',list, optional
        Parameters to show on the plot. The default is 'from_range'.
    scenarios : 'all','range'/list, optional
        Scenarios to include in the plot. 'range' is a given range_id in the nominalapproach.
    Returns
    -------
    table : pandas DataFrame
        Table with the metrics of interest layed out over the input parameters for the set of scenarios in endclasses
    """
    if metrics=='all':              metrics = [*nomapp_endclasses[[*nomapp_endclasses][0]]]
    if scenarios=='all':            scens = [*nomapp_endclasses]
    elif type(scenarios)==str:      scens = nomapp.ranges[scenarios]['scenarios']
    elif not type(scenarios)==list: raise Exception("Invalid option for scenarios. Provide 'all'/'rangeid' or list")
    else:                           scens = scenarios
    if inputparams=='from_range': 
        ranges=[*nomapp.ranges]
        if not(scenarios=='all') and not(type(scenarios)==list):    app_range= scenarios
        elif len(ranges)==1:                                        app_range=ranges[0]
        else: raise Exception("Multiple approach ranges "+str(ranges)+" in approach. Use inputparams=`all` or inputparams=[param1, param2,...]")
        inputparams= [*nomapp.ranges[app_range]['inputranges']]
    elif inputparams=='all':    inputparams=[*nomapp.scenarios.values()][0]['properties']['inputparams']
    elif inputparams=='none':   inputparams=[]
    table_values=[]
    for inputparam in inputparams:
        table_values.append([nomapp.scenarios[e]['properties']['inputparams'][inputparam] for e in scens])
    for metric in metrics:
        table_values.append([nomapp_endclasses[e][metric] for e in scens])
    table = pd.DataFrame(table_values, columns=[*nomapp_endclasses], index=inputparams+metrics)
    return table

def nominal_factor_comparison(nomapp, endclasses, params, metrics='all', rangeid='default', nan_as=np.nan, percent=True, difference=True, give_ci=False, **kwargs):
    """
    Compares a metric for a given set of model parameters/factors over set of nominal scenarios.

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal Approach used to generate the simulations
    endclasses : dict
        dict of endclasses from propagate.nominal_approach or nested_approach with structure: 
            {scen_x:{metric1:x, metric2:x...}} or {scen_x:{fault:{metric1:x, metric2:x...}}} 
    params : list
        List of parameters to use for the factor levels in the comparison
    metrics : 'all'/list, optional
        Metrics to show in the table. The default is 'all'.
    rangeid : str, optional
        Nominal Approach range to use for the test (must be run over a single range). 
        The default is 'default', which picks the only range (if there is only one).
    nan_as : float, optional
        Number to parse NaNs as (if present). The default is np.nan.
    percent : bool, optional
        Whether to compare metrics as bools (True - results in a comparison of percentages of indicator variables) 
        or as averages (False - results in a comparison of average values of real valued variables). The default is True.
    difference : bool, optional
        Whether to tabulate the difference of the metric from the nominal over each scenario (True),
        or the value of the metric over all (False). The default is True.
    give_ci = bool:
        gives the bootstrap confidence interval for the given statistic using the given kwargs
        'combined' combines the values as a strings in the table (for display)
    kwargs : keyword arguments for bootstrap_confidence_interval (sample_size, num_samples, interval, seed)
    Returns
    -------
    table : pandas table
        Table with the metric statistic (percent or average) over the nominal scenario and each listed function/mode (as differences or averages)
    """
    if rangeid=='default':
        if len(nomapp.ranges.keys())==1: rangeid=[*nomapp.ranges.keys()][0]
        else:   raise Exception("More than one range in approach--please provide rangid in: "+str(nomapp.ranges.keys()))
    if [*endclasses.values()][0].get('nominal', False): endclasses ={scen:ec['nominal'] for scen, ec in endclasses.items()}
    if metrics=='all':              metrics = [ec for ec,val in [*endclasses.values()][0].items() if type(val) in [float, int]]
    
    
    factors = nomapp.get_param_scens(rangeid, *params)
    full_stats=[]
    for metric in metrics:
        factor_stats = []
        for factor, scens in factors.items():
            endclass_fact = {scen:endclass for scen, endclass in endclasses.items() if scen in scens}

            if not percent: nominal_metrics = [nan_to_x(scen[metric], nan_as) for scen in endclass_fact.values()]
            else:           nominal_metrics = [np.sign(nan_to_x(scen[metric], nan_as)) for scen in endclass_fact.values()]
            factor_stats= factor_stats + [sum(nominal_metrics)/len(nominal_metrics)]
            if give_ci: 
                factor_boot, factor_lb, factor_ub = bootstrap_confidence_interval(nominal_metrics, **kwargs)
                factor_stats = factor_stats + [factor_lb, factor_ub]
        full_stats.append(factor_stats)
    if give_ci=='combined': full_stats = [[str(round(v,3))+' ('+str(round(f[i+1],3))+','+str(round(f[i+2],3))+')' for i,v in enumerate(f) if not i%3] for f in full_stats]
    if give_ci !=True: 
        table = pd.DataFrame(full_stats, columns = factors, index=metrics)
        table.columns.name=tuple(params)
    else:           
        columns = [(f, stat) for f in factors for stat in ["", "LB", "UB"]]
        table = pd.DataFrame(full_stats, columns=columns, index=metrics)
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=['metric', ''])
        table.columns.name=tuple(params)
    return table

def resilience_factor_comparison(nomapp, nested_endclasses, params, value, faults='functions', rangeid='default', nan_as=np.nan, percent=True, difference=True, give_ci=False, **kwargs):
    """
    Compares a metric for a given set of model parameters/factors over a nested set of nominal and fault scenarios.

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal Approach used to generate the simulations
    nested_endclasses : dict
        dict of endclasses from propagate.nested_approach with structure: {scen_x:{fault:{metric1:x, metric2:x...}}}
    params : list
        List of parameters to use for the factor levels in the comparison
    value : string
        metric of the endclass (returned by mdl.find_classification) to use for the comparison.
    faults : str/list, optional
        Set of faults to run the comparison over
            --'modes' (all fault modes),
            --'functions' (modes for each function are grouped)
            --'mode type' (modes with the same name are grouped)
            -- or a set of specific modes/functions. The default is 'functions'.
    rangeid : str, optional
        Nominal Approach range to use for the test (must be run over a single range). 
        The default is 'default', which picks the only range (if there is only one).
    nan_as : float, optional
        Number to parse NaNs as (if present). The default is np.nan.
    percent : bool, optional
        Whether to compare metrics as bools (True - results in a comparison of percentages of indicator variables) 
        or as averages (False - results in a comparison of average values of real valued variables). The default is True.
    difference : bool, optional
        Whether to tabulate the difference of the metric from the nominal over each scenario (True),
        or the value of the metric over all (False). The default is True.
    give_ci = bool:
        gives the bootstrap confidence interval for the given statistic using the given kwargs
        'combined' combines the values as a strings in the table (for display)
    kwargs : keyword arguments for bootstrap_confidence_interval (sample_size, num_samples, interval, seed)
    Returns
    -------
    table : pandas table
        Table with the metric statistic (percent or average) over the nominal scenario and each listed function/mode (as differences or averages)
    """
    if rangeid=='default':
        if len(nomapp.ranges.keys())==1: rangeid=[*nomapp.ranges.keys()][0]
        else:   raise Exception("More than one range in approach--please provide rangid in: "+str(nomapp.ranges.keys()))
    if faults=='functions':     faultlist = set([e.partition(' ')[0] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif faults=='modes':       faultlist = set([e.partition(',')[0] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif faults=='mode type':   faultlist = set([e.partition(',')[0].partition(' ')[2] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif type(faults) ==str: raise Exception("Invalid faults option: "+faults)
    elif type(faults)==list:    faultlist =set(faults)
    else:                       faultlist=faults
    faultlist.discard('nominal'); faultlist.discard(' '); faultlist.discard('')
    
    factors = nomapp.get_param_scens(rangeid, *params)
    full_stats=[]
    for factor, scens in factors.items():
        endclass_fact = {scen:endclass for scen, endclass in nested_endclasses.items() if scen in scens}
        ec_metrics = overall_diff(endclass_fact, value, nan_as=nan_as, as_ind=percent, no_diff=not difference)

        if not percent: nominal_metrics = [nan_to_x(res_scens['nominal'][value], nan_as) for res_scens in endclass_fact.values()]
        else:           nominal_metrics = [np.sign(nan_to_x(res_scens['nominal'][value], nan_as)) for res_scens in endclass_fact.values()]
        factor_stats=[sum(nominal_metrics)/len(nominal_metrics)]
        if give_ci: 
            factor_boot, factor_lb, factor_ub = bootstrap_confidence_interval(nominal_metrics, **kwargs)
            factor_stats = factor_stats + [factor_lb, factor_ub]
        for fault in faultlist:
            if faults=='functions':     fault_metrics = [metric for res_scens in ec_metrics.values() for res_scen,metric in res_scens.items() if fault in res_scen.partition(' ')[0]]
            else:                       fault_metrics = [metric for res_scens in ec_metrics.values() for res_scen,metric in res_scens.items() if fault in res_scen.partition(',')[0]]
            if len(fault_metrics)>0:    
                factor_stats.append(sum(fault_metrics)/len(fault_metrics))
                if give_ci: 
                    factor_boot, factor_lb, factor_ub = bootstrap_confidence_interval(fault_metrics, **kwargs)
                    factor_stats= factor_stats+[factor_lb, factor_ub]
            else:                       
                if not give_ci: factor_stats.append(np.NaN)
                else:           factor_stats= factor_stats + [np.NaN,np.NaN,np.NaN]
        full_stats.append(factor_stats)
    if give_ci=='combined': full_stats = [[str(round(v,3))+' ('+str(round(f[i+1],3))+','+str(round(f[i+2],3))+')' for i,v in enumerate(f) if not i%3] for f in full_stats]
    if give_ci !=True: 
        table = pd.DataFrame(full_stats, columns = ['nominal']+list(faultlist), index=factors)
        table.columns.name=tuple(params)
    else:           
        columns = [(f, stat) for f in ['nominal']+list(faultlist) for stat in ["", "LB", "UB"]]
        table = pd.DataFrame(full_stats, columns=columns, index=factors)
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=['fault', ''])
        table.columns.name=tuple(params)
    return table

def nested_stats(nomapp, nested_endclasses, percent_metrics=[], rate_metrics=[], average_metrics=[], expected_metrics=[], inputparams='from_range', scenarios='all'):
    """
    Makes a table of quantities of interest from endclasses.

    Parameters
    ----------
    nomapp : NominalApproach
        NominalApproach used to generate the simulation.
    endclasses : dict
        End-state classifcations for the set of simulations from propagate.nested_approach()
    percent_metrics : list
        List of metrics to calculate a percent of (e.g. use with an indicator variable like failure=1/0 or True/False)
    rate_metrics : list
        List of metrics to calculate the probability of using the rate variable in endclasses
    average_metrics : list
        List of metrics to calculate an average of (e.g., use for float values like speed=25)
    expected_metrics : list
        List of metrics to calculate the expected value of using the rate variable in endclasses
    inputparams : 'from_range'/'all',list, optional
        Parameters to show on the table. The default is 'from_range'.
    scenarios : 'all','range'/list, optional
        Scenarios to include in the table. 'range' is a given range_id in the nominalapproach.
    Returns
    -------
    table : pandas DataFrame
        Table with the averages/percentages of interest layed out over the input parameters for the set of scenarios in endclasses
    """
    if scenarios=='all':            scens = [*nested_endclasses]
    elif type(scenarios)==str:      scens = nomapp.ranges[scenarios]['scenarios']
    elif not type(scenarios)==list: raise Exception("Invalid option for scenarios. Provide 'all'/'rangeid' or list")
    else:                           scens = scenarios
    if inputparams=='from_range': 
        ranges=[*nomapp.ranges]
        if not(scenarios=='all') and not(type(scenarios)==list):    app_range= scenarios
        elif len(ranges)==1:                                        app_range=ranges[0]
        else: raise Exception("Multiple approach ranges "+str(ranges)+" in approach. Use inputparams=`all` or inputparams=[param1, param2,...]")
        inputparams= [*nomapp.ranges[app_range]['inputranges']]
    elif inputparams=='all':
        inputparams=[*nomapp.scenarios.values()][0]['properties']['inputparams']
    table_values=[]; table_rows = inputparams
    for inputparam in inputparams:
        table_values.append([nomapp.scenarios[e]['properties']['inputparams'][inputparam] for e in scens])
    for metric in percent_metrics:  
        table_values.append([percent(nested_endclasses[e], metric) for e in scens])
        table_rows.append('perc_'+metric)
    for metric in rate_metrics:     
        table_values.append([rate(nested_endclasses[e], metric) for e in scens])
        table_rows.append('rate_'+metric)
    for metric in average_metrics:  
        table_values.append([average(nested_endclasses[e], metric) for e in scens])
        table_rows.append('ave_'+metric)
    for metric in expected_metrics: 
        table_values.append([expected(nested_endclasses[e], metric) for e in scens])
        table_rows.append('exp_'+metric)
    table = pd.DataFrame(table_values, columns=[*nested_endclasses], index=table_rows)
    return table

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

