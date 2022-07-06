"""
Description: Translates simulation outputs to pandas tables for display, export, etc.

Uses methods:
    - :meth:`hist`:           Returns formatted pandas dataframe of model history
    - :meth:`objtab`:         Make table of function OR flow value attributes - objtype = 'function' or 'flow'
    - :meth:`stats`:          Makes a table of #of degraded flows, # of degraded functions, and # of total faults over time given a single result history
    - :meth:`degflows`:       Makes a  of flows over time, where 0 is degraded and 1 is nominal
    - :meth:`degflowvals`:    Makes a table of individual flow state values over time, where 0 is degraded and 1 is nominal
    - :meth:`degfxns`:        Makes a table showing which functions are degraded over time (0 for degraded, 1 for nominal)
    - :meth:`deghist`:        Makes a table of all funcitons and flows that are degraded over time. If withstats=True, the total # of each type degraded is provided in the last columns
    - :meth:`heatmaps`:       Makes a table of a heatmap dictionary
    - :meth:`metricovertime`: Makes a table of the total metric, rate, and expected metric of all faults over time
    - :meth:`samptime`:       Makes a table of the times sampled for each phase given a dict (i.e. app.sampletimes)
    - :meth:`summary:`        Makes a table of a summary dictionary from a given model run
    - :meth:`result`:         Makes a table of results (degraded functions/flows, cost, rate, expected cost) of a single run
    - :meth:`dicttab`:           Makes table of a generic dictionary
    - :meth:`maptab`:            Makes table of a generic map
    - :meth:`nominal_stats`:  Makes a table of quantities of interest from endclasses from a nominal approach.
    - :meth:`nested_stats`:   Makes a table of quantities of interest from endclasses from a nested approach.
    - :meth:`nominal_factor_comparison`: Compares a metric for a given set of model parameters/factors over a set of nominal scenarios.
    - :meth:`nested_factor_comparison`: Compares a metric for a given set of model parameters/factors over a nested set of nominal and fault scenarios.
Also used for FMEA-like tables:
    - :meth:`simplefmea`:          Makes a simple fmea (rate, cost, expected cost) of the endclasses of a list of fault scenarios run
    - :meth:`phasefmea`:           Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by phase.
    - :meth:`summfmea`:            Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by fault.
    - :meth:`fullfmea`:            Makes full fmea table (degraded functions/flows, cost, rate, expected cost) of scenarios given endclasses dict (cost, rate, expected cost) and summaries dict (degraded functions, degraded flows)
"""
#File Name: resultdisp/tabulate.py
#Author: Daniel Hulse
#Created: November 2019 (Refactored April 2020)

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
                if type(val)==dict:
                    for subatt, subval in val.items():
                        if subatt!= 'faults':
                            label=(fxn, att+'_'+subatt)
                            labels=labels+[label]
                            df[label]=subval
                        else:
                            label_faults(hist[objtype][fxn][att].get('faults', {}), df, fxn+'_'+subatt, labels)
                else:
                    label=(fxn, att)
                    labels=labels+[label]
                    df[label]=val
        if objtype =='functions':
            label_faults(hist[objtype][fxn].get('faults', {}), df, fxn, labels)
    index = pd.MultiIndex.from_tuples(labels)
    df = df.reindex(index, axis="columns")
    return df
def label_faults(faulthist, df, fxnlab, labels):
    if type(faulthist)==dict:
        for fault in faulthist:
            label=(fxnlab, fault+' fault')
            labels+=[label]
            df[label]=faulthist[fault]
    elif len(faulthist)==1:
        label=(fxnlab, 'faults')
        labels+=[label]
        df[label]=faulthist

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
def metricovertime(endclasses, app, metric='cost'):
    """
    Makes a table of the total metric, rate, and expected metric of all faults over time

    Parameters
    ----------
    endclasses : dict
        dict with rate, metric, and expected metric values for each injected scenario
    app : sampleapproach
        sample approach used to generate the list of scenarios
    metric : str
        metric from dict to tabulate over time. Default is 'cost'
    Returns
    -------
    met_overtime : dataframe
        pandas dataframe with the total metric, rate, and expected metric for the set of scenarios
    """
    expected_metric = "expected "+metric
    met_overtime={metric:{time:0.0 for time in app.times}, 'rate':{time:0.0 for time in app.times}, expected_metric:{time:0.0 for time in app.times}}
    for scen in app.scenlist:
        met_overtime[metric][scen['properties']['time']]+=endclasses[scen['properties']['name']][metric]
        met_overtime['rate'][scen['properties']['time']]+=endclasses[scen['properties']['name']]['rate']
        met_overtime[expected_metric][scen['properties']['time']]+=endclasses[scen['properties']['name']][expected_metric] 
    return pd.DataFrame.from_dict(met_overtime)
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
    """Makes a table of results (degraded functions/flows, classification) of a single run"""
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

def nominal_factor_comparison(nomapp, endclasses, params, metrics='all', rangeid='default', nan_as=np.nan, percent=True,  give_ci=False, **kwargs):
    """
    Compares a metric for a given set of model parameters/factors over set of nominal scenarios.

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal Approach used to generate the simulations
    endclasses : dict
        dict of endclasses from propagate.nominal_approach or nested_approach with structure: 
            {scen_x:{metric1:x, metric2:x...}} or {scen_x:{fault:{metric1:x, metric2:x...}}} 
    params : list/str
        List of parameters (or parameter) to use for the factor levels in the comparison
    metrics : 'all'/list, optional
        Metrics to show in the table. The default is 'all'.
    rangeid : str, optional
        Nominal Approach range to use for the test, if run over a single range.
        The default is 'default', which either:
            - picks the only range (if there is only one), or
            - compares between ranges (if more than one)
    nan_as : float, optional
        Number to parse NaNs as (if present). The default is np.nan.
    percent : bool, optional
        Whether to compare metrics as bools (True - results in a comparison of percentages of indicator variables) 
        or as averages (False - results in a comparison of average values of real valued variables). The default is True.
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
        if len(nomapp.ranges.keys())==1: 
            rangeid=[*nomapp.ranges.keys()][0]
            factors = nomapp.get_param_scens(rangeid, *params)
        else:
            factors = {rangeid:nomapp.ranges[rangeid]['scenarios'] for rangeid in nomapp.ranges}
    else: factors = nomapp.get_param_scens(rangeid, *params)
    if [*endclasses.values()][0].get('nominal', False): endclasses ={scen:ec['nominal'] for scen, ec in endclasses.items()}
    if metrics=='all':              metrics = [ec for ec,val in [*endclasses.values()][0].items() if type(val) in [float, int]]
    
    if type(params)==str: params=[params]
    full_stats=[]
    for metric in metrics:
        factor_stats = []
        for factor, scens in factors.items():
            endclass_fact = {scen:endclass for scen, endclass in endclasses.items() if scen in scens}

            if not percent: nominal_metrics = [nan_to_x(scen[metric], nan_as) for scen in endclass_fact.values()]
            else:           nominal_metrics = [np.sign(float(nan_to_x(scen[metric], nan_as))) for scen in endclass_fact.values()]
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
    params : list/str
        List of parameters (or parameter) to use for the factor levels in the comparison
    value : string
        metric of the endclass (returned by mdl.find_classification) to use for the comparison.
    faults : str/list, optional
        Set of faults to run the comparison over
            --'modes' (all fault modes),
            --'functions' (modes for each function are grouped)
            --'mode type' (modes with the same name are grouped)
            -- or a set of specific modes/functions. The default is 'functions'.
            -- or a tuple of form (group_by, apps, *arg), where
                - group_by is an argument to SampleApproach.get_scenid_groups
                - apps is a dictionary of approaches corresponding to the endclasses (from prop.nested_approach)
                - arg is:
                    - when using 'fxnclassfault' and 'fxnclass' options: a model
                    - when using 'modetype' options: a dictionary grouping modes by type
    rangeid : str, optional
        Nominal Approach range to use for the test, if run over a single range.
        The default is 'default', which either:
            - picks the only range (if there is only one), or
            - compares between ranges (if more than one)
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
        if len(nomapp.ranges.keys())==1: 
            rangeid=[*nomapp.ranges.keys()][0]
            factors = nomapp.get_param_scens(rangeid, *params)
        else:
            factors = {rangeid:nomapp.ranges[rangeid]['scenarios'] for rangeid in nomapp.ranges}
    else: factors = nomapp.get_param_scens(rangeid, *params)
    if faults=='functions':     faultlist = set([e.partition(' ')[0] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif faults=='modes':       faultlist = set([e.partition(',')[0] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif faults=='mode type':   faultlist = set([e.partition(',')[0].partition(' ')[2] for scen in nested_endclasses for e in nested_endclasses[scen]])
    elif type(faults) ==str: raise Exception("Invalid faults option: "+faults)
    elif type(faults)==list:    faultlist =set(faults)
    elif type(faults)==tuple:   
        group_by=faults[0]; apps=faults[1]; group_dict={}
        if group_by in ['fxnclassfault','fxnclass']: 
            mdl=faults[2]
            group_dict = {cl:mdl.fxns_of_class(cl) for cl in mdl.fxnclasses()}
        elif group_by=='modetype':  group_dict=faults[2]
        fault_scen_groups = {factor:{scen:apps[scen].get_scenid_groups(group_by, group_dict) for scen in scens} for factor, scens in factors.items()}
        faultlist = {fsname:set() for dicts in fault_scen_groups.values() for group in dicts.values() for fsname in group}
    else:                       faultlist=faults
    if type(faults)==tuple: faultlist.pop('nominal', 'nothing')
    else:                   faultlist.discard('nominal'); faultlist.discard(' '); faultlist.discard('')
    if type(params)==str: params=[params]
    full_stats=[]
    for factor, scens in factors.items():
        endclass_fact = {scen:endclass for scen, endclass in nested_endclasses.items() if scen in scens}
        ec_metrics = overall_diff(endclass_fact, value, nan_as=nan_as, as_ind=percent, no_diff=not difference)

        if not percent: nominal_metrics = [nan_to_x(res_scens['nominal'][value], nan_as) for res_scens in endclass_fact.values()]
        else:           nominal_metrics = [np.sign(float(nan_to_x(nan_to_x(res_scens['nominal'][value]), nan_as))) for res_scens in endclass_fact.values()]
        factor_stats=[sum(nominal_metrics)/len(nominal_metrics)]
        if give_ci: 
            factor_boot, factor_lb, factor_ub = bootstrap_confidence_interval(nominal_metrics, **kwargs)
            factor_stats = factor_stats + [factor_lb, factor_ub]
        if type(faults)==tuple:
            faultlist = {f:set() for f in faultlist}
            for scen, groups in fault_scen_groups[factor].items():
                for group, faultscens in groups.items():
                    if not faultlist.get(group, False) and faultscens:  faultlist[group]=set(faultscens)
                    else:                                               faultlist[group].update(faultscens)
                faultlist.pop('nominal', 'nothing')
        for fault in faultlist:
            if type(faults)==tuple:     fault_metrics=[metric for res_scens in ec_metrics.values() for res_scen,metric in res_scens.items() if res_scen in faultlist[fault]]
            elif faults=='functions':     fault_metrics = [metric for res_scens in ec_metrics.values() for res_scen,metric in res_scens.items() if fault in res_scen.partition(' ')[0]]
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
def simplefmea(endclasses, metrics=["rate", "cost", "expected cost"]):
    """Makes a simple fmea (rate, classification) of the endclasses of a list of fault scenarios run"""
    table = pd.DataFrame(endclasses)
    table = table.transpose()
    if metrics=='all':          return table
    elif type(metrics)==list:   return table.loc[:, metrics]
    else: 
        raise Exception("invalid metrics option: "+str(metrics))
    return 
def fullfmea(endclasses, summaries):
    """Makes full fmea table (degraded functions/flows, all metrics in endclasses) of scenarios given endclasses dict and summaries dict (degraded functions, degraded flows)"""
    degradedtable = pd.DataFrame(summaries)
    simplefmea=pd.DataFrame(endclasses)
    fulltable = pd.concat([degradedtable, simplefmea])
    return fulltable.transpose()

def fmea(endclasses, app, metrics=[], weight_metrics=[], avg_metrics = [], perc_metrics=[],
         mult_metrics={}, extra_classes={}, group_by='none', sort_by=False, mdl={}, mode_types={}, ascending=False, empty_as=0.0):
    """
    Makes a user-definable fmea of the endclasses of a set of fault scenarios.

    Parameters
    ----------
    endclasses : dict
        dict of endclasses of the simulation runs
    app : sampleapproach
        sample approach used for the underlying probability model of the set of scenarios run
    metrics : list
        generic unweighted metrics to query. The default is []. 'all' presents all metrics.
        metrics are summed over grouped scenarios.
    weight_metrics: list
        weighted metrics to query. The default is ['rate']. 
        metrics are weighted according to the number in each phase and then averaged
    avg_metrics: list
        metrics to average and query. The default is ['cost']. 
        avg_metrics are averaged over groups, rather than a total.
    perc_metrics : list, optional
        metrics to treat as indicator variables to calculate a percentage. The default is [].
        perc_metrics are treated as indicator variables and averaged over groups.
    mult_metrics : dict, optional
        mult_metrics are new metrics calculated by multiplying existing metrics 
        (e.g., to calculate expectations or risk values like an expected cost or RPN)
        The default is {"expected cost":['rate', 'cost']}.
    extra_classes : dict, optional
        An additional set of endclasses to include in the table (e.g., summaries from process.hists). 
        The default is {}.
    group_by : str, optional
        Way of grouping fmea rows. The default is 'none'.
        - 'none':           All scenarios are displayed individually
        - 'phase':          All identical scenarios (fxn, mode) within a given phase are grouped 
        - 'fxnfault':       All identical scenarios (fxn, mode) are grouped
        - 'mode':           All scenarios with the same mode name are grouped
        - 'modetype':      All scenarios with the same mode type, where mode types are strings in the mode name. Mode types must be given.
        - 'functions':      All scenarios and modes from a given function are grouped.
        - 'times':          All scenarios and modes at a given time are grouped
        - 'fxnclassfault':  All scenarios (fxnclass, mode) from a given function class are grouped. A Model must be provided.
        - 'fxnclass':       All scenarios from a given function class are grouped. A Model must be provided.
    mode_types : set
        Mode types to group by in 'mode type' option
    mdl : Model
        Model for use in 'fxnclassfault' and 'fxnclass' options
    sort_by : str, optional
        Column value to sort the table by. The default is "expected cost".
    ascending : bool, optional
        Whether to sort in ascending order. The default is False.
    empty_as : float/'nan'
        How to calculate stats of empty variables (for avg_metrics). Default is 0.0.

    Returns
    -------
    fmea_table : DataFrame
        pandas table with given metrics grouped as 
    """
    group_dict={}
    if group_by in ['fxnclassfault','fxnclass']: 
        if not mdl: raise Exception("No model mdl provided.")
        group_dict = {cl:mdl.fxns_of_class(cl) for cl in mdl.fxnclasses()}
    elif group_by=='modetype':
        group_dict=mode_types
    grouped_scens = app.get_scenid_groups(group_by, group_dict)
    
    if type(metrics)==str:          metrics=[metrics]
    if type(weight_metrics)==str:   weight_metrics=[weight_metrics]
    if type(perc_metrics)==str:     perc_metrics=[perc_metrics]
    if type(avg_metrics)==str:      avg_metrics=[avg_metrics]
    
    if not metrics and not weight_metrics and not perc_metrics and not avg_metrics and not mult_metrics:
        #default fmea is a cost-based table
        weight_metrics=["rate"]; avg_metrics = ["cost"] 
        mult_metrics={"expected cost":['rate', 'cost']}
    
    endclasses.update(extra_classes)
    
    id_weights = app.get_id_weights()
    id_weights['nominal']=1.0
    
    allmetrics = metrics+weight_metrics+avg_metrics+perc_metrics+[*mult_metrics.keys()]
    
    if group_by=='modetype':
        a=1
    
    if not sort_by:
        if "expected cost" in mult_metrics: sort_by="expected_cost"
        else:                               sort_by=allmetrics[-1]
    
    fmeadict = {g:dict.fromkeys(allmetrics) for g in grouped_scens}
    for group, ids in grouped_scens.items():
        b=1
        for metric in metrics:
            fmeadict[group][metric] = sum([endclasses[scenid][metric] for scenid in ids])
        for metric in weight_metrics:
            fmeadict[group][metric] = sum([endclasses[scenid][metric]*id_weights[scenid] for scenid in ids])
        for metric in perc_metrics:
            fmeadict[group][metric] = percent({scenid:endclasses[scenid] for scenid in ids}, metric)
        for metric in avg_metrics:    
            fmeadict[group][metric] = average({scenid:endclasses[scenid] for scenid in ids}, metric, empty_as=empty_as)
        for metric, to_mult in mult_metrics.items():
            if set(to_mult).intersection(weight_metrics):
                fmeadict[group][metric] = sum([np.prod([endclasses[scenid][m] for m in to_mult])*id_weights[scenid] for scenid in ids])
            else:
                fmeadict[group][metric] = sum([np.prod([endclasses[scenid][m] for m in to_mult]) for scenid in ids])
    table=pd.DataFrame(fmeadict)
    table=table.transpose() 
    if sort_by not in allmetrics: sort_by = allmetrics[0]
    table=table.sort_values(sort_by, ascending=ascending)
    return table
        
    
def phasefmea(endclasses, app, metrics=["rate", "expected cost"], weight_metrics = ["cost"], sort_by=None, ascending=False):
    """
    (LEGACY FUNCTION) Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by phase.
    Use tabulate.fmea with option group_by='phase' instead.

    Parameters
    ----------
    endclasses : dict
        dict of endclasses of the simulation runs
    app : sampleapproach
        sample approach used for the underlying probability model of the set of scenarios run
    metrics : list
        unweighted metrics to query. The default is ['rate', 'expected cost']
    weight_metrics: list
        weighted metrics to query. The default is ['cost']. 
        Weights are used to calculate an average, rather than a total.
    sort_by : str
        metric to stort the table by. default is 'expected cost'
    ascending : bool
        whether to sort ascending. Default is False.
    Returns
    -------
    tab: dataframe
        table with metrics of each fault in each phase
    """
    tab =fmea(endclasses, app, group_by='phase', metrics=metrics, weight_metrics = weight_metrics, sort_by=sort_by, ascending=ascending)
    return tab
    
def summfmea(endclasses, app, metrics=["rate", "expected cost"], weight_metrics = ["cost"], sort_by=None, ascending=False):
    """
    (LEGACY FUNCTION) Makes a simple fmea of the endclasses of a set of fault scenarios run grouped by fault.
    Use tabulate.fmea with group_by='fxnfault' instead.
    
    Parameters
    ----------
    endclasses : dict
        dict of endclasses of the simulation runs
    app : sampleapproach
        sample approach used for the underlying probability model of the set of scenarios run
    metrics : list
        unweighted metrics to query. The default is ['rate', 'expected cost']
    weighted_metrics: list
        weighted metrics to query. The default is ['cost']. 
        Weights are used to calculate an average, rather than a total.
    sort_by : str
        metric to stort the table by. default is 'expected cost'
    ascending : bool
        whether to sort ascending. Default is False
    Returns
    -------
    tab: dataframe
        table with metrics of each fault (over all phases)
    """
    tab = fmea(endclasses, app, group_by='fxnfault', metrics=metrics, weight_metrics=weight_metrics, sort_by=sort_by, ascending=ascending)
    return tab

