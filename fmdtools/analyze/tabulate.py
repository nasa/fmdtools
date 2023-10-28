"""
Description: Translates simulation outputs to pandas tables for display, export, etc.

Uses methods:
- :meth:`fmea`: Make a simple fmea of the endclasses of a set of fault scenarios.
- :meth:`result_summary_fmea`: Make a table of endclass metrics, along with degraded
functions/flows.
- :meth:`metricovertime`: Make a table of the total metric, rate, and expected metric
of all faults over time.
- :meth:`result_summary:` Make a a table of a summary dictionary from a given model run.
- :meth:`nominal_stats`: Makes a table of quantities of interest from endclasses from a
nominal approach.
- :meth:`nested_stats`: Make a table of quantities of interest from endclasses from a
nested approach.
- :meth:`nominal_factor_comparison`: Compare a metric for a given set of model
parameters/factors over a set of nominal scenarios.
- :meth:`nested_factor_comparison`: Compare a metric for a given set of model
parameters/factors over a nested set of nominal and fault scenarios.
- :meth:`dicttab`: Make table of a generic dictionary.
- :meth:`maptab`: Make table of a generic map.
"""
# File Name: analyze/tabulate.py
# Author: Daniel Hulse
# Created: November 2019 (Refactored April 2020)

import pandas as pd
import numpy as np
from fmdtools.analyze.result import nan_to_x, Result, bootstrap_confidence_interval

# stable methods:


def result_summary_fmea(endresult, mdlhist, *attrs, metrics=()):
    """
    Make full fmea table with degraded attributes noted.

    Parameters
    ----------
    endresult : Result
        Result (over scenarios) to get metrics from
    mdlhist : History
        History (over scenarios) to get degradations/faults from
    *attrs : strs
        Model constructs to check if faulty/degraded.
    metrics : tuple, optional
        Metrics to include from endresult. The default is ().

    Returns
    -------
    pandas.DataFrame
        Table of metrics and degraded functions/flows over scenarios
    """
    from fmdtools.analyze.result import History
    deg_summaries = {}
    fault_summaries = {}
    mdlhist = mdlhist.nest(levels=1)
    for scen, hist in mdlhist.items():
        hist_comp = History(faulty=hist, nominal=mdlhist.nominal)
        hist_summary = hist_comp.get_fault_degradation_summary(*attrs)
        deg_summaries[scen] = str(hist_summary.degraded)
        fault_summaries[scen] = str(hist_summary.faulty)
    degradedtable = pd.DataFrame(deg_summaries, index=['degraded'])
    faulttable = pd.DataFrame(fault_summaries, index=['faulty'])
    simplefmea = endresult.create_simple_fmea(*metrics)
    fulltable = pd.concat([degradedtable, faulttable, simplefmea.transpose()])
    return fulltable.transpose()


def fmea(res, fs, metrics=[],
         weight_metrics=[], avg_metrics=[], perc_metrics=[], mult_metrics={},
         extra_classes={}, group_by=('function', 'fault'), sort_by=False, mdl={},
         mode_types={}, ascending=False, empty_as=0.0):
    """
    Make a user-definable fmea of the endclasses of a set of fault scenarios.

    Parameters
    ----------
    res : Result
        Result corresponding to the the simulation runs
    fs : sampleapproach/faultsample
        FaultSample used for the underlying probability model of the set of scenarios.
    metrics : list
        generic unweighted metrics to query. metrics are summed over grouped scenarios.
        The default is []. 'all' presents all metrics.
    weight_metrics: list
        weighted metrics to query. weight metrics are summed over groups.
        The default is ['rate'].
    avg_metrics: list
        metrics to average and query. The default is ['cost'].
        avg_metrics are averaged over groups, rather than a total.
    perc_metrics : list, optional
        metrics to treat as indicator variables to calculate a percentage.
        perc_metrics are treated as indicator variables and averaged over groups.
        The default is [].
    mult_metrics : dict, optional
        mult_metrics are new metrics calculated by multiplying existing metrics.
        (e.g., to calculate expectations or risk values like an expected cost or RPN)
        The default is {"expected cost":['rate', 'cost']}.
    extra_classes : dict, optional
        An additional set of endclasses to include in the table.
        The default is {}.
    group_by : tuple, optional
        Way of grouping fmea rows by scenario fields.
        The default is ('function', 'fault').
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
    grouped_scens = fs.get_scen_groups(*group_by)

    if type(metrics) == str:
        metrics = [metrics]
    if type(weight_metrics) == str:
        weight_metrics = [weight_metrics]
    if type(perc_metrics) == str:
        perc_metrics = [perc_metrics]
    if type(avg_metrics) == str:
        avg_metrics = [avg_metrics]

    if not metrics and not weight_metrics and not perc_metrics and not avg_metrics and not mult_metrics:
        # default fmea is a cost-based table
        weight_metrics = ["rate"]
        avg_metrics = ["cost"]
        mult_metrics = {"expected cost": ['rate', 'cost']}

    res.update(extra_classes)

    allmetrics = metrics+weight_metrics+avg_metrics+perc_metrics+[*mult_metrics.keys()]

    if not sort_by:
        if "expected cost" in mult_metrics:
            sort_by = "expected_cost"
        else:
            sort_by = allmetrics[-1]

    fmeadict = {g: dict.fromkeys(allmetrics) for g in grouped_scens}
    for group, ids in grouped_scens.items():
        sub_result = Result({scenid: res.get(scenid) for scenid in ids})
        for metric in metrics + weight_metrics:
            fmeadict[group][metric] = sum([res.get(scenid).get('endclass.'+metric)
                                           for scenid in ids])
        for metric in perc_metrics:
            fmeadict[group][metric] = sub_result.percent(metric)
        for metric in avg_metrics:
            fmeadict[group][metric] = sub_result.average(metric, empty_as=empty_as)
        for metric, to_mult in mult_metrics.items():
            fmeadict[group][metric] = sum([np.prod([res.get(scenid).get('endclass.'+m)
                                                    for m in to_mult])
                                           for scenid in ids])

    table = pd.DataFrame(fmeadict)
    table = table.transpose()
    if sort_by not in allmetrics:
        sort_by = allmetrics[0]
    table = table.sort_values(sort_by, ascending=ascending)
    return table


def result_summary(endresult, mdlhist, *attrs):
    """
    Make a table of results (degraded functions/flows, classification) of a single run.

    Parameters
    ----------
    endresult : Result
        Result with end-state classification
    mdlhist : History
        History of model states
    *attrs : str
        Names of attributes to check in the history for degradation/faulty.

    Returns
    -------
    table : pd.DataFrame
        Table with summary
    """
    hist_summary = mdlhist.get_fault_degradation_summary(*attrs)
    if 'endclass' in endresult:
        endresult = endresult['endclass']
    table = pd.DataFrame(endresult.data, index=[0])
    table['degraded'] = [hist_summary.degraded]
    table['faulty'] = [hist_summary.faulty]
    return table


def dicttab(dictionary):
    """Make table of a generic dictionary."""
    return pd.DataFrame(dictionary, index=[0])


def maptab(mapping):
    """Make table of a generic map."""
    table = pd.DataFrame(mapping)
    return table.transpose()


def factor_metrics(res, samp, metrics=['cost'], factors=["time"],
                   default_stat="expected", stats={}, ci_metrics=[], ci_kwargs={}):
    """
    Make a table of the statistic for given metrics over given factors.

    Parameters
    ----------
    res : Result
        Result with the given metrics over a number of scenarios.
    samp : BaseSample
        Sample object used to generate the scenarios
    metrics : list
        metrics in res to tabulate over time. Default is ['cost'].
    factors : list
        Factors (Scenario properties e.g., 'name', 'time', 'var') in samp to take the
        statistic over. Default is ['time']
    default_stat : str
        statistic to take for given metrics my default.
        (e.g., 'average', 'percent'... see Result methods). Default is 'expected'.
    stats : dict
        Non-default statistics to take for each individual metric.
        e.g. {'cost': 'average'}. Default is {}
    ci_metrics : list
        Metrics to calculate a confidence interval for (using bootstrap_ci).
        Default is [].
    ci_kwargs : dict
        kwargs to bootstrap_ci

    Returns
    -------
    met_table : dataframe
        pandas dataframe with the statistic of the metric over the corresponding
        set of scenarios for the given factor level.
    """
    scen_groups = samp.get_scen_groups(*factors)
    met_dict = {met: {} for met in metrics}
    met_dict.update({met+"_lb": {} for met in ci_metrics})
    met_dict.update({met+"_ub": {} for met in ci_metrics})

    for fact_tup, scens in scen_groups.items():
        sub_res = res.get_scens(*scens)
        for met in metrics+ci_metrics:
            if met in stats:
                stat = stats[met]
            else:
                stat = default_stat
            if met in ci_metrics:
                mv, lb, ub = sub_res.get_metric_ci(met, metric=stat, **ci_kwargs)
                met_dict[met][fact_tup] = mv
                met_dict[met+"_lb"][fact_tup] = lb
                met_dict[met+"_ub"][fact_tup] = ub
            else:
                met_dict[met][fact_tup] = sub_res.get_metric(met, metric=stat)

    return pd.DataFrame.from_dict(met_dict)
