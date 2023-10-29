"""
Description: Translates simulation outputs to pandas tables for display, export, etc.

Uses methods:
- :meth:`result_summary_fmea`: Make a table of endclass metrics, along with degraded
functions/flows.
- :meth:`result_summary:` Make a a table of a summary dictionary from a given model run.
- :meth:`nominal_stats`: Makes a table of quantities of interest from endclasses from a
nominal approach.

and classes:
- :class:`FMEA`: Class defining FMEA tables (with plotting/tabular export).
- :class:`Comparison`: Class defining metric comparison (with plot/tab export).
"""
# File Name: analyze/tabulate.py
# Author: Daniel Hulse
# Created: November 2019 (Refactored April 2020)

import pandas as pd
import numpy as np
from fmdtools.analyze.result import Result
from fmdtools.analyze.plot import multiplot_helper
from collections import UserDict

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


def result_summary(endresult, mdlhist, *attrs):
    """
    Make a pandas table of results (degraded functions/flows, etc.) of a single run.

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


class BaseTab(UserDict):
    """
    Base class for tables that extends Userdict.

    Userdict has structure {metric: {comp_group: value}} which enables plots/tables.
    """

    def sort_by_factor(self, factor, reverse=False):
        """
        Sort the table by the given factor.

        Parameters
        ----------
        factor : str/int
            Name or index of factor to sort by.
        reverse : bool, optional
            Whether to sort in descending order. The default is False.
        """
        metric = [*self.keys()][0]
        keys = [*self[metric].keys()]
        ex_key = keys[0]

        if hasattr(self, 'factors') and type(factor) == str:
            value = self.factors.index(factor)

        if len(ex_key) > 1:
            order = np.argsort(keys, axis=0)[value]
        else:
            order = np.argsort([k[0] for k in keys], axis=0)

        if reverse:
            order = order[::-1]
        ordered_keys = [keys[o] for o in order]
        for met in self.keys():
            self[met] = {k: self[met][k] for k in ordered_keys}

    def sort_by_metric(self, metric, reverse=False):
        """
        Sort the table by a given metric.

        Parameters
        ----------
        metric : str
            Name of metric to sort by.
        reverse : bool, optional
            Whether to sort in descending order. The default is False.
        """
        keys = [*self[metric].keys()]
        vals = [*self[metric].values()]
        order = np.argsort(vals)
        if reverse:
            order = order[::-1]
        ordered_keys = [keys[o] for o in order]
        for met in self.keys():
            self[met] = {k: self[met][k] for k in ordered_keys}

    def all_metrics(self):
        """Return metrics in Table."""
        return [*self.keys()]

    def as_table(self, sort_by=False, ascending=False):
        """
        Return pandas table of the Table.

        Parameters
        ----------
        sort_by : str, optional
            Column value to sort the table by. The default is False.
        ascending : bool, optional
            Whether to sort in ascending order. The default is False.

        Returns
        -------
        fmea_table : DataFrame
            pandas table with given metrics grouped as
        """
        if not sort_by:
            if "expected cost" in self.all_metrics():
                sort_by = "expected cost"
            else:
                sort_by = self.all_metrics()[-1]

        table = pd.DataFrame(self.data)
        if sort_by not in self.all_metrics():
            sort_by = self.all_metrics()[0]
        table = table.sort_values(sort_by, ascending=ascending)
        return table

    def as_plot(self, metric, title="", fig=False, ax=False, figsize=(6,4),
                xlab='', xlab_ang=-90, ylab='', **kwargs):
        """
        Return bar plot of a metric in the comparison.

        Parameters
        ----------
        metric : str
            Metric to plot.
        title : str, optional
            Title to use (if not default). The default is "".
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        figsize : tuple, optional
            Figsize (if fig not provided). The default is (6,4).
        xlab_ang : number
            Angle to tilt the xlabel at. The default is 90.
        xlab : str, optional
            label for x-axis. The default is ''.
        ylab : str, optional
            label for y-axis. The default is ''.
        **kwargs : kwargs
            Keyword arguments to ax.bar

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not ax:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
        met_dict = self[metric]
        factors = [str(k[0]) if len(k) == 1 else str(k) for k in met_dict.keys()]
        values = np.array([*met_dict.values()])
        if metric+"_lb" in self:
            lb_err = values - np.array([*self[metric+"_lb"].values()])
            ub_err = np.array([*self[metric+"_ub"].values()]) - values
            errs = [lb_err, ub_err]
        else:
            errs = 0.0
        ax.bar(factors, values, yerr=errs, **kwargs)
        if not xlab:
            if len(self.factors) == 1:
                ax.set_xlabel(self.factors[0])
            else:
                ax.set_xlabel(str(self.factors))
        ax.tick_params(axis='x', rotation=xlab_ang)
        if ylab:
            ax.set_ylab(ylab)
        if title:
            ax.set_title(title)
        return fig, ax

    def as_plots(self, *metrics, cols=1, figsize='default', titles={}, **kwargs):
        """
        Plot multiple metrics on multiple plots.

        Parameters
        ----------
        *metrics : str
            Metrics to plot.
        cols : int, optional
            Number of columns. The default is 2.
        figsize : str, optional
            Figure size. The default is 'default'.
        titles : dict, optional
            Individual plot titles. The default is {}.
        **kwargs : kwargs
            Keyword arguments to BaseTab.as_plot

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not metrics:
            metrics = self.all_metrics()
        fig, axs, cols, rows, subplot_titles = multiplot_helper(cols, *metrics,
                                                                figsize=figsize,
                                                                titles=titles)
        for i, metric in enumerate(metrics):
            fig, ax = self.as_plot(metric, title=subplot_titles[metric],
                                   ax=axs[i], fig=fig, **kwargs)
        return fig, axs

class FMEA(BaseTab):
    def __init__(self, res, fs, metrics=[], weight_metrics=[], avg_metrics=[],
                 perc_metrics=[], mult_metrics={}, extra_classes={},
                 group_by=('function', 'fault'), mdl={}, mode_types={},
                 empty_as=0.0):
        """
        Make a user-definable fmea of the endclasses of a set of fault scenarios.

        Parameters
        ----------
        res : Result
            Result corresponding to the the simulation runs
        fs : sampleapproach/faultsample
            FaultSample used for the underlying probability model of the set of scens.
        metrics : list
            generic unweighted metrics to query. metrics are summed over grouped scens.
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
            (e.g., to calculate expectations or risk values like an expected cost/RPN)
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
        empty_as : float/'nan'
            How to calculate stats of empty variables (for avg_metrics). Default is 0.0.
        """
        self.factors = group_by
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

        fmeadict = {m: dict.fromkeys(grouped_scens) for m in allmetrics}
        for group, ids in grouped_scens.items():
            sub_result = Result({scenid: res.get(scenid) for scenid in ids})
            for metric in metrics + weight_metrics:
                fmeadict[metric][group] = sum([res.get(scenid).get('endclass.'+metric)
                                               for scenid in ids])
            for metric in perc_metrics:
                fmeadict[metric][group] = sub_result.percent(metric)
            for metric in avg_metrics:
                fmeadict[metric][group] = sub_result.average(metric, empty_as=empty_as)
            for metric, to_mult in mult_metrics.items():
                fmeadict[metric][group] = sum([np.prod([res.get(scenid).get('endclass.'+m)
                                                        for m in to_mult])
                                               for scenid in ids])
        self.data = fmeadict


class Comparison(BaseTab):
    def __init__(self, res, samp, metrics=['cost'], factors=["time"],
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
        self.factors = factors
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
                    try:
                        mv, lb, ub = sub_res.get_metric_ci(met, metric=stat,
                                                           **ci_kwargs)
                    except TypeError as e:
                        raise Exception("Invalid method: " + str(stat) + ", " +
                                        "Can only use ci for metrics w- numpy method " +
                                        "provided for stat (not str).") from e
                    met_dict[met][fact_tup] = mv
                    met_dict[met+"_lb"][fact_tup] = lb
                    met_dict[met+"_ub"][fact_tup] = ub
                else:
                    met_dict[met][fact_tup] = sub_res.get_metric(met, metric=stat)
        self.data = met_dict
