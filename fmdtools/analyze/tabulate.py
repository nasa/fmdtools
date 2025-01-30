#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translates simulation outputs to pandas tables for display, export, etc.

Uses methods:

- :func:`result_summary_fmea`: Make a table of endclass metrics, along with degraded functions/flows.
- :func:`result_summary`: Make a a table of a summary dictionary from a given model run.

and classes:

- :class:`FMEA`: Class defining FMEA tables (with plotting/tabular export).
- :class:`Comparison`: Class defining metric comparison (with plot/tab export).
- :class:`NominalEnvelope`: Class defining performance envelope (with plot export).

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from fmdtools.define.base import is_numeric
from fmdtools.analyze.result import Result
from fmdtools.analyze.common import multiplot_helper, consolidate_legend
from fmdtools.analyze.common import set_empty_multiplots
from fmdtools.analyze.common import multiplot_legend_title, setup_plot

import pandas as pd
import numpy as np
from collections import UserDict
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


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

    Examples
    --------
    >>> from fmdtools.define.architecture.function import ExFxnArch
    >>> from fmdtools.sim.propagate import fault_sample
    >>> from fmdtools.sim.sample import exfs
    >>> mdl = ExFxnArch()
    >>> res, hist = fault_sample(mdl, exfs)
    >>> result_summary_fmea(res, hist, *mdl.fxns, *mdl.flows)
                                              degraded  ... expected_cost
    ex_fxn_no_charge_t1              ['ex_fxn', 'exf']  ...           0.0
    ex_fxn_no_charge_t2              ['ex_fxn', 'exf']  ...           0.0
    ex_fxn_short_t1                  ['ex_fxn', 'exf']  ...           0.0
    ex_fxn_short_t2                  ['ex_fxn', 'exf']  ...           0.0
    ex_fxn2_no_charge_t1  ['ex_fxn', 'ex_fxn2', 'exf']  ...           0.0
    ex_fxn2_no_charge_t2  ['ex_fxn', 'ex_fxn2', 'exf']  ...           0.0
    ex_fxn2_short_t1      ['ex_fxn', 'ex_fxn2', 'exf']  ...           0.0
    ex_fxn2_short_t2      ['ex_fxn', 'ex_fxn2', 'exf']  ...           0.0
    nominal                                         []  ...           1.0
    <BLANKLINE>
    [9 rows x 5 columns]
    """
    from fmdtools.analyze.history import History
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

    Examples
    --------
    >>> from fmdtools.define.architecture.function import ExFxnArch
    >>> from fmdtools.sim.propagate import one_fault
    >>> mdl = ExFxnArch()
    >>> res, hist = one_fault(mdl, "ex_fxn", "short", time=2)
    >>> result_summary(res, hist, *mdl.fxns, *mdl.flows)
       endclass.rate  endclass.cost  ...       degraded    faulty
    0        0.00001              1  ...  [ex_fxn, exf]  [ex_fxn]
    <BLANKLINE>
    [1 rows x 5 columns]
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

    Attributes
    ----------
    factors : list
        List of factors in the table
    """

    def sort_by_factors(self, *factors):
        """
        Sort the table by its factors.

        Parameters
        ----------
        *factor : str/int
            Name of factor(s) to sort by, in order of sorting.
            (non-included factors will be sorted last)
        """
        factors = list(factors)
        factors.reverse()
        other_factors = [f for f in self.factors if f not in factors]
        all_factors = other_factors + factors
        for factor in all_factors:
            self.sort_by_factor(factor)

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
        keys = [k for k in self[metric].keys()]
        ex_key = keys[0]

        if hasattr(self, 'factors') and isinstance(factor, str):
            value = self.factors.index(factor)

        order = np.argsort([k[value] for k in keys], axis=0, kind='stable')

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

    def as_table(self, sort_by=False, ascending=False, sort=True):
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
            if "expected_cost" in self.all_metrics():
                sort_by = "expected_cost"
            else:
                sort_by = self.all_metrics()[-1]

        table = pd.DataFrame(self.data)
        if sort_by not in self.all_metrics():
            sort_by = self.all_metrics()[0]
        if sort:
            table = table.sort_values(sort_by, ascending=ascending)
        return table

    def as_plot(self, metric, title="", fig=False, ax=False, figsize=(6, 4),
                xlab='', xlab_ang=-90, ylab='', color_factor='',
                pallette=[*mcolors.TABLEAU_COLORS.keys()], suppress_legend=False,
                suppress_ticklabels=False, **kwargs):
        """
        Return bar plot of a metric in the comparison.

        Parameters
        ----------
        metric : str
            Metric to plot.
        title : str, optional
            Title to use (if not default). The default is "".
        fig : figure
            Matplotlib figure object.
        ax : axis
            Corresponding matplotlib axis.
        figsize : tuple, optional
            Figsize (if fig not provided). The default is (6, 4).
        xlab : str, optional
            label for x-axis. The default is ''.
        xlab_ang : number
            Angle to tilt the xlabel at. The default is 90.
        ylab : str, optional
            label for y-axis. The default is ''.
        color_factor : ''
            Factor to label with a color (instead of the x-axis).
        pallette : list
            list of colors to . Defaults to matplotlib.colors.TABLEAU_COLORS.
        suppress_legend : bool
            Whether to suppress the generated legend (for multiplots).
        suppress_ticklabels : bool
            Whether to suppress tick labels.
        **kwargs : kwargs
            Keyword arguments to ax.bar.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        # add figure
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        # get values
        met_dict = self[metric]
        # sort into color vs tick bars
        all_factors = [*met_dict.keys()]
        if color_factor:
            if isinstance(color_factor, int):
                c_fact = color_factor
                color_factor = self.factors[c_fact]
            else:
                c_fact = self.factors.index(color_factor)
            color_factors = [k[c_fact] for k in all_factors]
            color_options = list(set(color_factors))
            colors = [pallette[color_options.index(c)] for c in color_factors]
            factors = [tuple([k for i, k in enumerate(k) if i != c_fact])
                       for k in all_factors]
        else:
            factors = all_factors
            color_factors = ['' for k in factors]
            colors = [pallette[0] for factor in factors]
        factors = [str(k[0]) if len(k) == 1 else str(k) for k in factors]
        x = [i for i, k in enumerate(factors)]
        values = np.array([*met_dict.values()])

        # degermine error bars
        if metric+"_lb" in self:
            lb_err = values - np.array([*self[metric+"_lb"].values()])
            ub_err = np.array([*self[metric+"_ub"].values()]) - values
            errs = [lb_err, ub_err]
        else:
            errs = 0.0

        # plot bars
        ax.bar(x, values, yerr=errs, color=colors, label=color_factors, **kwargs)

        # label axes
        if not xlab:
            non_color_factors = [f for f in self.factors if f != color_factor]
            if len(non_color_factors) == 1:
                ax.set_xlabel(non_color_factors[0])
            else:
                ax.set_xlabel(str(non_color_factors))

        if not suppress_ticklabels:
            ax.set_xticks(x)
            ax.set_xticklabels(factors)
        else:
            ax.set_xticks([])
        ax.tick_params(axis='x', rotation=xlab_ang)
        # legend, title, etc.
        if color_factor and not suppress_legend:
            consolidate_legend(ax, title=color_factor)
        if ylab:
            ax.set_ylab(ylab)
        if title:
            ax.set_title(title)
        return fig, ax

    def as_plots(self, *metrics, cols=1, figsize='default', titles={},
                 legend_loc=-1, title='', v_padding=None, h_padding=None,
                 title_padding=0.0, xlab='', **kwargs):
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
        legend_loc : str
            Plot to put the legend on. The default is -1 (the last plot).
        titles : str
            Overall title for the plots. the default is {}.
        v_padding : float
            Vertical padding between plots.
        h_padding : float
            Horizontal padding between plots.
        title_padding : float
            Padding for the overall title
        xlab : str
            Label for the x-axis. Default is '', which generates it automatically.
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
            if i >= (rows-1)*cols:
                xlabel = xlab
            else:
                xlabel = ' '
            fig, ax = self.as_plot(metric, title=subplot_titles[metric], xlab=xlabel,
                                   ax=axs[i], fig=fig, suppress_legend=True,
                                   **kwargs)

        set_empty_multiplots(axs, len(metrics), cols,
                             xlab_ang=kwargs.get('xlab_ang', -90))
        color_factor = kwargs.get('color_factor', '')
        if not color_factor:
            legend_loc = False

        multiplot_legend_title(metrics, axs, ax, title=title,
                               v_padding=v_padding, h_padding=h_padding,
                               title_padding=title_padding,
                               legend_loc=legend_loc,
                               legend_title=color_factor)
        return fig, axs


class FMEA(BaseTab):
    """
    Make a user-definable fmea of the endclasses of a set of fault scenarios.

    Parameters
    ----------
    res : Result
        Result corresponding to the the simulation runs
    fs : sampleapproach/faultsample
        FaultSample used for the underlying probability model of the set of scens.
    add_res : dict/Result, optional
        An additional set of metrics to include in the table. Should have similar
        key structure to res. The default is {}.
    group_by : tuple, optional
        Way of grouping fmea rows by scenario fields.
        The default is ('function', 'fault').
    prefix : str
        Prefix for the metrics to use for get_metric. Default is 'endclass.', which
        gets the metrics from endclass (output of find_classification method) only.
    rates/weights : str(s)
        Weighting or rate factor to use for weighted averages and expectations.
        Can be any value from the result (e,g. rates='rate') or the FaultSample
        (e.g., rates='scenario.rate').
    **kwargs: str/list
        Metrics to calculate, (e.g., rate_metric='rate', expected_metric='cost')
        Note that rate and rate metrics become inputs to average and expected.
        rates='scenario_'
        All other kwargs will be kwargs to Result.get_metric
        (e.g., rates or weights for get_expected and get_average)

    Examples
    --------
    >>> from fmdtools.sim.sample import exfs
    >>> res = Result({scen.name+'.endclass': {'rate': scen.time, 'cost': i} for i, scen in enumerate(exfs.scenarios())}).flatten()
    >>> FMEA(res, exfs).as_table(sort_by="sum_cost")
                       average_scenario_rate  sum_cost  expected_cost
    ex_fxn2 short                        0.0        13            0.0
            no_charge                    0.0         9            0.0
    ex_fxn  short                        0.0         5            0.0
            no_charge                    0.0         1            0.0
    >>> FMEA(res, exfs, average_metric=["rate"], sum_metric=["cost"], expected_metric=["cost"], rates="rate").as_table()
                       average_rate  sum_cost  expected_cost
    ex_fxn2 short               1.5        13             20
            no_charge           1.5         9             14
    ex_fxn  short               1.5         5              8
            no_charge           1.5         1              2
    >>> FMEA(res, exfs, sum_metric=["rate"], average_metric=["cost"]).as_table()
                       sum_rate  average_cost
    ex_fxn2 short             3           6.5
            no_charge         3           4.5
    ex_fxn  short             3           2.5
            no_charge         3           0.5
    """

    def __init__(self, res, fs, add_res={}, group_by=('function', 'fault'),
                 prefix="endclass.", **kwargs):
        self.factors = group_by
        grouped_scens = fs.get_scen_groups(*group_by)
        all_metrics = {k[:-7]: [v] if not isinstance(v, list) else v
                       for k, v in kwargs.items() if "_metric" in k}
        met_kwar = {k: v for k, v in kwargs.items() if "_metric" not in k}
        if not all_metrics:
            # default fmea is a cost-based table
            all_metrics = {'average': ['scenario_rate'],
                           'sum': ['cost'], "expected": ["cost"]}
            if not met_kwar:
                met_kwar = {'rates': 'scenario_rate'}
        for met, met_value in met_kwar.items():
            if isinstance(met_value, str) and met_value.startswith("scenario_"):
                met_kwar[met] = fs.get_scen_values(met_value[9:])

        res.update(add_res)

        fmeadict = {m+"_"+vi: dict.fromkeys(grouped_scens)
                    for m, v in all_metrics.items() for vi in v}
        for group, ids in grouped_scens.items():
            sub_result = Result({scenid: res.get(scenid) for scenid in ids})
            for method, values in all_metrics.items():
                for value in values:
                    met = method+"_"+value
                    if isinstance(value, str) and value.startswith("scenario_"):
                        sum_met = fs.get_metric(value[9:], ids=ids, method=method,
                                                **met_kwar)
                    else:
                        sum_met = sub_result.get_metric(value, method=method,
                                                        prefix=prefix, **met_kwar)
                    fmeadict[met][group] = sum_met
        self.data = fmeadict


class BaseComparison(BaseTab):
    """
    Base comparison class used for other comparisons.

    Parameters
    ----------
    res : Result
        Result with the given metrics over a number of scenarios.
    scen_groups : dict
        Grouped scenarios.
    metrics : list
        metrics in res to tabulate over time. Default is ['cost'].
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
    """

    def __init__(self, res, scen_groups, metrics=['cost'],
                 default_stat="expected", stats={}, ci_metrics=[], ci_kwargs={}):

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
                        mv, lb, ub = sub_res.get_metric_ci(met, method=stat,
                                                           **ci_kwargs)
                    except TypeError as e:
                        raise Exception("Invalid method: " + str(stat) + ", " +
                                        "Can only use ci for metrics w- numpy method " +
                                        "provided for stat (not str).") from e
                    met_dict[met][fact_tup] = mv
                    met_dict[met+"_lb"][fact_tup] = lb
                    met_dict[met+"_ub"][fact_tup] = ub
                else:
                    met_dict[met][fact_tup] = sub_res.get_metric(met, method=stat)
        self.data = met_dict


class Comparison(BaseComparison):
    """
    Make a table of the statistic for given metrics over given factors.

    Parameters
    ----------
    res : Result
        Result with the given metrics over a number of scenarios.
    samp : BaseSample
        Sample object used to generate the scenarios
    factors : list
        Factors (Scenario properties e.g., 'name', 'time', 'var') in samp to take
        statistic over. Default is ['time']
    **kwargs : kwargs
        keyword arguments to BaseComparison

    Returns
    -------
    met_table : dataframe
        pandas dataframe with the statistic of the metric over the corresponding
        set of scenarios for the given factor level.

    Examples
    --------
    >>> from fmdtools.sim.sample import exp_ps
    >>> from fmdtools.analyze.result import Result
    >>> res = Result({k.name: Result({'a': k.p['x']**2, "b": k.p['y']*k.p['x'], 'rate':k.rate}) for i, k in enumerate(exp_ps.scenarios())})
    >>> res = res.flatten()

    example 1: checking the x = x^2 accross variables

    >>> comp = Comparison(res, exp_ps, metrics=['a'], factors=['p.x'], default_stat='expected')
    >>> comp.sort_by_factors("p.x")
    >>> comp
    {'a': {(0,): 0.0, (1,): 1.0, (2,): 4.0, (3,): 9.0, (4,): 16.0, (5,): 25.0, (6,): 36.0, (7,): 49.0, (8,): 64.0, (9,): 81.0, (10,): 100.0}}
    >>> comp.as_table()
            a
    10  100.0
    9    81.0
    8    64.0
    7    49.0
    6    36.0
    5    25.0
    4    16.0
    3     9.0
    2     4.0
    1     1.0
    0     0.0
    >>> fig, ax = comp.as_plot("a")

    example 2: viewing interaction between x and y:

    >>> comp = Comparison(res, exp_ps, metrics=['b'], factors=['p.x', 'p.y'], default_stat='expected')
    >>> comp.sort_by_factors("p.x", "p.y")
    >>> comp.as_table(sort=False)
               b
    0  1.0   0.0
       2.0   0.0
       3.0   0.0
       4.0   0.0
    1  1.0   1.0
       2.0   2.0
       3.0   3.0
       4.0   4.0
    2  1.0   2.0
       2.0   4.0
       3.0   6.0
       4.0   8.0
    3  1.0   3.0
       2.0   6.0
       3.0   9.0
       4.0  12.0
    4  1.0   4.0
       2.0   8.0
       3.0  12.0
       4.0  16.0
    5  1.0   5.0
       2.0  10.0
       3.0  15.0
       4.0  20.0
    6  1.0   6.0
       2.0  12.0
       3.0  18.0
       4.0  24.0
    7  1.0   7.0
       2.0  14.0
       3.0  21.0
       4.0  28.0
    8  1.0   8.0
       2.0  16.0
       3.0  24.0
       4.0  32.0
    9  1.0   9.0
       2.0  18.0
       3.0  27.0
       4.0  36.0
    10 1.0  10.0
       2.0  20.0
       3.0  30.0
       4.0  40.0
    >>> fig, ax = comp.as_plot("b", color_factor="p.y", figsize=(10, 4))
    """

    def __init__(self, res, samp, factors=['time'], **kwargs):
        self.factors = factors
        scen_groups = samp.get_scen_groups(*factors)
        super().__init__(res, scen_groups, **kwargs)


class NestedComparison(BaseComparison):
    """
    Make a nested table of the statistic for samples taken in other samples.

    Parameters
    ----------
    res : Result
        Result with the given metrics over a number of scenarios.
    samp : BaseSample
        Sample object used to generate the scenarios
    samp_factors : list
        Factors (Scenario properties e.g., 'name', 'time', 'var') in samp to take
        statistic over. Default is ['time']
    samps : dict
        Sample objects used to generate the scenarios. {'name': samp}
    samps_factors : list
        Factors (Scenario properties e.g., 'name', 'time', 'var') in samp to take
        statistic over in the apps. Default is ['time']
    **kwargs : kwargs
        keyword arguments to BaseComparison
    """

    def __init__(self, res, samp, samp_factors, samps, samps_factors, **kwargs):
        overall_scen_groups = {}
        scen_groups = samp.get_scen_groups(*samp_factors)
        for n_samp in samps.values():
            n_scen_groups = n_samp.get_scen_groups(*samps_factors)
            for scen_group, scens in scen_groups.items():
                for n_scen_group, n_scens in n_scen_groups.items():
                    k = tuple(list(scen_group)+list(n_scen_group))
                    v = [s+"."+ns for s in scens for ns in n_scens]
                    overall_scen_groups[k] = v

        self.factors = samp_factors + samps_factors
        super().__init__(res, overall_scen_groups, **kwargs)


class NominalEnvelope(object):
    """
    Class defining nominal performance envelope.

    Attributes
    ----------
    params : tuple
        Parameters explored in the envelope.
    variable_groups : dict
        Variable groups and their corresponding scenarios.
    group_values : dict
        Nominal/Faulty values for the scenarios/groups in variable groups.

    Parameters
    ----------
    ps : ParameterSample
        ParameterSample sample approach simulated in the model.
    res : Result
        Result dict for the set of simulations produced by running the model over ps
    metric : str
        Value to get from endclasses for the scenario(s). The default is 'cost'.
    x_param : str
        Parameter range desired to visualize in the operational envelope. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparams.x_in`, `p.x`...)
    func : method, optional
        Function to classify metric values as "nominal".
        Default is lambda x: x == 0.0
    """

    def __init__(self, ps, res, metric, *params, func=lambda x: x == 0.0):
        """
        Make an object showing the nominal envelope of operations.

        Parameters
        ----------
        ps : ParameterSample
            ParameterSample sample approach simulated in the model.
        res : Result
            Result dict for the set of simulations produced by running the model over ps
        metric : str
            Value to get from endclasses for the scenario(s). The default is 'cost'.
        x_param : str
            Parameter range desired to visualize in the operational envelope. Can be any
            property that changes over the nomapp
            (e.g., `r.seed`, `inputparams.x_in`, `p.x`...)
        func : method, optional
            Function to classify metric values as "nominal".
            Default is lambda x: x == 0.0
        """
        self.params = params
        self.variable_groups = ps.get_scen_groups(*params)
        if not self.variable_groups:
            raise Exception("No matching scenarios--are parameters " +
                            params + " in the nomapp Scenarios?")
        gv = {group:
              [func(v) for v in res.get_scens(*scens).get_values("."+metric).values()]
              for group, scens in self.variable_groups.items()}
        self.group_values = gv

    def as_plot(self, **kwargs):
        """
        Plot nominal envelope. Overall function that calls plot_event/plot_scatter.

        Parameters
        ----------
        **kwargs : kwargs
            kwargs to plot_event/plot_scatter

        Returns
        -------
        fig : mpl figure
            Figure with scatter plot
        ax :mpl, axis
            Axis with scatter plot
        """
        if len(self.params) == 1:
            return self.plot_event(**kwargs)
        elif len(self.params) in [2, 3]:
            return self.plot_scatter(**kwargs)
        else:
            raise Exception("Must have 1, 2, or 3 params to plot.")

    def plot_event(self, n_kwargs={}, f_kwargs={}, figsize=(6, 4), legend_loc='best',
                   xlabel='', title=''):
        """
        Make an eventplot of the Nominal Envelope (for 1D).

        Parameters
        ----------
        n_kwargs : dict, optional
            Nominal kwargs to ax.scatter. The default is {}.
        f_kwargs : dict, optional
            Faulty kwargs to ax.scatter. The default is {}.
        figsize : tuple, optional
            Figure size. The default is (6, 4).
        legend_loc : str, optional
            Location for the legend. The default is 'best'.
        xlabel : str, optional
            label for x-axis (defaults to parameter name for x_param)
        title : str, optional
            title for the figure. The default is ''.

        Returns
        -------
        fig : mpl figure
            Figure with scatter plot
        ax :mpl, axis
            Axis with scatter plot
        """
        n_kwargs = {**dict(label='nominal', alpha=0.5, color='blue'), **n_kwargs}
        f_kwargs = {**dict(label='faulty', alpha=0.5, color='red'), **f_kwargs}

        fig, ax = setup_plot(figsize=figsize)

        data_values = [k[0] for k in self.variable_groups.keys()]
        if is_numeric(data_values):
            min_x = np.min(data_values)
            max_x = np.max(data_values)
            ax.hlines(1, min_x-1, max_x+1)
            ax.set_xlim(min_x-1, max_x+1)

        for var, vals in self.group_values.items():
            for val in vals:
                if val:
                    ax.eventplot(var, **n_kwargs)
                else:
                    ax.eventplot(var, **f_kwargs)
        consolidate_legend(ax, loc=legend_loc)
        ax.yaxis.set_ticklabels([])
        if not xlabel:
            xlabel = self.params[0]
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(which='both', axis='x')
        return fig, ax

    def plot_scatter(self, n_kwargs={}, f_kwargs={}, figsize=(6, 4), legend_loc='best',
                     xlabel='', ylabel='', zlabel='', title=''):
        """
        Make a scatter plot of the Nominal Envelope (for 2D/3D).

        Parameters
        ----------
        n_kwargs : dict, optional
            Nominal kwargs to ax.scatter. The default is {}.
        f_kwargs : dict, optional
            Faulty kwargs to ax.scatter. The default is {}.
        figsize : tuple, optional
            Figure size. The default is (6, 4).
        legend_loc : str, optional
            Location for the legend. The default is 'best'.
        xlabel : str, optional
            label for x-axis (defaults to parameter name for x_param)
        ylabel : str, optional
            label for y-axis (defaults to parameter name for y_param)
        zlabel : str, optional
            label for z-axis (defaults to parameter name for z_param)
        title : str, optional
            title for the figure. The default is ''.

        Returns
        -------
        fig : mpl figure
            Figure with scatter plot
        ax :mpl, axis
            Axis with scatter plot
        """
        default_n_kwargs = dict(label='nominal', alpha=0.5, color='blue', marker='o')
        default_f_kwargs = dict(label='faulty', alpha=0.5, color='red', marker='x')
        n_kwargs = {**default_n_kwargs, **n_kwargs}
        f_kwargs = {**default_f_kwargs, **f_kwargs}

        if len(self.params) == 3:
            z = 0
        else:
            z = False
        fig, ax = setup_plot(figsize=figsize, z=z)

        for var, vals in self.group_values.items():
            for val in vals:
                x = [[i] for i in var]
                if val:
                    ax.scatter(*x, **n_kwargs)
                else:
                    ax.scatter(*x, **f_kwargs)

        consolidate_legend(ax, loc=legend_loc)
        if not xlabel:
            xlabel = self.params[0]
        if not ylabel:
            ylabel = self.params[1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(self.params) == 3:
            if not zlabel:
                zlabel = self.params[2]
            ax.set_zlabel(zlabel)
        ax.set_title(title)
        return fig, ax


if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=True)
