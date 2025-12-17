#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some common methods for analysis used by other modules.

Has methods:

- :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap
- :func:`diff`: Helper function for finding inconsistent states between val1, val2, with
  the difftype option
- :func:`join_key`: Helper function for Result Class
- :func:`setup_plot`: initializes mpl figure
- :func:`plot_err_hist`: Plots a line with a given range of uncertainty around it
- :func:`plot_err_lines`: Plots error lines on the given plot
- :func:`multiplot_legend_title`: Helper function for multiplot legends and titles
- :func:`consolidate_legend`: Creates a single legend for a given multiplot where
  multiple groups are being compared
- :func:`load_folder`: Lists files to load in folder.
- :func:`file_check`: Check if files exists and whether to overwrite the file
- :func:`auto_filetype`: Helper function that automatically determines the filetype
  (npz, csv, or json) of a given filename
- :func:`create_indiv_filename`: Helper function that creates an individualized name for
  a file given the general filename and an individual id

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
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import inspect
from scipy.stats import bootstrap
from fmdtools.define.base import filter_kwargs


plt.rcParams['pdf.fonttype'] = 42


def get_sub_include(att, to_include):
    """Determine attributes of att to include based on provided dict/str/list/set."""
    if type(to_include) in [list, set, tuple, str]:
        if att in to_include:
            new_to_include = 'default'
        elif isinstance(to_include, str) and to_include == 'all':
            new_to_include = 'all'
        elif isinstance(to_include, str) and to_include == 'default':
            new_to_include = 'default'
        else:
            new_to_include = False
    elif isinstance(to_include, dict) and att in to_include:
        new_to_include = to_include[att]
    else:
        new_to_include = False
    return new_to_include


def to_include_keys(to_include):
    """
    Determine dict keys to include from Result given nested to_include dictionary.

    Examples
    --------
    >>> to_include_keys({"a":{"b": "c"}})
    ('a.b.c',)
    >>> ks = to_include_keys({"a":{"b": {"c", "d", "e"}}})
    >>> all([k in ks for k in ('a.b.c', 'a.b.e', 'a.b.d')])
    True
    >>> to_include_keys("hi")
    ('hi',)
    >>> to_include_keys(["a", "b", "c"])
    ('a', 'b', 'c')
    """
    if isinstance(to_include, str):
        return tuple([to_include])
    elif type(to_include) in [list, set, tuple]:
        return tuple([to_i for to_i in to_include])
    elif isinstance(to_include, dict):
        keys = []
        for k, v in to_include.items():
            add = to_include_keys(v)
            keys.extend([k+'.'+v for v in add])
        return tuple(keys)


def diff(val1, val2, difftype='bool'):
    """
    Find inconsistent states between val1, val2.

    The difftype option ('diff' (takes the difference), 'bool' (checks if the same),
                         and float (checks if under the provided tolerance))

    Examples
    --------
    >>> diff([1, 2, 3], [2, 2, 3])
    array([ True, False, False])
    >>> diff([1, 2, 3], [2, 2, 3], difftype="diff")
    array([-1,  0,  0])
    """
    try:
        if isinstance(val1, list):
            val1 = np.array(val1)
        if isinstance(val2, list):
            val2 = np.array(val2)
        if difftype == 'diff':
            return val1-val2
        elif difftype == 'bool':
            return val1 != val2
        elif isinstance(difftype, float):
            return abs(val1-val2) > difftype
    except ValueError as e:
        raise Exception("Unable to diff "+str(val1)+" and "+str(val2)) from e


def file_check(filename, overwrite):
    """Check if files exists and whether to overwrite the file."""
    if os.path.exists(filename):
        if not overwrite:
            raise Exception("File already exists: "+filename)
        else:
            print("File already exists: "+filename+", writing anyway...")
            os.remove(filename)
    if "/" in filename:
        last_split_index = filename.rfind("/")
        foldername = filename[:last_split_index]
        try:
            os.makedirs(foldername)
        except FileExistsError:
            0  # in this case, just don't make the directory


def auto_filetype(filename, filetype="", filetypes=['npz', 'csv', 'json']):
    """
    Automatically determine the filetype (npz, csv, or json) of a filename.

    Examples
    --------
    >>> auto_filetype("hi.npz")
    'npz'
    >>> auto_filetype("example.csv")
    'csv'
    >>> auto_filetype("example.json")
    'json'
    >>> auto_filetype("x.pdf")
    Traceback (most recent call last):
      ...
    Exception: Invalid filename in x.pdf, ensure extension is in ['npz', 'csv', 'json'].
    >>> auto_filetype("no_ext", "csv")
    'csv'
    """
    if not filetype:
        if '.' not in filename:
            raise Exception("No file extension in: " + filename)
        for ft in filetypes:
            len_ft = len(ft)
            if filename[-(len_ft+1):] == '.'+ft:
                filetype = ft
                break
        if not filetype:
            raise Exception("Invalid filename in " + filename +
                            ", ensure extension is in "+str(filetypes)+".")
    return filetype


def create_indiv_filename(filename, indiv_id, splitchar='_'):
    """
    Create filename name for a file given general filename and individual id.

    Examples
    --------
    >>> create_indiv_filename("hi.csv", "4")
    'hi_4.csv'
    """
    filename_parts = filename.split(".")
    filename_parts.insert(1, '.')
    filename_parts.insert(1, splitchar+indiv_id)
    return "".join(filename_parts)


def load_folder(folder, filetype):
    """
    Create list of files to be read from a folder.

    (e.g., that have been saved from multi-scenario propagate methods with 'indiv':True)

    Parameters
    ----------
    folder : str
        Name of the folder. Must be in the current directory
    filetype : str
        Type of files in the folder ('pickle', 'csv', or 'json')

    Returns
    -------
    files_to_read : list
        files to load for results/mdlhists.
    """
    files = os.listdir(folder)
    files_toread = []
    for file in files:
        read_filetype = auto_filetype(file)
        if read_filetype == filetype:
            files_toread.append(file)
    return files_toread


def metric_preamble(data, dtype=None, rates=None, r_dtype=None, r_norm=False):
    """Process data for weighted metrics used by calc_metric and calc_metric_ci."""
    vals = np.array(data, dtype=dtype)
    if rates is not None:
        rates = np.array(rates, dtype=r_dtype)
        if r_norm:
            rates = rates/np.sum(rates)
        if np.size(rates) > 1 and np.size(vals) > 1:
            vals = np.multiply(rates.T, vals.T).T
        else:
            vals = rates*vals
    return vals


def calc_metric(data, method=np.average, args=(), axis=None, dtype=None,
                rates=None, r_dtype=None, r_norm=False, **kwargs):
    """
    Calculate a metric from data.

    Parameters
    ----------
    data : array/list/tuple
        Data to take metric over.
    method : method, optional
        Method to call to calculate the metric. The default is np.average.
    args : tuple, optional
        Arguments to the method. The default is ().
    axis : int, optional
        Axis of the array to take the metric over. The default is None.
    dtype : type, optional
        Datatype to pre-process to, if the datat should be interpreted as a different
        data (e.g., preprocess as bool to calculate a rate). The default is None.
    rate : array/list/tuple, optional
        Array of rates corresponding to data to weight the data by before calculating
        the metric. The default is None.
    r_dtype : type, optional
        Datatype to preprocess the rates over. The default is None.
    r_norm : bool, optional
        Whether to normalize rates. The default is False.
    **kwargs : kwargs
        Keyword arguments to method.

    Returns
    -------
    metric: float
        Metric calculated by method over data.

    Examples
    --------
    >>> calc_metric([1,2,3]) # simple average
    np.float64(2.0)
    >>> calc_metric([1,2,3], rates=[0.1, 0.1, 0.0], method=np.sum) # weighted sum
    np.float64(0.30000000000000004)
    >>> calc_metric([0, 20, 30], dtype=bool, rates=[0.1, 0.1, 0.1], method=np.sum) # rate of nonzero event
    np.float64(0.2)
    >>> calc_metric([0, 1, 2], "total")
    np.int64(2)
    >>> calc_metric([0, 1, 2], "expected", rates=[1.0, 2.0, 1.0])
    np.float64(4.0)
    """
    if isinstance(method, str):
        method = eval("calc_"+method)
        return method(data, args=args, axis=axis, dtype=dtype, rates=rates,
                      r_dtype=r_dtype, r_norm=r_norm, **kwargs)
    else:
        vals = metric_preamble(data, dtype, rates, r_dtype, r_norm)
        return method(vals, *args, **filter_kwargs(method, **kwargs, axis=axis))


def calc_metric_ci(data, method=np.average, return_anyway=False, interval=None,
                   axis=0, **kwargs):
    """
    Return bootstrapped confidence interval over calc_metric (or other methods).

    Parameters
    ----------
    data : data
        data for calc_metric.
    return_anyway : bool, optional
        Whether to return even if the bootstrap cannot be taken (e.g., for plotting).
        The default is False.
    axis : int
        Axis along which to calculate the confidence interval(s). The default is 0.
    **kwargs : kwargs
        Keyword arguments to calc_metric or scipy.bootstrap.

    Returns
    -------
    metric : float
        Metric (e.g., mean)
    ci_low : float
        Lower bound of confidence interval
    ci_high : float
        Upper bound of confidence interval

    Examples
    --------
    >>> calc_metric_ci([1, 2, 3], method=np.average)
    (np.float64(2.0), np.float64(1.0), np.float64(3.0))
    >>> calc_metric_ci([[1,2,3], [2,3,4]], method=np.average)
    (array([1.5, 2.5, 3.5]), array([1., 2., 3.]), array([2., 3., 4.]))
    >>> calc_metric_ci([[1,2,3], [2,3,4]], rates=[0.01, 1.0], method=np.sum)
    (array([2.01, 3.02, 4.03]), array([0.02, 0.04, 0.06]), array([4., 6., 8.]))
    """
    bs_kwar = filter_kwargs(bootstrap, **kwargs)
    if interval is not None:
        bs_kwar['confidence_level'] = interval*0.01

    vals = metric_preamble(data, **filter_kwargs(metric_preamble, **kwargs))
    if len(np.shape(vals)) > 1:
        val = vals[axis, 0]
    else:
        val = np.array(vals).flatten()[0]
    if "weights" in kwargs and kwargs['weights'] is not None:
        raise Exception("Weights not able to be used w- bootstrap--use rates instead.")
    met_val = method(vals, axis=axis, **filter_kwargs(method, **kwargs))
    if not np.all(vals == val):
        bs = bootstrap([vals], method, axis=axis, **bs_kwar)
        return met_val, bs.confidence_interval.low, bs.confidence_interval.high
    elif return_anyway:
        return met_val, met_val, met_val
    else:
        raise Exception("All data are the same!")


def calc_rate(data, rates=None, weights=None, **kwargs):
    """
    Calculate a rate of a non-zero value in data using calc_metric.

    Examples
    --------
    >>> calc_rate([0, 10, 0]) # defaults to equal rate
    np.float64(0.3333333333333333)
    >>> calc_rate([0, 10, 100], [0.1, 0.1, 0.1]) # provided rates
    np.float64(0.2)
    """
    if rates is None:
        rates = 1/np.size(data)
    kwar = {**kwargs, 'method': np.sum, 'dtype': bool, 'rates': rates}
    return calc_metric(data, **kwar)


def calc_percent(data, weights=None, rates=None, **kwargs):
    """
    Calculate a percent of a non-zero value in data using calc_metric.

    Examples
    --------
    >>> calc_percent([0, 10, 0])
    np.float64(0.3333333333333333)
    """
    return calc_metric(data, **{**kwargs, 'dtype': bool})


def calc_total(data, weights=None, **kwargs):
    """
    Calculate the total number of non-zero values in data using calc_metric.

    Examples
    --------
    >>> calc_total([0, 10, 0])
    np.int64(1)
    >>> calc_total([0, 10, 100])
    np.int64(2)
    """
    return calc_metric(data, **{**kwargs, 'method': np.sum, 'dtype': bool})


def calc_expected(data, rates=None, weights=None, **kwargs):
    """
    Calculate the expected value of given data using calc_metric.

    Examples
    --------
    >>> calc_expected([0, 5, 10]) # defaults to average
    np.float64(5.0)
    >>> calc_expected([0, 5, 10], [0.1, 0.5, 0.1])
    np.float64(3.5)
    """
    if rates is None:
        rates = 1/np.size(data)
    return calc_metric(data, **{**kwargs, 'method': np.sum, 'rates': rates})


def calc_average(data, weights=None, rates=None, **kwargs):
    """
    Calculate the average value of given data using calc_metric.

    Examples
    --------
    >>> calc_average([0, 5, 10])
    np.float64(5.0)
    >>> calc_average([0, 5, 10], [0.0, 0.5, 0.5])
    np.float64(7.5)
    """
    return calc_metric(data, **{**kwargs, 'method': np.average, 'weights': weights})


def calc_sum(data, weights=None, rates=None, **kwargs):
    """
    Calculate the average value of given data using calc_metric.

    Examples
    --------
    >>> calc_sum([0, 5, 10.0])
    np.float64(15.0)
    >>> calc_sum([0, 5, 10.0], [0.0, 0.5, 0.5])
    np.float64(15.0)
    """
    return calc_metric(data, **{**kwargs, 'method': np.sum})


def join_key(k):
    """
    Join list of keys into single key separated by a '.'.

    Examples
    --------
    >>> join_key(["key", "subkey"])
    'key.subkey'
    >>> join_key("existing_key")
    'existing_key'
    """
    if not isinstance(k, str):
        return '.'.join(k)
    else:
        return k


def setup_plot(fig=None, ax=None, z=False, figsize=(6, 4)):
    """
    Initialize a 2d or 3d figure at a given size.

    If there is a pre-existing figure or axis, uses that instead.
    """
    if not fig:
        if z or (type(z) in (int, float)):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, figsize=figsize)
    if fig:
        if not ax:
            ax = fig.add_subplot(111)
    return fig, ax


def phase_overlay(ax, phasemap, label_phases=True):
    """Overlay phasemap information on plot."""
    ymin, ymax = ax.get_ylim()
    phaseseps = [i[0] for i in list(phasemap.phases.values())[1:]]
    ax.vlines(phaseseps, ymin, ymax, colors='gray', linestyles='dashed')
    if label_phases:
        for phase in phasemap.phases:
            if phasemap.modephases:
                phasetext = [m for m, p in phasemap.modephases.items() if phase in p][0]
            else:
                phasetext = phase
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", lw=0, alpha=0.5)
            ax.text(np.average(phasemap.phases[phase]),
                    (ymin+ymax)/2, phasetext, ha='center', bbox=bbox_props)


def plot_err_hist(err_hist, ax=None, fig=None, figsize=(6, 4), boundtype='fill',
                  boundcolor='gray', boundlinestyle='--', fillalpha=0.3, time='time',
                  xlabel='time', ylabel='', title='', xlim=(), ylim=(), **kwargs):
    """
    Plot a line with a given range of uncertainty around it.

    Parameters
    ----------
    err_hist : History
        hist of line, low, high values. Has the form ::
        {'time': times, 'stat': stat_values, 'low': low_values, 'high': high_values}
    ax : mpl axis (optional)
        axis to plot the line on
    fig : mpl figure (optional)
        figure to plot line on
    figsize : tuple
        figure size (optional)
    boundtype : 'fill' or 'line'
        Whether the bounds should be marked with lines or a fill
    boundcolor : str, optional
        Color for bound fill The default is 'gray'.
    boundlinestyle : str, optional
        linestyle for bound lines (if any). The default is '--'.
    fillalpha : float, optional
        Alpha for fill. The default is 0.3.
    time : str, optional
        history to use as time. The default is 'time'.
    **kwargs : kwargs
        kwargs for the line

    Returns
    -------
    fig : mpl figure
    ax :mpl, axis
    """
    fig, ax = setup_plot(fig, ax, figsize)
    ax.plot(err_hist['stat'], **kwargs)
    if boundtype == 'fill':
        col = ax.lines[-1].get_color()
        ax.fill_between(err_hist[time], err_hist['low'], err_hist['high'],
                        alpha=fillalpha, color=col)
        if 'med_high' in err_hist and 'med_low' in err_hist:
            ax.fill_between(err_hist[time], err_hist['med_low'], err_hist['med_high'],
                            alpha=fillalpha, color=col)
    elif boundtype == 'line':
        plot_err_lines(err_hist[time], err_hist['low'], err_hist['high'], ax=ax,
                       fig=fig, color=boundcolor, linestyle=boundlinestyle)
        if 'med_high' in err_hist and 'med_low' in err_hist:
            plot_err_lines(err_hist[time], err_hist['med_low'], err_hist['med_high'],
                           ax=ax, fig=fig, color=boundcolor, linestyle=boundlinestyle)
    else:
        raise Exception("Invalid bound type: "+boundtype)
    if not xlim:
        xlim = err_hist[time][0], err_hist[time][-1]
    add_title_xylabs(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                     xlim=xlim, ylim=ylim)
    return fig, ax


def add_title_xylabs(ax, title='', xlabel='', ylabel='', zlabel='',
                     xlim=(), ylim=(), zlim=(), aspect=''):
    """Add/set title, x/y labels, and limits to the given axis."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if ax.name == '3d':
        if zlim:
            ax.set_zlim(*zlim)
        if zlabel:
            ax.set_zlabel(zlabel)
    if aspect:
        ax.set_aspect(aspect)
    if title:
        ax.set_title(title)


def plot_err_lines(times, lows, highs, ax=None, fig=None, figsize=(6, 4), **kwargs):
    """
    Plot error lines on the given plot.

    Parameters
    ----------
    times : list/array
        x data (time, typically)
    line : list/array
        y center data to plot
    lows : list/array
        y lower bound to plot
    highs : list/array
        y upper bound to plot
    **kwargs : kwargs
        kwargs for the line
    """
    fig, ax = setup_plot(ax, fig, figsize)
    ax.plot(times, highs, **kwargs)
    ax.plot(times, lows, **kwargs)
    return fig, ax


def unpack_plot_values(plot_values):
    """Upack plot_values if provided as a dict or str."""
    if len(plot_values) == 1 and type(plot_values[0]) is dict:
        plot_values = to_include_keys(plot_values[0])
    if not plot_values:
        raise Exception("Empty plot_values--make sure to pass quantities to plot!")
    return plot_values


def prep_animation_title(time, title='', **kwargs):
    """Add time to titles for plot_from methods."""
    kwargs['title'] = title+' t='+str(time)
    return kwargs


def clear_prev_figure(**kwargs):
    """Clear previous animations for plot_from methods."""
    if 'fig' in kwargs:
        kwargs['fig'].clf()
    # clear figure/ax beforehand for speed
    if 'ax' in kwargs:
        kwargs.pop('ax')
    return kwargs


def multiplot_helper(cols, *plot_values, figsize='default', titles={}, sharex=True,
                     sharey=False, fig=None, axs=None):
    """Create multiple plot axes for plotting."""
    num_plots = len(plot_values)
    if num_plots == 1:
        cols = 1
    rows = int(np.ceil(num_plots/cols))
    if not fig or not axs:
        if figsize == 'default':
            figsize = (cols*3, 2*rows)
        if not fig:
            fig, axs = plt.subplots(rows, cols,
                                    sharex=sharex, sharey=sharey, figsize=figsize)
        if axs is None:
            if len(fig.axes) != num_plots:
                fig.clf()
                axs = fig.subplots(rows, cols, sharex=sharex, sharey=sharey)
            else:
                axs = fig.axes

        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        elif not isinstance(axs, list):
            axs = [axs]

    subplot_titles = {plot_value: plot_value for plot_value in plot_values}
    subplot_titles.update(titles)
    return fig, axs, cols, rows, subplot_titles


def set_empty_multiplots(axs, num_plots, cols, xlab_ang=-90, grid=False,
                         set_above=True):
    """Align empty axes with the rest of the multiplot."""
    num_empty = len(axs) - num_plots
    starting_ax = len(axs) - num_empty
    for i, ax in enumerate(axs):
        if i >= starting_ax:
            # clear empty box
            ax.set_frame_on(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # set x tick labels for axis above
            if set_above:
                ax_above = axs[i-cols]
                ax_set = axs[num_plots-1]
                ax_above.tick_params(axis='x', rotation=xlab_ang, reset=True)
                ax_above.set_xlabel(ax_set.get_xlabel())
                labels = [t.get_text() for t in ax_set.get_xticklabels()]
                ax_above.set_xticks(ax_set.get_xticks(), labels=labels)
                # turn back on the reset grid
                if grid:
                    ax_above.grid(axis='x')


def multiplot_legend_title(groupmetrics, axs, ax,
                           legend_loc=False, title='', v_padding=None, h_padding=None,
                           title_padding=0.0, legend_title=None, **kwargs):
    """Create multiplot legends and titles on shared axes."""
    if len(groupmetrics) > 1 and legend_loc != False:
        consolidate_legend(ax)
        ax_l = axs[legend_loc]
        if ax_l != ax and legend_loc in [-1, len(axs)]:
            kwarg = {'loc': "center", "bbox_to_anchor": (0.5, 0.5), **kwargs}
        else:
            kwarg = kwargs
        consolidate_legend(ax_l, old_legend=ax.get_legend(), **kwarg)
    plt.subplots_adjust(hspace=v_padding, wspace=h_padding)
    if title:
        plt.suptitle(title, y=1.0+title_padding)


def consolidate_legend(ax, loc='upper left', bbox_to_anchor=(1.05, 1),
                       add_handles=[], remove_empty=True, color='', new_labs={},
                       old_legend=False, **kwargs):
    """
    Create a single legend for plots where multiple groups are being compared.

    Parameters
    ----------
    ax : matplotlib axis
        Axis object to mark on
    loc : str
        Legend location (see matplotlib.legend). Default is 'upper left'.
    bbox_to_anchor : tuple
        Anchoring bounding box point (see matplotlib.legend). Default is (1.05, 1).
    add_handles : list
        Labelled plot handles to add to the legend. Default is [].
    remove_empty : bool
        If add_handles is used, this toggles whether to add unlabeled entries to the
        legend. Default is False.
    """
    # get handles/labels and any existing legend
    hands, labs = ax.get_legend_handles_labels()
    if not old_legend:
        old_legend = ax.get_legend()

    # if there is an old legend (which may not have these handles),
    # update it with new handles/labels
    if old_legend:
        old_hands = old_legend.legend_handles
        kwargs['title'] = old_legend.get_title().get_text()
        old_legend.remove()
        hands = old_hands + hands

    # if there are labels to add (add_handles argument), add them here
    handles, labels = [], []
    for handle in hands + add_handles:
        lab = handle.get_label()
        if not (not lab and remove_empty):
            handles.append(handle)
            labels.append(new_labs.get(lab, lab))
    by_label = dict(zip(labels, handles))

    # generate legend with consolidated labels/handles
    leg_k = [k.split('.')[1] for k in rcParams if 'legend' in k]
    kw = {k: v for k, v in kwargs.items() if k in leg_k}
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=bbox_to_anchor, loc=loc, **kw)


def mark_times(ax, tick, time, *plot_values, fontsize=8, rounddec=1, pretext="t="):
    """
    Mark times on an axis at a particular tick interval.

    Parameters
    ----------
    ax : matplotlib axis
        Axis object to mark on
    tick : float
        Tick frequency.
    time : np.array
        Time vector.
    *plot_values : np.array
        x,y,z vectors
    fontsize : int, optional
        Size of the font. The default is 8.
    """
    t_tick = -tick
    for st in zip(*plot_values, time):
        tt = st[-1]
        xyz = st[:-1]
        if tt >= t_tick+tick:
            if tt < t_tick+tick+tick:
                ax.text(*xyz, pretext+str(np.round(tt, rounddec)), fontsize=fontsize)
            t_tick += tick


def suite_for_plots(testclass, plottests=False):
    """
    Qualitative testing suite with or without plots in unittest.

    Plot tests should have "plot" in the title of their method, this enables this
    function tofilter them out (or include them).

    Parameters
    ----------
    testclass : unittest.TestCase
        Test-case to create the suite for.
    plottests : bool/list, optional
        Whether to show the plot tests (True) or the non-plot tests (False). If a
        list is provided, only tests provided in the list will be run.

    Returns
    -------
    suite : unittest.TestSuite
        Test Suite to run with unittest.TextTestRunner() using runner.run
        (e.g., runner = unittest.TextTestRunner();
        runner.run(suite_for_plots(UnitTests, plottests=False)))
    """
    import unittest
    suite = unittest.TestSuite()
    if not plottests:
        tests = [func for func in dir(testclass)
                 if (func.startswith("test") and not ('plot' in func))]
    elif isinstance(plottests, list):
        tests = [func for func in dir(testclass)
                 if (func.startswith("test") and func in plottests)]
    else:
        tests = [func for func in dir(testclass)
                 if (func.startswith("test") and 'plot' in func)]
    for test in tests:
        suite.addTest(testclass(test))
    return suite


if __name__ == "__main__":
    calc_metric_ci([[1,2,3], [2,3,4]], rates= [0.01, 0.2], method=np.average)
    import doctest
    doctest.testmod(verbose=True)
