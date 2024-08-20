#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some common methods for analysis used by other modules.

Has methods:

- :func:`bootstrap_confidence_interval`: Convenience wrapper for scipy.bootstrap
- :func:`nan_to_x`: Helper function for Result Class, returns nan as zero if present,
  otherwise returns the number
- :func:`is_numeric`: Helper function for Result Class, checks if a given value is
  numeric
- :func:`join_key`: Helper function for Result Class
- :func:`setup_plot`: initializes mpl figure
- :func:`plot_err_hist`: Plots a line with a given range of uncertainty around it
- :func:`plot_err_lines`: Plots error lines on the given plot
- :func:`multiplot_legend_title`: Helper function for multiplot legends and titles
- :func:`consolidate_legend`: Creates a single legend for a given multiplot where
  multiple groups are being compared

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
import matplotlib.pyplot as plt


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


def is_numeric(val):
    """
    Check if a given value is a number.

    Examples
    --------
    >>> is_numeric(1.0)
    True
    >>> is_numeric("hi")
    False
    >>> is_numeric(np.array([1.0])[0])
    True
    >>> is_numeric(np.array(["hi"])[0])
    False
    """
    try:
        return np.issubdtype(np.array(val).dtype, np.number)
    except TypeError:
        return type(val) in [float, bool, int]


def bootstrap_confidence_interval(data, method=np.mean, return_anyway=False, **kwargs):
    """
    Return bootstrap confidence interval (helper for scipy.bootstrap).

    Parameters
    ----------
    data : list/array/etc
        Iterable with the data. May be float (for mean) or indicator (for proportion)
    method : method
        numpy method to give scipy.bootstrap.
    return_anyway: bool
        Gives a dummy interval of (stat, stat) if no . Used for plotting
    Returns
    ----------
    statistic, lower bound, upper bound
    """
    from scipy.stats import bootstrap
    if 'interval' in kwargs:
        kwargs['confidence_level'] = kwargs.pop('interval')*0.01
    if data.count(data[0]) != len(data):
        bs = bootstrap([data], np.mean, **kwargs)
        return method(data), bs.confidence_interval.low, bs.confidence_interval.high
    elif return_anyway:
        return method(data), method(data), method(data)
    else:
        raise Exception("All data are the same!")


def nan_to_x(metric, x=0.0):
    """
    Return nan as zero if present, otherwise return the number.

    Examples
    --------
    >>> nan_to_x(1.0)
    1.0
    >>> nan_to_x(np.nan, 10.0)
    10.0
    """
    if np.isnan(metric):
        return x
    else:
        return metric


def is_bool(val):
    """
    Check if the value is a boolean.

    Examples
    --------
    >>> is_bool(True)
    True
    >>> is_bool(1.0)
    False
    >>> is_bool(np.array([True])[0])
    True
    >>> is_bool(np.array([1.0])[0])
    False
    """
    try:
        return val.dtype in ['bool']
    except AttributeError:
        return type(val) in [bool]


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
                  boundcolor='gray', boundlinestyle='--', fillalpha=0.3,
                  xlabel='time', ylabel='', title='', **kwargs):
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
        ax.fill_between(err_hist['time'], err_hist['low'], err_hist['high'],
                        alpha=fillalpha, color=col)
        if 'med_high' in err_hist and 'med_low' in err_hist:
            ax.fill_between(err_hist['time'], err_hist['med_low'], err_hist['med_high'],
                            alpha=fillalpha, color=col)
    elif boundtype == 'line':
        plot_err_lines(err_hist['time'], err_hist['low'], err_hist['high'], ax=ax,
                       fig=fig, color=boundcolor, linestyle=boundlinestyle)
        if 'med_high' in err_hist and 'med_low' in err_hist:
            plot_err_lines(err_hist['time'], err_hist['med_low'], err_hist['med_high'],
                           ax=ax, fig=fig, color=boundcolor, linestyle=boundlinestyle)
    else:
        raise Exception("Invalid bound type: "+boundtype)
    add_title_xylabs(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(err_hist['time'][0], err_hist['time'][-1])
    return fig, ax


def add_title_xylabs(ax, title='', xlabel='', ylabel=''):
    """Add title and x/y labels to the given axis."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
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
                           title_padding=0.0, legend_title=None):
    """Create multiplot legends and titles on shared axes."""
    if len(groupmetrics) > 1 and legend_loc != False:
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax_l = axs[legend_loc]
        by_label = dict(zip(labels, handles))
        if ax_l != ax and legend_loc in [-1, len(axs)]:
            ax_l.legend(by_label.values(), by_label.keys(),
                        prop={'size': 8}, loc='center', title=legend_title)
        else:
            ax_l.legend(by_label.values(), by_label.keys(),
                        prop={'size': 8}, title=legend_title)
    plt.subplots_adjust(hspace=v_padding, wspace=h_padding)
    if title:
        plt.suptitle(title, y=1.0+title_padding)


def consolidate_legend(ax, loc='upper left', bbox_to_anchor=(1.05, 1),
                       add_handles=[], remove_empty=True, color='', **kwargs):
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
    old_legend = ax.get_legend()

    # if there is an old legend (which may not have these handles),
    # update it with new handles/labels
    if old_legend:
        old_hands = old_legend.legend_handles
        ax.get_legend().remove()
        hands = old_hands + hands

    # if there are labels to add (add_handles argument), add them here
    handles, labels = [], []
    for handle in hands + add_handles:
        lab = handle.get_label()
        if not (not lab and remove_empty):
            handles.append(handle)
            labels.append(lab)
    by_label = dict(zip(labels, handles))

    # generate legend with consolidated labels/handles
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=bbox_to_anchor, loc=loc, **kwargs)


def mark_times(ax, tick, time, *plot_values, fontsize=8):
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
    for st in zip(*plot_values, time):
        tt = st[-1]
        xyz = st[:-1]
        if tt % tick == 0:
            ax.text(*xyz, 't='+str(tt), fontsize=fontsize)


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
    import doctest
    doctest.testmod(verbose=True)
