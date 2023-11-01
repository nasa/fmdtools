"""
Description: Some helper functions for working with matplotlib.

Provides the following methods:
- :func:`nominal_vals_1d`: plots the end-state classification of a system over a
  (1-D) range of nominal runs
- :func:`nominal_vals_2d`: plots the end-state classification of a system over a
  (2-D) range of nominal runs
- :func:`nominal_vals_3d`: plots the end-state classification of a system over a
  (3-D) range of nominal runs
- :func:`suite_for_plots`: enables plots to be checked and turned on/off when testing
  using unittest

Also provides the following library methods:
- :func: setup_plot : initializes mpl figure.
- :func:`plot_err_hist`: Plots a line with a given range of uncertainty around it.
- :func:`plot_err_lines`: Plots error lines on the given plot.
- :func:`multiplot_legend_title`: Helper function for multiplot legends and titles.
- :func:`consolidate_legend`: Creates a single legend for a given multiplot where
  multiple groups are being compared.
"""
# File Name: analyze/plot.py
# Author: Daniel Hulse
# Created: November 2019 (Refactored April 2020, Feb 2022)

import numpy as np
import matplotlib.pyplot as plt
from fmdtools.analyze.common import to_include_keys, is_numeric

plt.rcParams['pdf.fonttype'] = 42


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
        hist of line, low, high values. Has the form:
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
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


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
    """Helper function for enabling both dict and str plot_values."""
    if len(plot_values) == 1 and type(plot_values[0]) == dict:
        plot_values = to_include_keys(plot_values[0])
    if not plot_values:
        raise Exception("Empty plot_values--make sure to pass quantities to plot!")
    return plot_values


def multiplot_helper(cols, *plot_values, figsize='default', titles={}, sharex=True,
                     sharey=False):
    """Create multiple plot axes for plotting."""
    num_plots = len(plot_values)
    if num_plots == 1:
        cols = 1
    rows = int(np.ceil(num_plots/cols))
    if figsize == 'default':
        figsize = (cols*3, 2*rows)
    fig, axs = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=figsize)

    if type(axs) == np.ndarray:
        axs = axs.flatten()
    else:
        axs = [axs]

    subplot_titles = {plot_value: plot_value for plot_value in plot_values}
    subplot_titles.update(titles)
    return fig, axs, cols, rows, subplot_titles


def multiplot_legend_title(groupmetrics, axs, ax,
                           legend_loc=False, title='', v_padding=None, h_padding=None,
                           title_padding=0.0, legend_title=None):
    """ Helper function for multiplot legends and titles"""
    if len(groupmetrics) > 1 and legend_loc != False:
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax_l = axs[legend_loc]
        by_label = dict(zip(labels, handles))
        if ax_l != ax and legend_loc in [-1, len(axs)]:
            ax_l.set_frame_on(False)
            ax_l.get_xaxis().set_visible(False)
            ax_l.get_yaxis().set_visible(False)
            ax_l.legend(by_label.values(), by_label.keys(),
                        prop={'size': 8}, loc='center', title=legend_title)
        else:
            ax_l.legend(by_label.values(), by_label.keys(),
                        prop={'size': 8}, title=legend_title)
    plt.subplots_adjust(hspace=v_padding, wspace=h_padding)
    if title:
        plt.suptitle(title, y=1.0+title_padding)


def consolidate_legend(ax, loc='upper left', bbox_to_anchor=(1.05, 1),
                       add_handles=[], **kwargs):
    """Create a single legend for a given multiplot where multiple groups are
    being compared"""
    ax.legend()
    hands, labels = ax.get_legend_handles_labels()
    ax.legend(handles=add_handles+hands)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.get_legend().remove()
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


def get_nominal_classes(ps, endclasses, params, metric):
    """helper function for nominal_values_xd functions that gets the parameters and
    metrics to plot"""
    variable_groups = ps.get_scen_groups(*params)
    if not variable_groups:
        raise Exception("No matching scenarios--are parameters " +
                        params + " in the nomapp Scenarios?")
    group_values = {group:
                    [*endclasses.get_scens(*scens).get_values("."+metric).values()]
                    for group, scens in variable_groups.items()}

    if not group_values:
        raise Exception("No scenarios--is metric " + metric + " in endclasses?")
    return variable_groups, group_values


def get_dim_data(data, classifications, cl, ind):
    return [d[0][ind] for i, d in enumerate(data) if classifications[i] == cl]


def nominal_vals_1d(ps, endclasses, x_param,
                    title="Nominal Operational Envelope", nom_func=lambda x: x == 0.0,
                    metric='cost', figsize=(6, 4), xlabel='',
                    nom_alpha=0.5, nom_color="blue",
                    fault_alpha=0.5, fault_color="red"):
    """
    Visualizes the nominal operational envelope along one given parameter.

    Parameters
    ----------
    ps : ParameterSample
        ParameterSample sample approach simulated in the model.
    endclasses : Result
        Result dict for the set of simulations produced by running the model over ps
    x_param : str
        Parameter range desired to visualize in the operational envelope. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nom_func : method, optional
        Function to classify metric values as "nominal".
        Default is lambda x: x == 0.0
    metric : str
        Value to get from endclasses for the scenario(s). The default is 'cost'.
    figsize : bool
        figsize argument to plt.figure
    xlabel : str, optional
        label for x-axis (defaults to parameter name for x_param)
    nom_alpha : float, optional
        alpha value for nominal values. Default is 0.5.
    nom_color : str, optional
        color for nominal values
    fault_alpha : float, optional
        alpha value for off-nominal values. Default is 0.5.
    fault_color : str, optional
        color for off-nominal values

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure(figsize=figsize)

    nom_c = get_nominal_classes(ps, endclasses, (x_param,), metric)
    variable_groups, group_classes = nom_c

    data_values = [k[0] for k in variable_groups.keys()]
    if is_numeric(data_values):
        min_x = np.min(data_values)
        max_x = np.max(data_values)
        plt.hlines(1, min_x-1, max_x+1)
        plt.xlim(min_x-1, max_x+1)

    for var, vals in group_classes.items():
        for val in vals:
            if nom_func(val):
                plt.eventplot(var, label='nominal', color=nom_color, alpha=nom_alpha)
            else:
                plt.eventplot(var, label='faulty', color=fault_color, alpha=fault_alpha)

    axis = plt.gca()
    consolidate_legend(axis)
    axis.yaxis.set_ticklabels([])
    if not xlabel:
        xlabel = x_param
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(which='both', axis='x')
    return fig


def nominal_vals_2d(ps, endclasses, x_param, y_param,
                    title="Nominal Operational Envelope", nom_func=lambda x: x == 0.0,
                    metric='cost', figsize=(6, 4), xlabel='', ylabel='',
                    nom_alpha=0.5, nom_color="blue", nom_marker="o",
                    fault_alpha=0.5, fault_color="red", fault_marker="X",
                    legend_loc="best"):
    """
    Visualizes the nominal operational envelope along two given parameters

    Parameters
    ----------
    ps : ParameterSample
        ParameterSample sample approach simulated in the model.
    endclasses : Result
        Result dict for the set of simulations produced by running the model over ps
    x_param : str
        Parameter range desired to visualize on the x-axis. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    y_param : str
        Parameter range desired to visualize on the y-axis. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nom_func : method, optional
        Function to classify metric values as "nominal".
        Default is lambda x: x == 0.0
    metric : str
        Value to get from endclasses for the scenario(s). The default is 'cost'.
    figsize : bool
        figsize argument to plt.figure
    xlabel : str, optional
        label for x-axis (defaults to parameter name for x_param)
    ylabel : str, optional
        label for y-axis (defaults to parameter name for x_param)
    nom_alpha : float, optional
        alpha value for nominal values. Default is 0.5.
    nom_color : str, optional
        color for nominal values
    nom_marker : str, optional
        marker for nominal values. Default is 'o'.
    fault_alpha : float, optional
        alpha value for off-nominal values. Default is 0.5.
    fault_color : str, optional
        color for off-nominal values
    fault_marker : str, optional
        marker for nominal values. Default is 'X'.
    legend_loc : str, optional
        location for the legend (see matplotlib docs). Default is 'best'.

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure(figsize=figsize)
    nom_c = get_nominal_classes(ps, endclasses, (x_param, y_param), metric)
    variable_groups, group_classes = nom_c

    for var, vals in group_classes.items():
        for val in vals:
            if nom_func(val):
                plt.scatter([var[0]], [var[1]], label='nominal', marker=nom_marker,
                            alpha=nom_alpha, color=nom_color)
            else:
                plt.scatter([var[0]], [var[1]], label='faulty', marker=fault_marker,
                            alpha=fault_alpha, color=fault_color)

    axis = plt.gca()
    consolidate_legend(axis, loc=legend_loc)
    if not xlabel:
        xlabel = x_param
    if not ylabel:
        ylabel = y_param
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(which='both')
    return fig


def nominal_vals_3d(ps, endclasses, x_param, y_param, z_param,
                    title="Nominal Operational Envelope", nom_func=lambda x: x == 0.0,
                    metric='cost', figsize=(6, 4), xlabel='', ylabel='', zlabel='',
                    nom_alpha=0.5, nom_color="blue", nom_marker="o",
                    fault_alpha=0.5, fault_color="red", fault_marker="X",
                    legend_loc="best", markersize=50):
    """
    Visualize the nominal operational envelope along two given parameters.

    Parameters
    ----------
    ps : ParameterSample
        ParameterSample sample approach simulated in the model.
    endclasses : Result
        Result dict for the set of simulations produced by running the model over ps
    x_param : str
        Parameter range desired to visualize on the x-axis. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    y_param : str
        Parameter range desired to visualize on the y-axis. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    z_param : str
        Parameter range desired to visualize on the y-axis. Can be any
        property that changes over the nomapp
        (e.g., `r.seed`, `inputparam.x_in`, `p.x`...)
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nom_func : method, optional
        Function to classify metric values as "nominal".
        Default is lambda x: x == 0.0
    metric : str
        Value to get from endclasses for the scenario(s). The default is 'cost'.
    figsize : bool
        figsize argument to plt.figure
    xlabel, ylabel, zlabel : str, optional
        label for x/y/z-axis (defaults to parameter name for x_param/2/3)
    nom_alpha : float, optional
        alpha value for nominal values. Default is 0.5.
    nom_color : str, optional
        color for nominal values
    nom_marker : str, optional
        marker for nominal values. Default is 'o'.
    fault_alpha : float, optional
        alpha value for off-nominal values. Default is 0.5.
    fault_color : str, optional
        color for off-nominal values
    fault_marker : str, optional
        marker for nominal values. Default is 'X'.
    legend_loc : str, optional
        location for the legend (see matplotlib docs). Default is 'best'.

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    nom_c = get_nominal_classes(ps, endclasses, (x_param, y_param, z_param), metric)
    variable_groups, group_classes = nom_c

    for var, vals in group_classes.items():
        for val in vals:
            if nom_func(val):
                ax.scatter([var[0]], [var[1]], [var[2]], label='nominal',
                           marker=nom_marker, alpha=nom_alpha, color=nom_color)
            else:
                ax.scatter([var[0]], [var[1]], [var[2]], label='faulty',
                           marker=fault_marker, alpha=fault_alpha, color=fault_color)

    consolidate_legend(ax, loc=legend_loc)
    if not xlabel:
        xlabel = x_param
    if not ylabel:
        ylabel = y_param
    if not zlabel:
        zlabel = z_param
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.grid(which='both')
    return fig


def suite_for_plots(testclass, plottests=False):
    """
    Enables qualitative testing suite with or without plots in unittest. Plot tests
    should have "plot" in the title of their method, this enables this function to
    filter them out (or include them).

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
    elif type(plottests) == list:
        tests = [func for func in dir(testclass)
                 if (func.startswith("test") and func in plottests)]
    else:
        tests = [func for func in dir(testclass)
                 if (func.startswith("test") and 'plot' in func)]
    for test in tests:
        suite.addTest(testclass(test))
    return suite
