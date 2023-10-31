"""
Description: Plots quantities of interest over time using matplotlib.

Uses the following methods:

- :func:`hist`: plots function and flow histories over time
  (with different plots for each function/flow)
- :func:`plot_line_and_err`: Plots a line with a given range of uncertainty around it.
- :func:`plot_err_lines`: Plots error lines on the given plot.
- :func:`multiplot_legend_title`: Helper function for multiplot legends and titles.
- :func:`make_consolidated_legend`: Creates a single legend for a given multiplot where
  multiple groups are being compared.
- :func:`metric_dist`: Plots the histogram of given metric(s) separated by
  comparison groups over a set of scenarios.
- :func:`metric_dist_from`:Plot the distribution of model history function/flow value
  over at defined time(s) over a number of scenarios.
- :func:`nominal_vals_1d`: plots the end-state classification of a system over a
  (1-D) range of nominal runs
- :func:`nominal_vals_2d`: plots the end-state classification of a system over a
  (2-D) range of nominal runs
- :func:`nominal_vals_3d`: plots the end-state classification of a system over a
  (3-D) range of nominal runs
- :func:`dyn_order`: Plots the run order for the model during the dynamic propagation
  step used by dynamic_behavior() methods.
  over given factors
- :func:`suite_for_plots`: enables plots to be checked and turned on/off when testing
  using unittest
"""
# File Name: analyze/plot.py
# Author: Daniel Hulse
# Created: November 2019 (Refactored April 2020, Feb 2022)

import numpy as np
import matplotlib.pyplot as plt
from fmdtools.analyze.result import bootstrap_confidence_interval, to_include_keys, is_numeric
from fmdtools.analyze.result import History, Result
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator

plt.rcParams['pdf.fonttype'] = 42


def hist(simhists, *plot_values, cols=2, aggregation='individual',
         legend_loc=-1, xlabel='time', ylabels={}, max_ind='max', titles={},
         title='', indiv_kwargs={}, time_slice=[], time_slice_label=None,
         figsize='default', comp_groups={},
         v_padding=None, h_padding=None, title_padding=0.0,
         phases={}, phase_kwargs={}, legend_title=None,  **kwargs):
    """
    Plot the behavior over time of the given function/flow values
    over a set of scenarios, with ability to aggregate behaviors as needed.

    Parameters
    ----------
    simhists : History
        Simulation history
    *plot_values : strs
        names of values to pull from the history (e.g., 'fxns.move_water.s.flowrate').
        Can also be specified as a dict (e.g. {'fxns': 'move_water'}) to get all keys
        from a given fxn/flow/mode/etc.
    cols : int, optional
        columns to use in the figure. The default is 2.
    aggregation : str, optional
        Way of aggregating the plot values. The default is 'individual'.

        Note that only the `individual` option can be used for histories of non-numeric
        quantities (e.g., modes, which are recorded as strings):

        - 'individual' plots each run individually.
        - 'mean_std' plots the mean values over the sim with standard deviation error
          bars.
        - 'mean_ci' plots the mean values over the sim with mean confidence interval
          error bars given by optional argument `ci`.
        - 'mean_bound' plots the mean values over the sim with variable bound error 
          bars.
        - 'percentile' plots the percentile distribution of the sim over time
          (does not reject outliers). Modified by 'perc_range' argument.
    comp_groups : dict, optional
        Dictionary for comparison groups (if more than one) with structure given by:
        ::
            {'group1': ('scen1', 'scen2'),
             'group2':('scen3', 'scen4')}.

        Default is {}, which compares nominal and faulty.
        If {'default': 'default'} is passed, all scenarios will be put in one group.
        If a legend is shown, group names are used as labels.
    legend_loc : int, optional
        Specifies the plot to place the legend on, if runs are bine compared. Default is
        -1 (the last plot). To remove the legend, give a value of False
    xlabel : str, optional
        Label for the x-axes. Default is 'time'
    ylabels : dict, optional
        Label for the y-axes.
        Has structure::
            {(fxnflowname, value): 'label'}

    max_ind : int, optional
        index (usually correlates to time) cutoff for the simulation. Default is 'max',
        which uses the first simulation termination time.
    boundtype : str, optional
        Way to represent the bound ('fill' or 'line'):

        - 'fill' plots the error bounds as a filled area with alpha define by optional 
         'fillalpha' argument.
        - 'line' plots the error bounds as lines modified by the optional 'boundcolor'
          and 'boundlinestyle' arguments.
    fillalpha : float
        alpha value for fill in aggregated plots. Default is 0.3.
    boundcolor : str, optional
        color to make the bounds in aggregated plots. Default is 'gray'
    boundlinestyle : str, optional
        linestyle to use for bounds in aggregated plots. Default is '--'
    perc_range : int, optional
        Percentile range of inner bars when using the 'percentile' option. Default is 50
    title : str, optional
        overall title for the plot. Default is ''
    indiv_kwargs : dict, optional
        Dict of kwargs to use to differentiate each comparison group.
        Has structure::
            {comp1: kwargs1, comp2: kwargs2}

        where kwargs is an individual dict of plt.plot arguments for the
        comparison group comp (or scenario, if not aggregated) which overrides
        the global kwargs (or default behavior). If no comparison groups are given,
        use 'default' for a single history or 'nominal'/'faulty' for a fault history
        e.g.::
            kwargs = {'nominal': {color: 'green'}} 

        would make the nominal color green. Default is {}.
    time_slice : int/list, optional
        overlays a bar or bars at the given index when the fault was injected (if any).
        Default is []
    time_slice_label : str, optional
        label to use for the time slice bars in the legend. Default is None.
    figsize : tuple (float,float)
        x-y size for the figure. The default is 'default', which dymanically gives 3 for
        each column and 2 for each row.
    v_padding : float
        vertical padding between subplots as a fraction of axis height.
    h_padding : float
        horizontal padding between subplots as a fraction of axis width.
    title_padding : float
        padding for title as a fraction of figure height.
    phases : dict, optional
        Provide to overlay phases on the individual function histories, where phases
        is a dict of PhaseMaps from analyze.phases.from_hist. Default is {}.
    phase_kwargs : dict
        kwargs to plot.phase_overlay.
    legend_title : str, optional
        title for the legend. Default is None
    **kwargs : kwargs
        Global/default keyword arguments to plot_line_and_err.

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis
    """
    simhists, plot_values, grouphists, indiv_kwargs = prep_hists(simhists,
                                                                 plot_values,
                                                                 comp_groups,
                                                                 indiv_kwargs)
    fig, axs, cols, rows, subplot_titles = multiplot_helper(cols, *plot_values,
                                                            figsize, titles=titles)

    for i, plot_value in enumerate(plot_values):
        ax = axs[i]
        ax.set_title(subplot_titles[plot_value])
        ax.grid()
        if i >= (rows-1)*cols and xlabel:
            ax.set_xlabel(xlabel)
        if ylabels.get(plot_value, False):
            ax.set_ylabel(ylabels[plot_value])
        for group, hists in grouphists.items():
            # TODO: find a better way to do this that will be compatible with timers
            loc_kwargs = {**kwargs, **indiv_kwargs.get(group, {}), 'label': group}
            try:
                if aggregation == 'individual':
                    individual_hists(hists, plot_value, group, ax, fig, **loc_kwargs)
                elif aggregation == 'mean_std':
                    mean_std(hists, plot_value, group, ax, fig, **loc_kwargs)
                elif aggregation == 'mean_ci':
                    mean_ci(hists, plot_value, group, ax, fig, max_ind=max_ind,
                            **loc_kwargs)
                elif aggregation == 'mean_bound':
                    mean_bound(hists, plot_value, group, ax, fig, **loc_kwargs)
                elif aggregation == 'percentile':
                    percentile(hists, plot_value, group, ax, fig, **loc_kwargs)
                else:
                    raise Exception("Invalid aggregation option: "+aggregation)
            except Exception as e:
                raise Exception("Error at plot_value " + str(plot_value)
                                + " and group: " + str(group)) from e
            if plot_value[1] in phases:
                phase_overlay(ax, phases[plot_value[1]], plot_value[1])
        if type(time_slice) == int:
            ax.axvline(x=time_slice, color='k', label=time_slice_label)
        else:
            for ts in time_slice:
                ax.axvline(x=ts, color='k', label=time_slice_label)
    multiplot_legend_title(grouphists, axs, ax, legend_loc, title,
                           v_padding, h_padding, title_padding, legend_title)
    return fig, axs


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


def hist_plot_helper(hists, ax=None, fig=None, figsize=(6, 4)):
    """Set up hist plot time/fig/ax."""
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    times = hists.get_metric('time', axis=0)
    return fig, ax, times


def individual_hists(hists, value, ax=None, fig=None, figsize=(6, 4),
                     **kwargs):
    """Plot plot_value in hist as individual lines."""
    fig, ax, times = hist_plot_helper(hists, ax=ax, fig=fig, figsize=figsize)
    hist_to_plot = hists.get_values(value)
    if 'color' not in kwargs:
        kwargs['color'] = next(ax._get_lines.prop_cycler)['color']
    for h in hist_to_plot.values():
        ax.plot(times, h, **kwargs)
    return fig, ax


def mean_std(hists, value, ax=None, fig=None, figsize=(6, 4), **kwargs):
    """Plot plot_value in hist aggregated by mean and standard devation."""
    fig, ax, times = hist_plot_helper(hists, ax=ax, fig=fig, figsize=figsize)
    mean = hists.get_metric(value, np.mean, axis=0)
    std_dev = hists.get_metric(value, np.std)
    plot_line_and_err(ax, times, mean, mean-std_dev/2, mean+std_dev/2, **kwargs)
    return fig, ax


def mean_ci(hists, value, ax=None, fig=None, figsize=(6, 4),
            max_ind='max', ci=0.95, **kwargs):
    """Plot plot_value in hist aggregated by bootstrap confidence interval for mean."""
    fig, ax, times = hist_plot_helper(hists, ax=ax, fig=fig, figsize=figsize)
    mean = hists.get_metric(value, np.mean, axis=0)
    if max_ind == 'max':
        max_ind = min([len(h) for h in hists.values()])
    vals = [[hist[t] for hist in hists.get_values(value).values()]
            for t in range(max_ind)]
    boot_stats = np.array([bootstrap_confidence_interval(val, return_anyway=True,
                                                         confidence_level=ci)
                           for val in vals]).transpose()
    plot_line_and_err(ax, times, mean, boot_stats[1], boot_stats[2], **kwargs)
    return fig, ax


def mean_bound(hists, value, ax=None, fig=None, figsize=(6, 4), **kwargs):
    """Plot the value in hist aggregated by the mean and variable bounds."""
    fig, ax, times = hist_plot_helper(hists, ax=ax, fig=fig, figsize=figsize)
    mean = hists.get_metric(value, np.mean, axis=0)
    maxs = hists.get_metric(value, np.max, axis=0)
    mins = hists.get_metric(value, np.min, axis=0)
    plot_line_and_err(ax, times, mean, mins, maxs, **kwargs)
    return fig, ax


def percentile(hists, value, ax=None, fig=None, figsize=(6, 4), prange=50,
               boundtype='fill', fillalpha=0.3, boundcolor='gray', boundlinestyle='--',
               **kwargs):
    """Plot the value in hist aggregated by percentiles."""
    fig, ax, times = hist_plot_helper(hists, ax=ax, fig=fig, figsize=figsize)
    median = hists.get_metric(value, np.median, axis=0)
    maxs = hists.get_metric(value, np.max, axis=0)
    mins = hists.get_metric(value, np.min, axis=0)
    plot_line_and_err(ax, times, median, mins, maxs, fillalpha=fillalpha,
                      boundcolor=boundcolor, boundlinestyle=boundlinestyle, **kwargs)

    low_perc = hists.get_metric(value, np.percentile, args=(50-prange/2,), axis=0)
    high_perc = hists.get_metric(value, np.percentile, args=(50+prange/2,), axis=0)
    if boundtype == 'fill':
        col = ax.lines[-1].get_color()
        ax.fill_between(times, low_perc, high_perc, alpha=fillalpha, color=col)
    elif boundtype == 'line':
        plot_err_lines(ax, times, low_perc, high_perc, color=boundcolor,
                       linestyle=boundlinestyle)
    return fig, ax


def prep_hists(simhists, plot_values, comp_groups, indiv_kwargs):
    """Prepare hists for plotting."""
    # Process data - clip and flatten
    if "time" in simhists:
        simhists = History(nominal=simhists).flatten()
    else:
        simhists = simhists.flatten()

    plot_values = unpack_plot_values(plot_values)

    grouphists = simhists.get_comp_groups(*plot_values, **comp_groups)

    # Set up plots and iteration
    if 'nominal' in grouphists.keys() and len(grouphists) > 1:
        indiv_kwargs['nominal'] = indiv_kwargs.get(
            'nominal', {'color': 'blue', 'ls': '--'})
    else:
        indiv_kwargs.pop('nominal', '')

    if 'faulty' in grouphists.keys():
        indiv_kwargs['faulty'] = indiv_kwargs.get('faulty', {'color': 'red'})
    else:
        indiv_kwargs.pop('faulty', '')

    return simhists, plot_values, grouphists, indiv_kwargs


def multiplot_helper(cols, *plot_values, figsize='default', titles={}):
    """Create multiple plot axes for plotting."""
    num_plots = len(plot_values)
    if num_plots == 1:
        cols = 1
    rows = int(np.ceil(num_plots/cols))
    if figsize == 'default':
        figsize = (cols*3, 2*rows)
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=figsize)

    if type(axs) == np.ndarray:
        axs = axs.flatten()
    else:
        axs = [axs]

    subplot_titles = {plot_value: plot_value for plot_value in plot_values}
    subplot_titles.update(titles)
    return fig, axs, cols, rows, subplot_titles


def plot_line_and_err(ax, times, line, lows, highs, boundtype='fill',
                      boundcolor='gray', boundlinestyle='--', fillalpha=0.3, **kwargs):
    """
    Plot a line with a given range of uncertainty around it.

    Parameters
    ----------
    ax : mpl axis
        axis to plot the line on
    times : list/array
        x data (time, typically)
    line : list/array
        y center data to plot
    lows : list/array
        y lower bound to plot
    highs : list/array
        y upper bound to plot
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
    """
    ax.plot(line, **kwargs)
    if boundtype == 'fill':
        ax.fill_between(times, lows, highs,
                        alpha=fillalpha, color=ax.lines[-1].get_color())
    elif boundtype == 'line':
        plot_err_lines(ax, times, lows, highs,
                       color=boundcolor, linestyle=boundlinestyle)
    else:
        raise Exception("Invalid bound type: "+boundtype)


def unpack_plot_values(plot_values):
    """Helper function for enabling both dict and str plot_values."""
    if len(plot_values) == 1 and type(plot_values[0]) == dict:
        plot_values = to_include_keys(plot_values[0])
    if not plot_values:
        raise Exception("Empty plot_values--make sure to pass quantities to plot!")
    return plot_values


def plot_err_lines(ax, times, lows, highs, **kwargs):
    """
    Plots error lines on the given plot

    Parameters
    ----------
    ax : mpl axis
        axis to plot the line on
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
    ax.plot(times, highs, **kwargs)
    ax.plot(times, lows, **kwargs)


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


def make_consolidated_legend(ax, loc='upper left', bbox_to_anchor=(1.05, 1),
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


def metric_dist(result, *plot_values, cols=2, comp_groups={},
                bins=10, metric_bins={},
                legend_loc=-1, xlabels={}, ylabel='count', title='', indiv_kwargs={},
                figsize='default',  v_padding=0.4, h_padding=0.05, title_padding=0.1,
                legend_title=None, **kwargs):
    """
    Plots the histogram of given metric(s) separated by comparison groups over a set of
    scenarios.

    Parameters
    ----------
    result : Result
        Result dictionary of metrics over set of scenarios
    *plot_values : str
        names of values to pull from the result (e.g., 'fxns.move_water.s.flowrate').
        Can also be specified as a dict (e.g. {'fxns':'move_water'}) to get all keys
        from a given fxn/flow/mode/etc.
    cols : int, optional
        columns to use in the figure. The default is 2.
    comp_groups : dict, optional
        Dictionary for comparison groups (if more than one).
        Has structure::
            {'group1': ('scen1', 'scen2'), 'group2': ('scen3', 'scen4')}.

        Default is {}, which compares nominal and faulty.
        If {'default': 'default'} is passed, all scenarios will be put in one group.
        If a legend is shown, group names are used as labels.
    bins : int
        Number of bins to use (for all plots). Default is None
    metric_bins : dict,
        Dictionary of number of bins to use for each metric.
        Has structure::
            {'metric':num}.
        
        Default is {}
    legend_loc : int, optional
        Specifies the plot to place the legend on, if runs are being compared.
        Default is -1 (the last plot)
        To remove the legend, give a value of False
    xlabels : dict, optional
        Label for the x-axes.
        Has structure::
            {'metric':'label'}

    ylabel : str, optional
        Label for the y-axes. Default is 'time'
    title : str, optional
        overall title for the plot. Default is ''
    indiv_kwargs : dict, optional
        dict of kwargs to differentiate the comparison groups.
        Has structure::
            {comp1: kwargs1, comp2: kwargs2}

        where kwargs is an individual dict of keyword arguments for the
        comparison group comp (or scenario, if not aggregated) which overrides
        the global kwargs (or default behavior).
    figsize : tuple (float,float)
        x-y size for the figure. The default is 'default', which dymanically gives 3 for
        each column and 2 for each row
    v_padding : float
        vertical padding between subplots as a fraction of axis height.
    h_padding : float
        horizontal padding between subplots as a fraction of axis width.
    title_padding : float
        padding for title as a fraction of figure height.
    legend_title : str, optional
        title for the legend. Default is None.
    **kwargs : kwargs
        keyword arguments to mpl.hist e.g. bins, etc.
    """
    # Sort into comparison groups
    plot_values = unpack_plot_values(plot_values)
    groupmetrics = result.get_comp_groups(*plot_values, **comp_groups)

    num_plots = len(plot_values)
    if num_plots == 1:
        cols = 1
    rows = int(np.ceil(num_plots/cols))
    if figsize == 'default':
        figsize = (cols*3, 2*rows)
    fig, axs = plt.subplots(rows, cols, sharey=True, sharex=False, figsize=figsize)
    if type(axs) == np.ndarray:
        axs = axs.flatten()
    else:
        axs = [axs]
    num_bins = bins
    for i, plot_value in enumerate(plot_values):
        ax = axs[i]
        xlabel = xlabels.get(plot_value, plot_value)
        if type(xlabel) == str:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(' '.join(xlabel))
        ax.grid(axis='y')
        fulldata = [i for endc in groupmetrics.values()
                    for i in [*endc.get_values(plot_value).values()]]
        bins = np.histogram(fulldata, metric_bins.get(plot_value, num_bins))[1]
        if not i % cols:
            ax.set_ylabel(ylabel)
        for group, endclasses in groupmetrics.items():
            local_kwargs = {**kwargs, **indiv_kwargs.get(group, {})}
            x = [*endclasses.get_values(plot_value).values()]
            ax.hist(x, bins, label=group, **local_kwargs)

    multiplot_legend_title(groupmetrics, axs, ax, legend_loc, title,
                           v_padding, h_padding, title_padding, legend_title)
    return fig, axs


def metric_dist_from(mdlhists, times, *plot_values, **kwargs):
    """
    Plot the distribution of model history function/flow value over at defined time(s)
    over a number of scenarios.

    Parameters
    ----------
    mdlhists : History
        Aggregate model histories over a set of scenarios.
    times : list/int
        List of times (or single time) to key the model history from.
        If more than one time is provided, it takes the place of comp_groups.
    *plot_values : strs
        names of values to pull from the history (e.g., 'fxns.move_water.s.flowrate').
        Can also be specified as a dict (e.g. {'fxns':'move_water'}) to get all keys
        from a given fxn/flow/mode/etc.
    **kwargs : kwargs
        keyword arguments to plot.metric_dist
    """
    flat_mdlhists = mdlhists.nest(levels=1)
    if type(times) in [int, float]:
        times = [times]
    if len(times) == 1 and kwargs.get('comp_groups', False):
        time_classes = Result({scen: Result(flat_hist.get_slice(times[0]))
                               for scen, flat_hist in flat_mdlhists.items()})
        comp_groups = kwargs.pop('comp_groups')
    elif kwargs.get('comp_groups', False):
        raise Exception("Cannot compare times and comp_groups at the same time")
    else:
        time_classes = Result({str(t)+'_'+scen: Result(flat_hist.get_slice(t))
                               for scen, flat_hist in flat_mdlhists.items()
                               for t in times})
        comp_groups = {str(t): {str(t)+'_'+scen for scen in flat_mdlhists}
                       for t in times}
    fig, axs = metric_dist(time_classes.flatten(), *plot_values,
                           comp_groups=comp_groups, **kwargs)
    return fig, axs

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
    make_consolidated_legend(axis)
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
    make_consolidated_legend(axis, loc=legend_loc)
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

    make_consolidated_legend(ax, loc=legend_loc)
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


def dyn_order(mdl, rotateticks=False, title="Dynamic Run Order"):
    """
    Plot the run order for the model during the dynamic propagation step used
    by dynamic_behavior() methods, where the x-direction is the order of each
    function executed and the y are the corresponding flows acted on by the
    given methods.

    Parameters
    ----------
    mdl : Model
        fmdtools model
    rotateticks : Bool, optional
        Whether to rotate the x-ticks (for bigger plots). The default is False.
    title : str, optional
        String to use for the title (if any). The default is "Dynamic Run Order".

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis

    """
    fxnorder = list(mdl.dynamicfxns)
    times = [i+0.5 for i in range(len(fxnorder))]
    fxntimes = {f: i for i, f in enumerate(fxnorder)}

    flowtimes = {f: [fxntimes[n] for n in mdl.graph.neighbors(
        f) if n in mdl.dynamicfxns] for f in mdl.flows}

    lengthorder = {k: v for k, v in
                   sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True) if len(v) > 0}
    starttimeorder = {k: v for k, v in
                      sorted(lengthorder.items(), key=lambda x: x[1][0], reverse=True)}
    endtimeorder = [k for k, v in
                    sorted(starttimeorder.items(), key=lambda x: x[1][-1], reverse=True)]
    flowtimedict = {flow: i for i, flow in enumerate(endtimeorder)}

    fig, ax = plt.subplots()

    for flow in flowtimes:
        phaseboxes = [((t, flowtimedict[flow]-0.5),
                       (t, flowtimedict[flow]+0.5),
                       (t+1.0, flowtimedict[flow]+0.5),
                       (t+1.0, flowtimedict[flow]-0.5))
                      for t in flowtimes[flow]]
        bars = PolyCollection(phaseboxes)
        ax.add_collection(bars)

    flowtimes = [i+0.5 for i in range(len(mdl.flows))]
    ax.set_yticks(list(flowtimedict.values()))
    ax.set_yticklabels(list(flowtimedict.keys()))
    ax.set_ylim(-0.5, len(flowtimes)-0.5)
    ax.set_xticks(times)
    ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
    ax.set_xlim(0, len(times))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor',  linewidth=2)
    ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labeltop=True)
    if title:
        if rotateticks:
            fig.suptitle(title, fontweight='bold', y=1.15)
        else:
            fig.suptitle(title, fontweight='bold')
    return fig, ax


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
