"""
Description: Plots quantities of interest over time using matplotlib.

Uses the following methods:
    - :func:`mdlhist`:         plots function and flow histories over time (with different plots for each function/flow)
    - :func:`mdlhists`:        plots function and flow histories over time on the same plot
    - :func:`metric_dist`:     Plots the histogram of given metric(s) separated by comparison groups over a set of scenarios
    - :func:`metric_dist_from`:Plot the distribution of model history function/flow value over at defined time(s) over a number of scenarios.
    - :func:`nominal_vals_1d`: plots the end-state classification of a system over a (1-D) range of nominal runs
    - :func:`nominal_vals_2d`: plots the end-state classification of a system over a (2-D) range of nominal runs
    - :func:`nominal_vals_3d`: plots the end-state classification of a system over a (3-D) range of nominal runs
    - :func:`nominal_factor_comparison`:    gives a bar plot of nominal simulation statistics over given factors
    - :func:`resilience_factor_comparison`: gives a bar plot of fault simulation statistics over given factors
    - :func:`phases`:          plots the phases of operation that the model progresses through.
    - :func:`samplemetric`:      plots a metric for a single fault sampled by a SampleApproach over time with rates
    - :func:`samplemetrics`:     plots a metric for a set of faults sampled by a SampleApproach over time with rates on separate plots
    - :func:`metricovertime`:    plots the total metric/explected metric of a set of faults sampled by a SampleApproach over time
"""
#File Name: resultdisp/plot.py
#Author: Daniel Hulse
#Created: November 2019 (Refactored April 2020, Feb 2022)

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
import copy
import warnings
import numpy as np
from fmdtools.resultdisp.tabulate import metricovertime as metric_table
from fmdtools.resultdisp.process import bootstrap_confidence_interval, flatten_hist
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from fmdtools.faultsim.propagate import cut_mdlhist


def mdlhists(mdlhists, fxnflowvals='all', cols=2, aggregation='individual', comp_groups={}, 
             legend_loc=-1, xlabel='time', ylabels={}, max_ind='max', boundtype='fill', 
             fillalpha=0.3, boundcolor='gray',boundlinestyle='--', ci=0.95, titles={},
             title='', indiv_kwargs={}, time_slice=[],time_slice_label=None, figsize='default',
             v_padding=None, h_padding=None, title_padding=0.0,
             phases={}, modephases={}, label_phases=True, legend_title=None,  **kwargs):
    """
    Plot the behavior over time of the given function/flow values 
    over a set of scenarios, with ability to aggregate behaviors as needed.

    Parameters
    ----------
    mdlhists : dict
        Aggregate model history with structure {'scen':mdlhist} (or single mdlhist)
    fxnflowsvals : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals], fxnflow:'val'/all, fxnflow:{'comp':[vals]}}. 
        The default is 'all', which returns all.
    cols : int, optional
        columns to use in the figure. The default is 2.
    aggregation : str, optional
        Way of aggregating the plot values. The default is 'individual'
        Note that only the `individual` option can be used for histories of non-numeric quantities
        (e.g., modes, which are recorded as strings)
        - 'individual' plots each run individually. 
        - 'mean_std' plots the mean values over the sim with standard deviation error bars
        - 'mean_ci'  plots the mean values over the sim with mean confidence interval error bars
            - optional argument ci (float 0.0-1.0) specifies the confidence interval (Default:0.95)
        - 'mean_bound' plots the mean values over the sim with variable bound error bars
        - 'percentile' plots the percentile distribution of the sim over time (does not reject outliers)
            - optional argument 'perc_range' (int 0-100) specifies the percentile range of the inner bars (Default: 50) 
    comp_groups : dict, optional
        Dictionary for comparison groups (if more than one) with structure:
            {'group1':('scen1', 'scen2'), 'group2':('scen3', 'scen4')} Default is {}
            If a legend is shown, group names are used as labels.
    legend_loc : int, optional
        Specifies the plot to place the legend on, if runs are bine compared. Default is -1 (the last plot)
        To remove the legend, give a value of False
    xlabel : str, optional
        Label for the x-axes. Default is 'time'
    ylabels : dict, optional
        Label for the y-axes with structure {(fxnflowname, value):'label'}
    max_ind : int, optional
        index (usually correlates to time) cutoff for the simulation. Default is 'max' which uses the first simulation termination time.
    boundtype : 'fill' or 'line', optional
        -'fill' plots the error bounds as a filled area
            - optional fillalpha (float) changes the alpha of this area.
        -'line' plots the error bounds as lines
            - optional boundcolor (str) changes the color of the bounds (default 'gray')
            - optional boundlinestyle (str) changes the style of the bound lines (default '--')
    fillalpha : float
        alpha value for fill in aggregated plots. Default is 0.3.
    boundcolor : str, optional
        color to make the bounds in aggregated plots. Default is 'gray'
    boundlinestyle : str, optional
        linestyle to use for bounds in aggregated plots. Default is '--'
    ci : float, optional
        Bootstrap confidence interval (0-1) to use in 'mean_ci' bound argument. Default is 0.95.
    title : str, optional
        overall title for the plot. Default is ''
    indiv_kwargs : dict, optional
        dict of kwargs with structure {comp1:kwargs1, comp2:kwargs2}, where 
        where kwargs is an individual dict of keyword arguments for the
        comparison group comp (or scenario, if not aggregated) which overrides 
        the global kwargs (or default behavior). If no comparison groups are given,
        use 'default' for a single history or 'nominal'/'faulty' for a fault history
        e.g. kwargs = {'nominal':{color:'green'}} would make the nominal color green   
        Default is {}.
    time_slice : int/list, optional
        overlays a bar or bars at the given index when the fault was injected (if any). Default is []
    time_slice_label : str, optional
        label to use for the time slice bars in the legend. Default is None
    figsize : tuple (float,float)
        x-y size for the figure. The default is 'default', which dymanically gives 3 for each column and 2 for each row
    v_padding : float
        vertical padding between subplots as a fraction of axis height
    h_padding : float
        horizontal padding between subplots as a fraction of axis width
    title_padding : float
        padding for title as a fraction of figure height
    phases : dict, optional
        Provide to overlay phases on the individual function histories, where phases
        is from rd.process.mdlhist and of structure {'fxnname':'phase':[start, end]}. 
        Default is {}.
    modephases : dict, optional
        dictionary that maps the phases to operational modes, if it is desired to track the progression
        through modes
    legend_title : str, optional
        title for the legend. Default is None
    **kwargs : kwargs
        keyword arguments to mpl.plot e.g. linestyle, color, etc. See 'aggregation' for specification.
        phases={}, modephases={}, label_phases=True,  **kwargs):
    """
    #Process data - clip and flatten
    if 'time' in mdlhists: mdlhists={'nominal':mdlhists}
    if max_ind=='max': max_ind = np.min([len(mdlhists[scen]['time']) for scen in mdlhists])-1
    inds = [i for i in range(len(mdlhists[[*mdlhists.keys()][0]]['time']))]
    for scen in mdlhists:
        mdlhists[scen] = cut_mdlhist(mdlhists[scen], max_ind)
    times = mdlhists[[*mdlhists.keys()][0]]['time']
    flat_mdlhists = {scen:flatten_hist(mdlhist,newhist={}, to_include=fxnflowvals) for scen, mdlhist in mdlhists.items()}
    #Sort into comparison groups
    if not comp_groups: 
        if aggregation=='individual':   grouphists = flat_mdlhists
        else:                           grouphists = {'default':flat_mdlhists}
    else:   grouphists = {group:{scen:flat_mdlhists[scen] for scen in scens} for group, scens in comp_groups.items()}
    # Set up plots and iteration
    if 'nominal' in grouphists.keys() and len(grouphists)>1: 
        indiv_kwargs['nominal'] = indiv_kwargs.get('nominal', {'color':'blue', 'ls':'--'})
    else: indiv_kwargs.pop('nominal','')
    if 'faulty' in grouphists.keys(): 
        indiv_kwargs['faulty'] = indiv_kwargs.get('faulty', {'color':'red'})  
    else: indiv_kwargs.pop('faulty','')
    template = [*flat_mdlhists.values()][0]
    plot_values = [i for i in template.keys() if i!='time']
    num_plots = len(plot_values)
    if num_plots==1: cols=1
    rows = int(np.ceil(num_plots/cols))
    if figsize=='default': figsize=(cols*3, 2*rows)
    fig, axs = plt.subplots(rows,cols, sharex=True, figsize=figsize) 
    if type(axs)==np.ndarray:   axs = axs.flatten()
    else:                       axs=[axs]
    
    subplot_titles = {plot_value:': '.join(plot_value[1:]) for plot_value in plot_values}
    subplot_titles.update(titles)
    
    for i, plot_value in enumerate(plot_values):
        ax = axs[i]
        ax.set_title(subplot_titles[plot_value])
        ax.grid()
        if i >= (rows-1)*cols and xlabel: ax.set_xlabel(xlabel)
        if ylabels.get(plot_value, False): ax.set_ylabel(ylabels[plot_value])
        for group, hists in grouphists.items():
            local_kwargs = {**kwargs, **indiv_kwargs.get(group,{})}
            if aggregation=='individual':
                if any([type(h)==tuple for h in hists.keys()]):
                    ax.plot(times, hists[plot_value], label=group, **local_kwargs)
                else:
                    if 'color' not in local_kwargs: local_kwargs['color'] = next(ax._get_lines.prop_cycler)['color']
                    for hist in hists.values():
                        ax.plot(times, hist[plot_value], label=group, **local_kwargs)
            elif aggregation=='mean_std':
                mean = np.mean([hist[plot_value] for hist in hists.values()], axis=0)
                std_dev = np.std([hist[plot_value] for hist in hists.values()], axis=0)
                plot_line_and_err(ax, times, mean, mean-std_dev/2, mean+std_dev/2,boundtype,boundcolor, boundlinestyle,fillalpha,label=group, **local_kwargs)
            elif aggregation=='mean_ci':
                mean = np.mean([hist[plot_value] for hist in hists.values()], axis=0)
                vals = [[hist[plot_value][t] for hist in hists.values()] for t in inds]
                boot_stats = np.array([bootstrap_confidence_interval(val, return_anyway=True, confidence_level=ci) for val in vals]).transpose()
                plot_line_and_err(ax, times, mean, boot_stats[1], boot_stats[2],boundtype,boundcolor, boundlinestyle,fillalpha,label=group, **local_kwargs)
            elif aggregation=='mean_bound':
                mean = np.mean([hist[plot_value] for hist in hists.values()], axis=0)
                maxs = np.max([hist[plot_value] for hist in hists.values()], axis=0)
                mins = np.min([hist[plot_value] for hist in hists.values()], axis=0)
                plot_line_and_err(ax, times, mean, mins, maxs,boundtype,boundcolor, boundlinestyle,fillalpha,label=group, **local_kwargs)
            elif aggregation=='percentile':
                median= np.median([hist[plot_value] for hist in hists.values()], axis=0)
                maxs = np.max([hist[plot_value] for hist in hists.values()], axis=0)
                mins = np.min([hist[plot_value] for hist in hists.values()], axis=0)
                low_perc = np.percentile([hist[plot_value] for hist in hists.values()],50-kwargs.get('perc_range',50)/2, axis=0)
                high_perc = np.percentile([hist[plot_value] for hist in hists.values()],50+kwargs.get('perc_range',50)/2, axis=0)
                plot_line_and_err(ax, times, median, mins, maxs,boundtype,boundcolor, boundlinestyle,fillalpha,label=group, **local_kwargs)
                if boundtype=='fill':       ax.fill_between(times,low_perc, high_perc, alpha=fillalpha, color=ax.lines[-1].get_color())
                elif boundtype=='line':     plot_err_lines(ax, times,low_perc,high_perc, color=boundcolor, linestyle=boundlinestyle)
            else: raise Exception("Invalid aggregation option: "+aggregation)
            if phases.get(plot_value[1]):
                ymin, ymax = ax.get_ylim()
                phaseseps = [i[0] for i in list(phases[plot_value[1]].values())[1:]]
                ax.vlines(phaseseps,ymin, ymax, colors='gray',linestyles='dashed')
                if label_phases:
                    for phase in phases[plot_value[1]]:
                        if modephases: phasetext = [m for m,p in modephases[plot_value[1]].items() if phase in p][0]
                        else: phasetext = phase
                        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", lw=0, alpha=0.5)
                        ax.text(np.average(phases[plot_value[1]][phase]), (ymin+ymax)/2, phasetext, ha='center', bbox=bbox_props)
        if type(time_slice)==int: ax.axvline(x=time_slice, color='k', label=time_slice_label)
        else:   
            for ts in time_slice: ax.axvline(x=ts, color='k', label=time_slice_label)
    multiplot_legend_title(grouphists, axs, ax, legend_loc, title, v_padding, h_padding, title_padding, legend_title)
    return fig, axs
def indiv_mdlhists(mdlhist, fxnflows={}, cols=2, aggregation='individual', comp_groups={}, 
             legend_loc=-1, xlabel='time', ylabels={}, max_ind='max', boundtype='fill', 
             fillalpha=0.3, boundcolor='gray',boundlinestyle='--', ci=0.95,
             title='', indiv_kwargs={}, time_slice=[],time_slice_label=None, figsize='default',
             phases={}, modephases={}, label_phases=True,  **kwargs):
    """
    Successively calls rd.plot.mdlhists to plot a number of seperate plots. Each
    function or flow is given its own plot.

    Parameters
    ----------
    mdlhists : dict
        Aggregate model history with structure {'scen':mdlhist} (or single mdlhist)
    fxnflows : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals], fxnflow:'val'/all, fxnflow:{'comp':[vals]}}.
        Each mdlhist is given an individual plot.
        The default is {}, which returns all.
    **kwargs : see kwargs for rd.plot.mdlhists
    Returns
    -------
    figs : list
        List of individual figures.
    """
    inputs = copy.copy(locals())
    inputs.pop('mdlhist',''); inputs.pop('fxnflows',''); inputs.pop('kwargs','')
    inputs = {**inputs, **kwargs}
    figs = []
    if 'time' in mdlhist: mdlhist={'nominal':mdlhist}
    if not fxnflows: fxnflows = {fxnflow:"all" for fxnflow in list(mdlhist['nominal']['functions'].keys())+list(mdlhist['nominal']['flows'].keys()) if any(mdlhist['nominal']['functions'].get(fxnflow, [])) or any(mdlhist['nominal']['flows'].get(fxnflow, []))}
    for fxnflow in fxnflows:
        fig = mdlhists(mdlhist.copy(), fxnflowvals={fxnflow:fxnflows.get(fxnflow,'all')}, **inputs, **kwargs)
        figs.append(fig)
    return figs
def mdlhist(mdlhist, fault='', time=0, fxnflows={},cols=2, returnfigs=False, legend=-1, timelabel='Time', units={}, phases={}, modephases={}, label_phases=True):
    """
    Deprecated legacy plotting function. Use resultdisp.plot.indiv_mdlhists instead.
    """
    warnings.warn("Deprecated function. Use resultdisp.plot.indiv_mdlhists instead.")
    figs = indiv_mdlhists(mdlhist, fxnflows, cols, title=fault, time_slice=time, legend_loc=legend, xlabel=timelabel, ylabels=units, phases=phases, modephases=modephases, label_phases=label_phases)
    if returnfigs: return figs
def mdlhistvals(mdlhist, fault='', time=0, fxnflowvals='all', cols=2, returnfig=True, legend=-1, timelabel="time", units={}, phases={}, modephases={}, label_phases=True):
    """
    Deprecated legacy plotting function. Use resultdisp.plot.mdlhists instead.
    """
    warnings.warn("Deprecated function. Use resultdisp.plot.mdlhists instead.")
    fig = mdlhists(mdlhist, fxnflowvals, cols, title=fault, time_slice=time, legend_loc=legend, xlabel=timelabel, ylabels=units, phases=phases, modephases=modephases, label_phases=label_phases)
    if returnfig: return fig
def plot_line_and_err(ax, times, line, lows, highs, boundtype, boundcolor='gray', boundlinestyle='--', fillalpha=0.3, **kwargs):
    """
    Plots a line with a given range of uncertainty around it.

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
    if boundtype=='fill':   ax.fill_between(times,lows, highs,alpha=fillalpha, color=ax.lines[-1].get_color())
    elif boundtype=='line': plot_err_lines(ax, times, lows, highs, color=boundcolor, linestyle=boundlinestyle)
    else:                   raise Exception("Invalid bound type: "+boundtype)
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
    ax.plot(times, highs **kwargs)
    ax.plot(times, lows, **kwargs)
def multiplot_legend_title(groupmetrics, axs, ax, legend_loc=False, title='', v_padding=None, h_padding=None, title_padding=0.0, legend_title=None):
    """ Helper function for multiplot legends and titles"""
    if len(groupmetrics)>1 and legend_loc!=False:
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax_l = axs[legend_loc]
        by_label = dict(zip(labels, handles))
        if ax_l !=ax and legend_loc in [-1, len(axs)]:
            ax_l.set_frame_on(False)
            ax_l.get_xaxis().set_visible(False)
            ax_l.get_yaxis().set_visible(False)
            ax_l.legend(by_label.values(), by_label.keys(), prop={'size': 8}, loc='center', title=legend_title)
        else: ax_l.legend(by_label.values(), by_label.keys(), prop={'size': 8}, title=legend_title)
    plt.subplots_adjust(hspace=v_padding, wspace=h_padding)
    if title: plt.suptitle(title, y=1.0+title_padding)
    

def metric_dist(endclasses, metrics='all', cols=2, comp_groups={}, bins=10, metric_bins={}, legend_loc=-1, 
                xlabels={}, ylabel='count', title='', indiv_kwargs={}, figsize='default', 
                v_padding=0.4, h_padding=0.05, title_padding=0.1, legend_title=None, **kwargs):
    """
    Plots the histogram of given metric(s) separated by comparison groups over a set of scenarios

    Parameters
    ----------
    endclasses : dict
        Dictionary of metrics with structure {'scen':{'metric':value}}
    metrics : list, optional
        list of metrics in the dictionary to plot
    cols : int, optional
        columns to use in the figure. The default is 2. 
    comp_groups : dict, optional
        Dictionary for comparison groups (if more than one) with structure:
            {'group1':('scen1', 'scen2'), 'group2':('scen3', 'scen4')} Default is {}
            If a legend is shown, group names are used as labels.
    bins : int
        Number of bins to use (for all plots). Default is None
    metric_bins : dict,
        Dictionary of number of bins to use for each metric with structure {'metric':num}
        Default is {}
    legend_loc : int, optional
        Specifies the plot to place the legend on, if runs are being compared. Default is -1 (the last plot)
        To remove the legend, give a value of False
    xlabels : dict, optional
        Label for the x-axes with structure {'metric':'label'}
    ylabel : str, optional
        Label for the y-axes. Default is 'time'
    title : str, optional
        overall title for the plot. Default is ''
    indiv_kwargs : dict, optional
        dict of kwargs with structure {comp1:kwargs1, comp2:kwargs2}, where 
        where kwargs is an individual dict of keyword arguments for the
        comparison group comp (or scenario, if not aggregated) which overrides 
        the global kwargs (or default behavior).
    figsize : tuple (float,float)
        x-y size for the figure. The default is 'default', which dymanically gives 3 for each column and 2 for each row
    v_padding : float
        vertical padding between subplots as a fraction of axis height
    h_padding : float
        horizontal padding between subplots as a fraction of axis width
    title_padding : float
        padding for title as a fraction of figure height
    legend_title : str, optional
        title for the legend. Default is None
    **kwargs : kwargs
        keyword arguments to mpl.hist e.g. bins, etc
    """
    #Sort into comparison groups
    if not comp_groups:     groupmetrics = {'default':endclasses}
    else:                   groupmetrics = {group:{ec:cl for ec,cl in endclasses.items() if ec in groupscens} for group, groupscens in comp_groups.items()}
    template = [*endclasses.values()][0]
    if metrics=='all':  plot_values = [i for i in template.keys()]
    else:               plot_values = [i for i in template.keys() if i in metrics]
    num_plots = len(plot_values)
    if num_plots==1: cols=1
    rows = int(np.ceil(num_plots/cols))
    if figsize=='default': figsize=(cols*3, 2*rows)
    fig, axs = plt.subplots(rows,cols, sharey=True, sharex=False, figsize=figsize) 
    if type(axs)==np.ndarray:   axs = axs.flatten()
    else:                       axs=[axs]
    num_bins = bins
    for i, plot_value in enumerate(plot_values):
        ax = axs[i]
        xlabel = xlabels.get(plot_value, plot_value)
        if type(xlabel)==str:   ax.set_xlabel(xlabel)
        else:                   ax.set_xlabel(' '.join(xlabel))
        ax.grid(axis='y')
        fulldata = [ec[plot_value] for endc in groupmetrics.values() for ec in endc.values()]
        bins = np.histogram(fulldata, metric_bins.get(plot_value, num_bins))[1]
        if not i%cols: ax.set_ylabel(ylabel)
        for group, endclasses in groupmetrics.items():
            local_kwargs = {**kwargs, **indiv_kwargs.get(group,{})}
            x = [ec[plot_value] for ec in endclasses.values()]
            ax.hist(x, bins, label=group, **local_kwargs)
    
    multiplot_legend_title(groupmetrics, axs, ax, legend_loc, title,v_padding, h_padding, title_padding, legend_title)
    return fig, axs

def metric_dist_from(mdlhists, times, fxnflowvals='all', **kwargs):
    """
    Plot the distribution of model history function/flow value over at defined time(s) over a number of scenarios.

    Parameters
    ----------
    mdlhists : dict
        Aggregate model history with structure {'scen':mdlhist} (or single mdlhist)
    times : list/int
        List of times (or single time) to key the model history from. 
        If more than one time is provided, it takes the place of comp_groups.
    fxnflowsvals : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals], fxnflow:'val'/all, fxnflow:{'comp':[vals]}}. 
        The default is 'all', which returns all.
    comp_groups : dict, optional
        Dictionary for comparison groups (if not comparing times) with structure:
            {'group1':('scen1', 'scen2'), 'group2':('scen3', 'scen4')} Default is {}
            If a legend is shown, group names are used as labels.
    **kwargs : kwargs
        keyword arguments to plot.metric_dist
    """
    flat_mdlhists = {scen:flatten_hist(mdlhist,newhist={}, to_include=fxnflowvals) for scen, mdlhist in mdlhists.items()}
    if type(times) in [int, float]: times=[times]
    if len(times)==1 and kwargs.get('comp_groups', False):
        time_classes = {scen:{metric:val[times[0]] for metric, val in flat_hist.items()} for scen, flat_hist in flat_mdlhists.items()}
        comp_groups=kwargs.pop('comp_groups')
    elif kwargs.get('comp_groups', False): raise Exception("Cannot compare times and comp_groups at the same time")
    else:
        time_classes = {str(t)+'_'+scen:{metric:val[t] for metric, val in flat_hist.items()} for scen, flat_hist in flat_mdlhists.items() for t in times}
        comp_groups = {t:{str(t)+'_'+scen for scen in flat_mdlhists} for t in times}
    fig, axs= metric_dist(time_classes,comp_groups=comp_groups, **kwargs)
    return fig, axs

def nominal_vals_1d(nomapp, nomapp_endclasses, param1, title="Nominal Operational Envelope", nomlabel = 'nominal', metric='classification', figsize=(6,4), xlabel=''):
    """
    Visualizes the nominal operational envelope along one given parameter

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal sample approach simulated in the model.
    nomapp_endclasses : dict
        End-classifications for the set of simulations in the model.
    param1 : str
        Parameter range desired to visualize in the operational envelope
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nomlabel : str, optional
        Flag for nominal end-states. The default is 'nominal'.
    xlabel: str, optional
        label for x-axis (defaults to parameter name for param1)

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.

    """
    
    fig = plt.figure(figsize=figsize)
    
    data = [(x, scen['properties']['inputparams'][param1]) for x,scen in nomapp.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False)!=False)]
    names = [d[0] for d in data]
    classifications = [str(nomapp_endclasses[name][metric]) for name in names] 
    all_classes = set(classifications)
    nom_classes = [c for c in all_classes if nomlabel in c]
    non_nom_classes = [c for c in all_classes if nomlabel not in c]
    discrete_classes = nom_classes + non_nom_classes
    
    min_x = np.min([d[1] for i,d in enumerate(data)])
    max_x = np.max([d[1] for i,d in enumerate(data)])
    plt.hlines(1,min_x-1, max_x+1)
    
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        if str(nomlabel) in cl:  plt.eventplot(xdata, label=cl, color='blue', alpha=0.5)
        else:                   plt.eventplot(xdata, label=cl, color='red', alpha=0.5)
    plt.legend()
    plt.xlim(min_x-1, max_x+1)
    axis = plt.gca()
    axis.yaxis.set_ticklabels([])
    if not xlabel: xlabel=param1
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(which='both', axis='x')
    return fig

def nominal_vals_2d(nomapp, nomapp_endclasses, param1, param2, title="Nominal Operational Envelope", nomlabel = 'nominal', metric='classification', legendloc='best', figsize=(6,4), xlabel='', ylabel=''):
    """
    Visualizes the nominal operational envelope along two given parameters

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal sample approach simulated in the model.
    nomapp_endclasses : dict
        End-classifications for the set of simulations in the model.
    param1 : str
        First parameter (x) desired to visualize in the operational envelope
    param2 : str
        Second arameter (y) desired to visualize in the operational envelope
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nomlabel : str, optional
        Flag for nominal end-states. The default is 'nominal'.
    xlabel: str, optional
        label for x-axis (defaults to parameter name for param1)
    ylabel: str, optional
        label for y-axis (defaults to parameter name for param2)

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure(figsize=figsize)
    
    data = [(x, scen['properties']['inputparams'][param1], scen['properties']['inputparams'][param2]) for x,scen in nomapp.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False)!=False and scen['properties']['inputparams'].get(param2,False)!=False)]
    names = [d[0] for d in data]
    classifications = [str(nomapp_endclasses[name][metric]) for name in names] 
    all_classes = set(classifications)
    nom_classes = [c for c in all_classes if nomlabel in c]
    non_nom_classes = [c for c in all_classes if nomlabel not in c]
    discrete_classes = nom_classes + non_nom_classes
    
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        ydata = [d[2] for i,d in enumerate(data) if classifications[i]==cl]
        if str(nomlabel) in cl:     plt.scatter(xdata, ydata, label=cl, marker="o")
        else:                       plt.scatter(xdata, ydata, label=cl, marker="X")
    plt.legend(loc=legendloc)
    if not xlabel: xlabel=param1
    if not ylabel: ylabel=param2
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(which='both')
    return fig

def nominal_vals_3d(nomapp, nomapp_endclasses, param1, param2, param3, title="Nominal Operational Envelope", nomlabel = 'nominal', metric='classification', figsize=(6,4), xlabel='', ylabel='', zlabel=''):
    """
    Visualizes the nominal operational envelope along three given parameters

    Parameters
    ----------
    nomapp : NominalApproach
        Nominal sample approach simulated in the model.
    nomapp_endclasses : dict
        End-classifications for the set of simulations in the model.
    param1 : str
        First parameter (x) desired to visualize in the operational envelope
    param2 : str
        Second parameter (y) desired to visualize in the operational envelope
    param3 : str
        Third parameter (y) desired to visualize in the operational envelope
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nomlabel : str, optional
        Flag for nominal end-states. The default is 'nominal'.
    xlabel: str, optional
        label for x-axis (defaults to parameter name for param1)
    ylabel: str, optional
        label for y-axis (defaults to parameter name for param2)
    zlabel: str, optional
        label for z-axis (defaults to parameter name for param3)

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    
    data = [(x, scen['properties']['inputparams'][param1], scen['properties']['inputparams'][param2], scen['properties']['inputparams'][param3]) for x,scen in nomapp.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False)!=False and scen['properties']['inputparams'].get(param2,False)!=False and scen['properties']['inputparams'].get(param3,False)!=False)]
    names = [d[0] for d in data]
    classifications = [str(nomapp_endclasses[name][metric]) for name in names] 
    all_classes = set(classifications)
    nom_classes = [c for c in all_classes if nomlabel in c]
    non_nom_classes = [c for c in all_classes if nomlabel not in c]
    discrete_classes = nom_classes + non_nom_classes
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        ydata = [d[2] for i,d in enumerate(data) if classifications[i]==cl]
        zdata = [d[3] for i,d in enumerate(data) if classifications[i]==cl]
        if str(nomlabel) in cl:  ax.scatter(xdata, ydata, zdata, label=cl, marker="o")
        else:                   ax.scatter(xdata, ydata, zdata, label=cl, marker="X")
    ax.legend()
    if not xlabel: xlabel=param1
    if not ylabel: ylabel=param2
    if not zlabel: xlabel=param3
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.grid(which='both')
    return fig

def dyn_order(mdl, rotateticks=False, title="Dynamic Run Order"):
    """
    Plots the run order for the model during the dynamic propagation step used 
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
    fxntimes = {f:i for i,f in enumerate(fxnorder)}
    
    flowtimes = {f:[fxntimes[n] for n in mdl.bipartite.neighbors(f) if n in mdl.dynamicfxns] for f in mdl.flows}
    
    lengthorder = {k:v for k,v in sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True) if len(v)>0}
    starttimeorder = {k:v for k,v in sorted(lengthorder.items(), key=lambda x: x[1][0], reverse=True)}
    endtimeorder = [k for k,v in sorted(starttimeorder.items(), key=lambda x: x[1][-1], reverse=True)]
    flowtimedict = {flow:i for i,flow in enumerate(endtimeorder)}
    
    fig, ax = plt.subplots()
    
    for flow in flowtimes:
        phaseboxes = [((t,flowtimedict[flow]-0.5),(t,flowtimedict[flow]+0.5),(t+1.0,flowtimedict[flow]+0.5),(t+1.0,flowtimedict[flow]-0.5)) for t in flowtimes[flow]]
        bars = PolyCollection(phaseboxes)
        ax.add_collection(bars)
        
    flowtimes = [i+0.5 for i in range(len(mdl.flows))]
    ax.set_yticks(list(flowtimedict.values()))
    ax.set_yticklabels(list(flowtimedict.keys()))
    ax.set_ylim(-0.5,len(flowtimes)-0.5)
    ax.set_xticks(times)
    ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
    ax.set_xlim(0,len(times))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor',  linewidth=2)
    ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labeltop=True)
    if title: 
        if rotateticks: fig.suptitle(title,fontweight='bold',y=1.15)
        else:           fig.suptitle(title,fontweight='bold')
    return fig, ax

def phases(mdlphases, modephases=[], mdl=[], singleplot = True, phase_ticks = 'both'):
    """
    Plots the phases of operation that the model progresses through.

    Parameters
    ----------
    mdlphases : dict
        phases that the functions of the model progresses through (e.g. from rd.process.mdlhist)
        of structure {'fxnname':'phase':[start, end]}
    modephases : dict, optional
        dictionary that maps the phases to operational modes, if it is desired to track the progression
        through modes
    mdl : Model, optional
        model, if it is desired to additionally plot the phases of the model with the function phases
    singleplot : bool, optional
        Whether the functions' progressions through phases are plotted on the same plot or on different plots.
        The default is True.
    phase_ticks : 'std'/'phases'/'both'
        x-ticks to use (standard, at the edge of phases, or both). Default is 'both'
    Returns
    -------
    fig/figs : Figure or list of Figures
        Matplotlib figures to edit/use.

    """
    if mdl: mdlphases["Model"] = mdl.phases
    
    if singleplot:
        num_plots = len(mdlphases)
        fig = plt.figure()
    else: figs = []
    
    for i,(fxn, fxnphases) in enumerate(mdlphases.items()):
        if singleplot:  ax = plt.subplot(num_plots, 1,i+1, label=fxn)
        else:           fig, ax = plt.subplots()
        
        if modephases and modephases.get(fxn, False): 
            mode_nums = {ph:i for i,(k,v) in enumerate(modephases[fxn].items()) for ph in v}
            ylabels = list(modephases[fxn].keys())
        else:
            mode_nums = {ph:i for i,ph in enumerate(fxnphases)}
            ylabels = list(mode_nums.keys())
        
        phaseboxes = [((v[0]-.5,mode_nums[k]-.4),(v[0]-.5,mode_nums[k]+.4),(v[1]+.5,mode_nums[k]+.4),(v[1]+.5,mode_nums[k]-.4)) for k,v in fxnphases.items()]
        color_options = list(mcolors.TABLEAU_COLORS.keys())[0:len(ylabels)]
        colors = [color_options[mode_nums[phase]] for phase in fxnphases]
        bars = PolyCollection(phaseboxes, facecolors=colors)
        
        ax.add_collection(bars)
        ax.autoscale()
        
        ax.set_yticks(list(set(mode_nums.values())))
        ax.set_yticklabels(ylabels)
        
        times = [0]+[v[1] for k,v in fxnphases.items()]
        if phase_ticks=='both':     ax.set_xticks(list(set(list(ax.get_xticks())+times)))
        elif phase_ticks=='phases':  ax.set_xticks(times)
        ax.set_xlim(times[0], times[-1])
        plt.grid(which='both', axis='x')
        if singleplot:
            plt.title(fxn)
        else:
            plt.title("Progression of "+fxn+" through operational phases")
            figs.append(fig)
    if singleplot:
        plt.suptitle("Progression of model through operational phases")
        plt.tight_layout(pad=1)
        plt.subplots_adjust(top=1-0.15-0.05/num_plots)
        return fig
    else:           return figs
             

def samplemetric(app, endclasses, fxnmode, samptype='std', title="", metric='cost', ylims=None):
    """
    Plots the sample metric and rate of a given fault over the injection times defined in the app sampleapproach
    
    (note: not currently compatible with joint fault modes)
    
    Parameters
    ----------
    app : sampleapproach
        Sample approach defining the underlying samples to take and probability model of the list of scenarios.
    endclasses : dict
        A dict with the end classification of each fault (metrics, etc)
    fxnmode : tuple
        tuple (or tuple of tuples) with structure ('function name', 'mode name') defining the fault mode
    metric : str
        Metric to plot. The default is 'cost'
    samptype : str, optional
        The type of sample approach used:
            - 'std' for a single point for each interval
            - 'quadrature' for a set of points with weights defined by a quadrature
            - 'pruned piecewise-linear' for a set of points with weights defined by a pruned approach (from app.prune_scenarios())
            - 'fullint' for the full integral (sampling every possible time)
    """
    associated_scens=[]
    for phasetup in app.mode_phase_map[fxnmode]:
        associated_scens = associated_scens + app.scenids.get((fxnmode, phasetup), [])
    associated_scens = list(set(associated_scens))
    costs = np.array([endclasses[scen][metric] for scen in associated_scens])
    #times = np.array(list(set([time  for phase, timemodes in app.sampletimes.items() if timemodes for time in timemodes if fxnmode in timemodes.get(time)])))  
    times = np.array([[a['properties']['time'] for a in app.scenlist if a['properties']['name']==scen][0] for scen in associated_scens])
    timesort = np.argsort(times)
    times = times[timesort]; costs=costs[timesort]
    a=1
    tPlot, axes = plt.subplots(2, 1, sharey=False, gridspec_kw={'height_ratios': [3, 1]})
    
    phasetimes_start=[]; phasetimes_end=[]; ratesvect=[]; phaselabels=[]
    for phase, ptimes in app.mode_phase_map[fxnmode].items():
        if type(ptimes[0])==list:
            phasetimes_start+=[t[0] for t in ptimes]
            phasetimes_end+=[t[1] for t in ptimes]
            ratesvect += [app.rates_timeless[fxnmode][phase] for t in ptimes] *2
            phaselabels+=[phase[1] for t in ptimes]
        else: 
            phasetimes_start.append(ptimes[0]); phasetimes_end.append(ptimes[1])
            ratesvect = ratesvect + [app.rates_timeless[fxnmode][phase]]*2
            phaselabels.append(phase[1])
    ratetimes =[]
    phaselocs = []
    for (ind, phasetime) in enumerate(phasetimes_start):
        axes[0].axvline(phasetime, color="black")        
        phaselocs= phaselocs +[(phasetimes_end[ind]-phasetimes_start[ind])/2 + phasetimes_start[ind]]

        axes[1].axvline(phasetime, color="black") 
        ratetimes = ratetimes + [phasetimes_start[ind]] + [phasetimes_end[ind]]
        
        #axes[1].text(middletime, 0.5*max(rates),  list(app.phases.keys())[ind], ha='center', backgroundcolor="white")
    #rate plots
    axes[1].set_xticks(phaselocs)
    axes[1].set_xticklabels(phaselabels)
    
    sorty = np.argsort(phasetimes_start)
    phasetimes_start=np.array(phasetimes_start)[sorty]; phasetimes_end=np.array(phasetimes_end)[sorty]
    sortx = np.argsort(ratetimes)
    axes[1].plot(np.array(ratetimes)[sortx], np.array(ratesvect)[sortx])
    axes[1].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    axes[1].set_ylim(0, np.max(ratesvect)*1.2 )
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Time ("+str(app.units)+")")
    axes[1].grid()
    #cost plots
    axes[0].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    if not ylims:
        ylims = [min(1.2*np.min(costs),-1e-5),max(1.2*np.max(costs),1e-5)]
    axes[0].set_ylim(*ylims)
    if samptype=='fullint':
        axes[0].plot(times, costs, label=metric)
    else:
        if samptype=='quadrature' or samptype=='pruned piecewise-linear': 
            sizes =  1000*np.array([weight if weight !=1/len(timeweights) else 0.0 for (phasetype, phase), timeweights in app.weights[fxnmode].items() if timeweights for time, weight in timeweights.items() if time in times])
            axes[0].scatter(times, costs,s=sizes, label=metric, alpha=0.5)
        axes[0].stem(times, costs, label=metric, markerfmt=",", use_line_collection=True)
    
    axes[0].set_ylabel(metric)
    axes[0].grid()
    if title: axes[0].set_title(title)
    elif type(fxnmode[0])==tuple: axes[0].set_title(metric+" function of "+str(fxnmode)+" over time")
    else:                       axes[0].set_title(metric+" function of "+fxnmode[0]+": "+fxnmode[1]+" over time")
    #plt.subplot_adjust()
    plt.tight_layout()
def samplemetrics(app, endclasses, joint=False, title="", metric='cost'):
    """
    Plots the costs and rates of a set of faults injected over time according to the approach app

    Parameters
    ----------
    app : sampleapproach
        The sample approach used to run the list of faults
    endclasses : dict
        A dict of results for each of the scenarios.
    joint : bool, optional
        Whether to include joint fault scenarios. The default is False.
    title : str
        Optional title.
    metric : str
        Metric to plot. The default is 'cost'
    """
    for fxnmode in app.list_modes(joint):
        if any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='fullint' and fm==fxnmode]):
            st='fullint'
        elif any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='quadrature' and fm==fxnmode]):
            st='quadrature'
        else: 
            st='std'
        samplemetric(app, endclasses, fxnmode, samptype=st, title="", metric=metric)

def metricovertime(endclasses, app, metric='cost', metrictype='expected cost'):
    """
    Plots the total cost or total expected cost of faults over time.

    Parameters
    ----------
    endclasses : dict
        dict with rate,cost, and expected cost for each injected scenario (e.g. from run_approach())
    app : sampleapproach
        sample approach used to generate the list of scenarios
    metric : str
        metric to plot ovre time. Default is 'cost'
    metrictype : str, optional
        type of cost to plot (e.g,'cost', 'expected cost' or 'rate'). The default is 'expected cost'.
    Returns
    -------
    figure: matplotlib figure
    """
    costovertime = metric_table(endclasses, app, metric=metric)
    plt.plot(list(costovertime.index), costovertime[metrictype])
    plt.title('Total '+metrictype+' of all faults over time.')
    plt.ylabel(metrictype)
    plt.xlabel("Time ("+str(app.units)+")")
    plt.grid()
    return plt.gcf()

def nominal_factor_comparison(comparison_table, metric, ylabel='proportion', figsize=(6,4), title='', maxy='max', xlabel=True, error_bars=False):
    """
    Compares/plots a comparison table from tabulate.nominal_factor_comparison as a bar plot for a given metric.

    Parameters
    ----------
    comparison_table : pandas table
        Table from tabulate.nominal_factor_comparison
    metrics : string
        Metric to use in the plot
    ylabel : string, optional
        label for the y-axis. The default is 'proportion'.
    figsize : tuple, optional
        Size for the plot. The default is (12,8).
    title : str, optional
        Plot title. The default is ''.
    maxy : float
        Cutoff for the y-axis (to use if the default is bad). The default is 'max'
    xlabel : TYPE, optional
        DESCRIPTION. The default is True.
    error_bars : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    figure: matplotlib figure
    """
    figure = plt.figure(figsize=figsize)
    
    if type(comparison_table.columns[0])==tuple and '' in comparison_table.columns[0]: #bounded table
        bar = np.array([comparison_table.at[metric,col] for col in comparison_table.columns if col[1]==''])
        labels= [str(i[0]) for i in comparison_table.columns if i[1]=='']
        if error_bars:
            UB = np.array([comparison_table.at[metric,col] for col in comparison_table.columns if col[1]=='UB'])
            LB = np.array([comparison_table.at[metric,col] for col in comparison_table.columns if col[1]=='LB'])
            yerr= [bar-LB, UB-bar]
            if maxy=='max': maxy = comparison_table.loc[metric].max()
        else: 
            yerr=[]
            if maxy=='max': maxy = max(bar)
    else:  
        bar = [*comparison_table.loc[metric]]; yerr=[]; labels=[str(i) for i in comparison_table.columns]
        if maxy=='max': maxy = max(bar)
    
    ax = figure.add_subplot(1,1,1)
    multibar_helper(ax,comparison_table.columns, maxy)
    ax.set_ylabel(ylabel)
    if title: plt.title(title)
    xs = np.array([ i for i in range(len(bar))])
    if yerr:    plt.bar(xs,bar, tick_label=labels, linewidth=4, yerr=yerr, error_kw={'elinewidth':3})
    else:       plt.bar(xs,bar, tick_label=labels, linewidth=4)
    return figure

def resilience_factor_comparison(comparison_table, faults='all', rows=1, stat='proportion', figsize=(12,8), title='', maxy='max', legend="single", stack=False, xlabel=True, error_bars=False):
    """
    Plots a comparison_table from tabulate.resilience_factor_comparison as a bar plot for each fault scenario/set of fault scenarios.

    Parameters
    ----------
    comparison_table : pandas table
        Table from tabulate.resilience_factor_test with factors as rows and fault scenarios as columns
    faults : list, optional
        iterable of faults/fault types to include in the bar plot (the columns of the table). The default is 'all'.
        a dictionary {'fault':'title'} will associate the given fault with a title (otherwise 'fault' is used)
    rows : int, optional
        Number of rows in the multplot. The default is 1.
    stat : str, optional
        Metric being presented in the table (for the y-axis). The default is 'proportion'.
    figsize : tuple(int, int), optional
        Size of the figure in (width, height). The default is (12,8).
    title : string, optional
        Overall title for the plots. The default is ''.
    maxy : float, optional
        Maximum y-value (to ensure same scale). The default is 'max' (finds max value of table).
    legend : str, optional
        'all'/'single'/'none'. The default is "single".
    stack : bool, optional
        Whether or not to stack the nominal and resilience plots. The default is False.
    xlabel : bool/str
        The x-label descriptor for the design factors. Defaults to the column values.
    error_bars : bool
        Whether to include error bars for the factor. Requires comparison_table to have lower and upper bound information

    Returns
    -------
    figure: matplotlib figure
        Plot handle of the figure.
    """
    figure = plt.figure(figsize=figsize)
    if type(comparison_table.columns[0])==tuple:    has_bounds = True
    else:                                           has_bounds=False
    if faults=='all': 
        if has_bounds: faults=[f[0] for f in comparison_table][0:-1:3]
        else:          faults=[*comparison_table.columns]
        faults.remove('nominal')
    columns = int(np.ceil(len(faults)/rows))
    n=0
    if maxy=='max':
        if stack==False:    maxy = comparison_table.max().max()
        else:               maxy=comparison_table.iloc[:,1:].max().max()+comparison_table['nominal'].max()
    for fault in faults:
        n+=1
        ax = figure.add_subplot(rows, columns, n, label=str(n))
        multibar_helper(ax,comparison_table.index, maxy)
        xs = np.array([ i for i in range(len(comparison_table.index))])
        if has_bounds: 
            nominal_bars = [*comparison_table['nominal','']]
            fault_bars = [*comparison_table[fault,'']]
        else:
            nominal_bars = [*comparison_table['nominal']]
            fault_bars = [*comparison_table[fault]]
        if stack:   bottom=nominal_bars
        else:       bottom=np.zeros(len(fault_bars))
        
        if error_bars:
            if not has_bounds: raise Exception("No bounds in the data to construct error bars out of")
            lower_nom_error = comparison_table['nominal', ''] - comparison_table['nominal', 'LB']
            upper_nom_error =  comparison_table['nominal', 'UB'] - comparison_table['nominal', '']
            yerror_nom=[[*lower_nom_error],[*upper_nom_error]]
            lower_error = comparison_table[fault, ''] - comparison_table[fault, 'LB']
            upper_error =  comparison_table[fault, 'UB'] - comparison_table[fault, '']
            yerror = [[*lower_error],[*upper_error]]
        else: yerror_nom=None; yerror=None
        
        plt.bar(xs,nominal_bars, tick_label=[str(i) for i in comparison_table.index], linewidth=4, fill=False, hatch='//', edgecolor='grey', label='nominal', yerr=yerror_nom, ecolor='grey', error_kw={'elinewidth':6})
        plt.bar(xs,fault_bars, tick_label=[str(i) for i in comparison_table.index], alpha=0.75, linewidth=4, label='fault scenarios', bottom=bottom, yerr=yerror, ecolor='red', error_kw={'elinewidth':2})
        if len(faults)>1:   
            if type(faults)==dict:      plt.title(faults[fault])
            else:                       plt.title(fault)
        elif title:         plt.title(title)
        if not (n-1)%columns:    
            ax.set_ylabel(stat)
        else: 
            ax.set_ylabel('')
            ax.axes.yaxis.set_ticklabels([])
        if (n-1) >= (rows-1)*columns: 
            if xlabel==True:    ax.set_xlabel(comparison_table.columns.name)
            else:               ax.set_xlabel(xlabel)
        if legend=='all': plt.legend()
        elif legend=='single' and n==1: plt.legend()
        elif legend==n:                 plt.legend() 
    figure.tight_layout(pad=0.3)
    if title and len(faults)>1:               figure.suptitle(title)
    return figure

def multibar_helper(ax, bar_index, maxy):
    """Shared plotting helper for resilience_factor_comparison and nominal_factor_comparison.
    Adds seperators to table groups (if any), limits the bounds of the plot, adds a grid, etc."""
    ax.set_ylim(top=maxy)
    plt.grid(axis='y')
    if 'MultiIndex' in str(type(bar_index)):
        if bar_index[0][1]=='': bar_index = [ind[0] for ind in bar_index][0:-1:3] # catches error bars
        if len(bar_index[0])>2: # color top-level categories
            first_inds = [i[0] for i in bar_index]
            reverse_inds = [i[0] for i in bar_index]
            reverse_inds.reverse()
            first_vals = set(first_inds)
            first_areas = {i:[first_inds.index(i), len(reverse_inds)-reverse_inds.index(i)] for i in first_vals}
            for i, area in enumerate(first_areas.values()):
                if i%2:
                    ax.axvspan(area[0]-0.5, area[1]-0.5, color="ivory", zorder=-2)
        if len(bar_index[0])>1: # put lines between mid-level categories
            second_inds = { i[0:len(bar_index[0])-1]:pos for pos,i in enumerate(bar_index)}
            second_inds = [*second_inds.values()]
            for i in second_inds[:-1]:
                ax.axvline(i+0.5, color='black')
    ax.set_xlim(-0.5, len(bar_index)-0.5)
