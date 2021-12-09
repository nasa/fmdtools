"""
File Name: resultdisp/plot.py
Author: Daniel Hulse
Created: November 2019 (Refactored April 2020)

Description: Plots quantities of interest over time using matplotlib.

Uses the following methods:
    - mdlhist:         plots function and flow histories over time (with different plots for each funciton/flow)
    - mdlhistvals:     plots function and flow histories over time on a single plot 
    - nominal_vals_1d: plots the end-state classification of a system over a (1-D) range of nominal runs
    - nominal_vals_2d: plots the end-state classification of a system over a (2-D) range of nominal runs
    - nominal_vals_3d: plots the end-state classification of a system over a (3-D) range of nominal runs
    - phases:          plots the phases of operation that the model progresses through.
    - samplecost:      plots the costs for a single fault sampled by a SampleApproach over time with rates
    - samplecosts:     plots the costs for a set of faults sampled by a SampleApproach over time with rates on separate plots
    - costovertime:    plots the total cost/explected cost of a set of faults sampled by a SampleApproach over time
"""
import matplotlib.pyplot as plt
import copy
import numpy as np
from fmdtools.resultdisp.tabulate import costovertime as cost_table
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

def mdlhist(mdlhist, fault='', time=0, fxnflows=[],cols=2, returnfigs=False, legend=True, timelabel='Time', units=[], phases={}, modephases={}, label_phases=True):
    """
    Plots all states of the model at a time given a model history on separate plots.

    Parameters
    ----------
    mdlhist : dict
        History of states over time. Can be just the scenario states or a dict of scenario states and nominal states per {'nominal':nomhist,'faulty':mdlhist}
    fault : str, optional
        Name of the fault (for the title). The default is ''.
    time : float, optional
        Time of fault injection. The default is 0.
    fxnflows : list, optional
        List of functions and flows to plot. The default is [], which returns all.
    cols : int, optional
        columns to use in the figure. The default is 2.
    returnfigs: bool, optional
        Whether to return the figure objects in a list. The default is False.
    legend: bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true.
    timelabel : str, optional
        Label to use for the x-axes (e.g., seconds, minutes). Default is "time".
    units : dict, optional
        Labels to use for the y-axes (e.g., power, voltage). Default is ''
    phases : dict, optional
        Phase dictionary from process.modephases. Overlays lines over function values corresponding to the phase progression.
    modephases : dict, optional
        Modephase dictionary from process.modephases. Makes the phase overlay labels correspond to mode names instead of phases.
    label_phases : book, optional
        Whether to overlay labels on phases (or just leave lines)
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    figs = []
    if not fxnflows: fxnflows = {fxnflow:"all" for fxnflow in list(mdlhists['nominal']['functions'].keys())+list(mdlhists['nominal']['flows'].keys()) if any(mdlhists['nominal']['functions'].get(fxnflow, [])) or any(mdlhists['nominal']['flows'].get(fxnflow, []))}
    for fxnflow in fxnflows:
        fig = mdlhistvals(mdlhists.copy(), fault=fault, time=time, fxnflowvals={fxnflow:'all'}, cols=cols, returnfig=True, legend=legend, timelabel=timelabel, units=units, phases=phases, modephases=modephases, label_phases=label_phases)
        figs.append(fig)
    if returnfigs:
        return figs
def mdlhistvals(mdlhist, fault='', time=0, fxnflowvals={}, cols=2, returnfig=True, legend=True, timelabel="time", units=[], phases={}, modephases={}, label_phases=True):
    """
    Plots the states of a model over time given a history.

    Parameters
    ----------
    mdlhist : dict
        History of states over time. Can be just the scenario states or a dict of scenario states and nominal states per {'nominal':nomhist,'faulty':mdlhist}
    fault : str, optional
        Name of the fault (for the title). The default is ''.
    time : float, optional
        Time of fault injection. The default is 0.
    fxnflowsvals : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals]}. The default is {}, which returns all.
    cols : int, optional
        columns to use in the figure. The default is 2.
    returnfig : bool, optional
        Whether to return the figure. The default is False.
    legend : bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true
    timelabel : str, optional
        Label to use for the x-axes (e.g., seconds, minutes). Default is "time".
    units : dict, optional
        Labels to use for the y-axes (e.g., power, voltage). Default is ''
    phases : dict, optional
        Phase dictionary from process.modephases. Overlays lines over function values corresponding to the phase progression.
    modephases : dict, optional
        Modephase dictionary from process.modephases. Makes the phase overlay labels correspond to mode names instead of phases.
    label_phases : book, optional
        Whether to overlay labels on phases (or just leave lines)
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    if 'faulty' in mdlhist: f_times = mdlhists["faulty"]["time"]
    
    unitdict = dict(enumerate(units))

    if fxnflowvals: 
        all_vals = [f for f,v in fxnflowvals.items() if v=='all']
        for fxnflow in all_vals:
            if fxnflow in mdlhist['nominal']['functions']:  fxnflowvals[fxnflow]=list(mdlhist['nominal']['functions'][fxnflow].keys())
            elif fxnflow in mdlhist['nominal']['flows']:    fxnflowvals[fxnflow]=list(mdlhist['nominal']['flows'][fxnflow].keys())
        num_plots = sum([len(val) for k,val in fxnflowvals.items()]) + int(legend)
    else: 
        num_flow_plots = sum([len(flow) for flow in mdlhists['nominal']['flows'].values()])
        num_fxn_plots = sum([len([a for a in atts if a!='faults']) for fname, atts in mdlhists['nominal'].get('functions',{}).items()])
        num_plots = num_fxn_plots + num_flow_plots + int(legend)
    fig = plt.figure(figsize=(cols*3, 2*num_plots/cols))
    n=1
    objtypes = set(mdlhists['nominal'].keys()).difference({'time'})
    for objtype in objtypes:
        for fxnflow in mdlhists['nominal'][objtype]:
            if fxnflowvals: #if in the list 
                if fxnflow not in fxnflowvals: continue
            
            if objtype =="flows":
                nomhist=mdlhists['nominal']["flows"][fxnflow]
                if 'faulty' in mdlhists: hist = mdlhists['faulty']["flows"][fxnflow]
            elif objtype=="functions":
                nomhist=copy.deepcopy(mdlhists['nominal']["functions"][fxnflow])
                if len(nomhist.get('faults',[])) > 0:
                    if type(nomhist.get('faults',[]))!=np.ndarray: del nomhist['faults']
                if 'faulty' in mdlhists: 
                    hist = copy.deepcopy(mdlhists['faulty']["functions"][fxnflow])
                    if len(hist.get('faults',[])) > 0:
                        if type(hist.get('faults',[]))!=np.ndarray: del hist['faults']

            for var in nomhist:
                if fxnflowvals: #if in the list of values
                    if var not in fxnflowvals[fxnflow]: continue
                if var=='faults': continue 
                plt.subplot(int(np.ceil((num_plots)/cols)),cols,n, label=fxnflow+var)
                n+=1
                if 'faulty' in mdlhists:
                    a, = plt.plot(f_times[:len(hist[var])], hist[var], color='r')
                    c = plt.axvline(x=time, color='k')
                    b, =plt.plot(times, nomhist[var], ls='--', color='b')
                else:
                    b, =plt.plot(times, nomhist[var], color='b')
                if phases.get(fxnflow):
                    ymin, ymax = plt.ylim()
                    phaseseps = [i[0] for i in list(phases[fxnflow].values())[1:]]
                    plt.vlines(phaseseps,ymin, ymax, colors='gray',linestyles='dashed')
                    if label_phases:
                        for phase in phases[fxnflow]:
                            if modephases: phasetext = [m for m,p in modephases[fxnflow].items() if phase in p][0]
                            else: phasetext = phase
                            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", lw=0, alpha=0.5)
                            plt.text(np.average(phases[fxnflow][phase]), (ymin+ymax)/2, phasetext, ha='center', bbox=bbox_props)
                plt.title(fxnflow+": "+var)
                plt.xlabel(timelabel)
                plt.ylabel(unitdict.get(n-2, ''))
    if 'faulty' in mdlhists and any(nomhist):
        if fxnflowvals: 
            if len(fxnflowvals)==1: fig.suptitle('Dynamic Response of '+list(fxnflowvals.keys())[0]+' to fault'+' '+fault)
            else:                   fig.suptitle('Dynamic Response of '+str(list(fxnflowvals.keys()))+' to fault'+' '+fault)
        else:           fig.suptitle('Dynamic Response of Model States to fault'+' '+fault)
        if legend:
            ax_l = plt.subplot(int(np.ceil((num_plots)/cols)),cols,n, label='legend')
            plt.legend([a,b],['faulty', 'nominal'], loc='center')
            plt.box(on=None)
            ax_l.get_xaxis().set_visible(False)
            ax_l.get_yaxis().set_visible(False)
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=1-0.05-0.15/(num_plots/cols))
    if returnfig: return fig
    else: plt.show()

def aggregate_mdlhistvals(mdlhists, fxnflowvals, cols=2, aggragation='individual', legend_loc=-1):
    """
    Plot the aggregate behavior over time of the given function/flow values 
    over a set of nominal scenarios

    Parameters
    ----------
    mdlhists : dict
        Aggregate model history with structure {'scen':mdlhist}
    fxnflowsvals : dict, optional
        dict of flow values to plot with structure {fxnflow:[vals]}. The default is {}, which returns all.
    cols : int, optional
        columns to use in the figure. The default is 2.
    aggratation : str
        Way of aggregating the plot values. The default is 'individual'
    legend_loc : int/bool
        Location for the legend. If not legend, provide False. Default is -1 (the last plot)
    """
    
    plot_values = [(objname, objval) for objname in fxnflowvals for objval in fxnflowvals[objname]]
    num_plots = len(plot_values)
    rows = int(np.ceil(num_plots/cols))
    fig, axs = plt.subplots(cols, rows, sharex=True,figsize=(cols*3, 2*num_plots/cols)) 
    axs = axs.flatten()
    for i, plot_value in enumerate(plot_values):
        ax = axs[i]
        ax.set_title(' '.join(plot_value))
        for scen in mdlhists:
            if plot_value[0] in mdlhists[scen]['flows']:        ax.plot(mdlhists[scen]['flows'][plot_value[0]][plot_value[1]], label=scen)
            elif plot_value[0] in mdlhists[scen]['functions']:  ax.plot(mdlhists[scen]['functions'][plot_value[0]][plot_value[1]], label=scen)
    if legend_loc!=False: 
        if legend_loc==-1:  ax.legend(prop={'size': 8})
        else:               axs[legend_loc].legend(prop={'size': 8})
        


def nominal_vals_1d(app, endclasses, param1, title="Nominal Operational Envelope", nomlabel = 'nominal'):
    """
    Visualizes the nominal operational envelope along one given parameter

    Parameters
    ----------
    app : NominalApproach
        Nominal sample approach simulated in the model.
    endclasses : dict
        End-classifications for the set of simulations in the model.
    param1 : str
        Parameter range desired to visualize in the operational envelope
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nomlabel : str, optional
        Flag for nominal end-states. The default is 'nominal'.

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.

    """
    
    fig = plt.figure()
    
    data = [(x, scen['properties']['inputparams'][param1]) for x,scen in app.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False))]
    names = [d[0] for d in data]
    classifications = [endclasses[name]['classification'] for name in names] 
    discrete_classes = set(classifications)
    min_x = np.min([d[1] for i,d in enumerate(data)])
    max_x = np.max([d[1] for i,d in enumerate(data)])
    plt.hlines(1,min_x-1, max_x+1)
    
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        if nomlabel in cl:  plt.eventplot(xdata, label=cl, color='blue', alpha=0.5)
        else:               plt.eventplot(xdata, label=cl, color='red', alpha=0.5)
    plt.legend()
    plt.xlim(min_x-1, max_x+1)
    axis = plt.gca()
    axis.yaxis.set_ticklabels([])
    plt.xlabel(param1)
    plt.title(title)
    plt.grid(which='both', axis='x')
    return fig

def nominal_vals_2d(app, endclasses, param1, param2, title="Nominal Operational Envelope", nomlabel = 'nominal'):
    """
    Visualizes the nominal operational envelope along two given parameters

    Parameters
    ----------
    app : NominalApproach
        Nominal sample approach simulated in the model.
    endclasses : dict
        End-classifications for the set of simulations in the model.
    param1 : str
        First parameter (x) desired to visualize in the operational envelope
    param2 : str
        Second arameter (y) desired to visualize in the operational envelope
    title : str, optional
        Plot title. The default is "Nominal Operational Envelope".
    nomlabel : str, optional
        Flag for nominal end-states. The default is 'nominal'.

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure()
    
    data = [(x, scen['properties']['inputparams'][param1], scen['properties']['inputparams'][param2]) for x,scen in app.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False) and scen['properties']['inputparams'].get(param2,False))]
    names = [d[0] for d in data]
    classifications = [endclasses[name]['classification'] for name in names] 
    discrete_classes = set(classifications)
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        ydata = [d[2] for i,d in enumerate(data) if classifications[i]==cl]
        if nomlabel in cl:  plt.scatter(xdata, ydata, label=cl, marker="o")
        else:               plt.scatter(xdata, ydata, label=cl, marker="X")
    plt.legend()
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(title)
    plt.grid(which='both')
    return fig

def nominal_vals_3d(app, endclasses, param1, param2, param3, title="Nominal Operational Envelope", nomlabel = 'nominal'):
    """
    Visualizes the nominal operational envelope along three given parameters

    Parameters
    ----------
    app : NominalApproach
        Nominal sample approach simulated in the model.
    endclasses : dict
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

    Returns
    -------
    fig : matplotlib figure
        Figure for the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    data = [(x, scen['properties']['inputparams'][param1], scen['properties']['inputparams'][param2], scen['properties']['inputparams'][param3]) for x,scen in app.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False) and scen['properties']['inputparams'].get(param2,False)and scen['properties']['inputparams'].get(param3,False))]
    names = [d[0] for d in data]
    classifications = [endclasses[name]['classification'] for name in names] 
    discrete_classes = set(classifications)
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        ydata = [d[2] for i,d in enumerate(data) if classifications[i]==cl]
        zdata = [d[3] for i,d in enumerate(data) if classifications[i]==cl]
        if nomlabel in cl:  ax.scatter(xdata, ydata, zdata, label=cl, marker="o")
        else:               ax.scatter(xdata, ydata, zdata, label=cl, marker="X")
    ax.legend()
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel(param3)
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
             

def samplecost(app, endclasses, fxnmode, samptype='std', title=""):
    """
    Plots the sample cost and rate of a given fault over the injection times defined in the app sampleapproach
    
    (note: not currently compatible with joint fault modes)
    
    Parameters
    ----------
    app : sampleapproach
        Sample approach defining the underlying samples to take and probability model of the list of scenarios.
    endclasses : dict
        A dict with the end classification of each fault (costs, etc)
    fxnmode : tuple
        tuple (or tuple of tuples) with structure ('function name', 'mode name') defining the fault mode
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
    costs = np.array([endclasses[scen]['cost'] for scen in associated_scens])
    times = np.array([time  for phase, timemodes in app.sampletimes.items() if timemodes for time in timemodes if fxnmode in timemodes.get(time)] )  
    times = sorted(times)
    rates = np.array(list(app.rates_timeless[fxnmode].values()))
    
    tPlot, axes = plt.subplots(2, 1, sharey=False, gridspec_kw={'height_ratios': [3, 1]})
    
    phasetimes_start =[times[0] for phase, times in app.mode_phase_map[fxnmode].items()]
    phasetimes_end =[times[1] for phase, times in app.mode_phase_map[fxnmode].items()]
    ratetimes =[]
    ratesvect =[]
    phaselocs = []
    for (ind, phasetime) in enumerate(phasetimes_start):
        axes[0].axvline(phasetime, color="black")        
        phaselocs= phaselocs +[(phasetimes_end[ind]-phasetimes_start[ind])/2 + phasetimes_start[ind]]

        axes[1].axvline(phasetime, color="black") 
        ratetimes = ratetimes + [phasetimes_start[ind]] + [phasetimes_end[ind]]
        ratesvect = ratesvect + [rates[ind]] + [rates[ind]]
        #axes[1].text(middletime, 0.5*max(rates),  list(app.phases.keys())[ind], ha='center', backgroundcolor="white")
    #rate plots
    axes[1].set_xticks(phaselocs)
    axes[1].set_xticklabels([phasetup[1] for phasetup in app.mode_phase_map[fxnmode]])
    
    axes[1].plot(ratetimes, ratesvect)
    axes[1].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    axes[1].set_ylim(0, np.max(ratesvect)*1.2 )
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Time ("+str(app.units)+")")
    axes[1].grid()
    #cost plots
    axes[0].set_xlim(phasetimes_start[0], phasetimes_end[-1])
    axes[0].set_ylim(0, 1.2*np.max(costs))
    if samptype=='fullint':
        axes[0].plot(times, costs, label="cost")
    else:
        if samptype=='quadrature' or samptype=='pruned piecewise-linear': 
            sizes =  1000*np.array([weight if weight !=1/len(timeweights) else 0.0 for (phasetype, phase), timeweights in app.weights[fxnmode].items() if timeweights for time, weight in timeweights.items() if time in times])
            axes[0].scatter(times, costs,s=sizes, label="cost", alpha=0.5)
        axes[0].stem(times, costs, label="cost", markerfmt=",", use_line_collection=True)
    
    axes[0].set_ylabel("Cost")
    axes[0].grid()
    if title: axes[0].set_title(title)
    elif type(fxnmode[0])==tuple: axes[0].set_title("Cost function of "+str(fxnmode)+" over time")
    else:                       axes[0].set_title("Cost function of "+fxnmode[0]+": "+fxnmode[1]+" over time")
    #plt.subplot_adjust()
    plt.tight_layout()
def samplecosts(app, endclasses, joint=False, title=""):
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
    """
    for fxnmode in app.list_modes(joint):
        if any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='fullint' and fm==fxnmode]):
            st='fullint'
        elif any([True for (fm, phase), val in app.sampparams.items() if val['samp']=='quadrature' and fm==fxnmode]):
            st='quadrature'
        else: 
            st='std'
        samplecost(app, endclasses, fxnmode, samptype=st, title="")

def costovertime(endclasses, app, costtype='expected cost'):
    """
    Plots the total cost or total expected cost of faults over time.

    Parameters
    ----------
    endclasses : dict
        dict with rate,cost, and expected cost for each injected scenario (e.g. from run_approach())
    app : sampleapproach
        sample approach used to generate the list of scenarios
    costtype : str, optional
        type of cost to plot ('cost', 'expected cost' or 'rate'). The default is 'expected cost'.
    """
    costovertime = cost_table(endclasses, app)
    plt.plot(list(costovertime.index), costovertime[costtype])
    plt.title('Total '+costtype+' of all faults over time.')
    plt.ylabel(costtype)
    plt.xlabel("Time ("+str(app.units)+")")
    plt.grid()

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
    
    if '' in comparison_table.columns[0]: #bounded table
        bar = np.array([comparison_table.loc[metric,col[0]][''] for col in comparison_table.columns if col[1]==''])
        labels= [str(i[0]) for i in comparison_table.columns if i[1]=='']
        if error_bars:
            UB = np.array([comparison_table.loc[metric,col[0]]['UB'] for col in comparison_table.columns if col[1]=='UB'])
            LB = np.array([comparison_table.loc[metric,col[0]]['LB'] for col in comparison_table.columns if col[1]=='LB'])
            yerr= [bar-LB, UB-bar]
            if maxy=='max': maxy = comparison_table.loc[metric].max()
        else: 
            yerr=[]
            if maxy=='max': maxy = max(bar)
    else:  
        bar = [*comparison_table.loc[metric]]; yerr=[]; labels=[str(i) for i in comparison_table.columns]
        if maxy=='max': maxy = max(bar)
    
    ax = figure.add_subplot(1,1,1)
    
    plt.grid(axis='y')
    ax.set_ylabel(ylabel)
    ax.set_ylim(top=maxy)
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
        if has_bounds: faults=list({f[0] for f in comparison_table})
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
        ax.set_ylim(top=maxy)
        xs = np.array([ i for i in range(len(comparison_table.index))])
        if has_bounds: 
            nominal_bars = [*comparison_table['nominal','']]
            fault_bars = [*comparison_table[fault,'']]
        else:
            nominal_bars = [*comparison_table['nominal']]
            fault_bars = [*comparison_table[fault]]
        if stack:   bottom=fault_bars
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
        plt.grid(axis='y')
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
    figure.tight_layout(pad=0.3)
    if title and len(faults)>1:               figure.suptitle(title)
    return figure
    


