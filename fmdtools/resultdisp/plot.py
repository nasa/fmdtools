"""
File Name: resultdisp/plot.py
Author: Daniel Hulse
Created: November 2019 (Refactored April 2020)

Description: Plots quantities of interest over time using matplotlib.

Uses the following methods:
    - mdlhist:         plots function and flow histories over time (with different plots for each funciton/flow)
    - mdlhistvals:     plots function and flow histories over time on a single plot 
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

def mdlhist(mdlhist, fault='', time=0, fxnflows=[], returnfigs=False, legend=True, timelabel='Time', units=[]):
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
    fxnflows : list, optional
        List of functions and flows to plot. The default is [], which returns all.
    returnfigs: bool, optional
        Whether to return the figure objects in a list. The default is False.
    legend: bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    unitdict = dict(enumerate(units))
    z=0
    figs =[]
    objtypes = list(set(mdlhists['nominal'].keys()).difference({'time'}))
    for objtype in objtypes:
        for fxnflow in mdlhists['nominal'][objtype]:
            if fxnflows: #if in the list 
                if fxnflow not in fxnflows: continue
            
            if objtype =="flows":
                nomhist=mdlhists['nominal']["flows"][fxnflow]
                if 'faulty' in mdlhists: hist = mdlhists['faulty']["flows"][fxnflow]
            elif objtype=="functions":
                nomhist=copy.deepcopy(mdlhists['nominal']["functions"][fxnflow])
                if nomhist.get('faults',False): del nomhist['faults']
                if 'faulty' in mdlhists: 
                    hist = copy.deepcopy(mdlhists['faulty']["functions"][fxnflow])
                    del hist['faults']
            plots=len(nomhist)
            if plots:
                fig = plt.figure()
                figs = figs +[fig]
                if legend: fig.add_subplot(np.ceil((plots+1)/2),2,plots)
                else: fig.add_subplot(np.ceil((plots)/2),2,plots)
                
                plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])
                n=1
                for var in nomhist:
                    plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+var)
                    n+=1
                    if 'faulty' in mdlhists:
                        a, = plt.plot(times, hist[var], color='r')
                        c = plt.axvline(x=time, color='k')
                        b, =plt.plot(times, nomhist[var], ls='--', color='b')
                    else:
                        b, =plt.plot(times, nomhist[var], color='b')
                    plt.title(var)
                    plt.xlabel(timelabel)
                    plt.ylabel(unitdict.get(z, ''))
                    z+=1
                if 'faulty' in mdlhists:
                    fig.suptitle('Dynamic Response of '+fxnflow+' to fault'+' '+fault)
                    if legend:
                        ax_l = plt.subplot(np.ceil((plots+1)/2),2,n, label=fxnflow+'legend')
                        plt.legend([a,b],['faulty', 'nominal'], loc='center')
                        plt.box(on=None)
                        ax_l.get_xaxis().set_visible(False)
                        ax_l.get_yaxis().set_visible(False)
                plt.show()
    if returnfigs: return figs

def mdlhistvals(mdlhist, fault='', time=0, fxnflowvals={}, cols=2, returnfig=False, legend=True, timelabel="time", units=[]):
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
    cols: int, optional
        columns to use in the figure. The default is 2.
    returnfig: bool, optional
        Whether to return the figure. The default is False.
    legend: bool, optional
        Whether the plot should have a legend for faulty and nominal states. The default is true
        
    """
    mdlhists={}
    if 'nominal' not in mdlhist: mdlhists['nominal']=mdlhist
    else: mdlhists=mdlhist
    times = mdlhists["nominal"]["time"]
    
    unitdict = dict(enumerate(units))
    
    if fxnflowvals: num_plots = sum([len(val) for k,val in fxnflowvals.items()])
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
                plt.subplot(int(np.ceil((num_plots)/cols)),cols,n, label=fxnflow+var)
                n+=1
                if 'faulty' in mdlhists:
                    a, = plt.plot(times, hist[var], color='r')
                    c = plt.axvline(x=time, color='k')
                    b, =plt.plot(times, nomhist[var], ls='--', color='b')
                else:
                    b, =plt.plot(times, nomhist[var], color='b')
                plt.title(fxnflow+": "+var)
                plt.xlabel(timelabel)
                plt.ylabel(unitdict.get(n-2, ''))
    if 'faulty' in mdlhists:
        if fxnflowvals: fig.suptitle('Dynamic Response of '+str(list(fxnflowvals.keys()))+' to fault'+' '+fault)
        else:           fig.suptitle('Dynamic Response of Model States to fault'+' '+fault)
        if legend:
            ax_l = plt.subplot(np.ceil((num_plots+1)/cols),cols,n, label='legend')
            plt.legend([a,b],['faulty', 'nominal'], loc='center')
            plt.box(on=None)
            ax_l.get_xaxis().set_visible(False)
            ax_l.get_yaxis().set_visible(False)
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=1-0.05-0.15/(num_plots/cols))
    if returnfig: return fig
    else: plt.show()
    
def nominal_vals(app, endclasses, param1, param2, param3=0, title="Nominal Operational Envelope"):
    fig = plt.figure()
    
    data = [(x, scen['properties']['inputparams'][param1], scen['properties']['inputparams'][param2]) for x,scen in app.scenarios.items()\
            if (scen['properties']['inputparams'].get(param1,False) and scen['properties']['inputparams'].get(param2,False))]
    names = [d[0] for d in data]
    classifications = [endclasses[name]['classification'] for name in names] 
    discrete_classes = set(classifications)
    for cl in discrete_classes:
        xdata = [d[1] for i,d in enumerate(data) if classifications[i]==cl]
        ydata = [d[2] for i,d in enumerate(data) if classifications[i]==cl]
        plt.scatter(xdata, ydata, label=cl)
    plt.legend()
    plt.xlabel(param1)
    plt.ylabel(param2)
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


