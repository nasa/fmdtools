# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:17:31 2020

@author: danie
"""

import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import time
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns; sns.set(style="ticks", color_codes=True)

from drone_mdl import *
import time

params={'start': [0.0,0.0, 10, 10], 'target': [0, 150, 160, 160], 'safe': [0, 50, 10, 10], # areas
        'loc':'rural',
        'flightplan':{ 1:[0,0,50], 2:[100, 200, 50], 3:[100, 100, 85], 4:[-25, 150, 20],5:[75, 300, 20],6:[0, 300, 20], 7:[0,0,50], 8:[0,0,0] },
        'bat':'series-split',                           #'monolithic', 'series-split', 'paralel-split', 'split-both'
        'linearch':'quad',                              #quad, hex, oct
        'respolicy':{'bat':'emland','line':'emland'},   #continue, to_home, to_nearest, emland
        'landtime':12}                                  #time when the drone lands (get from nominal simulation)
mdl = Drone(params=params)

# Design Model
def calc_des(mdl):
    batcostdict = {'monolithic':0, 'series-split':300, 'parallel-split':300, 'split-both':600}
    linecostdict = {'quad':0, 'hex':1000, 'oct':2000}
    descost = batcostdict[mdl.params['bat']] + linecostdict[mdl.params['linearch']]
    return descost
def x_to_dcost(xdes):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    batcostdict = {'monolithic':0, 'series-split':300, 'parallel-split':300, 'split-both':600}
    linecostdict = {'quad':0, 'hex':1000, 'oct':2000}
    descost = batcostdict[bats[xdes[0]]] + linecostdict[linarchs[xdes[1]]]
    return descost

# Operations Model
# Obj - flight time
# Constraints   - batteries stay above 20% (to avoid damage)
#               - no faults at end of simulation
#               - cannot fly above 122 m (400 ft)
def find_landtime(mdlhist):
    return min([i for i,a in enumerate(mdlhist['functions']['Planpath']['mode']) if a=='taxi']+[15])
def calc_oper(mdl):
    
    endresults_nom, resgraph, mdlhist =propagate.nominal(mdl)
    opercost = endresults_nom['classification']['expected cost']
    g_soc = 20 - mdlhist['functions']['StoreEE']['soc'][-1] 
    g_faults = any(endresults_nom['faults'])
    g_max_height = sum([i for i in mdlhist['flows']['DOFs']['elev']-122 if i>0])
    
    landtime = find_landtime(mdlhist)
    mdl.params['landtime']=landtime
    mdl.phases['forward'][1] = landtime
    mdl.phases['taxis'][0] = landtime
    return opercost, g_soc, g_faults, g_max_height
def x_to_ocost(xdes, xoper, loc='rural'):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(xoper[0], sq, start[0:2]+[0])
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':'continue','line':'continue'}, 'target':target,'safe':safe,'start':start, 'loc':loc, 'landtime':12}
    mdl = Drone(params=params)
    return calc_oper(mdl)

# Resilience Model
def calc_res(mdl, fullcosts=False, faultmodes = 'all'):
    app = SampleApproach(mdl, faults='single-component', phases={'forward'})   
    if faultmodes == 'battery':     app.scenlist = [scen for scen in app.scenlist if list(scen['faults'].keys())[0]=='StoreEE']
    elif faultmodes == 'line':      app.scenlist = [scen for scen in app.scenlist if list(scen['faults'].keys())[0]=='AffectDOF']
    elif faultmodes == 'notvars':   app.scenlist = [scen for scen in app.scenlist if list(scen['faults'].keys())[0] not in {'StoreEE', 'AffectDOF'}]
    
    
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    if fullcosts: 
        rescosts = {'cost':0,'repcost':0, 'landcost':0, 'safecost':0, 'lost value':0}
        for scen in endclasses:
            number = endclasses[scen]['expected cost']/endclasses[scen]['cost']
            rescosts['cost'] += endclasses[scen]['expected cost']
            rescosts['repcost'] += number *  endclasses[scen]['repcost']
            rescosts['landcost'] += number *  endclasses[scen]['landcost']
            rescosts['safecost'] += number *  endclasses[scen]['safecost']
            rescosts['lost value'] += number * -endclasses[scen]['viewed value']
        return rescosts
    else: return rescost
def x_to_rcost(xdes, xoper, xres, loc='rural', fullcosts=False, faultmodes = 'all'):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(xoper[0], sq, start[0:2]+[0])
    
    params = {'bat':bats[xdes[0]], 'linearch':linarchs[xdes[1]], 'flightplan':fp, 'respolicy':{'bat':respols[xres[0]],'line':respols[xres[1]]}, 'target':target,'safe':safe,'start':start,'loc':loc, 'landtime':12 }
    mdl = Drone(params=params)
    a,b,c,d = calc_oper(mdl) #used to form flight phases
    return calc_res(mdl, fullcosts=fullcosts, faultmodes = faultmodes)

#creates model from design variables (note: does not get flight time)
def x_to_mdl(x, loc='rural'):
    bats = ['monolithic', 'series-split', 'parallel-split', 'split-both']
    linarchs = ['quad', 'hex', 'oct']
    respols = ['continue', 'to_home', 'to_nearest', 'emland']
    #start locs
    target = [0, 150, 160, 160]
    safe = [0, 50, 10, 10]
    start = [0.0,0.0, 10, 10]
    
    sq = square(target[0:2],target[2],target[3])
    fp = plan_flight(x[2], sq, start[0:2]+[0])
    
    params = {'bat':bats[x[0]], 'linearch':linarchs[x[1]], 'flightplan':fp, 'respolicy':{'bat':respols[x[3]],'line':respols[x[4]]}, 'target':target,'safe':safe,'start':start,'loc':loc, 'landtime':12}
    mdl = Drone(params=params)
    return mdl
# all-in-one-model
def x_to_cost(x, loc='rural', fullcosts=False):
    mdl = x_to_mdl(x, loc=loc)
    dcost = calc_des(mdl)
    oper = calc_oper(mdl)
    rcost = calc_res(mdl, fullcosts=fullcosts)
    if fullcosts:   return dcost, oper, rcost
    else:           return dcost + oper[0] + rcost, oper[1:]



def plot_nomtraj(mdlhist, params, title='Trajectory'):
    xnom=mdlhist['flows']['DOFs']['x']
    ynom=mdlhist['flows']['DOFs']['y']
    znom=mdlhist['flows']['DOFs']['elev']
    
    time = mdlhist['time']
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title(title)
    plt.show()

def plot_faulttraj(mdlhist, params):
    xnom=mdlhist['nominal']['flows']['DOFs']['x']
    ynom=mdlhist['nominal']['flows']['DOFs']['y']
    znom=mdlhist['nominal']['flows']['DOFs']['elev']
    #
    x=mdlhist['faulty']['flows']['DOFs']['x']
    y=mdlhist['faulty']['flows']['DOFs']['y']
    z=mdlhist['faulty']['flows']['DOFs']['elev']
    
    time = mdlhist['nominal']['time']
    
    
    fig2 = plt.figure()
    
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim3d(-50, 200)
    ax2.set_ylim3d(-50,200)
    ax2.set_zlim3d(0,100)
    ax2.plot(xnom,ynom,znom)
    ax2.plot(x,y,z)

    for xx,yy,zz,tt in zip(xnom,ynom,znom,time):
        if tt%20==0:
            ax2.text(xx,yy,zz, 't='+str(tt), fontsize=8)
    
    for goal,loc in params['flightplan'].items():
        ax2.text(loc[0],loc[1],loc[2], str(goal), fontweight='bold', fontsize=12)
        ax2.plot([loc[0]],[loc[1]],[loc[2]], marker='o', markersize=10, color='red', alpha=0.5)
    
    ax2.set_title('Fault response to RFpropbreak fault at t=20')
    ax2.legend(['Nominal Flightpath','Faulty Flighpath'], loc=4)
    #
    plt.show()
    
def plot_xy(mdlhist, endresults, title='', retfig=False, legend=False):
    plt.figure()
    plot_one_xy(mdlhist, endresults)
    
    plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue', label='Starting Area')
    plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red', label='Target Area')
    plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow', label='Emergency Landing Area')
    
    plt.title(title)
    if legend: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if retfig:  return plt.gcf(), plt.gca()
    else:       plt.show()
def plot_one_xy(mdlhist,endresults, retfig=False):
    xnom=mdlhist['flows']['DOFs']['x']
    ynom=mdlhist['flows']['DOFs']['y']
    znom=mdlhist['flows']['DOFs']['elev']
    
    plt.plot(xnom,ynom)
    
    
    xviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    yviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view!='unviewed']
    xunviewed = [x for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    yunviewed = [y for (x,y),view in endresults['classification']['viewed'].items() if view=='unviewed']
    
    plt.scatter(xviewed,yviewed, color='red', label='Viewed')
    plt.scatter(xunviewed,yunviewed, color='grey', label='Unviewed')
    if retfig: return plt.gca(), plt.gcf()
    
def plot_xys(mdlhists, endresultss, cols=2, title='', retfig=False, legend=False):
    
    num_plots = len(mdlhists)
    fig, axs = plt.subplots(nrows=int(np.ceil((num_plots)/cols)), ncols=cols, figsize=(cols*6, 5*num_plots/cols))
    n=1
    
    for paramlab, mdlhist in mdlhists.items():
        plt.subplot(int(np.ceil((num_plots)/cols)),cols,n, label=paramlab)
        a, _= plot_one_xy(mdlhist, endresultss[paramlab],retfig=True)
        b= plt.fill([x[0] for x in mdl.start_area],[x[1] for x in mdl.start_area], color='blue', label='Starting Area')
        c=plt.fill([x[0] for x in mdl.target_area],[x[1] for x in mdl.target_area], alpha=0.2, color='red', label='Target Area')
        d=plt.fill([x[0] for x in mdl.safe_area],[x[1] for x in mdl.safe_area], color='yellow', label='Emergency Landing Area')
        plt.title(paramlab)
        n+=1
    plt.suptitle(title)
    if legend: 
        plt.subplot(np.ceil((num_plots+1)/cols),cols,n, label='legend')
        plt.axis('off')
        legend_elements = [Line2D([0], [0], color='b', lw=1, label='Flightpath'),
                   Line2D([0], [0], marker='o', color='r', label='Viewed',
                          markerfacecolor='r', markersize=8),
                   Line2D([0], [0], marker='o', color='grey', label='Unviewed',
                          markerfacecolor='grey', markersize=8),
                   Patch(facecolor='red', edgecolor='red', alpha=0.2,
                         label='Target Area'),
                   Patch(facecolor='blue', edgecolor='blue',label='Landing Area'),
                   Patch(facecolor='yellow', edgecolor='yellow',label='Emergency Landing Area')]
        plt.legend( handles=legend_elements, loc='center')
    plt.subplots_adjust(top=1-0.05-0.05/(num_plots/cols))
    
    if retfig:  return fig
    else:       plt.show()
    

def explore_tradoffs(loc='rural', filename=''):
    """ Below Code is to conduct a pre-optimization analysis of the impact of design, operational and resilience models
    # with the design of battery, rotors (inceasing complexity) and operable heights at each of the resilience policy.
    # This will help to observe the trends of the model functions with the design variables and to see the overall
    # feasible design space. I (Arpan) will document it in Jupyter note after the analysis.
    
    # # The arguments of the Operational model changes for the sake of the analysis, thus I have kept the models,
    # rather than importing from the original file. The models in the original files will be used in optimization framework
    # It is better to separate this file (model analysis) with the original model and optimization file, since
    # this file is for analysis purpose only and is subject to frequent changes in model arguments.
    
    # # The file currently takes about 25-30 mins to run. The whole analysis will be saved in "Cost_eval.csv".
    # To save time on running the file again, you can look into the already saved .csv file instead
    """
        #################################################################################################
    ## Creating test design grid matrix
    xbat = pd.Series([0, 1, 2, 3])  # 4 choices
    xrot = pd.Series([0, 1, 2])  # 3 choices
    xh = pd.Series(np.arange(10, 122, 10))  # Discritized into 12 different heights for computational efficiency
    xrespolbat = pd.Series([0, 1, 2, 3])
    xrespolrot = xrespolbat
    
    # Generating grid matrix with each combination of design variables (bat, rotor and height) at any fixed level of resilience policy
    xbat_grid = pd.concat([pd.concat([xbat.repeat(xrot.size * xh.size)] * 1, ignore_index=True)] * xrespolbat.size,
                          ignore_index=True)
    xrot_grid = pd.concat([pd.concat([xrot.repeat(xh.size)] * xbat.size, ignore_index=True)] * xrespolbat.size,
                          ignore_index=True)
    xh_grid = pd.concat([pd.concat([xh] * (xbat.size * xrot.size), ignore_index=True)] * xrespolbat.size, ignore_index=True)
    xrespolbat_grid = pd.concat([xrespolbat.repeat(xbat.size * xrot.size * xh.size)] * 1, ignore_index=True)
    xrespolrot_grid = xrespolbat_grid
    col_names = ["Bat", "Rotor", "Height", "ResPolBat", "ResPolRot"]
    x_mat = pd.concat([xbat_grid, xrot_grid, xh_grid, xrespolbat_grid, xrespolrot_grid], axis=1, ignore_index=True)
    x_mat = pd.DataFrame(x_mat.values, columns=col_names)
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(x_mat)
    ## Initializing dataframe for storing respective cost values and constraint validation for each row of x_mat (design choices)
    Cost_eval = pd.DataFrame(columns=['desC', 'operC', 'resC', 'c1', 'c2', 'c3', 'c_cum'])
    # Evaluating design, operational and resilience models at the design grid points
    for ix in range(len(x_mat)):
        xdes = [x_mat.iloc[ix, 0], x_mat.iloc[ix, 1]]
        xoper = [x_mat.iloc[ix, 2]]
        xres = [x_mat.iloc[ix, 3], x_mat.iloc[ix, 4]]
        desC = x_to_dcost(xdes)  # Calling design model
        operC = x_to_ocost(xdes, xoper)  # Calling operational model
        resC = x_to_rcost(xdes, xoper, xres)  # Calling failure model
        Cost_eval.loc[ix, ['desC']] = desC
        Cost_eval.loc[ix, ['operC']] = operC[0]
        Cost_eval.loc[ix, ['resC']] = resC
        # Constraints validation at each design grid points: 1=violation, 0=no violation
        if operC[1] > 0:  # Violation: batteries below 20% (to avoid damage)
            Cost_eval.loc[ix, ['c1']] = 1
        else:
            Cost_eval.loc[ix, ['c1']] = 0
    
        if operC[2] == True:  # Violation: faults at end of simulation
            Cost_eval.loc[ix, ['c2']] = 1
        else:
            Cost_eval.loc[ix, ['c2']] = 0
    
        if operC[3] > 0:  # Violation: fly above 122 m
            Cost_eval.loc[ix, ['c3']] = 1
        else:
            Cost_eval.loc[ix, ['c3']] = 0
    
        if ((operC[1] > 0 or operC[2] == True) or (operC[3] > 0)):  # Infeasible design if any above constraints violated
            Cost_eval.loc[ix, ['c_cum']] = 1
        else:
            Cost_eval.loc[ix, ['c_cum']] = 0
    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(Cost_eval)
    # print(feasible_DS)
    # Build a large dataset with the design choice and the respective costs and constraint validation
    grid_results = pd.concat([x_mat, Cost_eval], axis=1, ignore_index=True)
    grid_results.columns = ['Bat', 'Rotor', 'Height', 'ResPolBat', 'ResPolRot','desC', 'operC', 'resC', 'c1', 'c2', 'c3', 'c_cum']
    # Saving the results in .csv file
    # We will work on this dataset for plotting (examples provided in the trade-off_Analysis_plot.py)
    # No need to rerun the file unless we change the cost models
    if filename:    grid_results.to_csv(filename, index=False)
    else:           grid_results.to_csv('grid_results'+loc+'.csv', index=False)
    return grid_results

def plot_tradeoffs(filename, retfig=True):
    grid_results= pd.read_csv(filename)
    #print(grid_results.head())
    #print(grid_results.shape)
    
    # Portion of feasible data among the whole dataset
    feasible_DS =(grid_results['c_cum'].isin([0]).sum())/len(grid_results)
    #print("The portion of feasible design space from the grid results")
    #print(feasible_DS)
    
    #Subsetting only feasible data
    grid_results_FS = grid_results[(grid_results['c_cum']==0)]
    g = sns.pairplot(grid_results_FS, hue="ResPolBat", vars=["Bat", "Rotor","Height","desC","operC","resC"], corner=True, diag_kind="kde",kind="reg")
    if retfig:  return g
    else:       plt.show()
    

def AAO_f(X, *ulparams):
    """AAO objective function for use in multiobjective optimization """
    xdes = [int(X[0]),int(X[1])];  xoper = [X[2]]; xres = [int(X[3]), int(X[4])]
    desC0, operC0, resC0,normalize, w1, w2,u1,u2,loc  = ulparams
    #find costs
    desC = x_to_dcost(xdes)
    operC = x_to_ocost(xdes, xoper, loc=loc)
    resC = x_to_rcost(xdes, xoper, xres, loc=loc)
    
    #Constraints validation: >0 or 1(Boolean) means violation and is penalized in Obj Func
    c_batlife = operC[1]; c_faults = int(operC[2]); c_maxh = operC[3]
    #Penalizing obj function with upper level contraints
    pen = max(c_batlife, 0)**2+100000*c_faults+max(c_maxh,0)**2
    
    if normalize: #Normalizing obj function to avoid issue on magnitudes
        ndesC = (desC-desC0[0])/(desC0[1]-desC0[0])
        noperC =(operC[0]-operC0[0])/(operC0[1]-operC0[0])
        nresC = (resC - resC0[0]) / (resC0[1] - resC0[0])
        #Penalized weighted Tchebycheff obj func.
        obj = max((w1*(ndesC + noperC) -u1), (w2*nresC- u2)) + ULpen #normalized cost with penalty value
    else: obj = desC + operC + resC + pen
    return obj


def bistage_optimization(loc='rural', printresults=True, normalize = False, finish=None, desC0 = [0, 300000],operC0 = [-630000, -37171.5989], resC0 = [5245622.35, 310771934] ):
    """
    The two-stage optimization framework for the drone model.
    Stage 1 considers the nominal scenario, optimizing battery, rotor config and oper height,
    minimizing design and operational cost.
    Stage 2 considers simulated fault environment, optimizing resilience policy, minimizing failure cost.
    Overall, it is a combinatorial optimization problem, with upper level is MIP (Discrete battery and rotor choices;
    continuous height variable) and lower level is IP (Discrete resilience policy)
    Grid search brute force is used as optimization algo and handling constraints with Penalty method
    Because of using grid search, the Stage 1 height variable has been discritized to gain computational efficiency at
    the expense of accuracy.
    
    (optional) arguments are:
        loc:                    urban/rural/conjested (scenario for the optimization to run over)
        printresults:           bool (whether to print results)
        normalize:              T/F (whether to normalize over utopia values)
        desC0, operC0, resC0    [max, min] (max/min feasible values for design/oper/resilience normalization)
        
    """
    starttime = time.time()
    ######################### Stage 1 optimization ####################################
    # Initializing design variables and parameters
    ULXbound=(slice(0, 4, 1), slice(0, 3, 1), slice(10, 122, 10))    
    ulparams = (desC0, operC0, resC0, normalize, loc)
    
    # Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
    ULoptmodel = optimize.brute(bistage_firststageobj, ULXbound, args=ulparams, full_output=True, finish=finish)
    UL_xopt = abs(np.around(ULoptmodel[0])); UL_fopt = np.around(ULoptmodel[1], decimals= 4)
    xdes_opt = [int(UL_xopt[0]), int(UL_xopt[1])]; xoper_opt = [UL_xopt[2]]                 #get optimal design values   
    desC_opt = x_to_dcost(xdes_opt); operC_opt = x_to_ocost(xdes_opt, xoper_opt, loc=loc)   #get objective values
    num_upper = ULoptmodel[2][0].size
    #################### Stage 2 optimization##############################################
    #Running Stage 2 model for the upper level optimal solution
    LL_opt = bistage_secondstagemodel(xdes_opt, xoper_opt, resC0, normalize, loc)
    LL_xopt = abs(LL_opt[0]);  LL_fopt = np.around(LL_opt[1], decimals= 4)
    xres_opt = [int(LL_xopt[0]), int(LL_xopt[1])]                   #get optimal design values
    resC_opt = x_to_rcost(xdes_opt, xoper_opt, xres_opt, loc=loc)   #get optimal resilience
    num_lower = LL_opt[2][0].size
    if printresults:
        print("#####################Two-Stage approach###############################")
        print("Stage 1 optimal solution:")
        print(UL_xopt); print(UL_fopt); print(desC_opt); print(operC_opt)
        print("#####################################################################")
        print("Stage 2 optimal solution:")
        print(LL_xopt); print(LL_fopt); print(resC_opt)
    return xdes_opt, xoper_opt, xres_opt, desC_opt, operC_opt, resC_opt, num_upper, num_lower, time.time()- starttime
def bistage_firststageobj(X, *ulparams):
    """Objective function for first stage of the two-stage optimization"""
    xdes = [int(X[0]),int(X[1])]; xoper = [X[2]]    #get variables values
    desC0, operC0, resC0, normalize, loc = ulparams                 #get params
    # execute design + operational models
    desC = x_to_dcost(xdes)
    operC = x_to_ocost(xdes, xoper, loc=loc)
    #Constraints validation: >0 or 1(Boolean) means violation and is penalized in Obj Func
    c_batlife = operC[1]; c_faults = int(operC[2]); c_maxh = operC[3]
    #Penalizing obj function with upper level contraints
    ULpen = max(c_batlife, 0)**2+10000*c_faults+max(c_maxh,0)**2
    #Normalizing obj function to avoid issue on magnitudes
    ndesC = (desC-desC0[0])/(desC0[1]-desC0[0])
    noperC =(operC[0]-operC0[0])/(operC0[1]-operC0[0])
    ULobj = ndesC + noperC + ULpen #normalized design and oper cost with penalty value
    return ULobj
def bistage_secondstageobj(ll_x, *llparams):
    """Objective function for second stage of the two-stage optimization"""
    xdes, xoper, resC0, normalize, loc = llparams
    xres = ll_x
    resC = x_to_rcost(xdes, xoper, xres, loc=loc)
    if normalize:   LLobj = (resC-resC0[0])/(resC0[1]-resC0[0])
    else:           LLobj = resC
    return LLobj
def bistage_secondstagemodel(xdes, xoper, resC0, normalize,loc):
    """ Defining the Stage2 optimization model (Using brute force exhaustive grid search algortihm)"""
    LLXbound = (slice(0, 4, 1), slice(0, 4, 1))
    llparams = (xdes, xoper, resC0, normalize, loc)
    LLoptmodel = optimize.brute(bistage_secondstageobj, LLXbound, args=llparams, full_output=True, finish=None)
    return LLoptmodel


def bilevel_optimization(loc='rural', printresults=True, normalize = False, finish=None, decomp = False, desC0 = [0, 300000],operC0 = [-630000, -37171.5989], resC0 = [5245622.35, 310771934] ):
    """
    Bi-level optimization framework for the drone model.
    
    Upper level considers the nominal scenario, optimizing battery, rotor config and oper height,
    minimizing design and operational cost.
    Lower level considers simulated fault environment, optimizing resilience policy, minimizing failure cost.
    Overall, it is a combinatorial optimization problem, with upper level is MIP (Discrete battery and rotor choices;
    continuous height variable) and lower level is IP (Discrete resilience policy)
    Grid search brute force is used as optimization algo and handling constraints with Penalty method
    Because of using grid search, the upper level height variable has been discritized to gain computational efficiency at
    the expense of accuracy

    (optional) arguments are:
        loc:                    urban/rural/conjested (scenario for the optimization to run over)
        printresults:           bool (whether to print results)
        normalize:              T/F (whether to normalize over utopia values)
        desC0, operC0, resC0    [max, min] (max/min feasible values for design/oper/resilience normalization)
    """
    starttime = time.time()
    # Initializing design variables and parameters
    ULXbound=(slice(0, 4, 1), slice(0, 3, 1), slice(10, 122, 10))  
    LL_progress = {'num_iters':0, 'x_opt':[], 'c_r_opt':np.inf, 'c_tot_opt':np.inf}
    ulparams = (desC0, operC0, resC0, normalize, loc,LL_progress, decomp)
    # Brute force algo with polishing optimal results of brute force using downhill simplex algorithm
    ULoptmodel = optimize.brute(bilevel_upperlevelobj, ULXbound, args=ulparams, full_output=True, finish=finish)
    UL_xopt = abs(np.around(ULoptmodel[0])); UL_fopt = np.around(ULoptmodel[1], decimals= 4)
    xdes_opt = [int(UL_xopt[0]), int(UL_xopt[1])]; xoper_opt = [UL_xopt[2]]
    desC_opt = x_to_dcost(xdes_opt); operC_opt = x_to_ocost(xdes_opt, xoper_opt, loc=loc)
    num_upper = ULoptmodel[2][0].size
    #Getting best design, cost, iterations from lower level
    resC_opt = LL_progress['c_r_opt']; xres_opt = LL_progress['x_opt']; num_lower = LL_progress['num_iters']
    resC_opt = x_to_rcost(xdes_opt, xoper_opt, xres_opt, loc=loc)
    if printresults:
        print("#####################Bi-level approach###############################")
        print("Upper level optimal solution:")
        print(UL_xopt); print(UL_fopt); print(desC_opt); print(operC_opt)
        print("#####################################################################")
        print("Lower level optimal solution:")
        print(xres_opt); print(resC_opt)
    return xdes_opt, xoper_opt, xres_opt, desC_opt, operC_opt, resC_opt, num_upper, num_lower, time.time() - starttime
def bilevel_lowerlevelobj(ll_x, *llparams):
    """Lower-level objective function in bilevel framework."""
    xdes, xoper, resC0, normalize,loc, faultmodes = llparams
    xres = ll_x
    resC = x_to_rcost(xdes, xoper, xres, loc=loc, faultmodes = faultmodes)
    if normalize:   LLobj = (resC-resC0[0])/(resC0[1]-resC0[0])
    else:           LLobj = resC
    return LLobj
def bilevel_lowerlevelmodel(xdes, xoper, resC0, normalize,loc, decomp):
    """Defining the lower level optimization model (Using brute force exhaustive grid search algortihm)"""
    if decomp == False:
        faultmodes = 'all'
        LLXbound = (slice(0, 4, 1), slice(0, 4, 1))
        llparams = (xdes, xoper, resC0, normalize,loc, faultmodes) 
        LL_opt = optimize.brute(bilevel_lowerlevelobj, LLXbound, args=llparams, full_output=True, finish=None)
        LL_x_opt = [int(LL_opt[0][0]),int(LL_opt[0][1])]; LL_obj_opt = LL_opt[1]
        num_iters = LL_opt[2][0].size
    else:
        # battery optimization
        faultmodes = 'battery'
        LLXbound = (slice(0, 4, 1), slice(0, 1, 1))
        llparams = (xdes, xoper, resC0, normalize,loc, faultmodes) 
        LL_opt = optimize.brute(bilevel_lowerlevelobj, LLXbound, args=llparams, full_output=True, finish=None)
        opt_bat = LL_opt[0][0]; bat_cost = LL_opt[1]
        num_iters = LL_opt[2][0].size
        # line optimization
        faultmodes = 'line'
        LLXbound = (slice(0, 1, 1), slice(0, 4, 1))
        llparams = (xdes, xoper, resC0, normalize,loc, faultmodes) 
        LL_opt = optimize.brute(bilevel_lowerlevelobj, LLXbound, args=llparams, full_output=True, finish=None)
        opt_line= LL_opt[0][1]; line_cost = LL_opt[1]
        num_iters += LL_opt[2][0].size
        # getting residual costs
        LL_x_opt = [int(opt_bat), int(opt_line)]
        faultmodes = 'notvars'
        llparams = (xdes, xoper, resC0, normalize,loc, faultmodes)
        rest_cost = bilevel_lowerlevelobj(LL_x_opt, *llparams) 
        LL_obj_opt = bat_cost + line_cost + rest_cost
    return LL_x_opt, LL_obj_opt, num_iters
def bilevel_upperlevelobj(X, *ulparams):
    """ upper level objective function in the bilevel framework """
    xdes = [int(X[0]),int(X[1])]; xoper = [X[2]]
    desC0, operC0, resC0, normalize, loc, LL_progress, decomp = ulparams
    desC = x_to_dcost(xdes)
    operC = x_to_ocost(xdes, xoper, loc=loc)
    #Constraints validation: >0 or 1(Boolean) means violation and is penalized in Obj Func
    #Constraints validation: >0 or 1(Boolean) means violation and is penalized in Obj Func
    c_batlife = operC[1]; c_faults = int(operC[2]); c_maxh = operC[3]
    #Penalizing obj function with upper level contraints
    if ((operC[1] > 0 or operC[2] == True) or (operC[3] > 0)):  # Infeasible design if any above constraints violated
        ULpen = 100000 * max(c_batlife, 0)**2+100000*c_faults+100000 *max(c_maxh,0)**2 # Exterior Penalty method
        LLpen = 1000000 # Giving a big penalty of lower level if upper level decision is infeasible
        LL_x_opt=[0,0];
    else: # Calling lower level only if all the upper level contraints are feasible: Reducing redundant lower level iterations
        ULpen = 0
        LL_x_opt, LL_obj_opt, num_iters  = bilevel_lowerlevelmodel(xdes, xoper, resC0, normalize,loc, decomp)
        
        LLpen = 1*LL_obj_opt # with increasing penalty term, optimal design decision is provided with lower risk to failure
        LL_progress['num_iters'] += num_iters
    #Penalized obj func.(both upper and lower level): Double Penalty method
    if normalize: #Normalizing obj function to avoid issue on magnitudes
        ndesC = (desC-desC0[0])/(desC0[1]-desC0[0])
        noperC =(operC[0]-operC0[0])/(operC0[1]-operC0[0])
        ULobj = ndesC + noperC + ULpen + LLpen #normalized design and oper cost with penalty value
    else: ULobj = desC + operC[0] + ULpen + LLpen
    if ULobj < LL_progress['c_tot_opt']: 
        LL_progress['x_opt']=LL_x_opt; LL_progress['c_r_opt']=LLpen; LL_progress['c_tot_opt'] = ULobj
    return ULobj

def brute_search(loc = 'rural', Xranges = [[0,4,1],[0,3,1],[10, 130, 10],[0,4,1],[0,4,1]]):
    starttime = time.time()
    Xvals = [ e for e in itertools.product(*(range(x[0],x[1],x[2]) for x in Xranges))]
    results = dict(); opt_hist = []
    
    for X in Xvals:
        dcost, ocost, rcosts = x_to_cost(X, loc=loc, fullcosts=True)
        if (ocost[1]<=0) & (not ocost[2]) & (ocost[3] <= 0):
            results[X] = [dcost, ocost[0]]+list(rcosts.values())
        totalcost = dcost + ocost[0] + rcosts['cost']
        if not opt_hist:                     opt_hist= [[totalcost, X]]
        elif totalcost < opt_hist[-1][0]:     opt_hist.append([totalcost, X])
            
    return results, opt_hist, len(Xvals), time.time() - starttime

def get_2dpareto(resultstab, ind1, ind2):
    pareto = dict()
    for x in resultstab:
        if not any([(resultstab[x][ind1] >= resultstab[i][ind1] and resultstab[x][ind2] > resultstab[i][ind2]) or (resultstab[x][ind1] > resultstab[i][ind1] and resultstab[x][ind2] >= resultstab[i][ind2]) for i in resultstab]):
            pareto[x] = resultstab[x][ind1], resultstab[x][ind2]
    return dict(sorted(pareto.items(), key=lambda x: x[1]))

def get_3dpareto(resultstab, ind1, ind2, ind3):
    pareto = dict()
    for x in resultstab:
        if not any([(resultstab[x][ind1] >= resultstab[i][ind1] and resultstab[x][ind2] >= resultstab[i][ind2] and resultstab[x][ind3] > resultstab[i][ind3]) \
                    or (resultstab[x][ind1] > resultstab[i][ind1] and resultstab[x][ind2] >= resultstab[i][ind2] and resultstab[x][ind3] >= resultstab[i][ind3]) \
                        or (resultstab[x][ind1] >= resultstab[i][ind1] and resultstab[x][ind2] > resultstab[i][ind2] and resultstab[x][ind3] >= resultstab[i][ind3]) \
                            for i in resultstab]):
            pareto[x] = resultstab[x][ind1], resultstab[x][ind2], resultstab[x][ind3]
    return dict(sorted(pareto.items(), key=lambda x: x[1]))

def plot_pareto3(pareto3, legend='equal-weight design'):
    x = np.array([x for x,y,z in pareto3.values()])
    y = np.array([y for x,y,z in pareto3.values()])
    z = np.array([z for x,y,z in pareto3.values()])
    mindes = np.argmin(x+y+z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25., azim=45)
    ax.scatter(x,y,z)
    if legend: ax.scatter(x[mindes],y[mindes],z[mindes], color='red',s=55,marker='X', label=legend)
    #ax.plot_trisurf(x,y,z)
    plt.legend()
    return fig, ax



