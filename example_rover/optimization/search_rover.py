# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:29:42 2022

@author: igirshfe
"""
import sys, os

sys.path.insert(0, os.path.join('..','..'))

import fmdtools.faultsim.propagate as prop
import fmdtools.resultdisp as rd
from fmdtools.modeldef import SampleApproach
import rover_model as rvr
import tqdm

import numpy as np
import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

## OBJECTIVES: FORMULATION 1
def line_dist(ind):
#    """Takes all of the individuals in a species and returns each of their distances
#    from the end line in an array"""
    mdl = rvr.Rover(params=rvr.gen_params('turn', start=5), valparams={'drive_modes':{'custom_fault':{'friction':ind[0],'drift':ind[1], 'transfer':ind[2]}}})
    endresults,resgraph, reshist = prop.one_fault(mdl,'Drive','custom_fault', time=fault_time, staged=True, protect=False, track={'functions':{'Environment':'all'}, 'flows':{'Ground':'all'}})
    dist = endresults['classification']['line_dist']
    enddist = endresults['classification']['end_dist']
    endpt = endresults['classification']['endpt']
    return dist,enddist,  endpt

def line_dist_faster(ind):
    newmdl=mdl_ft.copy()
    newmdl.fxns['Drive'].mode_state_dict['custom_fault']={'friction':ind[0],'drift':ind[1], 'transfer':ind[2]}
    
    scen = prop.construct_nomscen(newmdl)
    scen['faults']['Drive']="custom_fault"
    scen['properties']['function']="Drive"
    scen['properties']['fault']="custom_fault"
    scen['properties']['time']=fault_time
    faultmdlhist, _, t_end = prop.prop_one_scen(newmdl, scen, staged=True, prevhist=mdlhist_nom, track='none')
    
    dist = rvr.find_line_dist(newmdl.flows['Ground'].x,newmdl.flows['Ground'].y, mdlhist_nom['flows']['Ground']['linex'], mdlhist_nom['flows']['Ground']['liney'])
    enddist = np.sqrt((newmdl.params['end'][0] - newmdl.flows["Ground"].x)**2+(newmdl.params['end'][1] - newmdl.flows["Ground"].y)**2)
    endpt = [newmdl.flows["Ground"].x, newmdl.flows["Ground"].y]
    return dist, enddist, endpt

"""f_1 returns the sum of line_dist for all individuals in representatives (rover distances)
f_2 returns the sum of distances between nearest neighbors in hazard space (hazard space distances)"""

def f_1(sol):
    """Calculates total line distances accross a solution"""
    return np.sum([ind.linedist for ind in sol])

def f_2(sol):
    """Calculates total nearest neighbor distance accross all points in a solution"""
    hspace_sum = 0
    for i in range(len(sol)):
        hspace_min=100
        for j in range(len(sol)):
            if i != j:
                hspace_dist = math.sqrt(((sol[i][0]-sol[j][0])/DRIFT_RANGE)**2+((sol[i][1]-sol[j][1])/FRIC_RANGE)**2 
                                                    +((sol[i][2]-sol[j][2])/TRANSFER_RANGE)**2)
                if hspace_dist < hspace_min: hspace_min = hspace_dist
                #hspace_sum += math.sqrt(((pop[i][0]-pop[j][0])/DRIFT_RANGE)**2+((pop[i][1]-pop[j][1])/FRIC_RANGE)**2 
                #                                    +((pop[i][2]-pop[j][2])/TRANSFER_RANGE)**2)
        hspace_sum+=hspace_min
    return hspace_sum

def f_12_mult(w,sol):
    """Calculates sum of nearest neighbor health-state distance times line distance"""
    hspace_sum = 0
    for i in range(len(sol)):
        hspace_min=100
        for j in range(len(sol)):
            if i != j:
                hspace_dist = math.sqrt(((sol[i][0]-sol[j][0])/DRIFT_RANGE)**2+((sol[i][1]-sol[j][1])/FRIC_RANGE)**2 
                                                    +((sol[i][2]-sol[j][2])/TRANSFER_RANGE)**2)
                if hspace_dist < hspace_min: hspace_min = hspace_dist
                #hspace_sum += math.sqrt(((pop[i][0]-pop[j][0])/DRIFT_RANGE)**2+((pop[i][1]-pop[j][1])/FRIC_RANGE)**2 
                #                                    +((pop[i][2]-pop[j][2])/TRANSFER_RANGE)**2)
        hspace_sum+=hspace_min*sol[i].linedist
    return hspace_sum

def f_14_mult(w,sol):
    """Calculates total nearest neighbor distance of the end-point all points in a solution"""
    dist_sum = 0
    for i in range(len(sol)):
        dist_min=1000
        for j in range(len(sol)):
            if i != j:
                dist = math.sqrt((sol[i].endpt[0]-sol[j].endpt[0])**2+(sol[i].endpt[1]-sol[j].endpt[1])**2)
                if dist < dist_min: dist_min = dist
        dist_sum+=dist_min*sol[i].linedist
    return dist_sum
    
"""objective function that returns fitness values for individual which are an array
of points in the faulty-state space."""

def evalTotDist(w,individual): 
    """
    :param w: weight for objective function. Increasing w puts more weight on linedist
    """
    norm_f1 = 1/(1.5*IND_SIZE)
    norm_f2 = 1/MU_SPACE #note - this mu is probably what is getting us in trouble here
    return w*norm_f1*f_1(individual)+(1-w)*norm_f2*f_2(individual)

# OBJECTIVES: FORMULATION 2
def end_dists(ind):
    newmdl=mdl_ft.copy()
    newmdl.fxns['Drive'].mode_state_dict['custom_fault']={'friction':ind[0],'drift':ind[1], 'transfer':ind[2]}
    
    scen = prop.construct_nomscen(newmdl)
    scen['faults']['Drive']="custom_fault"
    scen['properties']['function']="Drive"
    scen['properties']['fault']="custom_fault"
    scen['properties']['time']=fault_time
    faultmdlhist, _, t_end = prop.prop_one_scen(newmdl, scen, staged=True, prevhist=mdlhist_nom, track='none')
    
    enddist = np.sqrt((mdl.flows["Ground"].x - newmdl.flows["Ground"].x)**2+(mdl.flows["Ground"].y - newmdl.flows["Ground"].y)**2)
    endpt = [newmdl.flows["Ground"].x, newmdl.flows["Ground"].y]
    return enddist, endpt
def f_3(sol):
    """Calculates total distance from the end=point for the solutions"""
    return np.sum([ind.enddist for ind in sol])
def f_4(sol):
    """Calculates total nearest neighbor distance of the end-point all points in a solution"""
    dist_sum = 0
    for i in range(len(sol)):
        dist_min=1000
        for j in range(len(sol)):
            if i != j:
                dist = math.sqrt((sol[i].endpt[0]-sol[j].endpt[0])**2+(sol[i].endpt[1]-sol[j].endpt[1])**2)
                if dist < dist_min: dist_min = dist
        dist_sum+=dist_min
    return dist_sum

def evalTotDist2(w, individual):
    #norm_f3 = 1/(fault_dist*IND_SIZE)
    norm_f1 = 1/IND_SIZE
    norm_f4 = 1/IND_SIZE
    return w*norm_f1*f_1(individual)+(1-w)*norm_f4*f_4(individual)

# OPTIMIZATION FUNCTIONS
"""crossover function"""
def cxHealthStates(ind1,ind2):
    cxpoint1, cxpoint2 = np.random.choice(3, size=2, replace=False)
    if random.random()>0.5: c1, c2 = ind2, ind1
    else:                   c2,c1 = ind2, ind1

    if cxpoint1 >= cxpoint2: cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
    c1[cxpoint1:cxpoint2] = c2[cxpoint1:cxpoint2]
    return  c1

"""mutation function"""
def mutHealthStates(individual, sigma=0.25):
    minv = [FRIC_LB, DRIFT_LB, TRANSFER_LB]
    maxv= [FRIC_UB, DRIFT_UB, TRANSFER_UB]
    rangev = [FRIC_RANGE, DRIFT_RANGE,TRANSFER_RANGE]
    if len(individual) > 0:
        for i,val in enumerate(individual):
            newval = random.gauss(val, sigma*rangev[i])
            if newval<minv[i]:      newval=minv[i]
            elif newval>maxv[i]:    newval=maxv[i]
            individual[i]=newval
            #if random.random()<move_frac:
            #    individual[i]= random.triangular(minv[i], maxv[i], val)
    else:
        individual.add(random.uniform(FRIC_LB,FRIC_UB),
                       random.uniform(DRIFT_LB, DRIFT_UB),
                       random.uniform(TRANSFER_LB, TRANSFER_UB))
    return individual

def eval_pop_linedist(pop):
    for k,ind in enumerate(pop):
        for i in range(len(ind)):
            ind[i].linedist,ind[i].enddist, ind[i].endpt = line_dist_faster(ind[i])
    return pop

"""part of evolutionary algorithm that applies crossover and mutation"""
def permute_ea(pop, toolbox, mut_crossover_fraction):
    '''
    :param mut_crossover_fraction: fraction that will be mutated vs be a crossover
    '''
    offspring = toolbox.clone(pop)

    """Apply crossover and mutation to the offspring"""
    for k,ind in enumerate(offspring):
        for i in range(len(ind)):
            if i==0 or (random.random() < mut_crossover_fraction):
                ind[i] = toolbox.mutate(ind[i])
            else:
                ind[i] = toolbox.mate(ind[i - 1], ind[i])
            ind[i].linedist,ind[i].enddist, ind[i].endpt = line_dist_faster(ind[i])
            del ind.fitness.values
    return offspring
"""evolutionary algorithm that applies crossover and mutation"""
def permute_ccea(population, toolbox, mut_crossover_fraction):
    '''
    :param mut_crossover_fraction: fraction that will be mutated vs be a crossover
    '''
    offspring = [toolbox.clone(ind) for ind in population]

    """Apply crossover and mutation to the offspring"""
    for i in range(len(population)):
        if i==0 or (random.random() < mut_crossover_fraction):
            offspring[i] = toolbox.mutate(offspring[i])
        else:
            offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
        offspring[i].linedist,offspring[i].enddist, offspring[i].endpt  = line_dist_faster(offspring[i])
        del offspring[i].fitness.values
    return offspring

def plot_hspace(species, title="Faulty State-Space", filename="", ax=False):
    '''Visualization'''
    '''Plot the tuples (friction,drift,transfer) for all individuals in 
    the population'''
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
    else: fig =plt.gcf()
    #ax = plt.axes(projection='3d')
    #set axes ranges
    ax.set_xlim(FRIC_LB,FRIC_UB)
    ax.set_ylim(DRIFT_LB,DRIFT_UB)
    ax.set_zlim(TRANSFER_LB,TRANSFER_UB)
    
    #labels
    ax.set_xlabel("Friction")
    ax.set_ylabel("Drift")
    ax.set_zlabel("Transfer")
    ax.set_title(title)
    
    #define points
    
    colors = iter(plt.cm.rainbow(np.linspace(0,1,POP_SIZE)))
    
    for i,s in enumerate(species):
        c = next(colors)
        for j in range(IND_SIZE):
            ax.scatter(s[j][0],s[j][1],s[j][2], color=c)
            #print("plotted points", i, s[j][0],s[j][1],s[j][2])
            
    if filename: fig.savefig(filename, format="pdf", bbox_inches = 'tight', pad_inches = 0.0)
    return fig

def solution(x,g):
    '''
    :param x: typically population or offspring
    :param g: current generation
    '''
    max_fit = max([ind.fitness.values[0] for ind in x])
    for ind in x:
        temp = (ind,ind.fitness.values[0],g)
        if temp[1] == max_fit:
            soln = temp
    return soln

def plot_fitness(generation,rep_fitness):       
    '''Plot for fitness values v. generation and a table of the corresponding
    tuples for each individual''' 
    
    figg = plt.figure()
    ax3 = plt.axes()
    
    #fig =  plt.figure()
    #ax2 = plt.axes()
    #define points
    x = generation
    y = rep_fitness
    
    #label axes 
    ax3.set_ylabel("Fitness Value")
    ax3.set_xlabel("Generation")
    ax3.set_title("Fitness Values v. Generation")
    #plot scatterplot
    ax3.scatter(x,y)
    
    return figg.savefig('perform_random.pdf', format="pdf", bbox_inches = 'tight', pad_inches = 0.0)

def montecarlo(extended=True, verbose=True, ngen=2, show_sol=True, weight=0.5, filename='rslt_random.csv', formulation=1):    
    start = time.time()
    toolbox = setup_opt(mc=True, formulation=formulation)
    g = 0 
    gens = []
    sol_hist = []
    fit_hist=[]
    print("MC PERFORMANCE:")
    for g in tqdm.tqdm(range(ngen)):
        """Generate offspring"""
        offspring = toolbox.population()
        offspring = eval_pop_linedist(offspring)
        
        #plot_hspace(offspring,g)

        """Compute fitness values for all representative vectors.
        Assign representative fitness values to individuals that appeared in
        the representative vector. Assign max fitness from that array to be the
        individual fitness value. Using weighted sum to deal with multiple 
        objectives"""
        
        for ind in offspring:
            ind.fitness.values = (toolbox.evaluate(weight,ind),)
                
        """storing fitness values for each generation"""
        sol, fit, gen = solution(offspring,g)
        if g==0:                  fit_hist.append(fit); sol_hist.append(sol)
        elif fit >= fit_hist[-1]: fit_hist.append(fit); sol_hist.append(sol)
        else:                     fit_hist.append(fit_hist[-1]); sol_hist.append(sol_hist[-1])  
        gens.append(g)
    
    end = time.time()  
    elapsed = end - start
    print("Best Fitness: "+str(fit_hist[-1]))
    t_hist = elapsed*np.array([i for i in range(1,len(gens)+1)])/len(gens)
    
    #plot_fitness(gens, [fits[i][1] for i in range(len(fits))])
    rslt = {"Generations" : gens, 
            "Random Fitness Values" : fit_hist,
            "Random Health States" : sol_hist,
            "time" : t_hist}  
    rslt_dataframe = pd.DataFrame(rslt)
    if filename: rslt_dataframe.to_csv(filename)
    
    soln = sol_hist[-1]
    if show_sol: visualizations(list(soln), method="MC")

    return rslt, soln

def ea(extended=True, verbose=True, ngen = 5, show_space_during_opt=False, show_sol=True, weight=0.5, filename='rslt_ea.csv', formulation=1):    
    start = time.time()
    toolbox = setup_opt(formulation=formulation)
    #initialize counter for generations
    g = 0 
    '''initialize array for collecting tuple of top performing individual
    and its  fitness values at each genearation'''
    fit_hist = []
    max_fit=0.0
    best_sol = []
    sol_hist=[]
    #list of generations
    gens = []    
    #generate intial population
    population = toolbox.population()
    
    #compute fitness values for initial population before application of algorithm
    for ind in population:
        ind.fitness.values = (0.0,)

    print("EA PERFORMANCE:")
    for g in tqdm.tqdm(range(ngen)):
        """Generate offspring. 
        Note: the initial offspring is the EA applied to the
        initial population. Thereafter, offspring is the top 50% individuals 
        from the previous generation + that top 50% with the EA applied to it"""
        if g == 0:
            offspring = permute_ea(population,toolbox,0.75)
        else:
            offspring = permute_ea(population,toolbox,0.75) + population
        
        #plot faulty-state space 
        if show_space_during_opt: plot_hspace(offspring,g, title="EA Population at Gen "+str(g))

        """Compute fitness values for all representative vectors.
        Assign representative fitness values to individuals that appeared in
        the representative vector. Assign max fitness from that array to be the
        individual fitness value. Using weighted sum to deal with multiple 
        objectives"""
        for ind in offspring:
            fit = toolbox.evaluate(weight,ind)
            ind.fitness.values = (fit,)
            if fit > max_fit:
                max_fit=fit
                best_sol = ind
        
        #select top 50% performing individuals
        population = toolbox.select(offspring)
                
        """storing best individual and its fitness for each generation"""
        fit_hist.append(max_fit)
        sol_hist.append(best_sol)
        gens.append(g)
    
    
    end = time.time()  
    elapsed = end - start
    print("Best Fitness: "+str(fit_hist[-1]))
    t_hist = elapsed*np.array([i for i in range(1,len(gens)+1)])/len(gens)
    rslt = {"Generations" : gens, 
            "EA Fitness Values" : fit_hist,
            "EA Health States" : sol_hist,
            "time": t_hist} 
    rslt_dataframe = pd.DataFrame(rslt)
    if filename: rslt_dataframe.to_csv(filename)
    
    sol = sol_hist[-1]
    if show_sol: visualizations(list(sol), method="EA")

    
    return rslt, sol

def ccea(extended=True, verbose=True, ngen = 2,show_space_during_opt=False, show_sol=True, weight=0.5, filename='rslt_ccea.csv', formulation=1):    
    start = time.time()
    
    toolbox = setup_opt(ccea=True, formulation=formulation)
    g = 0 
    gens = []
    POP = []
    
    if SUBPOP_SIZE*NUM_SUBPOP >= 1000:
        nrep = 1000
    else:
        nrep = SUBPOP_SIZE*NUM_SUBPOP
        
    population = [toolbox.subpopulation() for _ in range(NUM_SUBPOP)]
    best_fit_hist =[]
    best_sol_hist =[]
    best_fit=0.0
    print("CCEA PERFORMANCE:")
    for g in tqdm.tqdm(range(ngen)):
        """Generate offspring"""
        
        if g == 0:
            offspring = [permute_ccea(p,toolbox,0.5) for p in population]
        else:
            offspring = [p+permute_ccea(p,toolbox,0.75) for p in population]
        
        if show_space_during_opt: plot_hspace(offspring,g, title="CCEA Population at Gen "+str(g))
        
        
        """Generate all possible representatives"""
        R = [[random.choice(o) for o in offspring] for i in range(nrep)]
        """Compute fitness values for all representative vectors.
        Assign representative fitness values to individuals that appeared in
        the representative vector. Assign max fitness from that array to be the
        individual fitness value. Using weighted sum to deal with multiple 
        objectives"""
        
        for rep in R:
            fitness=toolbox.evaluate(weight,rep)
            if fitness>best_fit: #Keep track of best solution generated
                best_fit=fitness
                best_sol=rep
            for ind in rep:
                if not ind.fitness.values or fitness >ind.fitness.values[0]:
                    ind.fitness.values = (fitness,)
        for k,o in enumerate(offspring):         
            """Select top 50% best individuals in each subpopulation"""
            population[k] = toolbox.select(o)
        
        POP.append(population)
        #record = stats.compile(population)
        #print(record)
        best_sol_hist.append(best_sol)
        best_fit_hist.append(best_fit)
        gens.append(g)
  
    end = time.time()  
    elapsed = end - start

    print("Best Fitness: "+str(best_fit_hist[-1]))
    
    rslts = {"Population" : POP}
    rslts_dataframe = pd.DataFrame(rslts)
    rslts_dataframe.to_csv('ccea_pop.csv')
       
    sol = best_sol_hist[-1]
    if show_sol: 
        visualizations(list(sol), method="CCEA")
    
    t_hist = elapsed*np.array([i for i in range(1,len(gens)+1)])/len(gens)
    rslt = {"Generations" : gens, "CCEA Fitness Values" : best_fit_hist, "Best_Sol":best_sol_hist, "time":t_hist}  
    rslt_dataframe = pd.DataFrame(rslt)
    if filename: rslt_dataframe.to_csv(filename)    
    
    return rslt, sol, POP

def setup_opt(ccea=False, mc=False, formulation=1):
    from deap import base, creator, tools
    
    if hasattr(creator, "FitnessMax"):  del creator.FitnessMax
    if hasattr(creator, "Point"):       del creator.Point
    if hasattr(creator, "Individual"):  del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights =(1.0,))
    creator.create("Point", list, linedist = 100.0, enddist=100.0, endpt=[])
    if ccea:
        creator.create("Individual", list, fitness = creator.FitnessMax, linedist = 100.0, enddist=100.0, endpt=[], rep=[])
    else:
        creator.create("Individual", list, fitness = creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register('attr_fric', random.uniform, FRIC_LB,FRIC_UB)
    toolbox.register('attr_drift', random.uniform, DRIFT_LB,DRIFT_UB)
    toolbox.register('attr_transfer', random.uniform, TRANSFER_LB, TRANSFER_UB)
    toolbox.register('point', tools.initCycle, creator.Point,
                    (toolbox.attr_fric, toolbox.attr_drift, toolbox.attr_transfer), n=NCYCLES)
    if ccea:
        toolbox.register('individual', tools.initCycle, creator.Individual,
                        (toolbox.attr_fric, toolbox.attr_drift, toolbox.attr_transfer), n=NCYCLES)
        toolbox.register('subpopulation', tools.initRepeat, list, toolbox.individual, SUBPOP_SIZE)
    elif mc:
        toolbox.register('individual', tools.initRepeat, creator.Individual, (toolbox.point), IND_SIZE)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual, int(POP_SIZE/2))    
    else:
        toolbox.register('individual', tools.initRepeat, creator.Individual, (toolbox.point), IND_SIZE)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual, POP_SIZE)     
    if formulation==1:      toolbox.register("evaluate", evalTotDist)
    elif formulation==12:   toolbox.register("evaluate", f_12_mult)
    elif formulation==2:    toolbox.register("evaluate", evalTotDist2)
    elif formulation==22:   toolbox.register("evaluate", f_14_mult)
    toolbox.register("mate", cxHealthStates)
    toolbox.register("mutate", mutHealthStates)
    toolbox.register("select", tools.selBest, k = int(0.5*POP_SIZE))
    
    return toolbox


def plot_line_dist(sol_dict, figsize=(4,12), v_padding=0.2):
    fig, axs = plt.subplots(len(sol_dict), figsize=figsize)
    k=0
    for alg, sol in sol_dict.items():
        if hasattr(sol[0], 'linedist'): linedists = [i.linedist for i in sol]
        else:                           linedists = [line_dist_faster(i)[0] for i in sol]
        axs[k].hist(linedists, label=alg, bins=[j for j in np.arange(0,2.5, 0.25)])
        axs[k].set_title(alg)
        axs[k].grid(axis="y")
        axs[k].set_ylim([0,10])
        if k!=len(axs)-1: axs[k].set_xticks([])
        k+=1
    rd.plot.multiplot_legend_title(sol_dict, axs, axs[k-1], v_padding=v_padding)
    return fig

def plot_hspaces(sol_dict, figsize=(4,12), v_padding=0.2):
    fig, axs = plt.subplots(len(sol_dict), figsize=figsize,subplot_kw=dict(projection='3d'))
    k=0
    for alg, sol in sol_dict.items():
        plot_hspace([sol], title=alg, ax=axs[k])
        k+=1
    ax =plt.gca()
    rd.plot.multiplot_legend_title(sol_dict, axs, ax, v_padding=v_padding)
    return fig

def plot_trajs(sol_dict, figsize=(4,12), v_padding=0.3):
    fig, axs = plt.subplots(len(sol_dict), figsize=figsize)
    k=0
    for alg, sol in sol_dict.items():
        ax=axs[k]
        visualizations(sol, method=alg, ax=ax, legend=False)
        k=k+1
    rd.plot.multiplot_legend_title(sol_dict, axs, ax, v_padding=v_padding, legend_loc=2)
    return fig

def visualizations(soln, method="EA", figsize=(4,4), ax=False, legend=True, xlim=[15,25], ylim=[0,10]):

    mdl_range = rvr.Rover(params=rvr.gen_params('turn', start=5), valparams={'drive_modes':list(soln)})
    _,_, mdlhists = prop.nominal(mdl_range)
    phases, modephases = rd.process.modephases(mdlhists)
    app_range = SampleApproach(mdl_range, faults='Drive', phases={'drive':phases['Avionics']['drive']})
    endclasses_range, mdlhists_range = prop.approach(mdl_range, app_range, staged=True, showprogress=False)    
    fig = rvr.plot_trajectories(mdlhists_range, app=app_range, faultlabel='Faulty Scenarios', faultalpha=0.5,title="Trajectories-"+method,show_labels=False, figsize=figsize, ax=ax, legend=legend)
    if not ax:
        ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not ax: return fig

NCYCLES = 1
"""params for health states which are upper and lower bounds"""
FRIC_LB = 1
FRIC_UB = 20
DRIFT_LB = -0.5
DRIFT_UB = 0.5
TRANSFER_LB = 0
TRANSFER_UB = 1

IND_SIZE = 10 #number of points in individual SPECIES_SIZE
POP_SIZE = 50 #number of individuals in population NUM_SPECIES
NUM_HSTATES = 3 #number of health states

DRIFT_RANGE = DRIFT_UB-DRIFT_LB
FRIC_RANGE = FRIC_UB-FRIC_LB
TRANSFER_RANGE = TRANSFER_UB-TRANSFER_LB
MU_SPACE = DRIFT_RANGE*FRIC_RANGE*TRANSFER_RANGE
#measure (volume) of space  

#CCEA params
SUBPOP_SIZE = 50 #number of individuals in a subpopulation
NUM_SUBPOP = 10 #number of subpopulation

#nominal scenario info (for line_dist)
mdl = rvr.Rover(params=rvr.gen_params('turn', start=5), valparams={'drive_modes':{'custom_fault':{'friction':1.0,'drift':0.0, 'transfer':0.0}}})
_,_, mdlhists_nom = prop.nominal(mdl)
phases, modephases = rd.process.modephases(mdlhists_nom)
app= SampleApproach(mdl, faults='Drive', phases={'drive':phases['Avionics']['drive']})
fault_time = app.times[0]
end_time = phases['Avionics']['drive'][1]+25
mdl.times[1]=end_time

nomscen=prop.construct_nomscen(mdl)
mdlhist_nom, mdls, t_end = prop.prop_one_scen(mdl, nomscen, ctimes=[fault_time], staged=False)

mdl_ft = mdls[fault_time]
fault_loc = [mdl_ft.flows['Ground'].x,mdl_ft.flows['Ground'].y]
fault_dist = np.sqrt((mdl.flows["Ground"].x - mdl_ft.flows["Ground"].x)**2+(mdl.flows["Ground"].y - mdl_ft.flows["Ground"].y)**2)
#prevhist=nommdlhist
    
if __name__=="__main__":
    
    result_mc, sol_mc= montecarlo(ngen=10, weight=0.5, filename="")
    result_ea, sol_ea= ea(ngen=10, weight=0.5, filename="")
    result_ccea, sol_ccea, pop= ccea(ngen=10, weight=0.5, filename="")
    
    #result_mc, sol_mc= montecarlo(ngen=10, weight=0.5, filename="", formulation=2)
    #result_ea, sol_ea= ea(ngen=10, weight=0.5, filename="", formulation=2)
    #result_ccea, sol_ccea, pop= ccea(ngen=5, weight=0.5, filename="", formulation=2)
    
    #result_ea["EA Health States"][-1]
    

    #plot_hspace(pop, "CCEA")
    
    #result_mc, sol_mc= montecarlo(ngen=50, weight=0.5)
    #result_ea, sol_ea= ea(ngen=50, weight=0.5)
    #result_ccea, sol_ccea, pop= ccea(ngen=25, weight=0.5)
    
    #plot_hspace([result_mc["Random Health States"][-1]], "MC")
    #plot_hspace([result_ea["EA Health States"][-1]], "EA")
    #plot_hspace([result_ccea["Best_Sol"][-1]], "CCEA")
    
    #plot_line_dist({"Monte Carlo":sol_mc, "Evolutionary Algorithm":sol_ea, "Cooperative Coevolution":sol_ccea})
    #plot_hspaces({"Monte Carlo":sol_mc, "Evolutionary Algorithm":sol_ea, "Cooperative Coevolution":sol_ccea})
    plot_trajs({"Monte Carlo":sol_mc, "Evolutionary Algorithm":sol_ea, "Cooperative Coevolution":sol_ccea}, figsize=(4,12))    
    
    #checking results
    #evalTotDist(0.9, sol_ccea)
    #evalTotDist(0.9, sol_ea)
    #evalTotDist(0.9, sol_mc)
    
    #result_mc, sol_mc= montecarlo(ngen=5, weight=0.9)
    #result_ea, sol_ea= ea(ngen=25, weight=0.9)
    #result_ccea, sol_ccea, pop= ccea(ngen=5, weight=0.9)
    #result_ccea, sol_ccea, pop= ccea(ngen=50, weight=0.5, filename="result_formulation_12", formulation=12)
    #result_ccea, sol_ccea, pop = ccea(ngen=50, weight=0.5, filename="result_formulation_22", formulation=22)
    #plot_hspace([result_ccea["Best_Sol"][-1]], "CCEA")


