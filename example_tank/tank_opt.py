# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:01:45 2021

@author: dhulse
"""
import sys, os
sys.path.insert(0, os.path.join('..'))
import numpy as np
import scipy as sp
import time
import itertools
import random
from scipy.optimize import minimize

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
from example_tank.tank_optimization_model import Tank
from fmdtools.modeldef import SampleApproach
import multiprocessing as mp
from fmdtools.faultsim.search import ProblemInterface


params={'capacity':20, # size of the tank (10 - 100)
        'turnup':0.0,  # amount the pump can be "turned up" (0 - 1)
        **{(a-1,b-1,c-1,ul):0 for a,b,c in np.ndindex((3,3,3)) for ul in ["u", "l"]}} #state-action pairs for resilience policy: what to given fault signals
def x_to_descost(xdes, xres1=[], xres2=[]):
    pen = 0 #determining upper-level penalty
    if xdes[0]<10: pen+=1e5*(10-xdes[0])**2
    if xdes[0]>100: pen+=1e5*(100-xdes[0])**2
    if xdes[1]<0: pen+=1e5*(xdes[1])**2
    if xdes[1]>1: pen+=1e5*(1-xdes[1])**2
    return (xdes[0]-10)*1000 + (xdes[0]-10)**2*1000   + xdes[1]**2*10000 + pen

mdl= Tank()
prob = ProblemInterface("res_problem", mdl, staged=True)
app = SampleApproach(mdl)

prob.add_simulation("des_cost", "external", x_to_descost)
prob.add_objectives("des_cost", cd="cd")
prob.add_variables("des_cost",'capacity', 'turnup')

prob.add_simulation("res_sim", "multi", app.scenlist, upstream_sims = {"des_cost":{'params':{"capacity":"capacity", "turnup":"turnup"}}})
res_vars_i = {param:1 for param,v in mdl.params.items() if param not in ['capacity','turnup']}
res_vars = [(var, None) for var in res_vars_i.keys()]
prob.add_variables("res_sim", *res_vars, vartype="param")
prob.add_objectives("res_sim", cost="expected cost", objtype="endclass")
prob.add_combined_objective('tot_cost', 'cd', 'cost')

#prob.update_sim_vars("res_sim", newparams={'capacity':21,'turnup':0.5})
xdes = [1,2]
prob.cd(xdes)

prob.cost([*res_vars_i.values()])

prob.tot_cost(xdes, [*res_vars_i.values()])

prob.update_sim_vars("res_sim", newparams={'capacity':10, 'turnup':0.5})
inter_cost = prob.cost([*res_vars_i.values()])

rvar = [*res_vars_i.values()][:27]
lvar = [*res_vars_i.values()][27:]

#leg_cost = x_to_totcost_leg(xdes, rvar, lvar)

## Legacy objective functions (for verification)
def x_to_rcost_leg(xres1,xres2, xdes=[20,1], pool=False, staged=True):
    fp =      {(a-1,b-1,c-1, "l"):xres1[i] for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))}
    fp.update({(a-1,b-1,c-1, "u"):xres2[i] for i,(a,b,c) in enumerate(np.ndindex((3,3,3)))})
    mdl=Tank(params={'capacity':xdes[0],'turnup':xdes[1],**fp})
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=staged, showprogress=False, pool=pool)
    rescost = rd.process.totalcost(endclasses)
    return rescost
def x_to_totcost_leg(xdes, xres1, xres2, pool=False):
    do_cost = x_to_descost(xdes)
    rescost = x_to_rcost_leg(xres1, xres2, xdes=xdes, pool=pool)
    return do_cost + rescost

def lower_level(xdes, args):
    do_cost = x_to_descost(xdes) 
    bestsol, rcost, runtime = EA(popsize=20, mutations=6, crossovers=4, numselect=6, args=args, xdes=xdes)
    t = time.time()-args['starttime']
    f=do_cost+rcost
    args['fhist'].append(f); args['thist'].append(t); args['xdhist'].append(xdes)
    print('time: '+str(t)+' fval: '+str(f)+' xdes: '+str(xdes))
    return do_cost + rcost

def bilevel_opt(pool=False, xdes=[21,.5]):
    args = {'seed':seedpop(), 'll_opt':1e6, 'll_optx':[], 'fhist':[],'thist':[],'starttime':time.time(), 'pool':pool, 'xdhist':[xdes]}
    result = minimize(lower_level, xdes, method='Nelder-Mead', bounds =((10, 100),(0,1)), callback=callbackF1, args = args, options={'disp':True, 'adaptive':True, 'fatol':10, 'xtol':0.00001})
    fullfhist = args['fhist']; fullxdhist = args['xdhist']
    bestfhist=  [fullfhist[0]]+[min(fullfhist[:i]) for i,f in enumerate(fullfhist) if i!=0]
    bestxdhist = [fullxdhist[0]]+[fullxdhist[np.argmin(fullfhist[:i])] for i,f in enumerate(fullfhist) if i!=0]
    return result, args, bestfhist, bestxdhist

def alternating_opt(option='with_cr', pool=False, xdes=[21,.5]):
    xdes = np.array(xdes)
    args = {'seed':seedpop(), 'll_opt':1e6, 'll_optx':[]}
    newmin = 100000000
    lastmin = 1000000001
    bestsol = np.zeros((2,27))
    last_run = False
    if option=='with_cr':       ul_cost_func = x_to_totcost
    elif option=='without_cr':  ul_cost_func= x_to_descost
    starttime = time.time()
    fhist = [x_to_totcost(xdes,bestsol[0], bestsol[1], pool)]
    thist = [0]
    xdhist = [xdes]
    for n in range(10):
        result = minimize(ul_cost_func, [np.round(xdes[0],1), np.round(xdes[1],1)], method='Nelder-Mead', bounds =((10, 100),(0,1)), callback=callbackF1, args = (bestsol[0],bestsol[1]), options={'disp':True})
        xdes = result['x']
        #result = minimize(x_to_totcost, xdes, method='Powell', callback=callbackF1,  args = (bestsol[0],bestsol[1]), options={'disp':True,'ftol': 0.000001})
        # doesn't really work: trust-constr, SLSQP, Nelder-Mead (doesn't respect bounds), COBYLA (a bit better, but converges poorly), 
        # powell does okay but I'm not sure if it's actually searching the x-direction
        bestsol, rcost, runtime = EA(args=args, popsize=50, mutations=10,numselect=20, crossovers=5, iters=100, xdes = xdes, verbose="iters")
        lastmin = newmin; newmin = x_to_descost(xdes) + rcost
        fhist.append(newmin)
        xdhist.append(xdes)
        thist.append(time.time()-starttime)
        print(n, newmin, lastmin-newmin)
        if lastmin - newmin <0.1: 
            if last_run:    break
            else:           last_run = True
        else:               last_run = False
        fhist.append(newmin), thist.append(time.time()-starttime)
    return result, args, fhist, thist, xdhist


def callbackF(Xdes, result):
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(result['nit'], Xdes[0], Xdes[1], result['fun']))
def callbackF1(Xdes):
    print(Xdes)

def EA(popsize=10, iters=20, mutations=3, crossovers=2, numselect=3, args={}, xdes=[20,1], verbose = False):
    starttime = time.time()
    randpopsize = popsize-numselect-mutations- crossovers
    opers = [randpop, mutepop, crossover]
    numopers = [randpopsize, mutations, crossovers]
    used_opers = [oper for i, oper in enumerate(opers) if numopers[i]>0]
    used_numopers = [numoper for numoper in numopers if numoper>0]
    if args: pop=np.concatenate((args['seed'],seedpop(), randpop([],popsize-3))); pool=False
    else:    pop=np.concatenate((seedpop(), randpop([],popsize-3))); pool=args.get('pool', False)
    makefeasible(pop)
    values = np.array([x_to_rcost(x[0],x[1], xdes=xdes, pool=pool) for x in pop])
    for i in range(iters):
        goodpop, goodvals = select(pop, values, numselect)
        newpop =  np.concatenate(tuple([oper(goodpop,used_numopers[i]) for i,oper in enumerate(used_opers)]))
        makefeasible(newpop)
        newvals = np.array([x_to_rcost(x[0],x[1], xdes=xdes) for x in newpop])
        pop, values = np.concatenate((goodpop, newpop)), np.concatenate((goodvals, newvals)) 
        if verbose=="iters": print(["iter "+str(i)+": ",min(values)])
    minind = np.argmin(values)
    if args: args['seed'] = goodpop; args['ll_opt']= values[minind];  args['ll_optx']= pop[minind];
    if verbose=="final": print(values[minind])
    return pop[minind], values[minind], time.time() - starttime

possible_sols = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]

def randpop(goodpop,popsize):
    return np.array([[[random.randint(-1,1) for a in range(0,27)],[random.randint(-1,1) for a in range(0,27)]] for i in range(0,popsize)])
def seedpop():
    donothing = np.zeros((2,27))
    adjustup = np.ones((2,27))
    adjustdown = -np.ones((2,27))
    return np.array([donothing, adjustup, adjustdown])
def mutepop(goodpop, mutations):
    to_mutate = np.random.choice([i for i in range(len(goodpop))], size=mutations, replace=False)
    return np.array([permute(solution) for solution in goodpop[to_mutate]])
def permute(solution):
    mutation = possible_sols[random.randint(0,8)]
    to_mutate = random.randint(0,26)
    solution[0][to_mutate] = mutation[0]
    solution[1][to_mutate] = mutation[1]
    return solution
def crossover(goodpop, crossovers):
    to_cross = np.random.choice([i for i in range(len(goodpop))], size=crossovers, replace=False)
    divider = np.random.randint(1,25)
    swap = np.random.choice([i for i in range(crossovers)], size=crossovers, replace=False)
    return np.array([[np.concatenate((goodpop[to_cross[i]][0][:divider],goodpop[to_cross[swap[i]]][0][divider:])),np.concatenate((goodpop[to_cross[i]][1][:divider],goodpop[to_cross[swap[i]]][1][divider:]))] for i in range(crossovers)])
def select(solutions, values, numselect):
    selection = np.argsort(values)[0:numselect]
    return solutions[selection], values[selection]
def makefeasible(population):
    for sol in population:
        sol[0][13] = 0; sol[1][13] = 0

def time_rcost():
    starttime = time.time()
    mdl=Tank()
    app = SampleApproach(mdl)
    endclasses, mdlhists = propagate.approach(mdl, app, staged=True)
    rescost = rd.process.totalcost(endclasses)
    return time.time() - starttime

if __name__=="__main__":
    #pool=mp.Pool(5)
    #result, args, fhist, thist, xdhist = alternating_opt(pool=pool)
    #result, args, bestfhist, bestxdhist = bilevel_opt(pool=pool)
    from fmdtools.faultsim.search import ProblemInterface
    
    # mdl= Tank()
    # prob = ProblemInterface("res_problem", mdl, staged=True)
    # app = SampleApproach(mdl)
    # prob.add_simulation("res_sim", "multi", app.scenlist)
    # res_vars_i = {param:1 for param,v in mdl.params.items() if param not in ['capacity','turnup']}
    # res_vars = [(var, None) for var in res_vars_i.keys()]
    # prob.add_variables("res_sim", *res_vars, vartype="param")
    # prob.add_objectives("res_sim", "expected cost", objtype="endclass")
    
    # rvar = [*res_vars_i.values()][:27]
    # lvar = [*res_vars_i.values()][27:]
    # prob.f0([*res_vars_i.values()])
    # x_to_rcost(rvar, lvar)
    
    # prob.update_sim_vars("res_sim", newparams={'capacity':21,'turnup':0.5})

    # prob.f0([*res_vars_i.values()])


