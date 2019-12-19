# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np
import operator
import itertools
import networkx as nx
from scipy.stats import binom

# MAJOR CLASSES
class Block(object):
    def __init__(self, states={}, timely=True):
        self.timely=timely
        self._states=states.keys()
        self._initstates=states.copy()
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        if timely: self.time=0.0
    def assoc_modes(self, modes):
        self.faultmodes=dict.fromkeys(modes)
        for mode in self.faultmodes:
            self.faultmodes[mode]=dict.fromkeys(('dist', 'oppvect', 'rcost'))
            self.faultmodes[mode]['dist'] =     modes[mode][0]
            self.faultmodes[mode]['oppvect'] =  modes[mode][1]
            self.faultmodes[mode]['rcost'] =    modes[mode][2]
    def has_fault(self,fault):
        return self.faults.intersection(set([fault]))
    def has_faults(self,faults):
        return self.faults.intersection(set(faults))
    def add_fault(self,fault):
        self.faults.update([fault])
    def add_faults(self,faults):
        self.faults.update(faults)
    def replace_fault(self, fault_to_replace,fault_to_add):
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
    def reset(self):            #reset requires flows to be cleared first
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
        self.time=0
    def return_states(self):
        states={}
        for state in self._states:
            states[state]=getattr(self,state)
        return states, self.faults.copy()

#Function superclass 
class FxnBlock(Block):
    def __init__(self,flownames,flows, states={}, components={},timers={}, timely=True):
        self.type = 'function'
        self.flows=self.make_flowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow,self.flows[flow])
        self.components=components
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
        self.timers = timers
        for timername in timers:
            setattr(self, timername, Timer(timername))
        super().__init__(states, timely)
    def make_flowdict(self,flownames,flows):
        flowdict={}
        for ind, flowname in enumerate(flownames):
            flowdict[flowname]=flows[ind]
        return flowdict
    def condfaults(self,time):
        return 0
    def behavior(self,time):
        return 0
    def reset(self):            #reset requires flows to be cleared first
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
        for name, component in self.components.items():
            component.reset()
        for timername in self.timers:
            getattr(self, timername).reset()
        if hasattr(self, 'time'): self.time=0.0
        if hasattr(self, 'tstep'): self.tstep=self.tstep
        self.updatefxn(faults=['nom'], time=0)
    def copy(self, newflows, *attr):
        copy = self.__class__(newflows, *attr)
        copy.faults = self.faults.copy()
        for state in self._initstates.keys():
            setattr(copy, state, self._initstates[state])
        if hasattr(self, 'time'): copy.time=self.time
        if hasattr(self, 'tstep'): copy.tstep=self.tstep
        return copy
    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        self.behavior(time)
        self.time=time
        return
        
class Component(Block):
    def __init__(self,name, states={}, timely=True):
        self.type = 'component'
        self.name = name
        super().__init__(states, timely)
    def behavior(self,time):
        return 0

#Flow superclass
# - replace with attribute dictionary???
class Flow(object):
    def __init__(self, attributes, name):
        self.type='flow'
        self.flow=name
        self._initattributes=attributes.copy()
        self._attributes=attributes.keys()
        for attribute in self._attributes:
            setattr(self, attribute, attributes[attribute])
    def reset(self):
        for attribute in self._initattributes:
            setattr(self, attribute, self._initattributes[attribute])
    def status(self):
        attributes={}
        for attribute in self._attributes:
            attributes[attribute]=getattr(self,attribute)
        return attributes
    def copy(self):
        attributes={}
        for attribute in self._attributes:
            attributes[attribute]=getattr(self,attribute)
        if self.__class__==Flow:
            copy = self.__class__(attributes, self.flow)
        else:
            copy = self.__class__()
            for attribute in self._attributes:
                setattr(copy, attribute, getattr(self,attribute))
        return copy

#Model superclass    
class Model(object):
    def __init__(self):
        self.type='model'
        self.flows={}
        self.fxns={}
        self.timelyfxns=set()
        self._fxnflows=[]
        self._fxninput={}
    def add_flow(self,flowname, flowtype, flowdict):
        if type(flowdict) == dict:
            self.flows[flowname]=Flow(flowdict, flowtype)
        elif isinstance(flowdict, Flow):
            self.flows[flowname] = flowdict
        else: raise Exception('Invalid flow. Must be dict or flow')
    def add_fxn(self,name,classobj, flownames, *args):
        flows=self.get_flows(flownames)
        if args: 
            self.fxns[name]=classobj(flows,args)
            self._fxninput[name]={'flows': flownames, 'args': args}
        else: 
            self.fxns[name]=classobj(flows)
            self._fxninput[name]={'flows': flownames, 'args': []}
        for flowname in flownames:
            self._fxnflows.append((name, flowname))
        if self.fxns[name].timely: self.timelyfxns.update([name])
        self.fxns[name].tstep=self.tstep
    def get_flows(self,flownames):
        return [self.flows[flowname] for flowname in flownames]
    def construct_graph(self):
        self.bipartite=nx.Graph()
        self.bipartite.add_nodes_from(self.fxns, bipartite=0)
        self.bipartite.add_nodes_from(self.flows, bipartite=1)
        self.bipartite.add_edges_from(self._fxnflows)
        self.multgraph = nx.projected_graph(self.bipartite, self.fxns,multigraph=True)
        self.graph = nx.projected_graph(self.bipartite, self.fxns)
        attrs={}
        #do we still need to do this for the objects? maybe not--I don't think we use the info anymore
        for edge in self.graph.edges:
            midedges=list(self.multgraph.subgraph(edge).edges)
            flows= [midedge[2] for midedge in midedges]
            flowdict={}
            for flow in flows:
                flowdict[flow]=self.flows[flow]
            attrs[edge]=flowdict
        nx.set_edge_attributes(self.graph, attrs)
        
        nx.set_node_attributes(self.graph, self.fxns, 'obj')
        #self.graph=nx.DiGraph()
        #self.graph.add_nodes_from(self.fxn)
        #self.graph=
        return self.graph
    def reset(self):
        for flowname, flow in self.flows.items():
            flow.reset()
        for fxnname, fxn in self.fxns.items():
            fxn.reset()
    def return_stategraph(self, gtype='normal'):
        if gtype=='normal':
            graph=nx.projected_graph(self.bipartite, self.fxns)
        elif gtype=='bipartite':
            graph=self.bipartite.copy()
        edgevals={}
        fxnmodes={}
        fxnstates={}
        flowstates={}
        if gtype=='normal': #set edge values for normal graph
            for edge in graph.edges:
                midedges=list(self.multgraph.subgraph(edge).edges)
                flows= [midedge[2] for midedge in midedges]
                flowdict={}
                for flow in flows: 
                    flowdict[flow]=self.flows[flow].status()
                edgevals[edge]=flowdict
            nx.set_edge_attributes(graph, edgevals) 
        elif gtype=='bipartite': #set flow node values for bipartite graph
            for flowname, flow in self.flows.items():
                flowstates[flowname]=flow.status()
            nx.set_node_attributes(graph, flowstates, 'states')
        #set node values for functions
        for fxnname, fxn in self.fxns.items():
            fxnstates[fxnname], fxnmodes[fxnname] = fxn.return_states()
            if gtype=='normal': del graph.nodes[fxnname]['bipartite']
        nx.set_node_attributes(graph, fxnstates, 'states')
        nx.set_node_attributes(graph, fxnmodes, 'modes')            
        
        return graph
    def return_faultmodes(self):
        modes={}
        modeprops={}
        for fxnname, fxn in self.fxns.items():
            ms = [m for m in fxn.faults.copy() if m!='nom']
            if ms: 
                modeprops[fxnname] = {}
                modes[fxnname] = ms
            for mode in ms:
                if mode!='nom': 
                    modeprops[fxnname][mode] = fxn.faultmodes[mode]
        return modes, modeprops
    def copy(self):
        copy = self.__class__(params=self.params)
        for flowname, flow in self.flows.items():
            copy.flows[flowname]=flow.copy()
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            args=self._fxninput[fxnname]['args']
            flows = copy.get_flows(flownames)
            if args:    copy.fxns[fxnname]=fxn.copy(flows, args)
            else:       copy.fxns[fxnname]=fxn.copy(flows)
        _ = copy.construct_graph()
        return copy

#class for model timers (e.g. for conditional faults)    
class Timer():
    def __init__(self, name, tstep=1.0):
        self.name=name
        self.time=0
    def t(self):
        return self.time
    def inc(self, tstep):
        self.time+=tstep
    def reset(self):
        self.time=0

class SampleApproach():
    def __init__(self, mdl, faults='all', jointfaults={'faults':'None'}, condprob=0.1, sampparams={}, defaultsamp={'samp':'evenspacing','numpts':1}):
        self.phases = mdl.phases
        self.tstep = mdl.tstep
        self.init_modelist(mdl,faults, jointfaults)
        self.init_rates(jointfaults=jointfaults)
        self.create_sampletimes(sampparams, defaultsamp)
        self.create_scenarios(jointfaults)
    def init_modelist(self,mdl, faults, jointfaults={'faults':'None'}):
        if faults=='all':
            self.fxnrates=dict.fromkeys(mdl.fxns)
            self._fxnmodes={}
            for fxnname, fxn in  mdl.fxns.items():
                for mode, params in fxn.faultmodes.items():
                    self._fxnmodes[fxnname, mode]=params
                self.fxnrates[fxnname]=fxn.failrate
        else:
            self.fxnrates=dict.fromkeys([fxnname for (fxnname, mode) in faults])
            self._fxnmodes={}
            for fxnname, mode in faults:
                self._fxnmodes[fxnname, mode]=mdl.fxns[fxnname].faultmodes[mode]
                self.fxnrates[fxnname]=mdl.fxns[fxnname].failrate
        if type(jointfaults['faults'])==int:
            self.jointmodes=[]
            for numjoint in range(2, jointfaults['faults']+1):
                jointmodes = list(itertools.combinations(self._fxnmodes, numjoint))
                if not jointfaults.get('jointfuncs', False): 
                    jointmodes = [jm for jm in jointmodes if not any([jm[i-1][0] ==j[0] for i in range(1, len(jm)) for j in jm[i:]])]
                self.jointmodes = self.jointmodes + jointmodes
        elif type(jointfaults['faults'])==list: self.jointmodes = jointfaults['faults']
    def init_rates(self, jointfaults={'faults':'None'}):
        self.rates=dict.fromkeys(self._fxnmodes)
        self.rates_timeless=dict.fromkeys(self._fxnmodes)
        for (fxnname, mode) in self._fxnmodes:
            self.rates[fxnname, mode]=dict.fromkeys(self.phases)
            self.rates_timeless[fxnname, mode]=dict.fromkeys(self.phases)
            for ind, (phase, times) in enumerate(self.phases.items()):
                opp = self._fxnmodes[fxnname, mode]['oppvect'][ind]/sum(self._fxnmodes[fxnname, mode]['oppvect']) #forces to be a conditional probability
                dist = self._fxnmodes[fxnname, mode]['dist']
                dt = float(times[1]-times[0])
                self.rates[fxnname, mode][phase] = self.fxnrates[fxnname]*opp*dist*dt
                self.rates_timeless[fxnname, mode][phase] = self.fxnrates[fxnname]*opp*dist
    def create_sampletimes(self, params={}, default={'samp':'evenspacing','numpts':1}):
        self.sampletimes=dict.fromkeys(self.phases.keys())
        self.weights={fxnmode:dict.fromkeys(rate) for fxnmode,rate in self.rates.items()}
        self.sampparams={}
        for phase, times in self.phases.items():
            possible_phasetimes = list(np.arange(times[0], times[1], self.tstep))
            for fxnmode in self.rates:
                param = params.get((fxnmode,phase), default)
                self.sampparams[fxnmode, phase] = param
                if param['samp']=='likeliest':
                    weights=[]
                    if self.rates[fxnmode][phase] == max(list(self.rates[fxnmode].values())):
                        phasetimes = [round(np.quantile(possible_phasetimes, 0.5)/self.tstep)*self.tstep]
                    else: phasetimes = []
                else: 
                    pts, weights = self.select_points(param, [pt for pt, t in enumerate(possible_phasetimes)])
                    phasetimes = [possible_phasetimes[pt] for pt in pts]
                self.add_phasetimes(fxnmode, phase, phasetimes, weights=weights)
    def select_points(self, param, possible_pts):
        weights=[]
        if param['samp']=='fullint': pts = possible_pts
        elif param['samp']=='evenspacing':
            if param['numpts']+2 > len(possible_pts): pts = possible_pts
            else: pts= [int(round(np.quantile(possible_pts, p/(param['numpts']+1)))) for p in range(param['numpts']+2)][1:-1]
        elif param['samp']=='quadrature':
            quantiles = param['quad'].points/2 +0.5
            if len(quantiles) > len(possible_pts): pts = possible_pts
            else: 
                pts= [int(round(np.quantile(possible_pts, q))) for q in quantiles]
                weights=param['quad'].weights/2
        elif param['samp']=='randtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: pts= [possible_pts.pop(np.random.randint(len(possible_pts))) for i in range(min(param['numpts'], len(possible_pts)))]
        elif param['samp']=='symrandtimes':
            if param['numpts']>=len(possible_pts): pts = possible_pts
            else: 
                if len(possible_pts) %2 >0:  pts = [possible_pts.pop(int(np.floor(len(possible_pts)/2)))]
                else: pts = [] 
                possible_pts_halved = np.reshape(possible_pts, (2,int(len(possible_pts)/2)))
                possible_pts_halved[1] = np.flip(possible_pts_halved[1])
                possible_inds = [i for i in range(int(len(possible_pts)/2))]
                inds = [possible_inds.pop(np.random.randint(len(possible_inds))) for i in range(min(int(np.floor(param['numpts']/2)), len(possible_inds)))]
                pts= pts+ [possible_pts_halved[half][ind] for half in range(2) for ind in inds ]
                pts.sort()
        else: print("invalid option: ", param)
        if not any(weights): weights = [1/len(pts) for t in pts]
        return pts, weights
    def add_phasetimes(self, fxnmode, phase, phasetimes, weights=[]):
        if phasetimes:
            if not self.weights[fxnmode].get(phase): self.weights[fxnmode][phase] = {t: 1/len(phasetimes) for t in phasetimes}
            for (ind, time) in enumerate(phasetimes):
                if self.rates[fxnmode][phase]==0.0:
                    break
                if not self.sampletimes.get(phase): 
                    self.sampletimes[phase] = {time:[]}
                if self.sampletimes[phase].get(time): self.sampletimes[phase][time] = self.sampletimes[phase][time] + [fxnmode]
                else: self.sampletimes[phase][time] = [fxnmode]
                if any(weights): self.weights[fxnmode][phase][time] = weights[ind]
                else:       self.weights[fxnmode][phase][time] = 1/len(phasetimes)
    def create_nomscen(self, mdl):
        nomscen={'faults':{},'properties':{}}
        for fxnname in mdl.fxns:
            nomscen['faults'][fxnname]='nom'
        nomscen['properties']['time']=0.0
        nomscen['properties']['type']='nominal'
        nomscen['properties']['name']='nominal'
        nomscen['properties']['weight']=1.0
        return nomscen
    def create_scenarios(self,jointfaults): #need to add to create scenarios to run
        self.scenlist=[]
        self.times = []
        self.scenids = {}
        if not jointfaults:
            for phase, samples in self.sampletimes.items():
                if samples:
                    for time, faultlist in samples.items():
                        self.times+=[time]
                        for fxnmode in faultlist:
                            if self.sampparams[fxnmode, phase]['samp']=='maxlike':    
                                rate = sum(self.rates[fxnmode].values())
                            else: 
                                rate = self.rates[fxnmode][phase] * self.weights[fxnmode][phase][time]
                            name = fxnmode[0]+' '+fxnmode[1]+', t='+str(time)
                            scen={'faults':{fxnmode[0]:fxnmode[1]}, 'properties':{'type': 'single-fault', 'function': fxnmode[0],\
                                  'fault': fxnmode[1], 'rate': rate, 'time': time, 'name': name}}
                            self.scenlist=self.scenlist+[scen]
                            if self.scenids.get((fxnmode, phase)): self.scenids[fxnmode, phase] = self.scenids[fxnmode, phase] + [name]
                            else: self.scenids[fxnmode, phase] = [name]
        self.times.sort()
    def prune_scenarios(self,endclasses,samptype='piecewise', threshold=0.1, sampparam={'samp':'evenspacing','numpts':1}):
        newscenids = dict.fromkeys(self.scenids.keys())
        newsampletimes = {key:{} for key in self.sampletimes.keys()}
        newweights = {fault:dict.fromkeys(phasetimes) for fault, phasetimes in self.weights.items()}
        for modeinphase in self.scenids:
            costs= np.array([endclasses[scen]['cost'] for scen in self.scenids[modeinphase]])
            if samptype=='bestpt':
                errs = abs(np.mean(costs) - costs)
                mins = np.where(errs == errs.min())[0]
                pts=[mins[int(len(mins)/2)]]
                weights=[1]
            elif samptype=='piecewise':
                partlocs=[0, len(list(np.arange(self.phases[modeinphase[1]][0], self.phases[modeinphase[1]][1], self.tstep)))]
                reset=False
                for ind, cost in enumerate(costs[1:-1]): # find where fxn is no longer linear
                    if reset==True:
                        reset=False
                        continue
                    if abs(((cost-costs[ind]) - (costs[ind+2]-cost))/(costs[ind+2]-cost + 0.0001)) > threshold:  
                        partlocs = partlocs + [ind+2]
                        reset=True
                partlocs.sort()
                pts=[]
                weights=[]
                for (ind_part, partloc) in enumerate(partlocs[1:]): # add points in each section
                    partition = [i for i in range(partlocs[ind_part], partloc)]
                    part_pts, part_weights = self.select_points(sampparam, partition)
                    pts = pts + part_pts
                    overall_part_weight =  (partloc-partlocs[ind_part])/(partlocs[-1]-partlocs[0])
                    weights = weights + list(np.array(part_weights)*overall_part_weight)
                pts.sort()
            newscenids[modeinphase] =  [self.scenids[modeinphase][pt] for pt in pts]
            newscens = [scen for scen in self.scenlist if scen['properties']['name'] in newscenids[modeinphase]]
            newweights[modeinphase[0]][modeinphase[1]] = {scen['properties']['time']:weights[ind] for (ind, scen) in enumerate(newscens)}
            newscenids[modeinphase] =  [self.scenids[modeinphase][pt] for pt in pts]
            for newscen in newscens:
                if not newsampletimes[modeinphase[1]].get(newscen['properties']['time']):
                    newsampletimes[modeinphase[1]][newscen['properties']['time']] = [modeinphase[0]]
                else:
                    newsampletimes[modeinphase[1]][newscen['properties']['time']] = newsampletimes[modeinphase[1]][newscen['properties']['time']] + [modeinphase[0]]
        self.scenids = newscenids
        self.weights = newweights
        self.sampletimes = newsampletimes
        self.create_scenarios([])
        self.sampparams={key:{'samp':'pruned '+samptype} for key in self.sampparams}
    def list_modes(self):
        return [(fxn, mode) for fxn, mode in self._fxnmodes.keys()]
    def list_moderates(self):
        return {(fxn, mode): sum(self.rates[fxn,mode].values()) for (fxn, mode) in self.rates.keys()}

        
# mode constructor????
class Mode():
    def __init__(self, name, baserate, phases, modifiers=[], rateunits = 360):
        self.name=name
        self.baserate=baserate
        self.phases = phases
        if not modifiers:               self.modifiers = [1]*len(phases)
        elif type(modifiers)== list:    self.modifiers = {phase: modifiers[i] for i,phase in enumerate(modifiers)}
        else:                           self.modifiers = modifiers
        self.rateunits = rateunits
    def phaserates(self):
        return {phase:self.baserate*self.modifiers[phase] for phase in self.phases}
    def ratesperuse(self):
        return {phase:self.baserate*self.modifiers[phase]*np.diff(inter)/self.rateunits for phase, inter in self.phases.items()}
    def totrateperuse(self):
        return sum(self.ratesperuse().values())
    def ratespertime(self):
        times = np.array(list(self.phases.values()))
        tottime = np.max(times) - np.min(times)
        return {phase:self.baserate*self.modifiers[phase]*np.diff(inter)/(tottime*self.rateunits) for phase, inter in self.phases.items()}
    def totratepertime(self):
        return sum(self.ratespertime().values())
    def expnumsfromuses(self, uses):
        userates = self.ratesperuse()
        return {phase:userate*uses for phase, userate in userates.items()}
    def totnumfromuses(self, uses):
        return sum(self.expnumsfromuses.values())
    def expnumsfromlifetime(self,lifetime):
        timerates=self.ratespertime()
        return {phase:timerate*lifetime for phase, timerate in timerates.items()}
    def totnumfromlifetime(self,lifetime):
        return sum(self.expnumsfromlifetime().values())
    def probspertime(self):
        return {phase: 1-np.exp(-rate) for phase, rate in self.ratespertime().items()}
    def totprobpertime(self):
        return sum(self.probspertime().values())
    def totprobsfromtime(self,lifetime):
        return {phase: 1-np.exp(-rate) for phase, rate in self.expnumsfromlifetime(lifetime).items()}
    def totprobfromtime(self, lifetime):
        return 1-np.exp(self.totnumfromlifetime(lifetime))
    def probsperuses(self):
        return {phase: 1-np.exp(-rate) for phase, rate in self.ratesperuse().items()}
    def totprobsfromuses(self, uses):
        return {phase: 1-binom.pdf(1,uses, prob) for phase, prob in self.probsperuses().items()}
    def totprobfromuses(self, uses):
        return sum(self.totprobsfromuses(uses).values())
        
    
    
def phases(times, names=[]):
    if not names: names = range(len(times)-1)
    return {names[i]:[times[i], times[i+1]] for (i, _) in enumerate(times) if i < len(times)-1}

#def mode(rate,rcost):
#    return {'rate':rate,'rcost':rcost}


# USEFUL FUNCTIONS FOR MODEL CONSTRUCTION
#m2to1
# multiplies a list of numbers which may take on the values infinity or zero
# in deciding if num is inf or zero, the earlier values take precedence
def m2to1(x):
    if np.size(x)>2:
        x=[x[0], m2to1(x[1:])]
    if x[0]==np.inf:
        y=np.inf
    elif x[1]==np.inf:
        if x[0]==0.0:
            y=0.0
        else:
            y=np.inf
    else:
        y=x[0]*x[1]
    return y

#trunc
# truncates a value to 2 (useful if behavior unchanged by increases)
def trunc(x):
    if x>2.0:
        y=2.0
    else:
        y=x
    return y

#truncn
# truncates a value to n (useful if behavior unchanged by increases)
def truncn(x, n):
    if x>n:
        y=n
    else:
        y=x
    return y

    