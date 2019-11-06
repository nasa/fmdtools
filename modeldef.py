# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np
import networkx as nx

# MAJOR CLASSES
class block(object):
    def __init__(self, states={}, timely=True):
        self.timely=timely
        self._states=states.keys()
        self._initstates=states.copy()
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        if timely: self.time=0.0
    def hasfault(self,fault):
        return self.faults.intersection(set([fault]))
    def hasfaults(self,faults):
        return self.faults.intersection(set(faults))
    def addfault(self,fault):
        self.faults.update([fault])
    def addfaults(self,faults):
        self.faults.update(faults)
    def replacefault(self, fault_to_replace,fault_to_add):
        self.faults.add(fault_to_add)
        self.faults.remove(fault_to_replace)
    def reset(self):            #reset requires flows to be cleared first
        self.faults.clear()
        self.faults.add('nom')
        for state in self._initstates.keys():
            setattr(self, state,self._initstates[state])
        self.time=0
    def returnstates(self):
        states={}
        for state in self._states:
            states[state]=getattr(self,state)
        return states, self.faults.copy()

#Function superclass 
class fxnblock(block):
    def __init__(self,flownames,flows, states={}, components={},timers={}, timely=True):
        self.type = 'function'
        self.flows=self.makeflowdict(flownames,flows)
        for flow in self.flows.keys():
            setattr(self, flow,self.flows[flow])
        self.components=components
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
        self.timers = timers
        for timername in timers:
            setattr(self, timername, timer(timername))
        super().__init__(states, timely)
    def makeflowdict(self,flownames,flows):
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
        
class component(block):
    def __init__(self,name, states={}, timely=True):
        self.type = 'component'
        self.name = name
        super().__init__(states, timely)
    def behavior(self,time):
        return 0

#Flow superclass
# - replace with attribute dictionary???
class flow(object):
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
        if self.__class__==flow:
            copy = self.__class__(attributes, self.flow)
        else:
            copy = self.__class__()
            for attribute in self._attributes:
                setattr(copy, attribute, getattr(self,attribute))
        return copy

#Model superclass    
class model(object):
    def __init__(self):
        self.type='model'
        self.flows={}
        self.fxns={}
        self.timelyfxns=set()
        self._fxnflows=[]
        self._fxninput={}
    def addflow(self,flowname, flowtype, flowdict):
        if type(flowdict) == dict:
            self.flows[flowname]=flow(flowdict, flowtype)
        elif isinstance(flowdict, flow):
            self.flows[flowname] = flowdict
        else: raise Exception('Invalid flow. Must be dict or flow')
    def addfxn(self,name,classobj, flownames, *args):
        flows=self.getflows(flownames)
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
    def getflows(self,flownames):
        return [self.flows[flowname] for flowname in flownames]
    def constructgraph(self):
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
    def returnstategraph(self, gtype='normal'):
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
            fxnstates[fxnname], fxnmodes[fxnname] = fxn.returnstates()
            if gtype=='normal': del graph.nodes[fxnname]['bipartite']
        nx.set_node_attributes(graph, fxnstates, 'states')
        nx.set_node_attributes(graph, fxnmodes, 'modes')            
        
        return graph
    def returnfaultmodes(self):
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
        copy = self.__class__()
        for flowname, flow in self.flows.items():
            copy.flows[flowname]=flow.copy()
        for fxnname, fxn in self.fxns.items():
            flownames=self._fxninput[fxnname]['flows']
            args=self._fxninput[fxnname]['args']
            flows = copy.getflows(flownames)
            if args:    copy.fxns[fxnname]=fxn.copy(flows, args)
            else:       copy.fxns[fxnname]=fxn.copy(flows)
        _ = copy.constructgraph()
        return copy

#class for model timers (e.g. for conditional faults)    
class timer():
    def __init__(self, name, tstep=1.0):
        self.name=name
        self.time=0
    def t(self):
        return self.time
    def inc(self, tstep):
        self.time+=tstep
    def reset(self):
        self.time=0
        
# mode constructor????
def mode(rate,rcost):
    return {'rate':rate,'rcost':rcost}


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

    