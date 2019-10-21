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
    def __init__(self, states={}):
        self._states=states.keys()
        self._initstates=states.copy()
        for state in states.keys():
            setattr(self, state,states[state])
        self.faults=set(['nom'])
        self.time=0.0
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
        return states.copy(), self.faults

#Function superclass 
class fxnblock(block):
    def __init__(self,flownames,flows, states={}, components={}):
        self.type = 'function'
        flowdict=self.makeflowdict(flownames,flows)
        for flow in flowdict.keys():
            setattr(self, flow,flowdict[flow])
        self.components=components
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
        super().__init__(states)
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
        self.time=0
        self.updatefxn(faults=['nom'], time=0)
    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        self.behavior(time)
        self.time=time
        return
        
class component(block):
    def __init__(self,name, states={}):
        self.type = 'component'
        self.name = name
        super().__init__(states)
    def behavior(self,time):
        return 0

#Flow superclass
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
        return attributes.copy()

#Model superclass    
class model(object):
    def __init__(self):
        self.type='model'
        self.flows={}
        self.fxns={}
        self._fxnflows=[]
    def addflow(self,flowname, flowtype, flowdict):
        if type(flowdict) == dict:
            self.flows[flowname]=flow(flowdict, flowtype)
        elif isinstance(flowdict, flow):
            self.flows[flowname] = flowdict
        else: raise Exception('Invalid flow. Must be dict or flow')
    def addfxn(self,name,classobj, flownames, *args):
        flows=self.getflows(flownames)
        if args: self.fxns[name]=classobj(flows,args)
        else: self.fxns[name]=classobj(flows)
        for flowname in flownames:
            self._fxnflows.append((name, flowname))
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

    