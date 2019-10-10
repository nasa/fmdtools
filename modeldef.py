# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np

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
    def __init__(self,flows, states={}, components={}):
        self.type = 'function'
        for flow in flows.keys():
            setattr(self, flow,flows[flow])
        self.components=components
        for cname in components:
            self.faultmodes.update(components[cname].faultmodes)
        super().__init__(states)
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

    