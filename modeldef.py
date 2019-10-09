# -*- coding: utf-8 -*-
"""
File name: modeldef.py
Author: Daniel Hulse
Created: October 2019

Description: A module to simplify model definition
"""
import numpy as np

# MAJOR CLASSES
#Function superclass 
class fxnblock(object):
    def __init__(self,flows):
        self.type = 'function'
        for flow in flows.keys():
            setattr(self, flow,flows[flow])
        self.faults=set(['nom'])
    def condfaults(self,time):
        return 0
    def behavior(self,time):
        return 0
    def updatefxn(self,faults=['nom'], time=0): #fxns take faults and time as input
        self.faults.update(faults)  #if there is a fault, it is instantiated in the function
        self.condfaults(time)           #conditional faults and behavior are then run
        self.behavior(time)
        return
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
        self.updatefxn(faults=['nom'], time=0)
        
class component(object):
    def __init__(self,name):
        self.type = 'component'
        self.name = name
        self.faults=set(['nom'])
    def behavior(self,time):
        return 0
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
        self.updatefxn(faults=['nom'], time=0)

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

    