# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:06:53 2023

@author: dhulse
"""
from recordclass import dataobject, asdict
import warnings
import numpy as np


class Injection(dataobject, readonly=True, mapping=True):
    faults:         dict={}
    disturbances:   dict={}
    def get(self, entry, fallback):
        if entry not in self: return fallback
        else:                 return self[entry] 

def create_sequence(faultseq={}, disturbances={}):
    times = {*faultseq, *disturbances}
    return {t:Injection(faults=faultseq.get(t, {}), 
                        disturbances=disturbances.get(t, {})) 
            for t in times}
    

class Scenario(dataobject, readonly=True, mapping=True):
    sequence:   dict = dict()
    rate:       float = 1.0
    name:       str = "nominal"
    times:      tuple = ()
    time:       float = 0.0
    def copy_with(self, **kwargs):
        existing_kwargs = asdict(self)
        return self.__class__(**{**existing_kwargs, **kwargs})
    def get(self, entry, fallback):
        if entry not in self: return fallback
        else:                 return self[entry]

class SingleFaultScenario(Scenario):
    function:   str=''
    fault:      str=''
    time:       float = 0.0
    name:       str = 'faulty'

class JointFaultScenario(Scenario):
    joint_faults:   int=1
    functions:      tuple=()
    modes:          tuple=()
    name:           str = 'faulty'
    time:           float = 0.0

class NominalScenario(Scenario, readonly=True):
    p:          dict={}
    r:          dict={}
    inputparams:    dict = {}
    rangeid:    str=''
    prob:       float=1.0
    
    
class ParamScenario(NominalScenario):
    paramfunc:      callable 
    fixedargs:      tuple = ()
    fixedkwargs:    dict = {}
    prob:       float=1.0
    
    



a = Scenario()

b = SingleFaultScenario()

seq = create_sequence(faultseq={1:"fault"})


