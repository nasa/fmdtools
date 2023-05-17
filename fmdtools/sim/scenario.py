# -*- coding: utf-8 -*-
"""
Description: Classes for defining scenarios to simulate. 

Classes:
    - :class:`Injection`:           Defines faults and disturbances to inject at a specific time
    - :class:`Scenario`:            Defines a generic scenario to simulation
    - :class:`SingleFaultScenario`: Defines the scenario of a single fault injected in a function
    - :class:`JointFaultScenario`:  Defines the scenario of multiple faults injected in a function at the same time
    - :class:`NominalScenario`:     Defines the scenario of a model having given parameters at the outset
    - :class:`ParamScenario`:       Defines the scenario of a model having parameters from a given paramfunc

Functions::
    - :func:`create_sequence`:     Creates an overall sequence of Injections from a given sequence of faults and disturbances
"""
from recordclass import dataobject, asdict



class Injection(dataobject, readonly=True, mapping=True):
    faults:         dict={}
    disturbances:   dict={}
    def get(self, entry, fallback):
        if not hasattr(self, entry): return fallback
        else:                        return self[entry] 

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
        if not hasattr(self, entry): return fallback
        else:                        return self[entry] 

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
    sp:         dict={}
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


